use crate::questions::QuestionSet;
use crate::{ann, corpus, run_with_questions, synthetic, MemStore};
use quickdb_api::{Error, Metric, Result, Store};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

struct Args {
    command: String,
    options: BTreeMap<String, String>,
}
impl Args {
    fn required(&self, key: &str) -> Result<&str> {
        self.options
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| Error::Schema(format!("missing --{key}")))
    }
    fn number(&self, key: &str) -> Result<usize> {
        self.required(key)?
            .parse()
            .map_err(|_| Error::Schema(format!("--{key} must be a positive integer")))
    }
}
fn execute<S: Store>(args: &Args) -> Result<()> {
    if args.command == "conformance" {
        let path = std::env::var_os("QUICKDB_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join(format!(
                "quickdb-conformance-{}-{}",
                args.required("backend")?,
                std::process::id()
            ));
        if path.exists() {
            return Err(Error::Schema("conformance directory already exists".into()));
        }
        let failures = quickdb_api::conformance::run_all::<S>(&path);
        let result = serde_json::json!({"backend":args.required("backend")?, "conformance": if failures.is_empty() { "pass" } else { "fail" }, "failures": failures});
        println!("{result}");
        if result["conformance"] == "fail" {
            return Err(Error::Backend("conformance failed".into()));
        }
        return Ok(());
    }
    let out = Path::new(args.required("out")?);
    let data_path = Path::new(args.required("data-dir")?);
    if out.exists() {
        return Err(Error::Schema(
            "--out already exists, refusing to replace evidence".into(),
        ));
    }
    let start = std::time::Instant::now();
    let data = match args.command.as_str() {
        "synthetic" => synthetic(args.number("rows")?, args.number("dim")?)?,
        "corpus" => corpus(Path::new(args.required("corpus")?))?,
        "ann" | "ab" => ann(
            Path::new(args.required("base")?),
            Path::new(args.required("query")?),
            Path::new(args.required("truth")?),
            args.number("limit")?,
            args.number("queries")?,
            match args.required("metric")? {
                "l2" => Metric::L2,
                "cosine" => Metric::Cosine,
                _ => return Err(Error::Schema("--metric must be l2 or cosine".into())),
            },
        )?,
        _ => {
            return Err(Error::Schema(
                "command must be conformance, synthetic, corpus, ann or ab".into(),
            ))
        }
    };
    let prepare_seconds = start.elapsed().as_secs_f64();
    let questions = match args.options.get("questions") {
        Some(dir) if args.command == "corpus" => Some(QuestionSet::load(
            Path::new(args.required("corpus")?),
            Path::new(dir),
            data.dim,
        )?),
        Some(_) => {
            return Err(Error::Schema(
                "--questions applies to the corpus command only".into(),
            ))
        }
        None => None,
    };
    if args.command == "ab" {
        let comparison = args.required("comparison")?;
        if !["aa", "ab"].contains(&comparison) {
            return Err(Error::Schema("--comparison must be aa or ab".into()));
        }
        let report = crate::ab::run::<S>(
            &data,
            data_path,
            args.number("pairs")?,
            args.number("trials")?,
            comparison == "ab",
            args.number("ef")?,
        )?;
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out)?;
        serde_json::to_writer(&mut output, &report).map_err(|e| Error::Backend(e.to_string()))?;
        writeln!(output)?;
        output.sync_all()?;
        return Ok(());
    }
    let mut report = run_with_questions::<S>(&data, data_path, questions.as_ref())?;
    report
        .phases_seconds
        .insert("prepare_dataset".into(), prepare_seconds);
    let json = serde_json::to_string(&report).map_err(|e| Error::Backend(e.to_string()))?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)?;
    writeln!(output, "{json}")?;
    output.sync_all()?;
    println!("{json}");
    Ok(())
}
fn main_result<S: Store>(extra: &str) -> Result<()> {
    let mut input = std::env::args().skip(1);
    let command = input.next().ok_or_else(|| Error::Schema("usage: quickdb-bench <conformance|synthetic|corpus|ann|ab> --backend <mem|native|cozo> [options]".into()))?;
    let mut options = BTreeMap::new();
    while let Some(key) = input.next() {
        let name = key
            .strip_prefix("--")
            .ok_or_else(|| Error::Schema(format!("expected option, got {key}")))?;
        if ![
            "backend",
            "rows",
            "dim",
            "corpus",
            "data-dir",
            "out",
            "queries",
            "metric",
            "base",
            "query",
            "truth",
            "limit",
            "comparison",
            "pairs",
            "trials",
            "ef",
            "questions",
        ]
        .contains(&name)
        {
            return Err(Error::Schema(format!("unknown option {key}")));
        }
        let value = input
            .next()
            .filter(|v| !v.starts_with("--"))
            .ok_or_else(|| Error::Schema(format!("missing value for {key}")))?;
        if options.insert(name.into(), value).is_some() {
            return Err(Error::Schema(format!("duplicate {key}")));
        }
    }
    let args = Args { command, options };
    // The only place that knows the candidate backend types.
    match args.required("backend")? {
        "mem" => execute::<MemStore>(&args),
        b if b == extra => execute::<S>(&args),
        b => Err(Error::Schema(format!(
            "backend {b:?} is not in this binary"
        ))),
    }
}
/// Command line of the harness for one binary: backend `mem` plus the backend `S` under the name `extra`.
/// Each candidate backend builds its own binary, because two crates that both link SQLite cannot share one.
pub fn main_with<S: Store>(extra: &str) {
    if let Err(e) = main_result::<S>(extra) {
        eprintln!("quickdb-bench: {e}");
        std::process::exit(1);
    }
}
