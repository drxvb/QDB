//! Power loss: record every file operation of one `dur_writer` run with `tools/fsrecord.c`, build
//! the directory a power cut at chosen records could leave (`powerloss_common`), and check each
//! image with the kill tests' checker (`durability_common::verify`) against the same promise:
//! everything flushed (or, with `sync_each_write`, every returned call) survives, nothing partly.
//!
//! The recorder is an `LD_PRELOAD` shim, so these tests run on Linux only. Elsewhere the harness
//! reports them as ignored with the reason. `--cfg quickdb_sim_not_linux` in `RUSTFLAGS` marks
//! them the same way on Linux, to check that report. On Linux without a C compiler the tests
//! fail rather than pass without running; skip them with `-- --skip powerloss`.
// The simulation cfg is set only by hand through RUSTFLAGS; the manifest does not declare it.
#![allow(unexpected_cfgs)]
#[path = "durability_common/mod.rs"]
mod common;
#[path = "powerloss_common/mod.rs"]
mod pl;

use common::*;
use pl::{Fs, Mode, Rec};
use quickdb_api::conformance::Lcg;
use std::path::{Path, PathBuf};
use std::process::Command;

fn example(name: &str) -> PathBuf {
    let exe = std::env::current_exe().expect("test binary path");
    exe.parent()
        .and_then(Path::parent)
        .map(|p| p.join("examples").join(name))
        .expect("target layout")
}

fn data_root() -> PathBuf {
    std::env::var_os("QUICKDB_DATA").map_or_else(std::env::temp_dir, PathBuf::from)
}

/// Builds the recorder shim once per test binary: tests run in parallel, and a second build
/// would rewrite the library while a writer started by another test is loading it.
/// Panics when no C compiler builds it, so that a test that cannot run does not report `ok`.
fn shim() -> PathBuf {
    static SHIM: std::sync::OnceLock<Result<PathBuf, String>> = std::sync::OnceLock::new();
    match SHIM.get_or_init(build_shim) {
        Ok(p) => p.clone(),
        Err(e) => panic!("POWERLOSS cannot run: {e}; skip with `cargo test -- --skip powerloss`"),
    }
}

fn build_shim() -> Result<PathBuf, String> {
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fsrecord.so");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/fsrecord.c");
    let compilers: Vec<String> = std::env::var("CC")
        .into_iter()
        .chain(["cc".to_string(), "gcc".to_string()])
        .collect();
    for c in &compilers {
        let ok = Command::new(c)
            .args(["-shared", "-fPIC", "-O2", "-Wall", "-o"])
            .arg(&out)
            .arg(&src)
            .arg("-ldl")
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            return Ok(out);
        }
    }
    Err(format!(
        "none of {compilers:?} could build {}",
        src.display()
    ))
}

const LINUX_ONLY: &str = "power loss: the fsrecord.c recorder is an LD_PRELOAD shim, Linux only";

/// Runs the writer under the recorder; returns the parsed log.
fn record(so: &Path, cfg: &Config, nosync: bool) -> Vec<Rec> {
    let log = cfg.dir.with_extension("fslog");
    let _ = std::fs::remove_dir_all(&cfg.dir);
    let _ = std::fs::remove_file(&log);
    let mut cmd = Command::new(example("dur_writer"));
    cmd.args(cfg.to_args())
        .env("LD_PRELOAD", so)
        .env("FSREC_DIR", &cfg.dir)
        .env("FSREC_LOG", &log)
        .stdout(std::process::Stdio::piped());
    if nosync {
        cmd.env("FSREC_NOSYNC", "1");
    }
    let out = cmd.output().expect("run dur_writer");
    assert!(
        out.status.success(),
        "writer failed: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let bytes = std::fs::read(&log).expect("recorder log");
    pl::parse(&bytes).expect("parse recorder log")
}

#[derive(Clone, Copy, PartialEq)]
enum Moment {
    MidRows,
    MidVectors,
    InFlush,
    InSnapshot,
}

struct Scenario {
    name: &'static str,
    moment: Moment,
    sync: bool,
    replace: bool,
}

const SCENARIOS: [Scenario; 8] = [
    Scenario {
        name: "mid-rows",
        moment: Moment::MidRows,
        sync: false,
        replace: false,
    },
    Scenario {
        name: "mid-vectors",
        moment: Moment::MidVectors,
        sync: false,
        replace: false,
    },
    Scenario {
        name: "in-flush",
        moment: Moment::InFlush,
        sync: false,
        replace: false,
    },
    Scenario {
        name: "in-snapshot",
        moment: Moment::InSnapshot,
        sync: false,
        replace: false,
    },
    Scenario {
        name: "compaction",
        moment: Moment::InFlush,
        sync: false,
        replace: true,
    },
    Scenario {
        name: "sync-mid-rows",
        moment: Moment::MidRows,
        sync: true,
        replace: false,
    },
    Scenario {
        name: "sync-mid-vectors",
        moment: Moment::MidVectors,
        sync: true,
        replace: false,
    },
    Scenario {
        name: "sync-compaction",
        moment: Moment::MidVectors,
        sync: true,
        replace: true,
    },
];

fn scenario_cfg(sc: &Scenario, dir: &Path, seed: u64) -> Config {
    let mut cfg = Config::new(dir);
    cfg.seed = seed;
    cfg.sync = sc.sync;
    cfg.replace = sc.replace;
    cfg.batch = if sc.sync { 50 } else { 200 };
    cfg.flush_every = if sc.replace { 2 } else { 3 };
    cfg.max_batches = 2 * cfg.flush_every + 1;
    cfg
}

/// Crash points (number of records replayed) inside the scenario's moment, as ranges.
fn windows(recs: &[Rec], lines: &[(usize, String)], m: Moment) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (j, (a, l)) in lines.iter().enumerate() {
        let opens = match m {
            Moment::MidRows => l.starts_with("OPENED") || l.ends_with_edges(),
            Moment::MidVectors => l.split_whitespace().nth(2) == Some("rows"),
            Moment::InFlush | Moment::InSnapshot => l.starts_with("FLUSHING"),
        };
        let Some((b, _)) = lines.get(j + 1) else {
            continue;
        };
        if !opens {
            continue;
        }
        let mut b = *b;
        if m == Moment::InSnapshot {
            if let Some(r) = (a + 1..b).find(|&i| recs[i].op == pl::RENAME) {
                b = r;
            }
        }
        if b > *a {
            out.push((a + 1, b));
        }
    }
    out
}

trait EdgesLine {
    fn ends_with_edges(&self) -> bool;
}
impl EdgesLine for String {
    fn ends_with_edges(&self) -> bool {
        self.split_whitespace().nth(2) == Some("edges")
    }
}

fn pick(rng: &mut Lcg, w: &[(usize, usize)], n: usize) -> Vec<usize> {
    let total: usize = w.iter().map(|(a, b)| b - a + 1).sum();
    let mut out: Vec<usize> = (0..n)
        .filter_map(|_| {
            let mut k = rng.next_u32() as usize % total.max(1);
            w.iter().find_map(|(a, b)| {
                if k <= b - a {
                    Some(a + k)
                } else {
                    k -= b - a + 1;
                    None
                }
            })
        })
        .collect();
    out.sort_unstable();
    out
}

#[derive(Default)]
struct Tally {
    images: u64,
    lost_flushed: u64,
    partial: u64,
    panics: u64,
    other: u64,
    failures: Vec<String>,
}

/// Builds and checks one image; returns a failure class and message.
#[allow(clippy::too_many_arguments)]
fn check(
    cfg: &Config,
    fs: &Fs,
    recs: &[Rec],
    lines: &[(usize, String)],
    c: usize,
    mode: Mode,
    rng: &mut Lcg,
    img: &Path,
) -> Option<(&'static str, String)> {
    let (files, how) = fs.image(mode, rng);
    pl::write_image(img, &files).expect("write image");
    let (committed, flushed) = pl::progress(lines, c);
    let must = if cfg.sync { committed } else { [flushed; 3] };
    let top = committed.iter().flatten().max().map_or(0, |n| n + 1);
    let mut icfg = cfg.clone();
    icfg.dir = img.to_path_buf();
    let got = std::panic::catch_unwind(|| verify(&icfg, top + 2, must));
    let err = match got {
        Err(_) => Some(("panic", "checker panicked".to_string())),
        Ok(Ok(_)) => None,
        Ok(Err(_)) if !files.contains_key("quickdb.sqlite") && must.iter().all(Option::is_none) => {
            None
        }
        Ok(Err(e)) if e.contains("was durable") => Some(("lost", e)),
        // Before the first flush the table definition itself is not yet promised.
        Ok(Err(e)) if e.starts_with("not found") && must.iter().all(Option::is_none) => None,
        Ok(Err(e)) if e.starts_with("not found") => Some(("lost", e)),
        Ok(Err(e)) if e.starts_with("open after kill") && must.iter().any(Option::is_some) => {
            Some(("lost", e))
        }
        Ok(Err(e))
            if ["partly", "prefix", "order", "counts", "wrong"]
                .iter()
                .any(|k| e.contains(k)) =>
        {
            Some(("partial", e))
        }
        Ok(Err(e)) => Some(("other", e)),
    };
    let _ = std::fs::remove_dir_all(img);
    err.map(|(k, e)| {
        let tail: Vec<String> = recs[c.saturating_sub(30)..c]
            .iter()
            .enumerate()
            .map(|(i, r)| format!("    #{} {}", c.saturating_sub(30) + i, r.describe()))
            .collect();
        (
            k,
            format!(
                "crash after {c} records, mode {}, must={must:?}: {e}\n  unsynced ops (+landed -lost ~torn):{how}\n  last records:\n{}",
                mode.name(),
                tail.join("\n")
            ),
        )
    })
}

fn tally(t: &mut Tally, r: Option<(&'static str, String)>, label: &str) {
    t.images += 1;
    if let Some((k, msg)) = r {
        match k {
            "lost" => t.lost_flushed += 1,
            "partial" => t.partial += 1,
            "panic" => t.panics += 1,
            _ => t.other += 1,
        }
        t.failures.push(format!("{label} [{k}] {msg}"));
    }
}

fn campaign(points_per_scenario: usize, seed: u64) {
    let so = shim();
    let root = data_root().join(format!("powerloss-{seed}"));
    let _ = std::fs::create_dir_all(&root);
    let mut rng = Lcg(seed);
    let mut t = Tally::default();
    for (si, sc) in SCENARIOS.iter().enumerate() {
        let cfg = scenario_cfg(sc, &root.join(sc.name), seed + si as u64);
        let recs = record(&so, &cfg, false);
        let lines = pl::lines(&recs);
        let w = windows(&recs, &lines, sc.moment);
        let points = pick(&mut rng, &w, points_per_scenario);
        let before = t.images;
        let mut fs = Fs::default();
        let img = root.join(format!("{}-img", sc.name));
        for &c in &points {
            while fs.at < c {
                fs.step(&recs[fs.at]);
            }
            for mode in Mode::ALL {
                let r = check(&cfg, &fs, &recs, &lines, c, mode, &mut rng, &img);
                tally(&mut t, r, sc.name);
            }
        }
        println!(
            "POWERLOSS_SCENARIO {} records={} windows={} images={}",
            sc.name,
            recs.len(),
            w.len(),
            t.images - before
        );
        let _ = std::fs::remove_dir_all(&cfg.dir);
    }
    for f in t.failures.iter().take(4) {
        println!("POWERLOSS_FAIL {f}");
    }
    let _ = std::fs::write(root.with_extension("failures.txt"), t.failures.join("\n\n"));
    println!(
        "POWERLOSS images={} lost_flushed={} partial={} panics={} other={}",
        t.images, t.lost_flushed, t.partial, t.panics, t.other
    );
    assert!(
        t.failures.is_empty(),
        "{} power-loss failures",
        t.failures.len()
    );
}

/// Criterion 2: the recorder sees the calls recovery depends on, and the image at the last
/// fsync reads like the real directory.
#[test]
#[cfg_attr(
    any(not(target_os = "linux"), quickdb_sim_not_linux),
    ignore = "power loss: the fsrecord.c recorder is an LD_PRELOAD shim, Linux only"
)]
fn powerloss_recorder_sanity() {
    let so = shim();
    let dir = data_root().join("powerloss-sanity");
    let cfg = scenario_cfg(&SCENARIOS[4], &dir, 11);
    let recs = record(&so, &cfg, false);
    let has = |ops: &[u8], name: &dyn Fn(&str) -> bool| {
        recs.iter().any(|r| ops.contains(&r.op) && name(&r.name))
    };
    let synced = [pl::FSYNC, pl::FDATASYNC];
    let mut seen = Vec::new();
    for (label, f) in [
        (
            "quickdb.sqlite",
            &(|n: &str| n == "quickdb.sqlite") as &dyn Fn(&str) -> bool,
        ),
        ("quickdb.sqlite-wal", &|n: &str| n == "quickdb.sqlite-wal"),
        ("emb.hnsw.tmp", &|n: &str| n.starts_with("emb.hnsw.tmp")),
        ("<dir>", &|n: &str| n.is_empty()),
    ] {
        let pw = has(&[pl::PWRITE], f);
        let w = has(&[pl::WRITE], f);
        let s = has(&synced, f);
        println!("RECORDER {label} pwrite={pw} write={w} fsync={s}");
        seen.push((label, pw || w || label == "<dir>", s));
    }
    let renames = recs.iter().filter(|r| r.op == pl::RENAME).count();
    println!("RECORDER rename={renames} records={}", recs.len());
    assert!(
        seen.iter().all(|(_, w, s)| *w && *s) && renames > 0,
        "recorder missed calls: {seen:?}"
    );
    let last_sync = recs
        .iter()
        .rposition(|r| synced.contains(&r.op))
        .expect("an fsync")
        + 1;
    let mut fs = Fs::default();
    for r in &recs[..last_sync] {
        fs.step(r);
    }
    let lines = pl::lines(&recs);
    let (committed, _) = pl::progress(&lines, recs.len());
    let upto = committed.iter().flatten().max().map_or(0, |n| n + 1) + 2;
    let real = verify(&cfg, upto, [None; 3]).expect("real directory verifies");
    let (files, _) = fs.image(Mode::Drop, &mut Lcg(1));
    let img = data_root().join("powerloss-sanity-img");
    pl::write_image(&img, &files).expect("write image");
    let mut icfg = cfg.clone();
    icfg.dir = img.clone();
    let got = verify(&icfg, upto, [None; 3]).expect("image at the last fsync verifies");
    println!(
        "RECORDER last_fsync_image rows={} vectors={} edges={} real rows={} vectors={} edges={} digest_equal={}",
        got.rows, got.vectors, got.edges, real.rows, real.vectors, real.edges, got.digest == real.digest
    );
    assert_eq!(
        (got.rows, got.vectors, got.edges, &got.digest),
        (real.rows, real.vectors, real.edges, &real.digest)
    );
    let _ = std::fs::remove_dir_all(&img);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Criterion 3: with every fsync suppressed by the shim (`FSREC_NOSYNC`), the method must see
/// flushed batches vanish.
#[test]
#[cfg_attr(
    any(not(target_os = "linux"), quickdb_sim_not_linux),
    ignore = "power loss: the fsrecord.c recorder is an LD_PRELOAD shim, Linux only"
)]
fn powerloss_negative_control() {
    let so = shim();
    let dir = data_root().join("powerloss-nosync");
    let cfg = scenario_cfg(&SCENARIOS[0], &dir, 13);
    let recs = record(&so, &cfg, true);
    let lines = pl::lines(&recs);
    let first = lines
        .iter()
        .find(|(_, l)| l.starts_with("FLUSHED"))
        .map(|(i, _)| i + 1)
        .expect("a FLUSHED line");
    let mut rng = Lcg(5);
    let points = pick(&mut rng, &[(first, recs.len())], 10);
    let mut fs = Fs::default();
    let mut lost = 0;
    let img = data_root().join("powerloss-nosync-img");
    for &c in &points {
        while fs.at < c {
            fs.step(&recs[fs.at]);
        }
        if let Some(("lost", _)) = check(&cfg, &fs, &recs, &lines, c, Mode::Drop, &mut rng, &img) {
            lost += 1;
        }
    }
    println!("NEGATIVE images={} LOST={lost}", points.len());
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        lost,
        points.len(),
        "a writer without fsync must lose flushed data in every image"
    );
}

#[test]
#[cfg_attr(
    any(not(target_os = "linux"), quickdb_sim_not_linux),
    ignore = "power loss: the fsrecord.c recorder is an LD_PRELOAD shim, Linux only"
)]
fn powerloss_smoke() {
    campaign(1, 20261120);
}

#[test]
#[ignore = "long: 8 scenarios x 25 crash points x 3 images; run with --ignored (Linux only)"]
fn powerloss_long() {
    if cfg!(any(not(target_os = "linux"), quickdb_sim_not_linux)) {
        panic!("{LINUX_ONLY}");
    }
    let seed = std::env::var("QDB_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let points = std::env::var("PL_POINTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    campaign(points, seed);
}
