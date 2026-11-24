//! Builds the disk image a power cut could leave, from a `tools/fsrecord.c` log.
//! Usage: pl_rebuild <log> <records|list> <drop|reorder|torn> <seed> <out-dir>
//! `records` is how many log records completed before the cut; `list` prints the log instead.
#[path = "../tests/powerloss_common/mod.rs"]
mod pl;

use quickdb_api::conformance::Lcg;
use std::path::Path;

fn run(args: &[String]) -> Result<String, String> {
    let [log, at, mode, seed, out] = args else {
        return Err(
            "usage: pl_rebuild <log> <records|list> <drop|reorder|torn> <seed> <out-dir>".into(),
        );
    };
    let bytes = std::fs::read(log).map_err(|e| format!("{log}: {e}"))?;
    let recs = pl::parse(&bytes)?;
    if at == "list" {
        let all: Vec<String> = recs
            .iter()
            .enumerate()
            .map(|(i, r)| format!("#{i} {}", r.describe()))
            .collect();
        return Ok(all.join("\n"));
    }
    let at: usize = at.parse().map_err(|e| format!("records: {e}"))?;
    let mode = pl::Mode::parse(mode).ok_or("mode must be drop, reorder or torn")?;
    let seed: u64 = seed.parse().map_err(|e| format!("seed: {e}"))?;
    let mut fs = pl::Fs::default();
    for r in recs.iter().take(at) {
        fs.step(r);
    }
    let (files, how) = fs.image(mode, &mut Lcg(seed));
    pl::write_image(Path::new(out), &files).map_err(|e| format!("{out}: {e}"))?;
    let sizes: Vec<String> = files
        .iter()
        .map(|(n, d)| format!("{n}={}", d.len()))
        .collect();
    Ok(format!(
        "IMAGE records={at} mode={} files {}\nUNSYNCED{how}",
        mode.name(),
        sizes.join(" ")
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("ERROR {e}");
            std::process::exit(2);
        }
    }
}
