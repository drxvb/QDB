//! Area 1: SIGKILL a writer process at chosen moments and check what survives.
#[path = "durability_common/mod.rs"]
mod common;

use common::*;
use quickdb_api::conformance::Lcg;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub fn example(name: &str) -> PathBuf {
    // target/<profile>/deps/<test> -> target/<profile>/examples/<name>; `cargo test` builds examples.
    let exe = std::env::current_exe().expect("test binary path");
    exe.parent()
        .and_then(Path::parent)
        .map(|p| p.join("examples").join(name))
        .expect("target layout")
}

pub fn data_root() -> PathBuf {
    std::env::var_os("QUICKDB_DATA").map_or_else(std::env::temp_dir, PathBuf::from)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Moment {
    /// After the previous batch's edges: the next call is `upsert_rows`.
    MidRows,
    /// After `COMMITTED n rows`: inside `upsert_vectors`.
    MidVectors,
    /// After `FLUSHING n`: anywhere in `flush()` (snapshot write, rename, checkpoint).
    InFlush,
    /// Early in `flush()`, where the snapshot file is being written.
    InSnapshot,
}

struct Scenario {
    name: &'static str,
    moment: Moment,
    sync: bool,
    replace: bool,
    /// Upper bound of the random delay after the trigger line, in microseconds.
    jitter_us: u64,
}

const SCENARIOS: [Scenario; 8] = [
    Scenario {
        name: "mid-rows",
        moment: Moment::MidRows,
        sync: false,
        replace: false,
        jitter_us: 400,
    },
    Scenario {
        name: "mid-vectors",
        moment: Moment::MidVectors,
        sync: false,
        replace: false,
        jitter_us: 3000,
    },
    Scenario {
        name: "in-flush",
        moment: Moment::InFlush,
        sync: false,
        replace: false,
        jitter_us: 6000,
    },
    Scenario {
        name: "in-snapshot",
        moment: Moment::InSnapshot,
        sync: false,
        replace: false,
        jitter_us: 600,
    },
    Scenario {
        name: "compaction",
        moment: Moment::InFlush,
        sync: false,
        replace: true,
        jitter_us: 20000,
    },
    Scenario {
        name: "sync-mid-rows",
        moment: Moment::MidRows,
        sync: true,
        replace: false,
        jitter_us: 3000,
    },
    Scenario {
        name: "sync-mid-vectors",
        moment: Moment::MidVectors,
        sync: true,
        replace: false,
        jitter_us: 20000,
    },
    Scenario {
        name: "sync-compaction",
        moment: Moment::MidVectors,
        sync: true,
        replace: true,
        jitter_us: 40000,
    },
];

#[derive(Default)]
struct Progress {
    committed: [Option<u64>; 3],
    flushed: Option<u64>,
    top: u64,
    error: Option<String>,
}

impl Progress {
    fn note(&mut self, line: &str) {
        let w: Vec<&str> = line.split_whitespace().collect();
        let n = w.get(1).and_then(|x| x.parse::<u64>().ok());
        match (w.first().copied(), n) {
            (Some("COMMITTED"), Some(n)) => {
                let k = match w.get(2).copied() {
                    Some("rows") => 0,
                    Some("vectors") => 1,
                    _ => 2,
                };
                self.committed[k] = Some(n);
                self.top = self.top.max(n + 1);
            }
            (Some("FLUSHED"), Some(n)) => self.flushed = Some(n),
            (Some("ERROR"), _) => self.error = Some(line.into()),
            _ => {}
        }
    }
    fn triggered(&self, line: &str, moment: Moment, target: u64) -> bool {
        let w: Vec<&str> = line.split_whitespace().collect();
        let n = w.get(1).and_then(|x| x.parse::<u64>().ok());
        match (moment, w.first().copied(), w.get(2).copied()) {
            (Moment::MidRows, Some("COMMITTED"), Some("edges")) => {
                n.is_some_and(|n| n + 1 >= target)
            }
            (Moment::MidVectors, Some("COMMITTED"), Some("rows")) => n.is_some_and(|n| n >= target),
            (Moment::InFlush | Moment::InSnapshot, Some("FLUSHING"), _) => {
                n.is_some_and(|n| n >= target)
            }
            _ => false,
        }
    }
}

struct KillStats {
    kills: u64,
    rebuilds: u64,
    reopen_ms: Vec<f64>,
    ann_self_miss: u64,
    failures: Vec<String>,
}

fn spawn(cfg: &Config) -> Child {
    Command::new(example("dur_writer"))
        .args(cfg.to_args())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("start dur_writer (built by cargo test)")
}

/// One kill: start the writer where the store left off, wait for the trigger, sleep a random
/// jitter, SIGKILL it, then verify the store twice.
fn one_kill(cfg: &mut Config, sc: &Scenario, rng: &mut Lcg, st: &mut KillStats) {
    let mut child = spawn(cfg);
    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let target = cfg.start + rng.next_u32() as u64 % (2 * cfg.flush_every);
    let mut p = Progress::default();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(line) => {
                p.note(&line);
                if p.triggered(&line, sc.moment, target) {
                    let jitter = rng.next_u32() as u64 % sc.jitter_us.max(1);
                    std::thread::sleep(Duration::from_micros(jitter));
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) if Instant::now() < deadline => {}
            Err(_) => break,
        }
    }
    // Child::kill sends SIGKILL on Unix, to this child's pid only.
    let _ = child.kill();
    let _ = child.wait();
    // Lines written before death are still in the pipe; they count as returned calls.
    while let Ok(line) = rx.recv_timeout(Duration::from_millis(200)) {
        p.note(&line);
    }
    let _ = reader.join();
    st.kills += 1;
    if let Some(e) = &p.error {
        st.failures
            .push(format!("{}: writer reported {e}", sc.name));
    }
    let must = if sc.sync { p.committed } else { [p.flushed; 3] };
    let upto = p.top.max(cfg.start) + 2;
    match verify(cfg, upto, must).and_then(|a| verify(cfg, upto, must).map(|b| (a, b))) {
        Ok((a, b)) => {
            if a.digest != b.digest || (a.rows, a.vectors, a.edges) != (b.rows, b.vectors, b.edges)
            {
                st.failures
                    .push(format!("{}: second reopen differs", sc.name));
            }
            if b.rebuilds != 0 && a.rebuilds == 0 {
                st.failures.push(format!(
                    "{}: a reopen invalidated a valid snapshot",
                    sc.name
                ));
            }
            st.rebuilds += a.rebuilds as u64;
            st.reopen_ms.push(a.reopen_ms);
            st.ann_self_miss += a.ann_self_miss;
            cfg.start = a.edges;
        }
        Err(e) => {
            st.failures.push(format!(
                "{} kill {} (seed {}): {e}",
                sc.name, st.kills, cfg.seed
            ));
            cfg.start = 0;
            let _ = std::fs::remove_dir_all(&cfg.dir);
        }
    }
}

fn campaign(seed: u64, kills_per_scenario: u64) {
    println!("durability_kill seed={seed} kills_per_scenario={kills_per_scenario}");
    let mut rng = Lcg(seed);
    let mut all_failures = Vec::new();
    let mut total = 0;
    for (si, sc) in SCENARIOS.iter().enumerate() {
        let dir = data_root().join(format!("kill-{seed}-{}", sc.name));
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg = Config::new(&dir);
        cfg.seed = seed + si as u64;
        cfg.sync = sc.sync;
        cfg.replace = sc.replace;
        cfg.batch = if sc.sync { 50 } else { 200 };
        cfg.flush_every = if sc.replace { 2 } else { 3 };
        let mut st = KillStats {
            kills: 0,
            rebuilds: 0,
            reopen_ms: Vec::new(),
            ann_self_miss: 0,
            failures: Vec::new(),
        };
        for k in 0..kills_per_scenario {
            // A fresh store every 10 kills keeps run time bounded and also kills young stores.
            if k % 10 == 0 {
                let _ = std::fs::remove_dir_all(&dir);
                cfg.start = 0;
            }
            one_kill(&mut cfg, sc, &mut rng, &mut st);
        }
        let mut ms = st.reopen_ms.clone();
        ms.sort_by(f64::total_cmp);
        let pct = |q: f64| {
            ms.get(((ms.len() as f64 - 1.0) * q) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        println!(
            "SCENARIO {} kills={} failures={} rebuilds={} reopen_ms_p50={:.1} p95={:.1} max={:.1} ann_self_miss={}",
            sc.name,
            st.kills,
            st.failures.len(),
            st.rebuilds,
            pct(0.5),
            pct(0.95),
            pct(1.0),
            st.ann_self_miss
        );
        for f in st.failures.iter().take(5) {
            println!("  FAIL {f}");
        }
        total += st.kills;
        all_failures.extend(st.failures);
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("TOTAL kills={total} failures={}", all_failures.len());
    assert!(
        all_failures.is_empty(),
        "{} failures, first: {}",
        all_failures.len(),
        all_failures[0]
    );
}

#[test]
fn kill_short() {
    campaign(20261118, 3);
}

#[test]
#[ignore = "long: 8 scenarios x 30 kills; run with --ignored"]
fn kill_long() {
    let seed = std::env::var("QDB_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    campaign(seed, 30);
}
