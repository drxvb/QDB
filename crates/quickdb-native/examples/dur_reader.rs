//! Durability helper: a reader process for the multi-process test.
//! Usage: dur_reader <dir> <seed> <batch> <dim> <run_secs> <reopen_ms>
//! Queries a store that a `dur_writer` is filling, reopening it every `reopen_ms`, and prints one
//! `READER key=value ...` summary line plus `ERR <message>` for the first instance of each error.
#[path = "../tests/durability_common/mod.rs"]
mod common;

use common::*;
use quickdb_api::conformance::Lcg;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Default)]
struct Stats {
    queries: u64,
    knn: u64,
    opens: u64,
    rebuilds: u64,
    open_ms_max: f64,
    /// Row present, vector absent in this reader's index, row newer than the reader's open.
    stale_vector: u64,
    /// Row present at open time (older than the batch in flight), vector absent after open.
    missing_after_open: u64,
    wrong: u64,
    errors: BTreeMap<String, u64>,
}

fn arg(args: &[String], i: usize) -> u64 {
    args.get(i).and_then(|a| a.parse().ok()).unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        println!("ERR usage");
        std::process::exit(2);
    };
    let mut cfg = Config::new(std::path::Path::new(dir));
    cfg.seed = arg(&args, 1);
    cfg.batch = arg(&args, 2).max(2);
    cfg.dim = arg(&args, 3).max(1) as usize;
    let run = Duration::from_secs(arg(&args, 4));
    let reopen = Duration::from_millis(arg(&args, 5).max(1));
    let mut st = Stats::default();
    let mut rng = Lcg(cfg.seed ^ std::process::id() as u64);
    let began = Instant::now();
    let note = |st: &mut Stats, e: String| {
        let key: String = e.chars().take(80).collect();
        let c = st.errors.entry(key).or_insert(0);
        if *c == 0 {
            println!("ERR {e}");
        }
        *c += 1;
    };
    while began.elapsed() < run {
        // Row count before the open: the index loaded afterwards must cover all but the batch in flight.
        // Read-only SQLite peek (internal table name `t_docs`) so this count costs no index load.
        let peek = rusqlite::Connection::open_with_flags(
            cfg.dir.join("quickdb.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .and_then(|c| c.query_row("SELECT count(*) FROM t_docs", [], |r| r.get::<_, u64>(0)));
        let before = match peek {
            Ok(n) => n,
            Err(e) => {
                note(&mut st, format!("count before open: {e}"));
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
        };
        let t = Instant::now();
        let s = match NativeStore::open_read_only(&cfg.dir) {
            Ok(s) => s,
            Err(e) => {
                note(&mut st, format!("open: {e}"));
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
        };
        st.opens += 1;
        st.open_ms_max = st.open_ms_max.max(t.elapsed().as_secs_f64() * 1e3);
        st.rebuilds += s.index_rebuilds() as u64;
        let settled = (before / cfg.batch).saturating_sub(1);
        for n in 0..settled {
            for i in [0, cfg.batch - 1] {
                match has_vector(&s, &cfg, cfg.id(n, i)) {
                    Ok(true) => {}
                    Ok(false) => st.missing_after_open += 1,
                    Err(e) => note(&mut st, format!("check after open: {e}")),
                }
            }
        }
        let opened = Instant::now();
        while opened.elapsed() < reopen && began.elapsed() < run {
            let rows = match s.count_rows(TABLE) {
                Ok(n) => n,
                Err(e) => {
                    note(&mut st, format!("count_rows: {e}"));
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
            };
            if rows == 0 {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            let id = rng.next_u32() as u64 % rows;
            let (n, i) = (id / cfg.batch, id % cfg.batch);
            st.queries += 1;
            match s.get_row(TABLE, id) {
                Ok(Some(r)) if r == cfg.row(n, i) => {}
                Ok(other) => {
                    st.wrong += 1;
                    note(&mut st, format!("row {id} wrong: {other:?}"));
                }
                Err(e) => note(&mut st, format!("get_row: {e}")),
            }
            st.knn += 1;
            match s.knn(INDEX, &cfg.vector(id), 5, 64) {
                Ok(hits) if hits.iter().any(|h| h.id == id && h.distance == 0.0) => {}
                Ok(_) => match has_vector(&s, &cfg, id) {
                    Ok(true) => {}
                    Ok(false) if n + 1 >= settled => st.stale_vector += 1,
                    Ok(false) => st.missing_after_open += 1,
                    Err(e) => note(&mut st, format!("exact: {e}")),
                },
                Err(e) => note(&mut st, format!("knn: {e}")),
            }
        }
    }
    let secs = began.elapsed().as_secs_f64();
    let errors: u64 = st.errors.values().sum();
    println!(
        "READER queries={} knn={} qps={:.0} opens={} rebuilds={} open_ms_max={:.1} stale_vector={} missing_after_open={} wrong={} errors={} kinds={:?}",
        st.queries,
        st.knn,
        st.queries as f64 / secs,
        st.opens,
        st.rebuilds,
        st.open_ms_max,
        st.stale_vector,
        st.missing_after_open,
        st.wrong,
        errors,
        st.errors
    );
}
