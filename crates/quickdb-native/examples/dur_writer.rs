//! Durability helper: writes deterministic batches and reports progress on stdout.
//! Usage: see `Config::to_args` in `tests/durability_common/mod.rs`.
//! Per batch n: one `upsert_rows`, one `upsert_vectors`, one `upsert_edges`, then every
//! `flush_every` batches one `flush()`. Lines: `COMMITTED <n> <kind> <micros>`,
//! `FLUSHING <n>`, `FLUSHED <n> <micros>`, `ERROR <message>`.
#[path = "../tests/durability_common/mod.rs"]
mod common;

use common::*;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::io::Write;
use std::time::{Duration, Instant};

fn run(cfg: &Config) -> Result<()> {
    let out = std::io::stdout();
    let say = |line: String| {
        let mut o = out.lock();
        // A closed pipe means the parent has gone; nothing useful remains to do with the error.
        let _ = writeln!(o, "{line}");
        let _ = o.flush();
    };
    let mut s = NativeStore::open(
        &cfg.dir,
        &OpenOptions {
            create: true,
            sync_each_write: cfg.sync,
        },
    )?;
    s.define_table(&table_def())?;
    s.define_vector_index(&index_def(cfg.dim))?;
    say(format!("OPENED rebuilds={}", s.index_rebuilds()));
    // Concurrency runs keep writing after a failed flush, to count how often it fails.
    let tolerate = std::env::var_os("QDB_TOLERATE_FLUSH").is_some();
    let began = Instant::now();
    for n in cfg.start..cfg.start.saturating_add(cfg.max_batches) {
        if cfg.run_secs > 0 && began.elapsed().as_secs() >= cfg.run_secs {
            break;
        }
        let t = Instant::now();
        s.upsert_rows(TABLE, &cfg.rows(n))?;
        say(format!("COMMITTED {n} rows {}", t.elapsed().as_micros()));
        let mut vectors = cfg.vectors(n);
        if cfg.replace && n > 0 {
            vectors.extend(cfg.vectors(n - 1));
        }
        let items: Vec<_> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let t = Instant::now();
        s.upsert_vectors(INDEX, &items)?;
        say(format!("COMMITTED {n} vectors {}", t.elapsed().as_micros()));
        let t = Instant::now();
        s.upsert_edges(&cfg.edges(n))?;
        say(format!("COMMITTED {n} edges {}", t.elapsed().as_micros()));
        if (n + 1) % cfg.flush_every == 0 {
            say(format!("FLUSHING {n}"));
            let t = Instant::now();
            match s.flush() {
                Ok(()) => say(format!("FLUSHED {n} {}", t.elapsed().as_micros())),
                Err(e) if tolerate => say(format!("FLUSH_ERROR {n} {e}")),
                Err(e) => return Err(e),
            }
        }
        if cfg.sleep_ms > 0 {
            std::thread::sleep(Duration::from_millis(cfg.sleep_ms));
        }
    }
    if let Err(e) = s.flush() {
        if !tolerate {
            return Err(e);
        }
        say(format!("FLUSH_ERROR final {e}"));
    }
    say(format!("DONE {}", began.elapsed().as_millis()));
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match Config::from_args(&args) {
        Err(e) => {
            println!("ERROR {e}");
            2
        }
        Ok(cfg) => match run(&cfg) {
            Ok(()) => 0,
            Err(e) => {
                println!("ERROR {e}");
                1
            }
        },
    };
    std::process::exit(code);
}
