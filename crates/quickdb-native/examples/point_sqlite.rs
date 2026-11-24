//! Isolate SQLite row-page caching on the persisted point probe, without HNSW loading.
use quickdb_api::conformance::Lcg;
use rusqlite::{Connection, StatementStatus};
use std::hint::black_box;
use std::time::Instant;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let path = args.first().map_or("/data/point-million", String::as_str);
    let n: u64 = args.get(1).map_or(Ok(1_000_000), |v| v.parse())?;
    if n == 0 {
        return Err("count must be positive".into());
    }
    for (label, cache, mmap) in [
        ("default", 2000, 0),
        ("cache64", 65536, 0),
        ("default_again", 2000, 0),
        ("mmap256", 2000, 268435456),
        ("default_final", 2000, 0),
    ] {
        let db = Connection::open_with_flags(
            std::path::Path::new(path).join("quickdb.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        db.pragma_update(None, "cache_size", -cache)?;
        db.pragma_update(None, "mmap_size", mmap)?;
        let pages: u64 = db.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let size: u64 = db.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        let mut stmt = db.prepare("SELECT * FROM t_docs WHERE id=?")?;
        for pass in 0..3 {
            let mut rng = Lcg(0x5142494453);
            let mut times = Vec::with_capacity(10000);
            for _ in 0..10000 {
                let id = u64::from(rng.next_u32()) % n;
                let t = Instant::now();
                stmt.query_row([id], |row| {
                    for i in 0..row.as_ref().column_count() {
                        black_box(row.get_ref(i)?);
                    }
                    Ok(())
                })?;
                times.push(t.elapsed().as_nanos());
            }
            times.sort_unstable();
            println!("POINT_SQLITE config={label} pass={pass} cache_kib={cache} mmap={mmap} pages={pages} page_size={size} median_us={:.3} p95_us={:.3} reprepares={}", times[5000] as f64 / 1000.0, times[9500] as f64 / 1000.0, stmt.get_status(StatementStatus::RePrepare));
        }
    }
    let db = Connection::open_with_flags(
        std::path::Path::new(path).join("quickdb.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    for query in [
        "SELECT dst FROM edges WHERE src=?1",
        "SELECT src FROM edges WHERE dst=?1",
        "SELECT dst FROM edges WHERE src=?1 AND rel=?2",
        "SELECT src FROM edges WHERE dst=?1 AND rel=?2",
    ] {
        let mut stmt = db.prepare(&format!("EXPLAIN QUERY PLAN {query}"))?;
        let details = if query.contains("?2") {
            stmt.query_map(rusqlite::params![0, "rel0"], |r| r.get::<_, String>(3))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        } else {
            stmt.query_map([0], |r| r.get::<_, String>(3))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        println!("POINT_PLAN query={query:?} details={details:?}");
    }
    for bound in ["?4", "CAST(?4 AS INTEGER)"] {
        for depth in 1..=3 {
            let sql = format!("WITH RECURSIVE walk(id,depth) AS (SELECT ?1,0 UNION SELECT e.dst,w.depth+1 FROM walk w JOIN edges e ON e.src=w.id WHERE w.depth<?2 AND (?3 IS NULL OR e.rel=?3)) SELECT id,min(depth) AS d FROM walk WHERE id<>?1 GROUP BY id ORDER BY d,id LIMIT {bound}");
            let mut stmt = db.prepare(&sql)?;
            let mut rng = Lcg(0x504f494e54);
            let mut times = Vec::new();
            for _ in 0..500 {
                let id = u64::from(rng.next_u32()) % n;
                let t = Instant::now();
                let rows = stmt.query_map(
                    rusqlite::params![id, depth, Option::<&str>::None, 1000],
                    |r| Ok((r.get::<_, u64>(0)?, r.get::<_, u32>(1)?)),
                )?;
                black_box(rows.collect::<rusqlite::Result<Vec<_>>>()?);
                times.push(t.elapsed().as_nanos());
            }
            times.sort_unstable();
            println!(
                "POINT_SQL_LIMIT bound={bound} depth={depth} median_us={:.3} reprepares={}",
                times[250] as f64 / 1000.0,
                stmt.get_status(StatementStatus::RePrepare)
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("POINT_SQLITE ERROR {e}");
        std::process::exit(1);
    }
}
