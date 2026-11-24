//! Persistent native point-lookup/traversal probe.
//! Usage: point_probe [path=/data/point-million] [count=1000000] [build|run]
//! `build` creates only a missing store, then measures all lifecycle phases.
//! `run` never loads data. The separate point_sqlite example isolates SQLite
//! cache/mmap settings without changing the native connection's defaults.
use quickdb_api::conformance::Lcg;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn measure<T>(
    phase: &str,
    operation: &str,
    samples: usize,
    mut f: impl FnMut(usize) -> Result<T>,
) -> Result<()> {
    let mut times = Vec::with_capacity(samples);
    for i in 0..samples {
        let start = Instant::now();
        let result = f(i)?;
        times.push(start.elapsed().as_nanos());
        black_box(result);
    }
    times.sort_unstable();
    let median = (times[(samples - 1) / 2] as f64 + times[samples / 2] as f64) / 2000.0;
    println!("POINT phase={phase} operation={operation} samples={samples} median_us={median:.3}");
    Ok(())
}

fn timings(store: &NativeStore, phase: &str, count: u64) -> Result<()> {
    // Identical precomputed starts in every phase, without RNG work in timings.
    let mut rng = Lcg(0x504f494e54);
    let starts: Vec<_> = (0..10_000)
        .map(|_| u64::from(rng.next_u32()) % count)
        .collect();
    measure(phase, "get_row", 10_000, |i| {
        store
            .get_row("docs", starts[i])?
            .ok_or_else(|| Error::Backend("missing probe row".into()))
    })?;
    measure(phase, "neighbors", 500, |i| {
        store.neighbors(starts[i], None, Direction::Out, 1000)
    })?;
    for depth in 1..=3 {
        let mut reached = 0usize;
        for &start in &starts[..500] {
            reached += store.k_hop(start, None, Direction::Out, depth, 1000)?.len();
        }
        println!(
            "POINT phase={phase} depth={depth} mean_reached={:.3}",
            reached as f64 / 500.0
        );
        for method in ["k_hop", "k_hop_bfs", "k_hop_sql"] {
            measure(phase, &format!("{method}_depth{depth}"), 500, |i| {
                let args = (starts[i], None, Direction::Out, depth, 1000);
                match method {
                    "k_hop_bfs" => store.k_hop_bfs(args.0, args.1, args.2, args.3, args.4),
                    "k_hop_sql" => store.k_hop_sql(args.0, args.1, args.2, args.3, args.4),
                    _ => store.k_hop(args.0, args.1, args.2, args.3, args.4),
                }
            })?;
        }
    }
    Ok(())
}

fn load(store: &mut NativeStore, count: u64) -> Result<()> {
    store.define_table(&TableDef {
        name: "docs".into(),
        columns: vec![
            ColumnDef {
                name: "kind".into(),
                ty: ColumnType::Text,
                indexed: true,
            },
            ColumnDef {
                name: "title".into(),
                ty: ColumnType::Text,
                indexed: false,
            },
            ColumnDef {
                name: "size".into(),
                ty: ColumnType::Int,
                indexed: true,
            },
            ColumnDef {
                name: "score".into(),
                ty: ColumnType::Float,
                indexed: false,
            },
            ColumnDef {
                name: "body".into(),
                ty: ColumnType::Text,
                indexed: false,
            },
        ],
    })?;
    store.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 128,
        metric: Metric::L2,
        m: 8,
        ef_construction: 32,
    })?;
    let started = Instant::now();
    let mut rng = Lcg(42);
    // Keep rows contiguous, then vectors, so layout does not vary between trials.
    for base in (0..count).step_by(2000) {
        let rows: Vec<_> = (base..(base + 2000).min(count)).map(|id| Row {
            id,
            values: vec![
                Value::Text(["paper", "note", "source", "report"][id as usize % 4].into()),
                Value::Text(format!("Research document {id:08}: native database performance notes")),
                Value::Int(512 + (id % 131_072) as i64),
                Value::Float((id % 10_000) as f64 / 10_000.0),
                Value::Text(format!("Document {id:08}: measured native database row lookup, vector retrieval, and graph traversal results.")),
            ],
        }).collect();
        store.upsert_rows("docs", &rows)?;
    }
    for base in (0..count).step_by(2000) {
        let data: Vec<_> = (base..(base + 2000).min(count))
            .map(|id| (id, rng.vector(128)))
            .collect();
        let items: Vec<_> = data
            .iter()
            .map(|(id, vector)| (*id, vector.as_slice()))
            .collect();
        store.upsert_vectors("emb", &items)?;
        if base % 20_000 == 0 {
            println!(
                "POINT load_vectors={} elapsed_s={:.1}",
                base + data.len() as u64,
                started.elapsed().as_secs_f64()
            );
        }
    }
    let edge_count = if count == 22_666 { 45_000 } else { count * 5 };
    let mut edges = Vec::with_capacity(10_000);
    for src in 0..count {
        let degree = edge_count / count + u64::from(src < edge_count % count);
        let mut destinations = Vec::new();
        for slot in 0..degree {
            let mut dst = if count == 22_666 {
                let base = src / 4 * 4;
                base + u64::from(rng.next_u32()) % (count - base).min(4)
            } else {
                u64::from(rng.next_u32()) % count
            };
            while destinations.contains(&dst) && destinations.len() < count as usize {
                dst = if count == 22_666 {
                    let base = src / 4 * 4;
                    base + (dst - base + 1) % (count - base).min(4)
                } else {
                    (dst + 1) % count
                };
            }
            destinations.push(dst);
            edges.push(Edge {
                src,
                rel: format!("rel{slot}"),
                dst,
                weight: 1.0,
            });
        }
        if edges.len() >= 10_000 {
            store.upsert_edges(&edges)?;
            edges.clear();
        }
    }
    store.upsert_edges(&edges)?;
    println!(
        "POINT loaded rows={} vectors={} edges={} elapsed_s={:.1}",
        store.count_rows("docs")?,
        store.count_vectors("emb")?,
        store.count_edges()?,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn file_state(path: &Path, phase: &str) {
    let size = |name: &str| std::fs::metadata(path.join(name)).map_or(0, |m| m.len());
    println!(
        "POINT_FILES phase={phase} database_bytes={} wal_bytes={} shm_bytes={}",
        size("quickdb.sqlite"),
        size("quickdb.sqlite-wal"),
        size("quickdb.sqlite-shm")
    );
}

fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let path = Path::new(args.first().map_or("/data/point-million", String::as_str));
    let count = args.get(1).map_or(Ok(1_000_000u64), |s| s.parse())?;
    let mode = args.get(2).map_or("build", String::as_str);
    if count == 0 || count > i64::MAX as u64 / 5 || !matches!(mode, "build" | "run") {
        return Err("expected positive count and mode build|run".into());
    }
    // A marker is written only after the durable flush. Incomplete stores are not
    // silently reused or rebuilt, since either would invalidate comparisons.
    let marker = path.join("point-probe-count");
    let existing = marker.exists();
    if mode == "run" && !existing {
        return Err("run requires a completed build (point-probe-count missing)".into());
    }
    if existing && std::fs::read_to_string(&marker)?.trim() != count.to_string() {
        return Err("stored count differs from requested count".into());
    }
    if !existing && path.exists() && std::fs::read_dir(path)?.next().is_some() {
        return Err(
            "unmarked nonempty store: choose another path or inspect incomplete build".into(),
        );
    }
    println!(
        "POINT path={} count={count} mode={mode} existing={existing}",
        path.display()
    );
    let mut store = NativeStore::open(
        path,
        &OpenOptions {
            create: !existing,
            sync_each_write: false,
        },
    )?;
    if !existing {
        load(&mut store, count)?;
    }
    let expected_edges = if count == 22_666 { 45_000 } else { count * 5 };
    if store.count_rows("docs")? != count
        || store.count_vectors("emb")? != count
        || store.count_edges()? != expected_edges
    {
        return Err("stored row/vector/edge counts differ".into());
    }
    file_state(path, "before_flush");
    timings(&store, "before_flush", count)?;
    let start = Instant::now();
    store.flush()?;
    println!("POINT flush_s={:.3}", start.elapsed().as_secs_f64());
    if !existing {
        std::fs::write(&marker, count.to_string())?;
    }
    file_state(path, "after_flush");
    timings(&store, "after_flush", count)?;
    let mut rng = Lcg(743);
    let queries: Vec<_> = (0..100).map(|_| rng.vector(128)).collect();
    measure("knn", "knn", queries.len(), |i| {
        store.knn("emb", &queries[i], 10, 64)
    })?;
    timings(&store, "after_knn", count)?;
    drop(store);
    let start = Instant::now();
    let store = NativeStore::open(
        path,
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )?;
    println!(
        "POINT reopen_s={:.3} rebuilds={} disk_bytes={}",
        start.elapsed().as_secs_f64(),
        store.index_rebuilds(),
        dir_size(path)?
    );
    file_state(path, "after_reopen");
    timings(&store, "after_reopen", count)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("POINT ERROR {error}");
        std::process::exit(1);
    }
}
