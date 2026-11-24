//! Filtered-search probe on SIFT: `cargo run --release -p quickdb-native --example filter_probe -- <n> [queries] [repeats]`.
//! Builds `$QUICKDB_DATA/filter-probe-<n>` once (kept between runs), then reports for each
//! selectivity the cost of fetching the allowed ids from SQLite, of building a `HashSet`, and the median
//! `knn_filtered` latency and recall@10 against exact filtered truth.
use quickdb_api::{
    ColumnDef, ColumnType, Filter, Hit, Metric, OpenOptions, Row, Store, TableDef, Value,
    VectorIndexDef,
};
use quickdb_native::NativeStore;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SELECTIVITY: [f64; 6] = [0.001, 0.01, 0.05, 0.2, 0.5, 0.9];

fn fvecs(path: &Path, take: usize) -> Vec<Vec<f32>> {
    let mut input = BufReader::new(File::open(path).expect("open fvecs"));
    let mut out = Vec::with_capacity(take);
    let mut header = [0u8; 4];
    while out.len() < take && input.read_exact(&mut header).is_ok() {
        let dim = i32::from_le_bytes(header) as usize;
        let mut payload = vec![0u8; dim * 4];
        input.read_exact(&mut payload).expect("truncated fvecs");
        out.push(
            payload
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        );
    }
    out
}

fn unit(id: u64, salt: u64) -> f64 {
    let mut h = id
        .wrapping_add(salt.wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add(0x9e3779b97f4a7c15);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn table() -> TableDef {
    TableDef {
        name: "docs".into(),
        columns: (0..SELECTIVITY.len())
            .map(|i| ColumnDef {
                name: format!("f{i}"),
                ty: ColumnType::Text,
                indexed: true,
            })
            .collect(),
    }
}

fn matches(id: u64, column: usize) -> bool {
    unit(id, column as u64 + 1) < SELECTIVITY[column]
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn recall(hits: &[Hit], truth: &[u64]) -> f64 {
    let t: HashSet<_> = truth.iter().collect();
    hits.iter().filter(|h| t.contains(&h.id)).count() as f64 / truth.len().max(1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let nq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let repeats: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let sift = PathBuf::from(std::env::var("SIFT").unwrap_or_else(|_| "/datasets/sift".into()));
    let root = PathBuf::from(std::env::var("QUICKDB_DATA").unwrap_or_else(|_| "/data".into()));
    let dir = root.join(format!("filter-probe-{n}"));
    let base = fvecs(&sift.join("sift_base.fvecs"), n);
    let queries = fvecs(&sift.join("sift_query.fvecs"), nq);
    assert_eq!(base.len(), n, "base file too short");
    if !dir.join("built").exists() {
        let _ = std::fs::remove_dir_all(&dir);
        let t = Instant::now();
        let mut s = NativeStore::open(&dir, &OpenOptions::default()).expect("open");
        s.define_table(&table()).expect("table");
        s.define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: 128,
            metric: Metric::L2,
            m: 16,
            ef_construction: 200,
        })
        .expect("index");
        let rows: Vec<Row> = (0..n as u64)
            .map(|id| Row {
                id,
                values: (0..SELECTIVITY.len())
                    .map(|c| Value::Text(if matches(id, c) { "y" } else { "n" }.into()))
                    .collect(),
            })
            .collect();
        for chunk in rows.chunks(10_000) {
            s.upsert_rows("docs", chunk).expect("rows");
        }
        for (i, chunk) in base.chunks(10_000).enumerate() {
            let items: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(j, v)| ((i * 10_000 + j) as u64, v.as_slice()))
                .collect();
            s.upsert_vectors("emb", &items).expect("vectors");
        }
        s.flush().expect("flush");
        drop(s);
        File::create(dir.join("built")).expect("marker");
        println!("BUILD n={n} seconds={:.1}", t.elapsed().as_secs_f64());
    }
    let t = Instant::now();
    let s = NativeStore::open(
        &dir,
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )
    .expect("reopen");
    println!(
        "REOPEN seconds={:.2} rebuilds={}",
        t.elapsed().as_secs_f64(),
        s.index_rebuilds()
    );
    let db = rusqlite::Connection::open_with_flags(
        dir.join("quickdb.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("sqlite");

    let mut unfiltered = Vec::new();
    let mut unfiltered_recall = 0.0;
    let all_truth: Vec<Vec<u64>> = queries.iter().map(|q| exact(&base, q, |_| true)).collect();
    for r in 0..repeats {
        let mut times = Vec::new();
        for (q, t) in queries.iter().zip(&all_truth) {
            let start = Instant::now();
            let hits = s.knn("emb", q, 10, 64).expect("knn");
            times.push(start.elapsed().as_secs_f64() * 1e3);
            if r == 0 {
                unfiltered_recall += recall(&hits, t) / nq as f64;
            }
        }
        unfiltered.push(median(times));
    }
    println!(
        "UNFILTERED n={n} ef=64 median_ms={:.3} runs={:?} recall={:.3}",
        median(unfiltered.clone()),
        unfiltered
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>(),
        unfiltered_recall
    );
    println!("| selectivity | allowed | fetch ms | set ms | knn_filtered median ms | search ms (derived) | recall@10 |");
    for (c, &p) in SELECTIVITY.iter().enumerate() {
        let sql = format!("SELECT id FROM \"t_docs\" WHERE \"c_f{c}\" = ?");
        let mut fetch = Vec::new();
        let mut set = Vec::new();
        let mut ids: Vec<u64> = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let mut stmt = db.prepare_cached(&sql).expect("prepare");
            ids = stmt
                .query_map(["y"], |r| r.get::<_, u64>(0))
                .expect("query")
                .collect::<rusqlite::Result<_>>()
                .expect("rows");
            fetch.push(start.elapsed().as_secs_f64() * 1e3);
            let start = Instant::now();
            let h: HashSet<u64> = ids.iter().copied().collect();
            set.push(start.elapsed().as_secs_f64() * 1e3);
            assert_eq!(h.len(), ids.len());
        }
        let allowed: HashSet<u64> = ids.iter().copied().collect();
        let truth: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| exact(&base, q, |id| allowed.contains(&id)))
            .collect();
        let filter = Filter::Eq {
            column: format!("f{c}"),
            value: Value::Text("y".into()),
        };
        let mut runs = Vec::new();
        let mut rec = 0.0;
        for r in 0..repeats {
            let mut times = Vec::new();
            for (q, t) in queries.iter().zip(&truth) {
                let start = Instant::now();
                let hits = s
                    .knn_filtered("emb", q, 10, 64, "docs", &filter)
                    .expect("filtered");
                times.push(start.elapsed().as_secs_f64() * 1e3);
                assert!(
                    hits.iter().all(|h| allowed.contains(&h.id)),
                    "hit outside filter"
                );
                if r == 0 {
                    rec += recall(&hits, t) / nq as f64;
                }
            }
            runs.push(median(times));
        }
        let total = median(runs.clone());
        let (f, st) = (median(fetch), median(set));
        println!(
            "| {:.1}% | {} | {:.3} | {:.3} | {:.3} (runs {:?}) | {:.3} | {:.3} |",
            p * 100.0,
            ids.len(),
            f,
            st,
            total,
            runs.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>(),
            total - f - st,
            rec
        );
    }
}

fn exact(base: &[Vec<f32>], q: &[f32], allowed: impl Fn(u64) -> bool) -> Vec<u64> {
    let mut d: Vec<(f32, u64)> = base
        .iter()
        .enumerate()
        .filter(|(i, _)| allowed(*i as u64))
        .map(|(i, v)| (l2(q, v), i as u64))
        .collect();
    let k = 10.min(d.len());
    if k == 0 {
        return Vec::new();
    }
    d.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.truncate(k);
    d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    d.into_iter().map(|x| x.1).collect()
}
