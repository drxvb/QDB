use quickdb_api::{
    conformance::{exact_distance, Lcg},
    *,
};
use quickdb_native::NativeStore;
use std::time::{Duration, Instant};

fn median(times: &mut [Duration]) -> f64 {
    times.sort_unstable();
    times[times.len() / 2].as_secs_f64()
}
#[test]
#[ignore = "100k x 1024 release benchmark, run explicitly on a dedicated benchmark host"]
fn targets_100k() {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::Builder::new()
        .prefix("native-target-")
        .tempdir_in(root)
        .unwrap();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    s.define_table(&TableDef {
        name: "docs".into(),
        columns: vec![ColumnDef {
            name: "kind".into(),
            ty: ColumnType::Int,
            indexed: true,
        }],
    })
    .unwrap();
    s.upsert_rows(
        "docs",
        &(0..100_000)
            .map(|id| Row {
                id,
                values: vec![Value::Int(id as i64 % 10)],
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 1024,
        metric: Metric::Cosine,
        m: 32,
        ef_construction: 200,
    })
    .unwrap();
    let mut rng = Lcg(771);
    let data: Vec<_> = (0..100_000).map(|id| (id, rng.vector(1024))).collect();
    let start = Instant::now();
    for chunk in data.chunks(1000) {
        s.upsert_vectors(
            "emb",
            &chunk
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    let build = start.elapsed().as_secs_f64();
    println!("TARGET_BUILD n=100000 dim=1024 metric=cosine m=32 ef_construction=200 seconds={build:.6} target_lt_seconds=120 pass={}", build < 120.0);
    let queries: Vec<_> = (0..40).map(|_| rng.vector(1024)).collect();
    let truth: Vec<Vec<_>> = queries
        .iter()
        .map(|q| {
            let mut scores: Vec<_> = data
                .iter()
                .map(|(id, v)| (exact_distance(Metric::Cosine, q, v), *id))
                .collect();
            scores.select_nth_unstable_by(9, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            scores[..10].iter().map(|x| x.1).collect()
        })
        .collect();
    let mut last_median = 0.0;
    let mut last_recall = 0.0;
    let mut last_ef = 0;
    for ef in [128, 512, 2048, 8192, 32768, 100_000] {
        let mut found = 0;
        let mut times = Vec::new();
        for (q, expected) in queries.iter().zip(&truth) {
            let start = Instant::now();
            let hits = s.knn("emb", q, 10, ef).unwrap();
            times.push(start.elapsed());
            found += hits.iter().filter(|h| expected.contains(&h.id)).count();
        }
        last_median = median(&mut times) * 1000.0;
        last_recall = found as f64 / (queries.len() * 10) as f64;
        last_ef = ef;
        println!(
            "SEARCH_SWEEP ef_search={ef} median_ms={last_median:.6} recall_at_10={last_recall:.6}"
        );
        if last_recall >= 0.95 {
            break;
        }
    }
    println!("TARGET_KNN k=10 ef_search={last_ef} median_ms={last_median:.6} recall_at_10={last_recall:.6} target_lt_ms=2 target_recall_ge=0.95 pass={}", last_median < 2.0 && last_recall >= 0.95);
    let mut times = Vec::new();
    for _ in 0..2000 {
        let id = u64::from(rng.next_u32()) % 100_000;
        let start = Instant::now();
        assert!(s.get_row("docs", id).unwrap().is_some());
        times.push(start.elapsed());
    }
    let get_us = median(&mut times) * 1_000_000.0;
    println!(
        "TARGET_GET_ROW median_us={get_us:.6} target_lt_us=20 pass={}",
        get_us < 20.0
    );
    let edges: Vec<_> = (0..100_000u64)
        .flat_map(|src| {
            (1..=5).map(move |j| Edge {
                src,
                dst: (src.wrapping_mul(17) + j * 193) % 100_000,
                rel: "link".into(),
                weight: 1.0,
            })
        })
        .collect();
    s.upsert_edges(&edges).unwrap();
    let mut bfs_times = Vec::new();
    let mut sql_times = Vec::new();
    for i in 0..200u64 {
        let start = Instant::now();
        let bfs = s
            .k_hop_bfs(i * 431, None, Direction::Out, 3, usize::MAX)
            .unwrap();
        bfs_times.push(start.elapsed());
        let start = Instant::now();
        let sql = s
            .k_hop_sql(i * 431, None, Direction::Out, 3, usize::MAX)
            .unwrap();
        sql_times.push(start.elapsed());
        assert_eq!(bfs, sql);
    }
    let bfs_us = median(&mut bfs_times) * 1_000_000.0;
    let sql_us = median(&mut sql_times) * 1_000_000.0;
    println!("GRAPH_COMPARISON nodes=100000 edges=500000 depth=3 queries=200 bfs_median_us={bfs_us:.6} recursive_sql_median_us={sql_us:.6} faster={}", if bfs_us < sql_us { "bfs" } else { "sql" });
    let start = Instant::now();
    s.flush().unwrap();
    println!("FLUSH seconds={:.6}", start.elapsed().as_secs_f64());
    drop(s);
    let start = Instant::now();
    let s = NativeStore::open(
        dir.path(),
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )
    .unwrap();
    let reopen = start.elapsed().as_secs_f64();
    assert_eq!(s.index_rebuilds(), 0);
    assert_eq!(s.count_vectors("emb").unwrap(), 100_000);
    println!(
        "TARGET_REOPEN seconds={reopen:.6} rebuilt={} target_lt_seconds=2 pass={}",
        s.index_rebuilds(),
        reopen < 2.0
    );
}
