use quickdb_api::*;
use quickdb_bench::{benchmark_table, corpus, exact_top10, run, synthetic, MemStore};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn scratch(label: &str) -> PathBuf {
    let base = std::env::var_os("QUICKDB_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join(format!(
        "bench-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
#[test]
fn mem_conformance() {
    let path = scratch("conformance");
    let failures = conformance::run_all::<MemStore>(&path);
    assert!(failures.is_empty(), "{failures:?}");
}
#[test]
fn fixture_corpus_loads_and_runs() {
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixture-corpus");
    let data = corpus(&input).expect("valid generated fixture");
    assert_eq!(data.rows.len(), 4);
    assert_eq!(data.vectors.len(), 4);
    assert_eq!(data.dim, 8);
    assert!(data.edges.iter().any(|e| e.rel == "links"));
    assert!(data
        .rows
        .iter()
        .any(|r| matches!(&r.values[4], Value::Text(t) if t.contains("العربية"))));
    let report = run::<MemStore>(&data, &scratch("fixture")).expect("corpus benchmark");
    assert_eq!(report.rows, 4);
    assert_eq!(report.queries["knn_ef64"].recall_at_10, Some(1.0));
    assert_eq!(report.queries["get_row"].samples, 10000);
    assert_eq!(report.queries["k_hop_depth3"].samples, 500);
}
#[test]
fn synthetic_is_deterministic_and_complete() {
    let a = synthetic(100, 16).expect("dataset");
    let b = synthetic(100, 16).expect("dataset");
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.vectors, b.vectors);
    assert_eq!(a.edges, b.edges);
    assert_eq!(a.edges.len(), 500);
    let report = run::<MemStore>(&a, &scratch("synthetic")).expect("synthetic benchmark");
    assert_eq!(report.queries.len(), 12);
    for ef in [32, 64, 128, 256] {
        assert_eq!(
            report.queries[&format!("knn_ef{ef}")].recall_at_10,
            Some(1.0)
        );
    }
    assert!(synthetic(0, 1).is_err());
    assert!(synthetic(1, 0).is_err());
}
#[test]
fn invalid_batches_are_atomic_and_ids_bounded() {
    let path = scratch("bad");
    let mut s = MemStore::open(&path, &OpenOptions::default()).expect("open");
    s.define_table(&benchmark_table()).expect("schema");
    let data = synthetic(2, 2).expect("dataset");
    let mut bad = data.rows[1].clone();
    bad.id = u64::MAX;
    assert!(matches!(
        s.upsert_rows("docs", &[data.rows[0].clone(), bad]),
        Err(Error::Schema(_))
    ));
    assert_eq!(s.count_rows("docs").expect("count"), 0);
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 2,
        metric: Metric::L2,
        m: 16,
        ef_construction: 100,
    })
    .expect("index");
    assert!(s
        .upsert_vectors("emb", &[(1, &[1.0, 2.0]), (2, &[f32::NAN, 0.0])])
        .is_err());
    assert_eq!(s.count_vectors("emb").expect("count"), 0);
    assert!(s
        .upsert_edges(&[
            Edge {
                src: 1,
                rel: "next".into(),
                dst: 2,
                weight: 1.0
            },
            Edge {
                src: u64::MAX,
                rel: "next".into(),
                dst: 1,
                weight: 1.0
            }
        ])
        .is_err());
    assert_eq!(s.count_edges().expect("count"), 0);
    assert!(matches!(
        s.k_hop(u64::MAX, None, Direction::Out, 1, 10),
        Err(Error::Schema(_))
    ));
    assert!(matches!(s.get_row("docs", u64::MAX), Err(Error::Schema(_))));
}
#[test]
fn graph_limit_orders_entire_frontier() {
    let mut s = MemStore::open(&scratch("bfs"), &OpenOptions::default()).expect("open");
    s.upsert_edges(
        &[(1, 20), (1, 10), (10, 100), (20, 2), (20, 1)].map(|(src, dst)| Edge {
            src,
            dst,
            rel: "next".into(),
            weight: 1.0,
        }),
    )
    .expect("edges");
    assert_eq!(
        s.k_hop(1, None, Direction::Out, 2, 3).expect("bfs"),
        vec![
            Reached { id: 10, depth: 1 },
            Reached { id: 20, depth: 1 },
            Reached { id: 2, depth: 2 }
        ]
    );
}
#[test]
fn exact_truth_ties_and_filter() {
    let data = vec![
        (3, vec![1.0, 0.0]),
        (1, vec![1.0, 0.0]),
        (2, vec![0.0, 1.0]),
    ];
    assert_eq!(exact_top10(&data, &[1.0, 0.0], None), vec![1, 3, 2]);
    assert_eq!(
        exact_top10(&data, &[1.0, 0.0], Some(&BTreeSet::from([2, 3]))),
        vec![3, 2]
    );
}
#[test]
fn refuses_nonempty_data_dir() {
    let path = scratch("existing");
    std::fs::create_dir_all(&path).expect("mkdir");
    std::fs::write(path.join("keep"), "keep").expect("write");
    assert!(run::<MemStore>(&synthetic(2, 2).expect("dataset"), &path).is_err());
    assert_eq!(
        std::fs::read_to_string(path.join("keep")).expect("read"),
        "keep"
    );
}
#[test]
fn sync_each_write_and_corrupt_snapshot() {
    let path = scratch("sync");
    let mut s = MemStore::open(
        &path,
        &OpenOptions {
            create: true,
            sync_each_write: true,
        },
    )
    .expect("open");
    s.define_table(&benchmark_table()).expect("schema");
    s.upsert_rows("docs", &synthetic(2, 2).expect("dataset").rows)
        .expect("write");
    drop(s);
    let reopened = MemStore::open(
        &path,
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )
    .expect("reopen");
    assert_eq!(reopened.count_rows("docs").expect("count"), 2);
    drop(reopened);
    std::fs::write(path.join("mem.json"), "not json").expect("corrupt");
    assert!(MemStore::open(&path, &OpenOptions::default()).is_err());
}
