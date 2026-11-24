use quickdb_api::{
    conformance::{exact_distance, Lcg},
    *,
};
use quickdb_native::NativeStore;
use std::collections::{HashMap, HashSet, VecDeque};

fn directory() -> tempfile::TempDir {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    tempfile::Builder::new()
        .prefix("native-test-")
        .tempdir_in(root)
        .unwrap()
}
fn definition(dim: usize, metric: Metric) -> VectorIndexDef {
    VectorIndexDef {
        name: "emb".into(),
        dim,
        metric,
        m: 32,
        ef_construction: 400,
    }
}
fn truth(data: &[(u64, Vec<f32>)], q: &[f32], metric: Metric, k: usize) -> Vec<u64> {
    let mut scores: Vec<_> = data
        .iter()
        .map(|(id, v)| (exact_distance(metric, q, v), *id))
        .collect();
    scores.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scores.iter().take(k).map(|x| x.1).collect()
}
#[test]
fn recall_20k_64_and_reopen_without_rebuild() {
    let dir = directory();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    s.define_vector_index(&definition(64, Metric::Cosine))
        .unwrap();
    let mut rng = Lcg(9927);
    let data: Vec<_> = (0..20_000).map(|id| (id, rng.vector(64))).collect();
    let items: Vec<_> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    s.upsert_vectors("emb", &items).unwrap();
    let queries: Vec<_> = (0..100).map(|_| rng.vector(64)).collect();
    let mut found = 0;
    for q in &queries {
        let expected = truth(&data, q, Metric::Cosine, 10);
        let hits = s.knn("emb", q, 10, 128).unwrap();
        found += hits.iter().filter(|h| expected.contains(&h.id)).count();
    }
    let recall = found as f64 / 1000.0;
    println!("RECALL_20K_64 ef_search=128 recall_at_10={recall:.4}");
    assert!(recall >= 0.95, "recall={recall}");
    let before = s.knn("emb", &queries[0], 10, 128).unwrap();
    s.flush().unwrap();
    drop(s);
    let s = NativeStore::open(
        dir.path(),
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )
    .unwrap();
    assert_eq!(s.index_rebuilds(), 0);
    assert_eq!(s.knn("emb", &queries[0], 10, 128).unwrap(), before);
    println!(
        "REOPEN_WITHOUT_REBUILD indexes_rebuilt={}",
        s.index_rebuilds()
    );
}
#[test]
fn snapshots_missing_stale_and_corrupt_recover_from_sql() {
    let dir = directory();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    s.define_vector_index(&definition(2, Metric::L2)).unwrap();
    s.upsert_vectors("emb", &[(1, &[1., 0.]), (2, &[0., 1.])])
        .unwrap();
    s.flush().unwrap();
    s.upsert_vectors("emb", &[(1, &[10., 10.])]).unwrap();
    drop(s);
    for mode in ["stale", "missing", "corrupt"] {
        if mode == "missing" {
            std::fs::remove_file(dir.path().join("emb.hnsw")).unwrap();
        }
        if mode == "corrupt" {
            std::fs::write(dir.path().join("emb.hnsw"), b"not an index").unwrap();
        }
        let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        assert_eq!(s.index_rebuilds(), 1, "{mode}");
        assert_eq!(s.count_vectors("emb").unwrap(), 2);
        assert_eq!(s.knn("emb", &[10., 10.], 1, 128).unwrap()[0].id, 1);
        s.flush().unwrap();
    }
}
#[test]
fn filtered_selective_broad_and_vector_only_ids() {
    let dir = directory();
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
    s.define_vector_index(&definition(8, Metric::L2)).unwrap();
    let mut rng = Lcg(31);
    let data: Vec<_> = (0..3000).map(|id| (id, rng.vector(8))).collect();
    s.upsert_vectors(
        "emb",
        &data
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    s.upsert_rows(
        "docs",
        &(0..2900)
            .map(|id| Row {
                id,
                values: vec![Value::Int(if id.is_multiple_of(100) { 1 } else { 0 })],
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let q = rng.vector(8);
    for kind in [0, 1] {
        let filter = Filter::Eq {
            column: "kind".into(),
            value: Value::Int(kind),
        };
        let eligible: Vec<_> = data
            .iter()
            .filter(|(id, _)| *id < 2900 && (if id.is_multiple_of(100) { 1 } else { 0 }) == kind)
            .cloned()
            .collect();
        let hits = s
            .knn_filtered("emb", &q, 10, 3000, "docs", &filter)
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id).collect::<Vec<_>>(),
            truth(&eligible, &q, Metric::L2, 10)
        );
    }
    assert_eq!(
        s.knn_filtered("emb", &q, usize::MAX, 1, "docs", &Filter::All)
            .unwrap()
            .len(),
        2900
    );
}
fn reference(
    edges: &[Edge],
    start: u64,
    rel: Option<&str>,
    dir: Direction,
    depth: u32,
    limit: usize,
) -> Vec<Reached> {
    let mut adj = HashMap::<u64, Vec<u64>>::new();
    for e in edges.iter().filter(|e| rel.is_none_or(|r| e.rel == r)) {
        if matches!(dir, Direction::Out | Direction::Both) {
            adj.entry(e.src).or_default().push(e.dst);
        }
        if matches!(dir, Direction::In | Direction::Both) {
            adj.entry(e.dst).or_default().push(e.src);
        }
    }
    let mut queue = VecDeque::from([(start, 0)]);
    let mut seen = HashSet::from([start]);
    let mut result = Vec::new();
    while let Some((id, d)) = queue.pop_front() {
        if d == depth {
            continue;
        }
        for &n in adj.get(&id).into_iter().flatten() {
            if seen.insert(n) {
                result.push(Reached {
                    id: n,
                    depth: d + 1,
                });
                queue.push_back((n, d + 1));
            }
        }
    }
    result.sort_by_key(|r| (r.depth, r.id));
    result.truncate(limit);
    result
}
#[test]
fn generated_graph_three_hops_matches_reference() {
    let dir = directory();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    let mut rng = Lcg(982);
    let edges: Vec<_> = (0..3000)
        .flat_map(|src| {
            (0..5)
                .map(|j| Edge {
                    src,
                    rel: if j & 1 == 0 { "a".into() } else { "b".into() },
                    dst: u64::from(rng.next_u32()) % 3000,
                    weight: 1.0,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    s.upsert_edges(&edges).unwrap();
    for dir in [Direction::Out, Direction::In, Direction::Both] {
        for rel in [None, Some("a"), Some("missing")] {
            for start in [0, 391, 2999] {
                let expected = reference(&edges, start, rel, dir, 3, 70);
                assert_eq!(s.k_hop(start, rel, dir, 3, 70).unwrap(), expected);
                assert_eq!(s.k_hop_bfs(start, rel, dir, 3, 70).unwrap(), expected);
                assert_eq!(s.k_hop_sql(start, rel, dir, 3, 70).unwrap(), expected);
            }
        }
    }
}
#[test]
fn bad_inputs_are_errors_and_batches_are_atomic() {
    let dir = directory();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    s.define_table(&TableDef {
        name: "empty".into(),
        columns: vec![],
    })
    .unwrap();
    s.define_vector_index(&definition(2, Metric::L2)).unwrap();
    let bad = i64::MAX as u64 + 1;
    assert!(matches!(
        s.upsert_rows(
            "empty",
            &[
                Row {
                    id: 1,
                    values: vec![]
                },
                Row {
                    id: bad,
                    values: vec![]
                }
            ]
        ),
        Err(Error::Schema(_))
    ));
    assert_eq!(s.count_rows("empty").unwrap(), 0);
    assert!(matches!(s.get_row("empty", bad), Err(Error::Schema(_))));
    assert!(matches!(
        s.delete_rows("empty", &[bad]),
        Err(Error::Schema(_))
    ));
    assert!(matches!(
        s.upsert_vectors("emb", &[(1, &[1., 0.]), (bad, &[0., 1.])]),
        Err(Error::Schema(_))
    ));
    assert_eq!(s.count_vectors("emb").unwrap(), 0);
    assert!(matches!(
        s.knn("emb", &[f32::NAN, 0.], 1, 128),
        Err(Error::Schema(_))
    ));
    assert!(matches!(
        s.upsert_edges(&[Edge {
            src: 1,
            dst: bad,
            rel: "x".into(),
            weight: 1.0
        }]),
        Err(Error::Schema(_))
    ));
    assert!(matches!(
        s.neighbors(bad, None, Direction::Out, 1),
        Err(Error::Schema(_))
    ));
    assert!(matches!(
        s.k_hop(bad, None, Direction::Out, 1, 1),
        Err(Error::Schema(_))
    ));
    s.upsert_rows(
        "empty",
        &[Row {
            id: i64::MAX as u64,
            values: vec![],
        }],
    )
    .unwrap();
    assert_eq!(
        s.get_row("empty", i64::MAX as u64).unwrap().unwrap().id,
        i64::MAX as u64
    );
}
#[test]
fn text_prefix_is_literal_case_sensitive_and_byte_wise() {
    let dir = directory();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    s.define_table(&TableDef {
        name: "texts".into(),
        columns: vec![ColumnDef {
            name: "text".into(),
            ty: ColumnType::Text,
            indexed: true,
        }],
    })
    .unwrap();
    s.upsert_rows(
        "texts",
        &["A%b", "a%b", "a_b", "عربي", "", "a\0b"]
            .iter()
            .enumerate()
            .map(|(id, t)| Row {
                id: id as u64,
                values: vec![Value::Text((*t).into())],
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (prefix, ids) in [
        ("a%", vec![1]),
        ("a_", vec![2]),
        ("ع", vec![3]),
        ("a\0", vec![5]),
        ("", vec![0, 1, 2, 3, 4, 5]),
    ] {
        let rows = s
            .select(
                "texts",
                &Filter::TextPrefix {
                    column: "text".into(),
                    prefix: prefix.into(),
                },
                usize::MAX,
            )
            .unwrap();
        assert_eq!(rows.iter().map(|r| r.id).collect::<Vec<_>>(), ids);
    }
}
