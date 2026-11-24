//! Regression tests for AGY review findings R1, R2, and R3.
use quickdb_api::{
    ColumnDef, ColumnType, Direction, Error, Filter, Hit, Metric, OpenOptions, Row, Store,
    TableDef, Value, VectorIndexDef,
};
use quickdb_native::NativeStore;
use std::collections::HashSet;

#[test]
fn r1_filtered_and_unfiltered_tie_breaking_orders_by_id_after_replace() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
    store
        .define_table(&TableDef {
            name: "docs".into(),
            columns: vec![ColumnDef {
                name: "p".into(),
                ty: ColumnType::Int,
                indexed: true,
            }],
        })
        .expect("define_table");
    store
        .define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: 2,
            metric: Metric::L2,
            m: 16,
            ef_construction: 64,
        })
        .expect("define_vector_index");

    // IDs 1..=20 sit at [1.0, 0.0] (distance 0.0 to query [1.0, 0.0]).
    // IDs 21..=5000 sit at [1.0, id as f32] so greedy routing descends smoothly to 1..=20.
    let vectors: Vec<(u64, [f32; 2])> = (1..=5000u64)
        .map(|id| {
            let v = if id <= 20 {
                [1.0, 0.0]
            } else {
                [1.0, id as f32]
            };
            (id, v)
        })
        .collect();
    let items: Vec<(u64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    store.upsert_vectors("emb", &items).expect("upsert_vectors");

    // Replace id = 1 so its live slot moves from node 0 to node 5000 (after nodes 1..=19).
    store
        .upsert_vectors("emb", &[(1, &[1.0, 0.0])])
        .expect("replace id 1");

    // Exclude the 256 sampled positions from the extra matches so estimate < lazy_from.
    let cap = 5001usize;
    let sampled_ids: HashSet<u64> = (0..256)
        .map(|i| {
            let pos = i * cap / 256;
            if pos < 5000 {
                (pos + 1) as u64
            } else {
                1
            }
        })
        .collect();

    let mut matching_ids: HashSet<u64> = (1..=20u64).collect();
    for id in 21..=5000u64 {
        if !sampled_ids.contains(&id) && matching_ids.len() < 1024 {
            matching_ids.insert(id);
        }
    }
    assert_eq!(matching_ids.len(), 1024);

    let rows: Vec<Row> = (1..=5000u64)
        .map(|id| Row {
            id,
            values: vec![Value::Int(i64::from(matching_ids.contains(&id)))],
        })
        .collect();
    store.upsert_rows("docs", &rows).expect("upsert_rows");

    let filter = Filter::Eq {
        column: "p".into(),
        value: Value::Int(1),
    };
    let query = [1.0f32, 0.0];

    // At 1,024 matches (allowed == exact_up_to), exact_nodes returns id = 1.
    let exact_hits = store
        .knn_filtered("emb", &query, 1, 16, "docs", &filter)
        .expect("exact knn_filtered");
    assert_eq!(
        exact_hits,
        vec![Hit {
            id: 1,
            distance: 0.0
        }]
    );

    // Adding one more match (allowed == 1,025 > exact_up_to) switches to masked search_filtered.
    let extra_id = (21..=5000u64)
        .find(|id| !sampled_ids.contains(id) && !matching_ids.contains(id))
        .expect("extra id");
    store
        .upsert_rows(
            "docs",
            &[Row {
                id: extra_id,
                values: vec![Value::Int(1)],
            }],
        )
        .expect("add 1025th match");

    let masked_hits = store
        .knn_filtered("emb", &query, 1, 16, "docs", &filter)
        .expect("masked knn_filtered");
    assert_eq!(
        masked_hits,
        vec![Hit {
            id: 1,
            distance: 0.0
        }],
        "masked search_filtered must break distance ties by ascending NodeId, not internal slot"
    );

    // Lazy search_filtered path (Filter::All gives estimate = 1.0 >= lazy_from).
    let lazy_hits = store
        .knn_filtered("emb", &query, 1, 16, "docs", &Filter::All)
        .expect("lazy knn_filtered");
    assert_eq!(
        lazy_hits,
        vec![Hit {
            id: 1,
            distance: 0.0
        }],
        "lazy search_filtered must break distance ties by ascending NodeId, not internal slot"
    );

    // Unfiltered Hnsw::search path (len = 5000 > 512, ef = 16 < cap).
    let unfiltered_hits = store.knn("emb", &query, 1, 16).expect("unfiltered knn");
    assert_eq!(
        unfiltered_hits,
        vec![Hit {
            id: 1,
            distance: 0.0
        }],
        "unfiltered knn must break distance ties by ascending NodeId, not internal slot"
    );
}

#[test]
fn r2_selectivity_sample_ignores_tombstones_and_handles_all_tombstone_sample() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
    store
        .define_table(&TableDef {
            name: "docs".into(),
            columns: vec![ColumnDef {
                name: "p".into(),
                ty: ColumnType::Int,
                indexed: true,
            }],
        })
        .expect("define_table");
    store
        .define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: 2,
            metric: Metric::L2,
            m: 16,
            ef_construction: 32,
        })
        .expect("define_vector_index");

    // 2,000 vectors:
    // - IDs 0..1999 sit at [10.0, 0.0] (distance 100.0 to [0.0, 0.0])
    // - ID 1999 sits at [0.0, 0.0] (distance 0.0 to [0.0, 0.0], disconnected from 0..32 because
    //   all 0..1999 are at [10.0, 0.0] and prune node 1999 when capping neighbor lists to 32).
    // Only 60 rows match p = 1: IDs 0..59 and ID 1999 (allowed = 60 <= exact_up_to = 1024).
    let vectors: Vec<(u64, [f32; 2])> = (0..2000u64)
        .map(|id| {
            let v = if id == 1999 { [0.0, 0.0] } else { [10.0, 0.0] };
            (id, v)
        })
        .collect();
    let items: Vec<(u64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    store.upsert_vectors("emb", &items).expect("upsert_vectors");

    let rows: Vec<Row> = (0..2000u64)
        .map(|id| Row {
            id,
            values: vec![Value::Int(i64::from(id < 59 || id == 1999))],
        })
        .collect();
    store.upsert_rows("docs", &rows).expect("upsert_rows");

    // Repeatedly replace the 58 matching vectors 1..59 without flush() (leaving id 0 live at
    // slot 0), filling 0..cap with tombstones whose IDs still have p = 1 in SQLite.
    let replacements: Vec<(u64, &[f32])> = (1..59u64).map(|id| (id, &[10.0f32, 0.0][..])).collect();
    for _ in 0..100 {
        store
            .upsert_vectors("emb", &replacements)
            .expect("replace matching subset");
    }

    let filter = Filter::Eq {
        column: "p".into(),
        value: Value::Int(1),
    };
    let hits = store
        .knn_filtered("emb", &[0.0, 0.0], 1, 16, "docs", &filter)
        .expect("knn_filtered after tombstone inflation");
    assert_eq!(
        hits,
        vec![Hit {
            id: 1999,
            distance: 0.0
        }],
        "sparse filter (60/2000 matches) must not be diverted to lazy search by tombstones"
    );

    // Second check: every one of the 256 evenly spaced positions (2 * i in 0..512) is a tombstone,
    // so zero live positions are sampled and the lazy path must be skipped cleanly.
    let dir2 = tempfile::tempdir().expect("tempdir2");
    let mut store2 = NativeStore::open(dir2.path(), &OpenOptions::default()).expect("open2");
    store2
        .define_table(&TableDef {
            name: "docs".into(),
            columns: vec![ColumnDef {
                name: "p".into(),
                ty: ColumnType::Int,
                indexed: true,
            }],
        })
        .expect("define_table2");
    store2
        .define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: 2,
            metric: Metric::L2,
            m: 8,
            ef_construction: 32,
        })
        .expect("define_vector_index2");
    let paired: Vec<(u64, [f32; 2])> = (0..256u64)
        .flat_map(|id| [(id, [id as f32, 0.0]), (id, [id as f32, 0.0])])
        .collect();
    let paired_items: Vec<(u64, &[f32])> =
        paired.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    store2
        .upsert_vectors("emb", &paired_items)
        .expect("paired upsert");
    let rows2: Vec<Row> = (0..256u64)
        .map(|id| Row {
            id,
            values: vec![Value::Int(1)],
        })
        .collect();
    store2.upsert_rows("docs", &rows2).expect("upsert_rows2");
    let hits2 = store2
        .knn_filtered("emb", &[42.0, 0.0], 1, 16, "docs", &filter)
        .expect("knn_filtered with all-tombstone sample");
    assert_eq!(
        hits2,
        vec![Hit {
            id: 42,
            distance: 0.0
        }]
    );
}

// R3: An unknown `Direction` variant cannot be constructed in safe external test code because
// `quickdb_api::Direction` is frozen and defines only `Out`, `In`, and `Both`. We verify that
// `k_hop_bfs` still validates `start` and returns empty on zero `max_depth` or `limit` across all
// known directions.
#[test]
fn r3_k_hop_bfs_validates_id_and_zero_bounds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
    assert!(matches!(
        store.k_hop_bfs(i64::MAX as u64 + 1, None, Direction::Out, 0, 0),
        Err(Error::Schema(_))
    ));
    for d in [Direction::Out, Direction::In, Direction::Both] {
        assert!(store.k_hop_bfs(1, None, d, 0, 10).unwrap().is_empty());
        assert!(store.k_hop_bfs(1, None, d, 2, 0).unwrap().is_empty());
    }
}
