//! A huge `ef_search` or `k` must not overflow inside `knn_filtered` (review finding).
use quickdb_api::{
    ColumnDef, ColumnType, Filter, Metric, OpenOptions, Row, Store, TableDef, Value, VectorIndexDef,
};
use quickdb_native::NativeStore;

#[test]
fn huge_ef_and_k_do_not_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = NativeStore::open(
        dir.path(),
        &OpenOptions {
            create: true,
            sync_each_write: false,
        },
    )
    .unwrap();
    s.define_table(&TableDef {
        name: "docs".into(),
        columns: vec![ColumnDef {
            name: "kind".into(),
            ty: ColumnType::Text,
            indexed: true,
        }],
    })
    .unwrap();
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 2,
        metric: Metric::L2,
        m: 16,
        ef_construction: 100,
    })
    .unwrap();
    let rows: Vec<Row> = (1..=400u64)
        .map(|id| Row {
            id,
            values: vec![Value::Text(if id % 2 == 0 { "even" } else { "odd" }.into())],
        })
        .collect();
    s.upsert_rows("docs", &rows).unwrap();
    let data: Vec<(u64, Vec<f32>)> = (1..=400u64).map(|id| (id, vec![id as f32, 0.0])).collect();
    let items: Vec<(u64, &[f32])> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    s.upsert_vectors("emb", &items).unwrap();
    for kind in ["even", "odd", "none"] {
        let f = Filter::Eq {
            column: "kind".into(),
            value: Value::Text(kind.into()),
        };
        for (k, ef) in [
            (1, usize::MAX),
            (3, usize::MAX / 2),
            (usize::MAX, 1),
            (usize::MAX, usize::MAX),
        ] {
            let hits = s
                .knn_filtered("emb", &[10.2, 0.0], k, ef, "docs", &f)
                .unwrap();
            match kind {
                "none" => assert!(hits.is_empty()),
                "even" => assert_eq!(hits[0].id, 10),
                _ => assert_eq!(hits[0].id, 11),
            }
            assert_eq!(hits.len(), k.min(if kind == "none" { 0 } else { 200 }));
        }
    }
}
