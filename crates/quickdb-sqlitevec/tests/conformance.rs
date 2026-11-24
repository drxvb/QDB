use quickdb_api::*;
use quickdb_sqlitevec::SqliteVecStore;

fn store() -> (tempfile::TempDir, SqliteVecStore) {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::tempdir_in(root).expect("temporary directory");
    let store = SqliteVecStore::open(dir.path(), &OpenOptions::default()).expect("open store");
    (dir, store)
}
fn index(store: &mut SqliteVecStore, metric: Metric) {
    store
        .define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: 2,
            metric,
            m: 16,
            ef_construction: 100,
        })
        .expect("define vector index");
}
fn docs(store: &mut SqliteVecStore) {
    store
        .define_table(&TableDef {
            name: "docs".into(),
            columns: vec![ColumnDef {
                name: "tag".into(),
                ty: ColumnType::Text,
                indexed: true,
            }],
        })
        .expect("define rows");
}
fn row(id: u64, tag: &str) -> Row {
    Row {
        id,
        values: vec![Value::Text(tag.into())],
    }
}

#[test]
fn conformance() {
    let (dir, s) = store();
    drop(s);
    let failures = conformance::run_all::<SqliteVecStore>(dir.path());
    for (check, message) in &failures {
        eprintln!("CONFORMANCE check={check} failure={message}");
    }
    assert!(failures.is_empty(), "{failures:?}");
    println!(
        "CONFORMANCE backend=sqlite-vec relational=pass vectors=pass graph=pass persistence=pass"
    );
}

#[test]
fn l2_is_squared_and_ef_is_ignored() {
    let (_dir, mut s) = store();
    assert_eq!(s.backend_name(), "sqlite-vec");
    index(&mut s, Metric::L2);
    s.upsert_vectors("emb", &[(0, &[3.0, 4.0]), (1, &[6.0, 8.0])])
        .unwrap();
    let hits = s.knn("emb", &[0.0, 0.0], 10, 0).unwrap();
    assert_eq!(
        hits,
        vec![
            Hit {
                id: 0,
                distance: 25.0
            },
            Hit {
                id: 1,
                distance: 100.0
            }
        ]
    );
    assert_eq!(hits, s.knn("emb", &[0.0, 0.0], 10, 1000).unwrap());
    assert!(s.knn("emb", &[0.0, 0.0], 0, 0).unwrap().is_empty());
}

#[test]
fn ties_are_ordered_by_id_even_at_cutoff() {
    let (_dir, mut s) = store();
    index(&mut s, Metric::L2);
    s.upsert_vectors(
        "emb",
        &[
            (50, &[1.0, 0.0]),
            (30, &[-1.0, 0.0]),
            (20, &[0.0, 1.0]),
            (10, &[0.0, -1.0]),
        ],
    )
    .unwrap();
    let hits = s.knn("emb", &[0.0, 0.0], 2, 0).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![10, 20]);
}

#[test]
fn filtering_is_before_top_k_and_tracks_row_changes() {
    let (_dir, mut s) = store();
    index(&mut s, Metric::L2);
    docs(&mut s);
    s.upsert_rows(
        "docs",
        &[row(1, "no"), row(2, "no"), row(3, "yes"), row(4, "yes")],
    )
    .unwrap();
    s.upsert_vectors(
        "emb",
        &[
            (0, &[0.0, 0.0]),
            (1, &[1.0, 0.0]),
            (2, &[2.0, 0.0]),
            (3, &[3.0, 0.0]),
            (4, &[4.0, 0.0]),
        ],
    )
    .unwrap();
    let f = Filter::Eq {
        column: "tag".into(),
        value: Value::Text("yes".into()),
    };
    let hits = s
        .knn_filtered("emb", &[0.0, 0.0], 2, 0, "docs", &f)
        .unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![3, 4]);
    s.upsert_rows("docs", &[row(1, "yes")]).unwrap();
    s.delete_rows("docs", &[3]).unwrap();
    let hits = s
        .knn_filtered("emb", &[0.0, 0.0], 2, 0, "docs", &f)
        .unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 4]);
    assert_eq!(s.count_vectors("emb").unwrap(), 5);
    assert_eq!(
        s.knn_filtered("emb", &[0.0, 0.0], 10, 0, "docs", &Filter::All)
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn cosine_zero_norm_and_zero_query_follow_contract() {
    let (_dir, mut s) = store();
    index(&mut s, Metric::Cosine);
    s.upsert_vectors(
        "emb",
        &[(3, &[0.0, 0.0]), (2, &[-1.0, 0.0]), (1, &[1.0, 0.0])],
    )
    .unwrap();
    assert_eq!(
        s.knn("emb", &[1.0, 0.0], 3, 0).unwrap(),
        vec![
            Hit {
                id: 1,
                distance: 0.0
            },
            Hit {
                id: 3,
                distance: 1.0
            },
            Hit {
                id: 2,
                distance: 2.0
            }
        ]
    );
    assert_eq!(
        s.knn("emb", &[0.0, 0.0], 2, 0).unwrap(),
        vec![
            Hit {
                id: 1,
                distance: 1.0
            },
            Hit {
                id: 2,
                distance: 1.0
            }
        ]
    );
    s.upsert_vectors("emb", &[(3, &[0.0, 1.0])]).unwrap();
    assert_eq!(s.knn("emb", &[0.0, 1.0], 1, 0).unwrap()[0].id, 3);
}

#[test]
fn vector_batch_rejection_is_atomic_and_duplicate_ids_replace() {
    let (_dir, mut s) = store();
    index(&mut s, Metric::L2);
    s.upsert_vectors("emb", &[(1, &[1.0, 0.0]), (1, &[2.0, 0.0])])
        .unwrap();
    assert_eq!(s.count_vectors("emb").unwrap(), 1);
    assert_eq!(s.knn("emb", &[0.0, 0.0], 1, 0).unwrap()[0].distance, 4.0);
    for bad in [vec![1.0], vec![f32::NAN, 1.0], vec![f32::INFINITY, 0.0]] {
        assert!(matches!(
            s.upsert_vectors("emb", &[(2, &[3.0, 0.0]), (3, &bad)]),
            Err(Error::Schema(_))
        ));
        assert_eq!(s.count_vectors("emb").unwrap(), 1);
    }
    assert!(matches!(
        s.upsert_vectors("emb", &[(i64::MAX as u64 + 1, &[1.0, 0.0])]),
        Err(Error::Schema(_))
    ));
    assert!(matches!(
        s.knn("emb", &[f32::NAN, 0.0], 0, 0),
        Err(Error::Schema(_))
    ));
}

#[test]
fn large_k_is_not_limited_to_vec0_match_cap() {
    let (_dir, mut s) = store();
    index(&mut s, Metric::L2);
    let data: Vec<_> = (0..4100u64).map(|id| (id, [id as f32, 1.0])).collect();
    let items: Vec<_> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    s.upsert_vectors("emb", &items).unwrap();
    let hits = s.knn("emb", &[0.0, 0.0], 5000, 0).unwrap();
    assert_eq!(hits.len(), 4100);
    assert_eq!(hits.first().unwrap().id, 0);
    assert_eq!(hits.last().unwrap().id, 4099);
}

#[test]
fn byte_prefixes_are_literal_case_sensitive_and_null_safe() {
    let (_dir, mut s) = store();
    docs(&mut s);
    s.upsert_rows(
        "docs",
        &[
            row(1, "a%_\0x"),
            row(2, "a%_\0y"),
            row(3, "A%_\0x"),
            row(4, "ax"),
            Row {
                id: 5,
                values: vec![Value::Null],
            },
        ],
    )
    .unwrap();
    let f = Filter::TextPrefix {
        column: "tag".into(),
        prefix: "a%_\0".into(),
    };
    assert_eq!(
        s.select("docs", &f, 10)
            .unwrap()
            .iter()
            .map(|r| r.id)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        s.select(
            "docs",
            &Filter::TextPrefix {
                column: "tag".into(),
                prefix: String::new()
            },
            10
        )
        .unwrap()
        .len(),
        4
    );
}

#[test]
fn one_writer_and_reopen_with_full_durability() {
    let (dir, mut s) = store();
    assert!(SqliteVecStore::open(dir.path(), &OpenOptions::default()).is_err());
    index(&mut s, Metric::L2);
    s.upsert_vectors("emb", &[(i64::MAX as u64, &[3.0, 4.0])])
        .unwrap();
    s.flush().unwrap();
    drop(s);
    let mut s = SqliteVecStore::open(
        dir.path(),
        &OpenOptions {
            create: false,
            sync_each_write: true,
        },
    )
    .unwrap();
    assert_eq!(
        s.knn("emb", &[0.0, 0.0], 1, 0).unwrap()[0].id,
        i64::MAX as u64
    );
    s.upsert_vectors("emb", &[(0, &[0.0, 0.0])]).unwrap();
    s.flush().unwrap();
}
