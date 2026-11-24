use quickdb_api::*;
use quickdb_native::NativeStore;
use rusqlite::{params, Connection};

fn directory() -> tempfile::TempDir {
    let root = std::env::var_os("QUICKDB_DATA").unwrap_or_else(|| std::env::temp_dir().into());
    tempfile::tempdir_in(root).unwrap()
}

fn def() -> VectorIndexDef {
    VectorIndexDef {
        name: "emb".into(),
        dim: 2,
        metric: Metric::L2,
        m: 8,
        ef_construction: 32,
    }
}

#[test]
fn stored_negative_row_and_edge_ids_are_backend_errors() {
    let dir = directory();
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    store
        .define_table(&TableDef {
            name: "docs".into(),
            columns: vec![],
        })
        .unwrap();
    let db = Connection::open(dir.path().join("quickdb.sqlite")).unwrap();
    db.execute("INSERT INTO t_docs VALUES(-1)", []).unwrap();
    assert!(matches!(
        store.select("docs", &Filter::All, 10),
        Err(Error::Backend(_))
    ));
    db.execute("INSERT INTO edges VALUES(1,'r',-1,1.0)", [])
        .unwrap();
    assert!(matches!(
        store.neighbors(1, None, Direction::Out, 10),
        Err(Error::Backend(_))
    ));
    assert!(matches!(
        store.k_hop(1, None, Direction::Out, 1, 10),
        Err(Error::Backend(_))
    ));
    println!("SQL_HARDENING item=15 negative_row_and_edge_ids=backend_errors");
}

#[test]
fn damaged_vectors_name_index_and_id() {
    for bytes in [
        vec![1, 2, 3],
        [f32::NAN.to_le_bytes(), 1.0f32.to_le_bytes()].concat(),
    ] {
        let dir = directory();
        let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        store.define_vector_index(&def()).unwrap();
        store.upsert_vectors("emb", &[(42, &[1.0, 2.0])]).unwrap();
        store.flush().unwrap();
        drop(store);
        let db = Connection::open(dir.path().join("quickdb.sqlite")).unwrap();
        db.execute(
            "UPDATE vectors SET data=? WHERE name='emb' AND id=42",
            [bytes],
        )
        .unwrap();
        drop(db);
        std::fs::remove_file(dir.path().join("emb.hnsw")).unwrap();
        let error = match NativeStore::open(dir.path(), &OpenOptions::default()) {
            Ok(_) => panic!("damaged vector accepted"),
            Err(e) => e,
        };
        assert!(matches!(error, Error::Backend(_)));
        let message = error.to_string();
        assert!(
            message.contains("emb") && message.contains("42"),
            "{message}"
        );
        println!("SQL_HARDENING item=14 {message}");
    }
}

#[test]
fn snapshot_temps_removed_on_failure_and_open() {
    let dir = directory();
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    store.define_vector_index(&def()).unwrap();
    store.upsert_vectors("emb", &[(1, &[1.0, 2.0])]).unwrap();
    let snapshot = dir.path().join("emb.hnsw");
    let temp = dir.path().join("emb.hnsw.tmp");
    std::fs::create_dir(&snapshot).unwrap();
    assert!(store.flush().is_err());
    assert!(!temp.exists(), "failed rename left snapshot temp");
    std::fs::remove_dir(&snapshot).unwrap();
    store.flush().unwrap();
    drop(store);
    std::fs::write(&temp, b"left by interrupted writer").unwrap();
    let store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    assert!(!temp.exists());
    assert_eq!(store.count_vectors("emb").unwrap(), 1);
    println!("SQL_HARDENING item=12 failed_write_cleanup=true interrupted_write_open_cleanup=true");
}

#[test]
fn open_enters_wal_and_legacy_binary_indexes_still_work() {
    let dir = directory();
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    store
        .define_table(&TableDef {
            name: "docs".into(),
            columns: vec![ColumnDef {
                name: "text".into(),
                ty: ColumnType::Text,
                indexed: true,
            }],
        })
        .unwrap();
    drop(store);
    let db = Connection::open(dir.path().join("quickdb.sqlite")).unwrap();
    let mode: String = db
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!(mode, "wal");
    // Recreate the original schema, with implicit rather than explicit BINARY.
    db.execute_batch("DROP TABLE t_docs; CREATE TABLE t_docs(id INTEGER PRIMARY KEY,c_text TEXT); CREATE INDEX idx_4_docs_text ON t_docs(c_text)").unwrap();
    db.execute(
        "INSERT INTO t_docs VALUES(1,?),(2,?)",
        params!["a\u{7f}", "a\u{80}"],
    )
    .unwrap();
    drop(db);
    let store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    let rows = store
        .select(
            "docs",
            &Filter::TextPrefix {
                column: "text".into(),
                prefix: "a\u{7f}".into(),
            },
            10,
        )
        .unwrap();
    assert_eq!(rows.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1]);
    println!("SQL_HARDENING item=13 journal_mode=wal legacy_binary_index=true invalid_utf8_upper_bound=true");
}

#[test]
fn public_prefix_matches_byte_reference_indexed_and_scan() {
    let mut rng = quickdb_api::conformance::Lcg(99);
    let chars = [
        'a',
        'A',
        'b',
        'ك',
        'ت',
        '%',
        '_',
        '\0',
        '\u{7f}',
        '\u{7ff}',
        '\u{10ffff}',
    ];
    let mut rows: Vec<_> = (0..400)
        .map(|id| {
            let n = 2 + rng.next_u32() as usize % 11;
            let text: String = (0..n)
                .map(|_| chars[rng.next_u32() as usize % chars.len()])
                .collect();
            Row {
                id,
                values: vec![Value::Text(text)],
            }
        })
        .collect();
    rows.push(Row {
        id: 400,
        values: vec![Value::Null],
    });
    rows.push(Row {
        id: 401,
        values: vec![Value::Text(String::new())],
    });
    let mut prefixes = vec![String::new(), "not present".into()];
    while prefixes.len() < 2000 {
        let row = &rows[rng.next_u32() as usize % 400];
        if let Value::Text(text) = &row.values[0] {
            let n = rng.next_u32() as usize % (text.chars().count() + 1);
            prefixes.push(text.chars().take(n).collect());
        }
    }
    for indexed in [false, true] {
        let dir = directory();
        let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        store
            .define_table(&TableDef {
                name: "docs".into(),
                columns: vec![ColumnDef {
                    name: "text".into(),
                    ty: ColumnType::Text,
                    indexed,
                }],
            })
            .unwrap();
        store.upsert_rows("docs", &rows).unwrap();
        for (i, prefix) in prefixes.iter().enumerate() {
            let limit = [0, 1, 17, 1000][i % 4];
            let expected: Vec<_> = rows.iter().filter(|r|matches!(&r.values[0], Value::Text(text) if text.as_bytes().starts_with(prefix.as_bytes()))).take(limit).cloned().collect();
            assert_eq!(
                store
                    .select(
                        "docs",
                        &Filter::TextPrefix {
                            column: "text".into(),
                            prefix: prefix.clone()
                        },
                        limit
                    )
                    .unwrap(),
                expected,
                "indexed={indexed}, prefix={prefix:?}"
            );
        }
    }
    println!(
        "SQL_PUBLIC_PREFIX_EQUIVALENCE prefixes=2000 indexed_and_scan=true ordered_equal=true"
    );
}
