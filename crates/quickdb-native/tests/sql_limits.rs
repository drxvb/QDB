use quickdb_api::*;
use quickdb_native::NativeStore;

#[test]
fn cached_limits_preserve_rows_and_all_neighbor_shapes() {
    let root = std::env::var_os("QUICKDB_DATA").unwrap_or_else(|| std::env::temp_dir().into());
    let dir = tempfile::tempdir_in(root).unwrap();
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    for indexed in [false, true] {
        let name = if indexed { "indexed" } else { "scan" };
        store
            .define_table(&TableDef {
                name: name.into(),
                columns: vec![ColumnDef {
                    name: "word".into(),
                    ty: ColumnType::Text,
                    indexed,
                }],
            })
            .unwrap();
        let rows: Vec<_> = (0..200)
            .rev()
            .map(|id| Row {
                id,
                values: vec![if id.is_multiple_of(3) {
                    Value::Null
                } else {
                    Value::Text(format!("word{}", id % 2))
                }],
            })
            .collect();
        store.upsert_rows(name, &rows).unwrap();
        for limit in [0, 1, 3, usize::MAX, 2, 0, 100] {
            for value in [
                Value::Text("word0".into()),
                Value::Text("missing".into()),
                Value::Null,
            ] {
                let mut expected: Vec<_> = rows
                    .iter()
                    .filter(|r| value != Value::Null && r.values[0] == value)
                    .cloned()
                    .collect();
                expected.sort_by_key(|r| r.id);
                expected.truncate(limit);
                assert_eq!(
                    store
                        .select(
                            name,
                            &Filter::Eq {
                                column: "word".into(),
                                value
                            },
                            limit
                        )
                        .unwrap(),
                    expected
                );
            }
        }
    }
    let mut edges = Vec::new();
    for id in 0..40 {
        edges.push(Edge {
            src: 7,
            rel: format!("r{}", id % 3),
            dst: id,
            weight: id as f32,
        });
        edges.push(Edge {
            src: id,
            rel: "back".into(),
            dst: 7,
            weight: 0.5,
        });
    }
    store.upsert_edges(&edges).unwrap();
    edges.sort_by(|a, b| {
        a.rel
            .cmp(&b.rel)
            .then(a.src.cmp(&b.src))
            .then(a.dst.cmp(&b.dst))
    });
    for dir in [Direction::Out, Direction::In, Direction::Both] {
        for rel in [None, Some("r0"), Some("back"), Some("missing")] {
            for limit in [0, 1, 17, usize::MAX, 3, 0, 100] {
                let expected: Vec<_> = edges
                    .iter()
                    .filter(|e| {
                        (match dir {
                            Direction::Out => e.src == 7,
                            Direction::In => e.dst == 7,
                            Direction::Both => e.src == 7 || e.dst == 7,
                            _ => unreachable!("the loop covers every direction"),
                        }) && rel.is_none_or(|r| e.rel == r)
                    })
                    .take(limit)
                    .cloned()
                    .collect();
                assert_eq!(store.neighbors(7, rel, dir, limit).unwrap(), expected);
            }
        }
    }
    println!("SQL_LIMIT_REGRESSION eq_index_and_scan=true directions=3 relation_shapes=4 zero_and_max_limits=true self_loops_once=true");
}
