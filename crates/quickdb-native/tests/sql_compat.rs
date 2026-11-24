//! Files written by one SQLite binding open with another. `scripts/sqlpair.sh` runs the
//! `write` phase with the old build and the `read` phase with the new one on the same directory.
//! Without `QUICKDB_COMPAT_DIR` both phases run in one temporary directory.
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::path::Path;

fn vector(id: u64) -> Vec<f32> {
    (0..16)
        .map(|j| ((id * 31 + j) % 97) as f32 / 97.0)
        .collect()
}

fn write(dir: &Path) {
    let mut s = NativeStore::open(dir, &OpenOptions::default()).unwrap();
    s.define_table(&TableDef {
        name: "items".into(),
        columns: vec![ColumnDef {
            name: "name".into(),
            ty: ColumnType::Text,
            indexed: true,
        }],
    })
    .unwrap();
    let rows: Vec<_> = (0..3000)
        .map(|id| Row {
            id,
            values: vec![Value::Text(format!("n{id:05}"))],
        })
        .collect();
    s.upsert_rows("items", &rows).unwrap();
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 16,
        metric: Metric::L2,
        m: 16,
        ef_construction: 100,
    })
    .unwrap();
    let vectors: Vec<_> = (0..3000).map(vector).collect();
    let items: Vec<_> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    s.upsert_vectors("emb", &items).unwrap();
    let edges: Vec<_> = (0..6000)
        .map(|i| Edge {
            src: i % 3000,
            rel: format!("r{}", i / 3000),
            dst: (i * 7) % 3000,
            weight: 1.0,
        })
        .collect();
    s.upsert_edges(&edges).unwrap();
    s.flush().unwrap();
}

fn read(dir: &Path) {
    let s = NativeStore::open(dir, &OpenOptions::default()).unwrap();
    assert_eq!(
        s.index_rebuilds(),
        0,
        "snapshot written by the other binding was rebuilt"
    );
    assert_eq!(s.count_rows("items").unwrap(), 3000);
    assert_eq!(s.count_vectors("emb").unwrap(), 3000);
    assert_eq!(s.count_edges().unwrap(), 6000);
    for id in [0u64, 1234, 2999] {
        let row = s.get_row("items", id).unwrap().unwrap();
        assert_eq!(row.values, vec![Value::Text(format!("n{id:05}"))]);
        let hits = s.knn("emb", &vector(id), 1, 64).unwrap();
        assert_eq!(hits[0].distance, 0.0);
        let mut out = s.neighbors(id, None, Direction::Out, 10).unwrap();
        out.sort_by_key(|e| (e.rel.clone(), e.dst));
        let mut expected: Vec<_> = (0..6000u64)
            .filter(|i| i % 3000 == id)
            .map(|i| Edge {
                src: id,
                rel: format!("r{}", i / 3000),
                dst: (i * 7) % 3000,
                weight: 1.0,
            })
            .collect();
        expected.sort_by_key(|e| (e.rel.clone(), e.dst));
        assert_eq!(out, expected);
    }
    let prefixed = s
        .select(
            "items",
            &Filter::TextPrefix {
                column: "name".into(),
                prefix: "n012".into(),
            },
            1000,
        )
        .unwrap();
    assert_eq!(prefixed.len(), 100);
    println!("SQL_COMPAT read ok rebuilds=0 rows=3000 vectors=3000 edges=6000");
}

#[test]
fn files_cross_sqlite_bindings() {
    match (
        std::env::var_os("QUICKDB_COMPAT_DIR"),
        std::env::var("QUICKDB_COMPAT_PHASE").ok(),
    ) {
        (Some(dir), Some(phase)) => {
            let dir = Path::new(&dir);
            if phase == "write" {
                std::fs::create_dir_all(dir).unwrap();
                write(dir);
                println!("SQL_COMPAT write ok");
            } else {
                read(dir);
            }
        }
        _ => {
            let root =
                std::env::var_os("QUICKDB_DATA").unwrap_or_else(|| std::env::temp_dir().into());
            let dir = tempfile::tempdir_in(root).unwrap();
            write(dir.path());
            read(dir.path());
        }
    }
}
