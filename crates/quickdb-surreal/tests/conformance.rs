use quickdb_api::{
    conformance, ColumnDef, ColumnType, Direction, Edge, Error, Filter, Metric, OpenOptions,
    Row, Store, TableDef, Value, VectorIndexDef,
};
use quickdb_surreal::{GraphLayout, SurrealStore};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let base = std::env::var_os("QUICKDB_DATA").map(PathBuf::from).unwrap_or_else(|| {
            if Path::new("/data").is_dir() { PathBuf::from("/data") } else { std::env::temp_dir() }
        });
        std::fs::create_dir_all(&base).expect("create test base");
        loop {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).expect("clock before epoch").as_nanos();
            let path = base.join(format!("quickdb-surreal-{label}-{}-{stamp}-{}",
                std::process::id(), NEXT_DIR.fetch_add(1, Ordering::Relaxed)));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create unique test directory: {e}"),
            }
        }
    }

    fn open(&self) -> SurrealStore {
        SurrealStore::open(&self.0.join("store"), &OpenOptions::default()).expect("open store")
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            eprintln!("cleanup own test directory {}: {e}", self.0.display());
        }
    }
}

fn table() -> TableDef {
    TableDef { name: "docs".into(), columns: vec![ColumnDef {
        name: "text".into(), ty: ColumnType::Text, indexed: true,
    }] }
}

fn index() -> VectorIndexDef {
    VectorIndexDef { name: "embedding".into(), dim: 2, metric: Metric::L2, m: 8, ef_construction: 32 }
}

fn schema<T>(result: quickdb_api::Result<T>) {
    assert!(matches!(result, Err(Error::Schema(_))), "expected Error::Schema");
}

#[test]
fn conformance() {
    let dir = TestDir::new("conformance");
    let failures = conformance::run_all::<SurrealStore>(&dir.0);
    for (name, failure) in &failures {
        eprintln!("conformance::{name}: {failure}");
    }
    assert!(failures.is_empty(), "{} named conformance failures", failures.len());
}

#[test]
fn input_validation() {
    let dir = TestDir::new("validation");
    let mut store = dir.open();
    for name in ["", "Bad", "a; DELETE docs", "a-b"] {
        schema(store.define_table(&TableDef { name: name.into(), columns: vec![] }));
        let mut def = index();
        def.name = name.into();
        schema(store.define_vector_index(&def));
    }
    let mut bad_column = table();
    bad_column.columns[0].name = "bad;name".into();
    schema(store.define_table(&bad_column));
    let mut duplicate = table();
    duplicate.columns.push(duplicate.columns[0].clone());
    schema(store.define_table(&duplicate));
    store.define_table(&table()).expect("table");
    schema(store.upsert_rows("docs", &[Row { id: 1, values: vec![] }]));
    schema(store.upsert_rows("docs", &[
        Row { id: 2, values: vec![Value::Text("valid".into())] },
        Row { id: 3, values: vec![Value::Int(7)] },
    ]));
    assert_eq!(store.count_rows("docs").expect("count"), 0, "invalid batch must be atomic");
    store.define_vector_index(&index()).expect("index");
    schema(store.upsert_vectors("embedding", &[(1, &[1.0])]));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        schema(store.upsert_vectors("embedding", &[(1, &[value, 0.0])]));
    }
    schema(store.knn("embedding", &[0.0], 1, 32));
    assert_eq!(store.count_vectors("embedding").expect("vector count"), 0);
}

#[test]
fn injection_safe_relation_and_text() {
    for layout in [GraphLayout::Plain, GraphLayout::Relation] {
        let dir = TestDir::new("injection");
        let mut store = dir.open();
        store.set_graph_layout(layout).expect("set layout");
        store.define_table(&table()).expect("table");
        let payload = "quote'\"; DELETE docs; --\n\\ العربية 🦀";
        let row = Row { id: 1, values: vec![Value::Text(payload.into())] };
        store.upsert_rows("docs", std::slice::from_ref(&row)).expect("parameterized text");
        assert_eq!(store.get_row("docs", 1).expect("get"), Some(row.clone()));
        for filter in [
            Filter::Eq { column: "text".into(), value: Value::Text(payload.into()) },
            Filter::TextPrefix { column: "text".into(), prefix: "quote'\";".into() },
        ] {
            assert_eq!(store.select("docs", &filter, 10).expect("text filter"), vec![row.clone()]);
        }
        let edge = Edge { src: 1, rel: payload.into(), dst: 2, weight: 0.5 };
        store.upsert_edges(std::slice::from_ref(&edge)).expect("parameterized relation");
        assert_eq!(store.neighbors(1, Some(payload), Direction::Out, 10).expect("neighbors"), vec![edge]);
        assert_eq!(store.k_hop(1, Some(payload), Direction::Out, 3, 10).expect("traverse").len(), 1);
        assert!(store.neighbors(1, Some("' OR true --"), Direction::Out, 10).expect("nonmatching relation").is_empty());
        assert_eq!(store.count_rows("docs").expect("table intact"), 1);
    }
}

#[test]
fn schema_and_graph_layout_persist() {
    for layout in [GraphLayout::Plain, GraphLayout::Relation] {
        let dir = TestDir::new("persistence");
        let row = Row { id: 7, values: vec![Value::Text("persisted".into())] };
        let edge = Edge { src: 7, rel: "link".into(), dst: 8, weight: 1.0 };
        {
            let mut store = dir.open();
            store.set_graph_layout(layout).expect("set layout");
            store.define_table(&table()).expect("table");
            store.define_vector_index(&index()).expect("index");
            store.upsert_rows("docs", std::slice::from_ref(&row)).expect("row");
            store.upsert_vectors("embedding", &[(7, &[1.0, 2.0])]).expect("vector");
            store.flush().expect("flush empty graph metadata");
        }
        {
            let mut store = dir.open();
            store.define_table(&table()).expect("identical persisted table");
            store.define_vector_index(&index()).expect("identical persisted index");
            let mut changed = table();
            changed.columns[0].ty = ColumnType::Int;
            schema(store.define_table(&changed));
            let mut changed = index();
            changed.dim = 3;
            schema(store.define_vector_index(&changed));
            assert_eq!(store.get_row("docs", 7).expect("row"), Some(row));
            assert_eq!(store.knn("embedding", &[1.0, 2.0], 1, 32).expect("search")[0].id, 7);
            store.upsert_edges(std::slice::from_ref(&edge)).expect("edge using persisted layout");
            store.flush().expect("flush graph");
        }
        let mut store = dir.open();
        assert_eq!(store.neighbors(7, Some("link"), Direction::Out, 10).expect("persisted edge"), vec![edge]);
        let other = match layout { GraphLayout::Plain => GraphLayout::Relation, GraphLayout::Relation => GraphLayout::Plain };
        schema(store.set_graph_layout(other));
        assert_eq!(store.count_edges().expect("graph intact after rejected switch"), 1);
    }
}

#[test]
fn node_id_bounds() {
    let dir = TestDir::new("bounds");
    let mut store = dir.open();
    store.define_table(&table()).expect("table");
    store.define_vector_index(&index()).expect("index");
    let max = i64::MAX as u64;
    let row = Row { id: max, values: vec![Value::Text("maximum".into())] };
    store.upsert_rows("docs", std::slice::from_ref(&row)).expect("maximum row id");
    assert_eq!(store.get_row("docs", max).expect("maximum get"), Some(row));
    store.upsert_vectors("embedding", &[(max, &[1.0, 0.0])]).expect("maximum vector id");
    store.upsert_edges(&[Edge { src: max, rel: "bound".into(), dst: 0, weight: 1.0 }]).expect("maximum endpoint");
    for bad in [max + 1, u64::MAX] {
        schema(store.upsert_rows("docs", &[Row { id: bad, values: vec![Value::Null] }]));
        schema(store.get_row("docs", bad));
        schema(store.delete_rows("docs", &[bad]));
        schema(store.upsert_vectors("embedding", &[(bad, &[1.0, 0.0])]));
        for (src, dst) in [(bad, 0), (0, bad)] {
            schema(store.upsert_edges(&[Edge { src, rel: "bound".into(), dst, weight: 1.0 }]));
        }
        schema(store.neighbors(bad, None, Direction::Both, 10));
        schema(store.k_hop(bad, None, Direction::Out, 3, 10));
    }
    assert_eq!(store.count_rows("docs").expect("count"), 1);
    assert_eq!(store.count_vectors("embedding").expect("count"), 1);
    assert_eq!(store.count_edges().expect("count"), 1);
}
