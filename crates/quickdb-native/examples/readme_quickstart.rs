//! Copy-and-run tour of QuickDB: one store holding rows, vectors and graph edges.
//!
//! Run with `cargo run --release --example readme_quickstart`. The store lives in a
//! temporary directory that the example removes at the end.
//!
//! The contract has no multi-call transaction. Each write call is atomic and `flush()`
//! makes everything written so far durable, so the example writes rows, vectors and edges
//! with three calls and one `flush()`.

use std::path::Path;

use quickdb_api::{
    ColumnDef, ColumnType, Direction, Edge, Filter, Metric, OpenOptions, Reached, Row, Store,
    TableDef, Value, VectorIndexDef,
};
use quickdb_native::NativeStore;

type BoxError = Box<dyn std::error::Error>;

/// What the quickstart observed, so the test can assert it and `main` can print it.
#[derive(Debug, PartialEq)]
pub struct Report {
    pub nearest: Vec<u64>,
    pub nearest_recent: Vec<u64>,
    pub two_hops: Vec<Reached>,
    pub rows_after_reopen: u64,
    pub vectors_after_reopen: u64,
    pub edges_after_reopen: u64,
    pub title_of_3_after_reopen: Option<String>,
    pub nearest_after_reopen: Vec<u64>,
}

/// Documents: (id, title, year, vector). Ids are shared by rows, vectors and edges.
const DOCS: [(u64, &str, i64, [f32; 4]); 6] = [
    (1, "intro", 2022, [1.0, 0.0, 0.0, 0.0]),
    (2, "basics", 2023, [0.9, 0.1, 0.0, 0.0]),
    (3, "indexes", 2024, [0.0, 1.0, 0.0, 0.0]),
    (4, "graphs", 2025, [0.0, 0.9, 0.1, 0.0]),
    (5, "tuning", 2025, [0.0, 0.0, 1.0, 0.0]),
    (6, "recap", 2024, [0.8, 0.2, 0.1, 0.0]),
];

/// `(src, dst)` pairs of the relation `cites`.
const CITES: [(u64, u64); 5] = [(1, 2), (2, 3), (2, 4), (3, 5), (6, 1)];

pub fn run(dir: &Path) -> Result<Report, BoxError> {
    let query = [1.0_f32, 0.0, 0.0, 0.0];
    let ids = |hits: Vec<quickdb_api::Hit>| hits.into_iter().map(|h| h.id).collect::<Vec<_>>();

    // 1. Open (creating) the store in `dir`.
    let mut db = NativeStore::open(dir, &OpenOptions::default())?;

    // 2. A table with an indexed integer column, and a vector index.
    db.define_table(&TableDef {
        name: "docs".into(),
        columns: vec![
            ColumnDef {
                name: "title".into(),
                ty: ColumnType::Text,
                indexed: false,
            },
            ColumnDef {
                name: "year".into(),
                ty: ColumnType::Int,
                indexed: true,
            },
        ],
    })?;
    db.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 4,
        metric: Metric::L2,
        m: 16,
        ef_construction: 100,
    })?;

    // 3. Rows, vectors and edges, then one flush.
    let rows: Vec<Row> = DOCS
        .iter()
        .map(|(id, title, year, _)| Row {
            id: *id,
            values: vec![Value::Text((*title).into()), Value::Int(*year)],
        })
        .collect();
    db.upsert_rows("docs", &rows)?;
    let vectors: Vec<(u64, &[f32])> = DOCS
        .iter()
        .map(|(id, _, _, v)| (*id, v.as_slice()))
        .collect();
    db.upsert_vectors("emb", &vectors)?;
    let edges: Vec<Edge> = CITES
        .iter()
        .map(|(src, dst)| Edge {
            src: *src,
            rel: "cites".into(),
            dst: *dst,
            weight: 1.0,
        })
        .collect();
    db.upsert_edges(&edges)?;
    db.flush()?;

    // 4. The three nearest vectors to `query`, then the two nearest among documents from 2024 or later.
    let nearest = ids(db.knn("emb", &query, 3, 64)?);
    let recent = Filter::IntRange {
        column: "year".into(),
        min: 2024,
        max: i64::MAX,
    };
    let nearest_recent = ids(db.knn_filtered("emb", &query, 2, 64, "docs", &recent)?);

    // 5. Everything reachable from document 1 in at most two `cites` hops.
    let two_hops = db.k_hop(1, Some("cites"), Direction::Out, 2, 100)?;

    // 6. Close and reopen: the data is still there.
    drop(db);
    let db = NativeStore::open(
        dir,
        &OpenOptions {
            create: false,
            ..OpenOptions::default()
        },
    )?;
    let title_of_3_after_reopen = match db
        .get_row("docs", 3)?
        .and_then(|r| r.values.into_iter().next())
    {
        Some(Value::Text(t)) => Some(t),
        _ => None,
    };
    Ok(Report {
        nearest,
        nearest_recent,
        two_hops,
        rows_after_reopen: db.count_rows("docs")?,
        vectors_after_reopen: db.count_vectors("emb")?,
        edges_after_reopen: db.count_edges()?,
        title_of_3_after_reopen,
        nearest_after_reopen: ids(db.knn("emb", &query, 3, 64)?),
    })
}

// Tests include this file as a module and never call `main`.
#[allow(dead_code)]
fn main() -> Result<(), BoxError> {
    let dir = std::env::temp_dir().join(format!("quickdb-quickstart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let report = run(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    println!("{:#?}", report?);
    Ok(())
}
