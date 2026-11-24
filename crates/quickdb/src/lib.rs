//! QDB (QuickDB): relational rows, vectors and a directed graph in one embedded, in-process store.
//!
//! This crate is the front door. It re-exports the contract from [`quickdb_api`] (the [`Store`]
//! trait, its types and the [`conformance`] suite) and the engine [`NativeStore`] from
//! [`quickdb_native`]: bundled SQLite for rows and edges, plus a persisted, pure-Rust HNSW index
//! for vector search. There is no server and no external C or C++ toolchain to install.
//!
//! - A store lives in one directory. Row ids, vector ids and edge endpoints share one
//!   [`NodeId`] space (`0..=i64::MAX`).
//! - Writes are upserts. After [`Store::flush`] returns, data survives drop and reopen.
//! - Bad input is an [`Error`], never a panic.
//!
//! # Quick start
//!
//! ```
//! use quickdb::{
//!     ColumnDef, ColumnType, Direction, Edge, Filter, Metric, NativeStore, OpenOptions, Row,
//!     Store, TableDef, Value, VectorIndexDef,
//! };
//!
//! # fn main() -> quickdb::Result<()> {
//! let dir = std::env::temp_dir().join(format!("quickdb-doc-{}", std::process::id()));
//! # let _ = std::fs::remove_dir_all(&dir);
//! let mut db = NativeStore::open(&dir, &OpenOptions::default())?;
//!
//! // A table with an indexed integer column, and a 2-dimensional vector index.
//! let column = |name: &str, ty, indexed| ColumnDef { name: name.into(), ty, indexed };
//! db.define_table(&TableDef {
//!     name: "docs".into(),
//!     columns: vec![column("title", ColumnType::Text, false), column("year", ColumnType::Int, true)],
//! })?;
//! db.define_vector_index(&VectorIndexDef {
//!     name: "emb".into(),
//!     dim: 2,
//!     metric: Metric::L2,
//!     m: 16,
//!     ef_construction: 100,
//! })?;
//!
//! // Rows, vectors and edges use the same ids.
//! db.upsert_rows("docs", &[
//!     Row { id: 1, values: vec![Value::Text("intro".into()), Value::Int(2023)] },
//!     Row { id: 2, values: vec![Value::Text("graphs".into()), Value::Int(2025)] },
//!     Row { id: 3, values: vec![Value::Text("tuning".into()), Value::Int(2025)] },
//! ])?;
//! db.upsert_vectors("emb", &[(1, &[1.0, 0.0][..]), (2, &[0.9, 0.1][..]), (3, &[0.0, 1.0][..])])?;
//! db.upsert_edges(&[
//!     Edge { src: 1, rel: "cites".into(), dst: 2, weight: 1.0 },
//!     Edge { src: 2, rel: "cites".into(), dst: 3, weight: 1.0 },
//! ])?;
//! db.flush()?;
//!
//! // Nearest vectors, nearest among rows that match a filter, and a two-hop walk.
//! let query = [1.0, 0.0];
//! let nearest: Vec<u64> = db.knn("emb", &query, 2, 64)?.iter().map(|h| h.id).collect();
//! assert_eq!(nearest, [1, 2]);
//! let recent = Filter::IntRange { column: "year".into(), min: 2025, max: i64::MAX };
//! let hits = db.knn_filtered("emb", &query, 1, 64, "docs", &recent)?;
//! assert_eq!(hits[0].id, 2);
//! let reached = db.k_hop(1, Some("cites"), Direction::Out, 2, 10)?;
//! assert_eq!(reached.len(), 2);
//!
//! drop(db);
//! std::fs::remove_dir_all(&dir).ok();
//! # Ok(())
//! # }
//! ```
//!
//! To write another backend, implement [`Store`] and run [`conformance::run_all`] against it.

#![warn(missing_docs)]

pub use quickdb_api::*;
pub use quickdb_native::{fts, NativeStore, TextIndexDef};
