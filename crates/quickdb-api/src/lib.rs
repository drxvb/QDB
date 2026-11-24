//! The QuickDB contract: one embedded, in-process store for relational rows, vectors and graph
//! edges that share one id space. Every backend implements [`Store`] and must pass
//! [`conformance::run_all`]; the benchmark harness is generic over `Store`.
//!
//! - A store lives in one directory. Everything it writes stays inside it.
//! - Row `id`, vector id and edge endpoints share one [`NodeId`] space, at most `i64::MAX`;
//!   a larger id is [`Error::Schema`] (checked at run time, `NodeId` is a plain `u64`).
//! - Writes are upserts. After [`Store::flush`] returns, data survives drop and reopen.
//! - Reads take `&self`, writes take `&mut self`. Bad input is an [`Error`], never a panic.
//! - [`Value`], [`Filter`], [`Metric`], [`ColumnType`], [`Direction`] and [`Error`] are
//!   `#[non_exhaustive]`: match them with a wildcard arm. A backend answers a variant it does not
//!   know with `Error::Schema("unsupported …")`.
//!
//! The crate ships no store, so this example is generic over `S: Store`: compiled, not run.
//! ```
//! use quickdb_api::{Filter, OpenOptions, Result, Row, Store, TableDef, Value};
//!
//! fn first_titles<S: Store>(dir: &std::path::Path, def: &TableDef) -> Result<Vec<Row>> {
//!     let mut db = S::open(dir, &OpenOptions::default())?;
//!     db.define_table(def)?;
//!     db.upsert_rows("docs", &[Row { id: 1, values: vec![Value::Text("a".into())] }])?;
//!     db.flush()?;
//!     db.select("docs", &Filter::All, 10)
//! }
//! ```

#![warn(missing_docs)]

use std::fmt;
use std::path::Path;

pub mod conformance;

/// Identifier shared by rows, vectors and graph nodes.
///
/// Valid values are `0..=i64::MAX`. The type is a plain `u64` so that call sites stay simple, and
/// the limit is enforced at run time: every [`Store`] method returns [`Error::Schema`] for a larger
/// id, so that backends can keep ids in a signed 64-bit column.
pub type NodeId = u64;

/// One cell of a row.
///
/// The variant must match the column's [`ColumnType`], except [`Value::Null`], which is allowed in
/// every column. A mismatch is [`Error::Schema`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Value {
    /// No value. Stored in any column type; never matches [`Filter::Eq`].
    Null,
    /// Signed 64-bit integer, for a [`ColumnType::Int`] column.
    Int(i64),
    /// 64-bit float, for a [`ColumnType::Float`] column.
    Float(f64),
    /// UTF-8 text, for a [`ColumnType::Text`] column. Compared byte-wise.
    Text(String),
    /// Arbitrary bytes, for a [`ColumnType::Bytes`] column.
    Bytes(Vec<u8>),
}

/// The type of a column. Each type accepts exactly the matching [`Value`] variant, and `Null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColumnType {
    /// Signed 64-bit integer ([`Value::Int`]). The only type [`Filter::IntRange`] accepts.
    Int,
    /// 64-bit float ([`Value::Float`]).
    Float,
    /// UTF-8 text ([`Value::Text`]). The only type [`Filter::TextPrefix`] accepts.
    Text,
    /// Arbitrary bytes ([`Value::Bytes`]).
    Bytes,
}

/// One column of a [`TableDef`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    /// Column name; must satisfy [`valid_name`]. Unique within the table.
    pub name: String,
    /// Type that every non-null value in this column must have.
    pub ty: ColumnType,
    /// Ask the backend for a secondary index on this column (used by `select`).
    /// It changes speed only, never results. It is part of the definition: redefining the table
    /// with a different `indexed` flag is [`Error::Schema`].
    pub indexed: bool,
}

/// A table. Every table has an implicit primary key `id: NodeId`; `columns` are the rest.
/// Table and column names match `[a-z][a-z0-9_]*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDef {
    /// Table name; must satisfy [`valid_name`].
    pub name: String,
    /// Columns after the implicit `id`, in the order that [`Row::values`] uses.
    pub columns: Vec<ColumnDef>,
}

/// One row. `values` are positional and match `TableDef::columns` in length and order.
/// `Value::Null` is allowed in any column.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Primary key, at most `i64::MAX`. Writing an existing id replaces the row.
    pub id: NodeId,
    /// One value per column; a wrong count or type is [`Error::Schema`].
    pub values: Vec<Value>,
}

/// A row predicate for [`Store::select`] and [`Store::knn_filtered`].
///
/// A column that does not exist is [`Error::NotFound`]. A filter that does not fit the column's
/// type (see each variant) is [`Error::Schema`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Filter {
    /// Every row.
    All,
    /// Column equals value (never matches `Null`).
    Eq {
        /// Column to compare.
        column: String,
        /// Value to look for; its variant must fit the column's type.
        value: Value,
    },
    /// Int column in `min..=max`.
    IntRange {
        /// An [`ColumnType::Int`] column.
        column: String,
        /// Smallest accepted value, inclusive.
        min: i64,
        /// Largest accepted value, inclusive. `max < min` matches nothing.
        max: i64,
    },
    /// Text column starts with `prefix` (byte-wise, case-sensitive).
    TextPrefix {
        /// A [`ColumnType::Text`] column.
        column: String,
        /// Required leading bytes. `%` and `_` are ordinary characters, not wildcards.
        prefix: String,
    },
}

/// How the distance between two vectors is measured. Smaller is nearer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Metric {
    /// distance = 1 - cosine similarity. If either vector has zero norm, distance is 1.
    Cosine,
    /// distance = squared Euclidean distance
    L2,
}

/// A vector index: a named set of fixed-length `f32` vectors.
///
/// Defining the same name again with an identical definition is a no-op; any difference is
/// [`Error::Schema`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorIndexDef {
    /// Index name; must satisfy [`valid_name`].
    pub name: String,
    /// Number of `f32` components of every vector, fixed for the life of the index.
    pub dim: usize,
    /// Distance measure used by every search of this index.
    pub metric: Metric,
    /// HNSW max neighbours per node (ignored by exact backends).
    pub m: usize,
    /// HNSW candidate-list size while building (ignored by exact backends). Larger builds slower
    /// and gives better recall.
    pub ef_construction: usize,
}

/// One nearest-neighbour result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Id of the vector.
    pub id: NodeId,
    /// Distance from the query under the index's [`Metric`]. Smaller is nearer.
    pub distance: f32,
}

/// A directed, weighted, labelled edge between two nodes.
///
/// An edge is identified by `(src, rel, dst)`; the weight is its only payload.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Source node, at most `i64::MAX`. It need not have a row or a vector.
    pub src: NodeId,
    /// Relation name. Any string: it is not restricted by [`valid_name`].
    pub rel: String,
    /// Destination node, at most `i64::MAX`. It need not have a row or a vector.
    pub dst: NodeId,
    /// Edge weight. Must be finite (not NaN or infinite), otherwise [`Error::Schema`].
    pub weight: f32,
}

/// Which edges to follow from a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Direction {
    /// Edges whose `src` is the node (follow to `dst`).
    Out,
    /// Edges whose `dst` is the node (follow to `src`).
    In,
    /// Both of the above. A self-loop counts once.
    Both,
}

/// One node found by [`Store::k_hop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reached {
    /// The reached node. Never the start node.
    pub id: NodeId,
    /// Smallest number of hops from the start node (1..=max_depth).
    pub depth: u32,
}

/// Options for [`Store::open`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenOptions {
    /// Create the directory and an empty store when missing.
    /// When false, opening a missing store is an error.
    pub create: bool,
    /// When false the backend may trade durability of the last writes for speed
    /// (bulk loading). `flush()` must still make everything durable.
    pub sync_each_write: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            create: true,
            sync_each_write: false,
        }
    }
}

/// The error type of every [`Store`] method.
///
/// Callers should match on the variant they can act on and treat the rest, including variants
/// added later, as a failure of the call.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Unknown table, index or column.
    NotFound(String),
    /// Row or vector does not match the definition; bad name; redefinition with a different shape.
    /// Also an id above `i64::MAX`, a non-finite vector component or edge weight, and a request
    /// the backend does not support.
    Schema(String),
    /// Operating-system error while reading or writing the store directory.
    Io(std::io::Error),
    /// Anything the underlying engine reports.
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound(s) => write!(f, "not found: {s}"),
            Error::Schema(s) => write!(f, "schema: {s}"),
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Backend(s) => write!(f, "backend: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// `Result` with the crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// The QuickDB contract.
/// Every id supplied to a method, including row/vector ids and both edge endpoints,
/// must be at most `i64::MAX`; otherwise the method returns `Error::Schema`.
/// Batch validation errors leave the entire batch unapplied. Other errors do not
/// promise rollback, for example an I/O error while making committed data durable.
pub trait Store {
    /// Open (or create) the store in directory `path`.
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self>
    where
        Self: Sized;

    /// Short stable name used in benchmark output, e.g. `"native"`, `"cozo-sqlite"`.
    fn backend_name(&self) -> &'static str;

    // ---- relational ----

    /// Create the table. Defining the same table again with an identical definition is a no-op;
    /// with a different definition it is `Error::Schema`.
    fn define_table(&mut self, def: &TableDef) -> Result<()>;
    /// Insert or replace rows by `id`, atomically for the whole slice.
    fn upsert_rows(&mut self, table: &str, rows: &[Row]) -> Result<()>;
    /// The row with this `id`, or `None` when there is none. An unknown table is `Error::NotFound`.
    fn get_row(&self, table: &str, id: NodeId) -> Result<Option<Row>>;
    /// Rows matching `filter`, ordered by ascending `id`, at most `limit`.
    /// A zero `limit` returns no rows.
    fn select(&self, table: &str, filter: &Filter, limit: usize) -> Result<Vec<Row>>;
    /// Number of rows in the table. An unknown table is `Error::NotFound`.
    fn count_rows(&self, table: &str) -> Result<u64>;
    /// Delete rows by id; returns the number of distinct existing rows removed.
    /// Duplicate and missing ids do not increase the count. Vectors and edges are not touched.
    fn delete_rows(&mut self, table: &str, ids: &[NodeId]) -> Result<u64>;

    // ---- vectors ----

    /// Create the vector index (same redefinition rule as `define_table`).
    fn define_vector_index(&mut self, def: &VectorIndexDef) -> Result<()>;
    /// Insert or replace vectors by id. Each slice must have length `dim`.
    /// After a replace, searches must use the new vector and never return a hit computed from the old one.
    /// A vector with a non-finite value (NaN or infinity) is `Error::Schema`.
    fn upsert_vectors(&mut self, index: &str, items: &[(NodeId, &[f32])]) -> Result<()>;
    /// Up to `k` nearest vectors, ascending distance (ties: ascending id).
    /// Ties are equal computed distances, without an epsilon tolerance.
    /// A query of the wrong dimension or containing NaN or infinity is `Error::Schema`,
    /// even when `k` is zero. With a valid query, `k == 0` returns no hits.
    /// `ef_search` is the HNSW search breadth; exact backends ignore it.
    fn knn(&self, index: &str, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<Hit>>;
    /// Like `knn`, including ordering, query validation and `k == 0`, but only ids
    /// that have a row in `table` matching `filter` may be returned.
    /// Deleting that row excludes its vector from this search without deleting the vector.
    /// Must return `k` hits when at least `k` matching vectors exist and `ef_search` is large enough.
    fn knn_filtered(
        &self,
        index: &str,
        query: &[f32],
        k: usize,
        ef_search: usize,
        table: &str,
        filter: &Filter,
    ) -> Result<Vec<Hit>>;
    /// Number of vectors in the index. An unknown index is `Error::NotFound`.
    fn count_vectors(&self, index: &str) -> Result<u64>;

    // ---- graph ----

    /// Insert or replace edges; an edge is identified by `(src, rel, dst)`.
    /// Later duplicates in a batch win. A non-finite weight is `Error::Schema`.
    /// Relations are arbitrary strings, not restricted to `valid_name`.
    fn upsert_edges(&mut self, edges: &[Edge]) -> Result<()>;
    /// Edges touching `node` in `dir`, optionally only relation `rel`.
    /// Ordered by `(rel, src, dst)`, at most `limit`. With `Both`, a self-loop appears once.
    /// A zero `limit` returns no edges.
    fn neighbors(
        &self,
        node: NodeId,
        rel: Option<&str>,
        dir: Direction,
        limit: usize,
    ) -> Result<Vec<Edge>>;
    /// Breadth-first search from `start` following edges in `dir` (optionally only `rel`).
    /// Returns every distinct node other than `start` reachable within `max_depth` hops, with its
    /// smallest depth, ordered by `(depth, id)`, truncated to `limit`.
    /// A zero `limit` or zero `max_depth` returns no reached nodes.
    fn k_hop(
        &self,
        start: NodeId,
        rel: Option<&str>,
        dir: Direction,
        max_depth: u32,
        limit: usize,
    ) -> Result<Vec<Reached>>;
    /// Number of distinct edges, that is of distinct `(src, rel, dst)` triples.
    fn count_edges(&self) -> Result<u64>;

    // ---- lifecycle ----

    /// Make everything written so far durable, including table and vector-index definitions.
    /// If the process dies before `flush()` returns, each earlier write call must be found either
    /// complete or absent on reopen, never partly applied.
    fn flush(&mut self) -> Result<()>;
}

/// Total size in bytes of all files under `path` (recursive). Used for "size on disk".
/// Fails with the OS error when `path` or a directory below it cannot be read.
pub fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in std::fs::read_dir(&p)? {
            let entry = entry?;
            let md = entry.metadata()?;
            if md.is_dir() {
                stack.push(entry.path());
            } else {
                total += md.len();
            }
        }
    }
    Ok(total)
}

/// True when `name` is a legal table, column or index name: `[a-z][a-z0-9_]*`, at most 48 bytes.
pub fn valid_name(name: &str) -> bool {
    let b = name.as_bytes();
    !b.is_empty()
        && b.len() <= 48
        && b[0].is_ascii_lowercase()
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
}
