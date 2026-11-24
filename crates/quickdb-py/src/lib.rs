//! Python bindings for QDB (QuickDB): one embedded store for relational rows, vectors and graph
//! edges that share one id space. The Python package is `quickdb`; this crate is its native
//! module `quickdb._quickdb`, and `python/quickdb/` re-exports it with type stubs.
//!
//! - One `Store` wraps one [`NativeStore`] behind a `Mutex`: the store is `Send` but not `Sync`.
//! - Every store call runs with the GIL released, so other Python threads keep running during a
//!   search, a bulk upsert or a flush. Python input is converted to Rust values first, with the
//!   GIL held, and results are converted back after the call.
//! - Every error the store reports is raised as a subclass of `quickdb.QuickDBError`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType,
};
use quickdb_api::{
    ColumnDef, ColumnType, Direction, Edge, Error, Filter, Metric, NodeId, OpenOptions, Row, Store,
    TableDef, Value, VectorIndexDef,
};
use quickdb_native::{NativeStore, TextIndexDef};

create_exception!(
    quickdb,
    QuickDBError,
    PyException,
    "Base class of every error raised by quickdb."
);
create_exception!(
    quickdb,
    NotFoundError,
    QuickDBError,
    "Unknown table, vector index or column."
);
create_exception!(
    quickdb,
    SchemaError,
    QuickDBError,
    "Input that does not fit a definition: wrong row shape or value type, bad name, a different \
     redefinition, an id above 2**63-1, a non-finite vector component or edge weight, a malformed \
     filter or vector buffer."
);
create_exception!(
    quickdb,
    StorageError,
    QuickDBError,
    "Operating-system error while reading or writing the store directory, including a missing store."
);
create_exception!(
    quickdb,
    BackendError,
    QuickDBError,
    "Error reported by the storage engine, including a store locked by another writer and a write \
     through a read-only handle."
);

const MAX_ID: u64 = i64::MAX as u64;

fn db_err(e: Error) -> PyErr {
    let msg = e.to_string();
    match e {
        Error::NotFound(_) => NotFoundError::new_err(msg),
        Error::Schema(_) => SchemaError::new_err(msg),
        Error::Io(_) => StorageError::new_err(msg),
        Error::Backend(_) => BackendError::new_err(msg),
        _ => QuickDBError::new_err(msg),
    }
}

fn schema_err(msg: impl std::fmt::Display) -> PyErr {
    SchemaError::new_err(format!("schema: {msg}"))
}

/// Why a store call did not reach the store, or what it returned.
enum Fail {
    Db(Error),
    Closed,
    Poisoned,
}

fn fail_err(f: Fail) -> PyErr {
    match f {
        Fail::Db(e) => db_err(e),
        Fail::Closed => QuickDBError::new_err("store is closed"),
        Fail::Poisoned => BackendError::new_err("backend: store is unusable after a panic"),
    }
}

/// Short description of a Python object for error messages: its type and a bounded repr.
fn describe(obj: &Bound<'_, PyAny>) -> String {
    let ty = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "?".into());
    let mut r = obj
        .repr()
        .map(|r| r.to_string())
        .unwrap_or_else(|_| "?".into());
    if r.chars().count() > 60 {
        r = r.chars().take(57).collect::<String>() + "...";
    }
    format!("{ty} {r}")
}

fn to_id(obj: &Bound<'_, PyAny>) -> PyResult<NodeId> {
    match obj.extract::<u64>() {
        Ok(v) if v <= MAX_ID && !obj.is_instance_of::<PyFloat>() => Ok(v),
        _ => Err(schema_err(format!(
            "an id is an int in 0..=2**63-1, got {}",
            describe(obj)
        ))),
    }
}

fn to_ids(obj: &Bound<'_, PyAny>) -> PyResult<Vec<NodeId>> {
    obj.try_iter()?.map(|i| to_id(&i?)).collect()
}

fn items<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if obj.is_instance_of::<PyString>() {
        return Err(schema_err(format!(
            "expected a sequence, got {}",
            describe(obj)
        )));
    }
    obj.try_iter()?.collect()
}

fn type_name(ty: ColumnType) -> &'static str {
    match ty {
        ColumnType::Int => "int",
        ColumnType::Float => "float",
        ColumnType::Text => "text",
        ColumnType::Bytes => "bytes",
        _ => "unknown",
    }
}

/// A column type from `"int" | "float" | "text" | "bytes"` (and a few aliases) or the Python
/// types `int`, `float`, `str`, `bytes`.
fn parse_type(obj: &Bound<'_, PyAny>) -> PyResult<ColumnType> {
    if let Ok(t) = obj.downcast::<PyType>() {
        // bool is a subclass of int and maps to Int.
        if t.is_subclass_of::<PyInt>()? {
            return Ok(ColumnType::Int);
        }
        if t.is_subclass_of::<PyFloat>()? {
            return Ok(ColumnType::Float);
        }
        if t.is_subclass_of::<PyString>()? {
            return Ok(ColumnType::Text);
        }
        if t.is_subclass_of::<PyBytes>()? {
            return Ok(ColumnType::Bytes);
        }
    } else if let Ok(s) = obj.extract::<String>() {
        match s.to_ascii_lowercase().as_str() {
            "int" | "integer" | "i64" => return Ok(ColumnType::Int),
            "float" | "real" | "f64" => return Ok(ColumnType::Float),
            "text" | "str" => return Ok(ColumnType::Text),
            "bytes" | "blob" => return Ok(ColumnType::Bytes),
            _ => {}
        }
    }
    Err(schema_err(format!(
        "a column type is \"int\", \"float\", \"text\" or \"bytes\" (or int, float, str, bytes), got {}",
        describe(obj)
    )))
}

fn parse_columns(columns: &Bound<'_, PyAny>) -> PyResult<Vec<ColumnDef>> {
    let mut out = Vec::new();
    for col in items(columns)? {
        let parts = items(&col)?;
        if parts.len() != 2 && parts.len() != 3 {
            return Err(schema_err(format!(
                "a column is a (name, type, indexed) tuple, got {}",
                describe(&col)
            )));
        }
        let name: String = parts[0].extract().map_err(|_| {
            schema_err(format!(
                "a column name is a str, got {}",
                describe(&parts[0])
            ))
        })?;
        let ty = parse_type(&parts[1])?;
        let indexed = match parts.get(2) {
            Some(v) => v.is_truthy()?,
            None => false,
        };
        out.push(ColumnDef { name, ty, indexed });
    }
    Ok(out)
}

/// Converts a Python value for column `col`. `None` is NULL in every column. An int is accepted
/// in a float column; nothing else is converted.
fn to_value(obj: &Bound<'_, PyAny>, col: &ColumnDef) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    let v = match col.ty {
        ColumnType::Int => obj.extract::<i64>().ok().map(Value::Int),
        ColumnType::Float => obj.extract::<f64>().ok().map(Value::Float),
        ColumnType::Text => {
            if obj.is_instance_of::<PyString>() {
                obj.extract::<String>().ok().map(Value::Text)
            } else {
                None
            }
        }
        ColumnType::Bytes => {
            if let Ok(b) = obj.downcast::<PyBytes>() {
                Some(Value::Bytes(b.as_bytes().to_vec()))
            } else if let Ok(b) = obj.downcast::<PyByteArray>() {
                Some(Value::Bytes(b.to_vec()))
            } else {
                None
            }
        }
        _ => None,
    };
    v.ok_or_else(|| {
        schema_err(format!(
            "column {:?} ({}) cannot hold {}",
            col.name,
            type_name(col.ty),
            describe(obj)
        ))
    })
}

fn value_to_py(py: Python<'_>, v: Value) -> PyResult<PyObject> {
    Ok(match v {
        Value::Null => py.None(),
        Value::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        Value::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
        Value::Text(s) => s.into_pyobject(py)?.into_any().unbind(),
        Value::Bytes(b) => PyBytes::new(py, &b).into_any().unbind(),
        _ => return Err(BackendError::new_err("backend: unsupported value type")),
    })
}

fn row_to_dict<'py>(py: Python<'py>, row: Row, def: &TableDef) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", row.id)?;
    for (c, v) in def.columns.iter().zip(row.values) {
        d.set_item(&c.name, value_to_py(py, v)?)?;
    }
    Ok(d)
}

fn parse_rows(rows: &Bound<'_, PyAny>, def: &TableDef) -> PyResult<Vec<Row>> {
    let mut out = Vec::new();
    for item in items(rows)? {
        if let Ok(d) = item.downcast::<PyDict>() {
            let id = d
                .get_item("id")?
                .ok_or_else(|| schema_err("a row dict needs an \"id\" key"))?;
            let id = to_id(&id)?;
            let mut used = 1;
            let mut values = Vec::with_capacity(def.columns.len());
            for c in &def.columns {
                match d.get_item(&c.name)? {
                    Some(v) => {
                        used += 1;
                        values.push(to_value(&v, c)?);
                    }
                    None => values.push(Value::Null),
                }
            }
            if used != d.len() {
                for (k, _) in d.iter() {
                    let k: String = k.extract().unwrap_or_default();
                    if k != "id" && !def.columns.iter().any(|c| c.name == k) {
                        return Err(schema_err(format!(
                            "table {:?} has no column {k:?}",
                            def.name
                        )));
                    }
                }
            }
            out.push(Row { id, values });
        } else {
            let pair = items(&item).map_err(|_| {
                schema_err(format!(
                    "a row is a dict with \"id\" or an (id, [values]) pair, got {}",
                    describe(&item)
                ))
            })?;
            if pair.len() != 2 {
                return Err(schema_err(format!(
                    "a row is a dict with \"id\" or an (id, [values]) pair, got {}",
                    describe(&item)
                )));
            }
            let id = to_id(&pair[0])?;
            let vals = items(&pair[1])?;
            if vals.len() != def.columns.len() {
                return Err(schema_err(format!(
                    "row {id}: {} values for the {} columns of table {:?}",
                    vals.len(),
                    def.columns.len(),
                    def.name
                )));
            }
            let values = vals
                .iter()
                .zip(&def.columns)
                .map(|(v, c)| to_value(v, c))
                .collect::<PyResult<Vec<_>>>()?;
            out.push(Row { id, values });
        }
    }
    Ok(out)
}

fn column<'a>(def: &'a TableDef, obj: &Bound<'_, PyAny>) -> PyResult<&'a ColumnDef> {
    let name: String = obj
        .extract()
        .map_err(|_| schema_err(format!("a column name is a str, got {}", describe(obj))))?;
    def.columns
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| NotFoundError::new_err(format!("not found: column {name} in {}", def.name)))
}

fn bound_i64(obj: &Bound<'_, PyAny>, default: i64) -> PyResult<i64> {
    if obj.is_none() {
        return Ok(default);
    }
    obj.extract::<i64>().map_err(|_| {
        schema_err(format!(
            "a range bound is an int or None, got {}",
            describe(obj)
        ))
    })
}

/// `None` or `{"all": ...}` is every row; `{"eq": [col, value]}`, `{"range": [col, lo, hi]}` (an
/// int column, inclusive, `None` for an open end) and `{"prefix": [col, text]}`.
fn parse_filter(obj: Option<&Bound<'_, PyAny>>, def: &TableDef) -> PyResult<Filter> {
    let obj = match obj {
        None => return Ok(Filter::All),
        Some(o) if o.is_none() => return Ok(Filter::All),
        Some(o) => o,
    };
    let shape = || {
        schema_err(format!(
            "a filter is {{\"eq\": [col, value]}}, {{\"range\": [col, lo, hi]}}, {{\"prefix\": [col, text]}} or None, got {}",
            describe(obj)
        ))
    };
    let d = obj.downcast::<PyDict>().map_err(|_| shape())?;
    if d.len() != 1 {
        return Err(shape());
    }
    let (key, arg) = d.iter().next().ok_or_else(shape)?;
    let key: String = key.extract().map_err(|_| shape())?;
    if key == "all" {
        return Ok(Filter::All);
    }
    let args = items(&arg).map_err(|_| shape())?;
    match (key.as_str(), args.len()) {
        ("eq", 2) => {
            let c = column(def, &args[0])?;
            let value = to_value(&args[1], c)?;
            Ok(Filter::Eq {
                column: c.name.clone(),
                value,
            })
        }
        ("range", 3) => {
            let c = column(def, &args[0])?;
            Ok(Filter::IntRange {
                column: c.name.clone(),
                min: bound_i64(&args[1], i64::MIN)?,
                max: bound_i64(&args[2], i64::MAX)?,
            })
        }
        ("prefix", 2) => {
            let c = column(def, &args[0])?;
            let prefix: String = args[1].extract().map_err(|_| {
                schema_err(format!("a prefix is a str, got {}", describe(&args[1])))
            })?;
            Ok(Filter::TextPrefix {
                column: c.name.clone(),
                prefix,
            })
        }
        _ => Err(shape()),
    }
}

fn parse_direction(s: &str) -> PyResult<Direction> {
    match s.to_ascii_lowercase().as_str() {
        "out" => Ok(Direction::Out),
        "in" => Ok(Direction::In),
        "both" => Ok(Direction::Both),
        _ => Err(schema_err(format!(
            "direction is \"out\", \"in\" or \"both\", got {s:?}"
        ))),
    }
}

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "l2" => Ok(Metric::L2),
        _ => Err(schema_err(format!(
            "metric is \"cosine\" or \"l2\", got {s:?}"
        ))),
    }
}

/// Row-major `f32` vectors.
struct Matrix {
    data: Vec<f32>,
    rows: usize,
    dim: usize,
}

impl Matrix {
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

#[derive(Clone, Copy)]
enum Order {
    Native,
    Little,
    Big,
}

fn decode<const N: usize>(b: &[u8], order: Order, f: fn([u8; N], Order) -> f32) -> Vec<f32> {
    b.chunks_exact(N)
        .map(|c| {
            let mut a = [0u8; N];
            a.copy_from_slice(c);
            f(a, order)
        })
        .collect()
}

fn f32_of(a: [u8; 4], o: Order) -> f32 {
    match o {
        Order::Native => f32::from_ne_bytes(a),
        Order::Little => f32::from_le_bytes(a),
        Order::Big => f32::from_be_bytes(a),
    }
}

fn f64_of(a: [u8; 8], o: Order) -> f32 {
    (match o {
        Order::Native => f64::from_ne_bytes(a),
        Order::Little => f64::from_le_bytes(a),
        Order::Big => f64::from_be_bytes(a),
    }) as f32
}

/// Reads an object that exports the buffer protocol (a numpy array, `array.array`, a memoryview)
/// with `ndim` dimensions of float32 (or float64, narrowed to float32). Returns `None` for an
/// object without the buffer protocol, and for lists and tuples.
///
/// The buffer is read through `memoryview(obj).tobytes()`, one copy. PyO3's zero-copy `PyBuffer`
/// is not part of the stable ABI before Python 3.11, and this module targets abi3 for 3.9+; the
/// copy is small next to building an HNSW index or searching it.
fn buffer_floats(obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Option<Matrix>> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return Ok(None);
    }
    let py = obj.py();
    let mv = match py.import("builtins")?.getattr("memoryview")?.call1((obj,)) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let read = || -> PyResult<Matrix> {
        let fmt: String = mv.getattr("format")?.extract()?;
        let got: usize = mv.getattr("ndim")?.extract()?;
        let shape: Vec<usize> = mv.getattr("shape")?.extract()?;
        if got != ndim {
            return Err(schema_err(format!(
                "expected a {ndim}-D float buffer, got {got}-D with shape {shape:?}"
            )));
        }
        let (width, order) = match fmt.as_str() {
            "f" | "@f" | "=f" => (4, Order::Native),
            "<f" => (4, Order::Little),
            ">f" | "!f" => (4, Order::Big),
            "d" | "@d" | "=d" => (8, Order::Native),
            "<d" => (8, Order::Little),
            ">d" | "!d" => (8, Order::Big),
            other => {
                return Err(schema_err(format!(
                    "vectors must be float32 or float64, got buffer format {other:?}"
                )))
            }
        };
        // tobytes() returns the elements in C order, also for a non-contiguous view.
        let bytes = mv.call_method0("tobytes")?;
        let b = bytes.downcast::<PyBytes>()?.as_bytes();
        let data = if width == 4 {
            decode::<4>(b, order, f32_of)
        } else {
            decode::<8>(b, order, f64_of)
        };
        let (rows, dim) = if ndim == 2 {
            (shape[0], shape[1])
        } else {
            (1, shape[0])
        };
        Ok(Matrix { data, rows, dim })
    };
    let result = read();
    let _ = mv.call_method0("release");
    result.map(Some)
}

fn parse_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Some(m) = buffer_floats(obj, 1)? {
        return Ok(m.data);
    }
    items(obj)?
        .iter()
        .map(|x| {
            if x.is_instance_of::<PyString>() {
                return Err(schema_err(format!(
                    "a vector component is a float, got {}",
                    describe(x)
                )));
            }
            x.extract::<f64>().map(|f| f as f32).map_err(|_| {
                schema_err(format!(
                    "a vector component is a float, got {}",
                    describe(x)
                ))
            })
        })
        .collect()
}

fn parse_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Matrix> {
    if let Some(m) = buffer_floats(obj, 2)? {
        return Ok(m);
    }
    let mut data = Vec::new();
    let mut rows = 0;
    let mut dim = None;
    for v in items(obj)? {
        let v = parse_vector(&v)?;
        match dim {
            None => dim = Some(v.len()),
            Some(d) if d != v.len() => {
                return Err(schema_err(format!(
                    "vector {rows} has {} components, vector 0 has {d}",
                    v.len()
                )))
            }
            Some(_) => {}
        }
        data.extend_from_slice(&v);
        rows += 1;
    }
    Ok(Matrix {
        data,
        rows,
        dim: dim.unwrap_or(0),
    })
}

fn parse_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<Edge>> {
    let mut out = Vec::new();
    for e in items(edges)? {
        let p = items(&e)?;
        if p.len() != 3 && p.len() != 4 {
            return Err(schema_err(format!(
                "an edge is a (src, rel, dst, weight) tuple, got {}",
                describe(&e)
            )));
        }
        let rel: String = p[1].extract().map_err(|_| {
            schema_err(format!(
                "an edge relation is a str, got {}",
                describe(&p[1])
            ))
        })?;
        let weight = match p.get(3) {
            Some(w) => w.extract::<f64>().map_err(|_| {
                schema_err(format!("an edge weight is a float, got {}", describe(w)))
            })? as f32,
            None => 1.0,
        };
        out.push(Edge {
            src: to_id(&p[0])?,
            rel,
            dst: to_id(&p[2])?,
            weight,
        });
    }
    Ok(out)
}

/// Reads the table definitions the store persisted, through a short-lived read-only SQLite
/// connection, once when a handle opens. `NativeStore` keeps these definitions but has no public
/// accessor for them, and dict rows need column names, also for tables defined by an earlier
/// process. Rows themselves are always read and written through `NativeStore`; this function
/// reads definitions only.
fn read_schema(path: &Path) -> Result<HashMap<String, TableDef>, String> {
    use rusqlite::{Connection, OpenFlags};
    let db = Connection::open_with_flags(
        path.join("quickdb.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| e.to_string())?;
    db.busy_timeout(Duration::from_secs(30))
        .map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare(
            "SELECT d.name, c.name, c.ty, c.indexed FROM table_defs d \
             LEFT JOIN column_defs c ON c.tbl = d.name ORDER BY d.name, c.pos",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<i64>>(2)?,
                r.get::<_, Option<bool>>(3)?,
            ))
        })
        .map_err(|e| e.to_string())?;
    let mut tables: HashMap<String, TableDef> = HashMap::new();
    for r in rows {
        let (table, col, ty, indexed) = r.map_err(|e| e.to_string())?;
        let def = tables.entry(table.clone()).or_insert_with(|| TableDef {
            name: table,
            columns: Vec::new(),
        });
        if let (Some(name), Some(ty), Some(indexed)) = (col, ty, indexed) {
            let ty = match ty {
                0 => ColumnType::Int,
                1 => ColumnType::Float,
                2 => ColumnType::Text,
                3 => ColumnType::Bytes,
                _ => return Err(format!("unknown stored column type {ty}")),
            };
            def.columns.push(ColumnDef { name, ty, indexed });
        }
    }
    Ok(tables)
}

/// An open QDB store: relational rows, vectors and graph edges in one directory.
///
/// ``Store(path)`` opens the store for writing, creating it when missing. Only one writable
/// handle per directory may exist at a time, in any process; a second one raises
/// ``BackendError``. Use ``Store.open_read_only(path)`` for any number of readers beside it.
///
/// Every write is atomic. ``flush()`` makes everything written so far durable; leaving a
/// ``with`` block without an exception flushes and closes the store.
#[pyclass(frozen, module = "quickdb", name = "Store")]
pub struct PyStore {
    store: Mutex<Option<NativeStore>>,
    tables: Mutex<HashMap<String, TableDef>>,
    path: PathBuf,
    read_only: bool,
}

impl PyStore {
    fn wrap(py: Python<'_>, store: NativeStore, path: PathBuf, read_only: bool) -> PyResult<Self> {
        let p = path.clone();
        let tables = py.allow_threads(move || read_schema(&p)).map_err(|e| {
            BackendError::new_err(format!("backend: reading table definitions: {e}"))
        })?;
        Ok(PyStore {
            store: Mutex::new(Some(store)),
            tables: Mutex::new(tables),
            path,
            read_only,
        })
    }

    /// Runs `f` on the store with the GIL released.
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut NativeStore) -> quickdb_api::Result<T> + Send,
    ) -> PyResult<T> {
        py.allow_threads(|| {
            let mut guard = self.store.lock().map_err(|_| Fail::Poisoned)?;
            let store = guard.as_mut().ok_or(Fail::Closed)?;
            f(store).map_err(Fail::Db)
        })
        .map_err(fail_err)
    }

    /// The definition of `table` as this handle knows it: the tables persisted when it opened,
    /// plus those it defined since. That is also the set `NativeStore` serves through this handle.
    fn table_def(&self, table: &str) -> PyResult<TableDef> {
        if self
            .store
            .lock()
            .map_err(|_| fail_err(Fail::Poisoned))?
            .is_none()
        {
            return Err(fail_err(Fail::Closed));
        }
        self.tables
            .lock()
            .map_err(|_| fail_err(Fail::Poisoned))?
            .get(table)
            .cloned()
            .ok_or_else(|| NotFoundError::new_err(format!("not found: {table}")))
    }
}

#[pymethods]
impl PyStore {
    /// Open the store in directory ``path`` for writing.
    ///
    /// ``sync_each_write=False`` trades durability of the last writes for speed (bulk loading);
    /// ``flush()`` still makes everything durable. ``create=False`` raises ``StorageError`` when
    /// the store does not exist.
    #[new]
    #[pyo3(signature = (path, sync_each_write = true, create = true))]
    fn new(py: Python<'_>, path: PathBuf, sync_each_write: bool, create: bool) -> PyResult<Self> {
        let opts = OpenOptions {
            create,
            sync_each_write,
        };
        let p = path.clone();
        let store = py
            .allow_threads(move || NativeStore::open(&p, &opts))
            .map_err(db_err)?;
        Self::wrap(py, store, path, false)
    }

    /// Open an existing store for reading, beside at most one writer and any number of readers.
    ///
    /// Rows and edges are read live; vectors, and the set of tables and indexes, are those
    /// committed when this handle opened (reopen to see a table created later). Every
    /// write method and ``flush()`` raise ``BackendError``.
    #[staticmethod]
    fn open_read_only(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let p = path.clone();
        let store = py
            .allow_threads(move || NativeStore::open_read_only(&p))
            .map_err(db_err)?;
        Self::wrap(py, store, path, true)
    }

    /// The store directory.
    #[getter]
    fn path(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }

    /// True for a handle from ``open_read_only``.
    #[getter]
    fn read_only(&self) -> bool {
        self.read_only
    }

    /// True after ``close()``.
    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self
            .store
            .lock()
            .map_err(|_| fail_err(Fail::Poisoned))?
            .is_none())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "quickdb.Store({:?}, read_only={}, closed={})",
            self.path.to_string_lossy(),
            if self.read_only { "True" } else { "False" },
            if self.closed()? { "True" } else { "False" }
        ))
    }

    // ---- relational ----

    /// Create table ``name``. ``columns`` is a list of ``(name, type, indexed)`` tuples; ``type``
    /// is ``"int"``, ``"float"``, ``"text"`` or ``"bytes"`` (or ``int``, ``float``, ``str``,
    /// ``bytes``), and ``indexed`` may be left out (False). Every table also has an implicit
    /// ``id`` key. Defining a table again with the identical definition does nothing.
    fn define_table(
        &self,
        py: Python<'_>,
        name: String,
        columns: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let def = TableDef {
            name,
            columns: parse_columns(columns)?,
        };
        let d = def.clone();
        self.with(py, move |s| s.define_table(&d))?;
        self.tables
            .lock()
            .map_err(|_| fail_err(Fail::Poisoned))?
            .insert(def.name.clone(), def);
        Ok(())
    }

    /// Insert or replace rows, atomically for the whole batch. Each row is a dict with an
    /// ``"id"`` key and column keys (a missing column is NULL), or an ``(id, [values])`` pair with
    /// one value per column in definition order.
    fn upsert_rows(&self, py: Python<'_>, table: String, rows: &Bound<'_, PyAny>) -> PyResult<()> {
        let def = self.table_def(&table)?;
        let rows = parse_rows(rows, &def)?;
        self.with(py, move |s| s.upsert_rows(&table, &rows))
    }

    /// The row with this id as a dict (``"id"`` and every column), or None.
    fn get_row(&self, py: Python<'_>, table: String, id: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let id = to_id(id)?;
        let t = table.clone();
        let row = self.with(py, move |s| s.get_row(&t, id))?;
        match row {
            None => Ok(py.None()),
            Some(r) => {
                let def = self.table_def(&table)?;
                Ok(row_to_dict(py, r, &def)?.into_any().unbind())
            }
        }
    }

    /// Rows matching ``filter`` as dicts, ascending id, at most ``limit``. ``filter`` is None
    /// (every row), ``{"eq": [col, value]}``, ``{"range": [col, lo, hi]}`` (int column,
    /// inclusive, None for an open end) or ``{"prefix": [col, text]}``.
    #[pyo3(signature = (table, filter = None, limit = 1000))]
    fn select<'py>(
        &self,
        py: Python<'py>,
        table: String,
        filter: Option<&Bound<'py, PyAny>>,
        limit: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let def = self.table_def(&table)?;
        let filter = parse_filter(filter, &def)?;
        let rows = self.with(py, move |s| s.select(&table, &filter, limit))?;
        let list = PyList::empty(py);
        for r in rows {
            list.append(row_to_dict(py, r, &def)?)?;
        }
        Ok(list)
    }

    /// Delete rows by id; returns how many existing rows were removed. Vectors and edges with the
    /// same ids are kept.
    fn delete_rows(&self, py: Python<'_>, table: String, ids: &Bound<'_, PyAny>) -> PyResult<u64> {
        let ids = to_ids(ids)?;
        self.with(py, move |s| s.delete_rows(&table, &ids))
    }

    /// Number of rows in ``table``.
    fn count_rows(&self, py: Python<'_>, table: String) -> PyResult<u64> {
        self.with(py, move |s| s.count_rows(&table))
    }

    // ---- vectors ----

    /// Create vector index ``name`` of ``dim`` float32 components. ``metric`` is ``"cosine"``
    /// (1 - cosine similarity) or ``"l2"`` (squared Euclidean). ``m`` and ``ef_construction`` are
    /// the HNSW graph degree and build breadth.
    #[pyo3(signature = (name, dim, metric = "cosine", m = 16, ef_construction = 200))]
    fn define_vector_index(
        &self,
        py: Python<'_>,
        name: String,
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
    ) -> PyResult<()> {
        let def = VectorIndexDef {
            name,
            dim,
            metric: parse_metric(metric)?,
            m,
            ef_construction,
        };
        self.with(py, move |s| s.define_vector_index(&def))
    }

    /// Insert or replace vectors by id. ``vectors`` is a list of float lists, or any 2-D float32
    /// (or float64) buffer such as a numpy array, with one row per id.
    fn upsert_vectors(
        &self,
        py: Python<'_>,
        index: String,
        ids: &Bound<'_, PyAny>,
        vectors: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let ids = to_ids(ids)?;
        let m = parse_matrix(vectors)?;
        if m.rows != ids.len() {
            return Err(schema_err(format!(
                "{} ids for {} vectors",
                ids.len(),
                m.rows
            )));
        }
        self.with(py, move |s| {
            let items: Vec<(NodeId, &[f32])> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| (*id, m.row(i)))
                .collect();
            s.upsert_vectors(&index, &items)
        })
    }

    /// Up to ``k`` nearest vectors as ``(id, distance)`` pairs, nearest first. ``query`` is a
    /// float list or a 1-D float buffer; ``ef`` is the HNSW search breadth.
    #[pyo3(signature = (index, query, k = 10, ef = 64))]
    fn knn(
        &self,
        py: Python<'_>,
        index: String,
        query: &Bound<'_, PyAny>,
        k: usize,
        ef: usize,
    ) -> PyResult<Vec<(NodeId, f32)>> {
        let q = parse_vector(query)?;
        let hits = self.with(py, move |s| s.knn(&index, &q, k, ef))?;
        Ok(hits.into_iter().map(|h| (h.id, h.distance)).collect())
    }

    /// Like ``knn``, but only ids whose row in ``table`` matches ``filter`` (see ``select``).
    #[allow(clippy::too_many_arguments)]
    fn knn_filtered(
        &self,
        py: Python<'_>,
        index: String,
        query: &Bound<'_, PyAny>,
        k: usize,
        ef: usize,
        table: String,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(NodeId, f32)>> {
        let q = parse_vector(query)?;
        let def = self.table_def(&table)?;
        let filter = parse_filter(filter, &def)?;
        let hits = self.with(py, move |s| {
            s.knn_filtered(&index, &q, k, ef, &table, &filter)
        })?;
        Ok(hits.into_iter().map(|h| (h.id, h.distance)).collect())
    }

    /// Define a full-text index ``name`` over the text ``columns`` of ``table`` (SQLite FTS5,
    /// BM25, Arabic diacritics and letter variants folded). Idempotent; existing rows are indexed.
    fn define_text_index(
        &self,
        py: Python<'_>,
        name: String,
        table: String,
        columns: Vec<String>,
    ) -> PyResult<()> {
        let def = TextIndexDef {
            name,
            table,
            columns,
        };
        self.with(py, move |s| s.define_text_index(&def))
    }

    /// Up to ``k`` ``(id, score)`` pairs, best first (higher score is better). ``match="all"``
    /// requires every word of ``query``; ``match="any"`` accepts any word, ranking rows with
    /// more and rarer words higher. The query is plain words, never search syntax.
    #[pyo3(signature = (index, query, k = 10, r#match = "all"))]
    fn text_search(
        &self,
        py: Python<'_>,
        index: String,
        query: String,
        k: usize,
        r#match: &str,
    ) -> PyResult<Vec<(NodeId, f64)>> {
        let any = match r#match {
            "all" => false,
            "any" => true,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "match must be \"all\" or \"any\", not {other:?}"
                )))
            }
        };
        self.with(py, move |s| {
            if any {
                s.text_search_any(&index, &query, k)
            } else {
                s.text_search(&index, &query, k)
            }
        })
    }

    /// Text and vector search fused by Reciprocal Rank Fusion (k = 60): up to ``k``
    /// ``(id, score)`` pairs, best first.
    #[pyo3(signature = (text_index, vector_index, query_text, query_vector, k = 10, ef = 64))]
    #[allow(clippy::too_many_arguments)]
    fn hybrid_search(
        &self,
        py: Python<'_>,
        text_index: String,
        vector_index: String,
        query_text: String,
        query_vector: &Bound<'_, PyAny>,
        k: usize,
        ef: usize,
    ) -> PyResult<Vec<(NodeId, f64)>> {
        let q = parse_vector(query_vector)?;
        self.with(py, move |s| {
            s.hybrid_search(&text_index, &vector_index, &query_text, &q, k, ef)
        })
    }

    /// Number of vectors in ``index``.
    fn count_vectors(&self, py: Python<'_>, index: String) -> PyResult<u64> {
        self.with(py, move |s| s.count_vectors(&index))
    }

    // ---- graph ----

    /// Insert or replace edges given as ``(src, rel, dst, weight)`` tuples (``weight`` defaults
    /// to 1.0). An edge is identified by ``(src, rel, dst)``; later duplicates win.
    fn upsert_edges(&self, py: Python<'_>, edges: &Bound<'_, PyAny>) -> PyResult<()> {
        let edges = parse_edges(edges)?;
        self.with(py, move |s| s.upsert_edges(&edges))
    }

    /// Edges touching ``node`` as ``(src, rel, dst, weight)`` tuples, ordered by
    /// ``(rel, src, dst)``. ``direction`` is ``"out"``, ``"in"`` or ``"both"``.
    #[pyo3(signature = (node, rel = None, direction = "out", limit = 1000))]
    fn neighbors(
        &self,
        py: Python<'_>,
        node: &Bound<'_, PyAny>,
        rel: Option<String>,
        direction: &str,
        limit: usize,
    ) -> PyResult<Vec<(NodeId, String, NodeId, f32)>> {
        let node = to_id(node)?;
        let dir = parse_direction(direction)?;
        let edges = self.with(py, move |s| s.neighbors(node, rel.as_deref(), dir, limit))?;
        Ok(edges
            .into_iter()
            .map(|e| (e.src, e.rel, e.dst, e.weight))
            .collect())
    }

    /// Nodes reachable from ``start`` within ``depth`` hops as ``(id, hops)`` pairs, ordered by
    /// ``(hops, id)``, at most ``limit``. The start node is never included.
    #[pyo3(signature = (start, rel = None, direction = "out", depth = 1, limit = 1000))]
    fn k_hop(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        rel: Option<String>,
        direction: &str,
        depth: u32,
        limit: usize,
    ) -> PyResult<Vec<(NodeId, u32)>> {
        let start = to_id(start)?;
        let dir = parse_direction(direction)?;
        let reached = self.with(py, move |s| {
            s.k_hop(start, rel.as_deref(), dir, depth, limit)
        })?;
        Ok(reached.into_iter().map(|r| (r.id, r.depth)).collect())
    }

    /// Number of distinct edges.
    fn count_edges(&self, py: Python<'_>) -> PyResult<u64> {
        self.with(py, |s| s.count_edges())
    }

    // ---- lifecycle ----

    /// Make everything written so far durable, including definitions.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, |s| s.flush())
    }

    /// Flush (for a writable handle) and close the store. Closing twice does nothing; any other
    /// call on a closed store raises ``QuickDBError``.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let read_only = self.read_only;
        py.allow_threads(|| {
            let mut guard = self.store.lock().map_err(|_| Fail::Poisoned)?;
            let result = match guard.as_mut() {
                Some(s) if !read_only => s.flush().map_err(Fail::Db),
                _ => Ok(()),
            };
            // The handle is released (and the writer lock with it) even when the flush fails.
            *guard = None;
            result
        })
        .map_err(fail_err)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// On a clean exit, flush and close; after an exception, close without flushing.
    #[pyo3(signature = (exc_type = None, _exc = None, _tb = None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if exc_type.is_none_or(|t| t.is_none()) {
            self.close(py)?;
        } else {
            let mut guard = self.store.lock().map_err(|_| fail_err(Fail::Poisoned))?;
            let store = guard.take();
            drop(guard);
            py.allow_threads(move || drop(store));
        }
        Ok(false)
    }
}

/// Native module of the `quickdb` package.
#[pymodule]
fn _quickdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyStore>()?;
    m.add("QuickDBError", py.get_type::<QuickDBError>())?;
    m.add("NotFoundError", py.get_type::<NotFoundError>())?;
    m.add("SchemaError", py.get_type::<SchemaError>())?;
    m.add("StorageError", py.get_type::<StorageError>())?;
    m.add("BackendError", py.get_type::<BackendError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
