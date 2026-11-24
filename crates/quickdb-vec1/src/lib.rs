//! SQLite rows and edges plus SQLite's `vec1` 0.8 vector extension (IVF + product quantisation).
//!
//! A vector index stays in vec1's exact "flat" mode until it holds `TRAIN_AT` vectors; the next
//! `flush` trains a model on a sample of the stored vectors and rebuilds the index, so that the
//! cost is part of the caller's load. Every search reranks vec1's candidates on the full vectors
//! with the contract's distance. See the crate README for settings and sources.
use quickdb_api::*;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_int};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Smallest vector count that gets an IVF model. vec1_train needs `4 * nbucket` vectors, and
/// below this size the exhaustive flat scan is exact and cheap.
pub const TRAIN_AT: i64 = 4 * NBUCKET as i64;
/// IVF bucket count, vec1's published setting for its SIFT and 1,024-dimension tests.
pub const NBUCKET: usize = 1024;
/// Training sample size (vec1 recommends about 100 vectors per bucket).
pub const TRAIN_SAMPLE: usize = 100_000;
/// vec1's hard limit on K (VEC1_MAX_K in vec1.c).
const MAX_K: usize = 32 * 1024 * 1024;

extern "C" {
    // vendor/vec1.c's `sqlite3_extension_init`, renamed by build.rs.
    fn sqlite3_vec1_init(
        db: *mut rusqlite::ffi::sqlite3,
        err: *mut *mut c_char,
        api: *const rusqlite::ffi::sqlite3_api_routines,
    ) -> c_int;
}

type SqlValue = rusqlite::types::Value;

fn backend(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}
fn schema(s: impl Into<String>) -> Error {
    Error::Schema(s.into())
}
fn check_id(id: NodeId) -> Result<()> {
    if id > i64::MAX as u64 {
        Err(schema("NodeId exceeds i64::MAX"))
    } else {
        Ok(())
    }
}
fn check_name(name: &str) -> Result<()> {
    if valid_name(name) {
        Ok(())
    } else {
        Err(schema(format!("invalid name: {name}")))
    }
}
fn sql_value(value: &Value) -> SqlValue {
    match value {
        Value::Null => SqlValue::Null,
        Value::Int(v) => SqlValue::Integer(*v),
        Value::Float(v) => SqlValue::Real(*v),
        Value::Text(v) => SqlValue::Text(v.clone()),
        Value::Bytes(v) => SqlValue::Blob(v.clone()),
        // Unreachable in practice: `compatible` rejects an unknown variant first.
        _ => SqlValue::Null,
    }
}
fn compatible(value: &Value, ty: ColumnType) -> bool {
    matches!(
        (value, ty),
        (Value::Null, _)
            | (Value::Int(_), ColumnType::Int)
            | (Value::Float(_), ColumnType::Float)
            | (Value::Text(_), ColumnType::Text)
            | (Value::Bytes(_), ColumnType::Bytes)
    ) && !matches!(value, Value::Float(x) if !x.is_finite())
}
fn ty_code(ty: ColumnType) -> i64 {
    match ty {
        ColumnType::Int => 0,
        ColumnType::Float => 1,
        ColumnType::Text => 2,
        ColumnType::Bytes => 3,
        // Unreachable in practice: `define_table` rejects an unknown type first.
        _ => -1,
    }
}
fn code_ty(code: i64) -> Result<ColumnType> {
    match code {
        0 => Ok(ColumnType::Int),
        1 => Ok(ColumnType::Float),
        2 => Ok(ColumnType::Text),
        3 => Ok(ColumnType::Bytes),
        _ => Err(backend("invalid stored column type")),
    }
}
fn projection(def: &TableDef) -> String {
    let mut sql = String::from("id");
    for col in &def.columns {
        sql.push_str(&format!(",\"c_{}\"", col.name));
    }
    sql
}
fn read_row(row: &rusqlite::Row<'_>, def: &TableDef) -> rusqlite::Result<Row> {
    let mut values = Vec::with_capacity(def.columns.len());
    for (i, col) in def.columns.iter().enumerate() {
        let p = i + 1;
        values.push(
            if row.get_ref(p)?.data_type() == rusqlite::types::Type::Null {
                Value::Null
            } else {
                match col.ty {
                    ColumnType::Int => Value::Int(row.get(p)?),
                    ColumnType::Float => Value::Float(row.get(p)?),
                    ColumnType::Text => Value::Text(row.get(p)?),
                    ColumnType::Bytes => Value::Bytes(row.get(p)?),
                    _ => return Err(rusqlite::Error::InvalidQuery),
                }
            },
        );
    }
    Ok(Row {
        id: row.get(0)?,
        values,
    })
}
fn bytes(vector: &[f32]) -> Vec<u8> {
    // vec1's native format: packed float32 in machine byte order.
    vector.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn floats(blob: &[u8], dim: usize) -> Result<Vec<f32>> {
    if blob.len() != dim * 4 {
        return Err(backend("vec1 returned a vector of the wrong size"));
    }
    Ok(blob
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}
/// The contract's distance, computed in f32 like `conformance::exact_distance`.
fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    if metric == Metric::L2 {
        return a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        1.0
    } else {
        1.0 - dot / (na * nb)
    }
}
fn limit_value(limit: usize) -> SqlValue {
    SqlValue::Integer(limit.min(i64::MAX as usize) as i64)
}
/// vec1's published model settings: PQ at dim/8 bytes for engineered L2 descriptors such as
/// SIFT, OPQ at 32 bytes without residuals for neural cosine embeddings.
fn model_params(def: &VectorIndexDef, threads: usize) -> String {
    match def.metric {
        Metric::L2 => format!(
            r#"{{"distance":"l2","nbucket":{NBUCKET},"quantizer":"pq","codesize":{},"nthread":{threads}}}"#,
            def.dim.div_ceil(8).clamp(1, 256)
        ),
        _ => format!(
            r#"{{"distance":"cos","nbucket":{NBUCKET},"quantizer":"opq","codesize":32,"residual":0,"nthread":{threads}}}"#
        ),
    }
}
fn env_factor(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(default)
}
fn register_extension() -> Result<()> {
    static REGISTERED: OnceLock<i32> = OnceLock::new();
    let rc = *REGISTERED.get_or_init(|| {
        // SAFETY: `sqlite3_vec1_init` is vec1's extension entry point (renamed at compile time),
        // with exactly the signature SQLite's auto-extension list expects. It is statically
        // linked, so it stays valid for the life of the process, and SQLite serialises access
        // to the auto-extension list itself.
        unsafe { rusqlite::ffi::sqlite3_auto_extension(Some(sqlite3_vec1_init)) }
    });
    if rc == rusqlite::ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(backend(format!("vec1 registration failed: {rc}")))
    }
}

/// The frozen Store contract on SQLite tables plus one vec1 virtual table per vector index.
/// A directory admits one writable adapter handle, enforced by an OS file lock.
pub struct Vec1Store {
    db: Connection,
    path: PathBuf,
    tables: HashMap<String, TableDef>,
    definitions: HashMap<String, VectorIndexDef>,
    /// Indexes that have a trained model, with the vector count it was trained at.
    trained: HashMap<String, i64>,
    sync_each_write: bool,
    threads: usize,
    nprobe_div: usize,
    rerank_factor: usize,
    _lock: File,
}

impl Vec1Store {
    fn table(&self, name: &str) -> Result<&TableDef> {
        self.tables
            .get(name)
            .ok_or_else(|| Error::NotFound(name.into()))
    }
    fn index(&self, name: &str) -> Result<&VectorIndexDef> {
        self.definitions
            .get(name)
            .ok_or_else(|| Error::NotFound(name.into()))
    }
    fn validate_query(&self, name: &str, query: &[f32]) -> Result<&VectorIndexDef> {
        let def = self.index(name)?;
        if query.len() != def.dim || query.iter().any(|x| !x.is_finite()) {
            return Err(schema("invalid vector dimension or non-finite coordinate"));
        }
        Ok(def)
    }
    fn finish_write(&mut self) -> Result<()> {
        if self.sync_each_write {
            self.flush()?;
        }
        Ok(())
    }
    fn load_definitions(&mut self) -> Result<()> {
        let mut stmt = self
            .db
            .prepare("SELECT name FROM table_defs ORDER BY name")
            .map_err(backend)?;
        let names = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(backend)?;
        for name in names {
            let name = name.map_err(backend)?;
            check_name(&name)?;
            let mut columns = Vec::new();
            let mut stmt = self
                .db
                .prepare("SELECT name,ty,indexed FROM column_defs WHERE tbl=? ORDER BY pos")
                .map_err(backend)?;
            let cols = stmt
                .query_map([&name], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, bool>(2)?,
                    ))
                })
                .map_err(backend)?;
            for col in cols {
                let (name, ty, indexed) = col.map_err(backend)?;
                check_name(&name)?;
                columns.push(ColumnDef {
                    name,
                    ty: code_ty(ty)?,
                    indexed,
                });
            }
            self.tables.insert(name.clone(), TableDef { name, columns });
        }
        let mut stmt = self
            .db
            .prepare("SELECT name,dim,metric,m,ef FROM vector_defs")
            .map_err(backend)?;
        let defs = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, usize>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, usize>(3)?,
                    r.get::<_, usize>(4)?,
                ))
            })
            .map_err(backend)?;
        for def in defs {
            let (name, dim, metric, m, ef_construction) = def.map_err(backend)?;
            check_name(&name)?;
            if dim == 0 || !matches!(metric, 0 | 1) {
                return Err(backend("invalid stored vector definition"));
            }
            self.definitions.insert(
                name.clone(),
                VectorIndexDef {
                    name,
                    dim,
                    metric: if metric == 0 {
                        Metric::Cosine
                    } else {
                        Metric::L2
                    },
                    m,
                    ef_construction,
                },
            );
        }
        let mut stmt = self
            .db
            .prepare("SELECT name,trained_count FROM vec1_models")
            .map_err(backend)?;
        let models = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            .map_err(backend)?;
        for model in models {
            let (name, count) = model.map_err(backend)?;
            self.trained.insert(name, count);
        }
        Ok(())
    }
    fn filter(&self, def: &TableDef, filter: &Filter) -> Result<(String, Vec<SqlValue>)> {
        let name = match filter {
            Filter::All => return Ok(("1".into(), vec![])),
            Filter::Eq { column, .. }
            | Filter::IntRange { column, .. }
            | Filter::TextPrefix { column, .. } => column,
            _ => return Err(schema("unsupported filter")),
        };
        let col = def
            .columns
            .iter()
            .find(|c| &c.name == name)
            .ok_or_else(|| Error::NotFound(name.clone()))?;
        let c = format!("\"c_{name}\"");
        match filter {
            Filter::All => unreachable!(),
            Filter::Eq { value, .. } => {
                if !compatible(value, col.ty) {
                    return Err(schema("filter type mismatch"));
                }
                Ok((format!("{c} = ?"), vec![sql_value(value)]))
            }
            Filter::IntRange { min, max, .. } => {
                if col.ty != ColumnType::Int {
                    return Err(schema("range requires Int"));
                }
                Ok((
                    format!("{c} BETWEEN ? AND ?"),
                    vec![(*min).into(), (*max).into()],
                ))
            }
            Filter::TextPrefix { prefix, .. } => {
                if col.ty != ColumnType::Text {
                    return Err(schema("prefix requires Text"));
                }
                if prefix.is_empty() {
                    return Ok((format!("{c} IS NOT NULL"), vec![]));
                }
                if col.indexed {
                    let mut upper = prefix.as_bytes().to_vec();
                    let mut successor = None;
                    while let Some(last) = upper.pop() {
                        if last != u8::MAX {
                            upper.push(last + 1);
                            successor = Some(upper);
                            break;
                        }
                    }
                    let mut sql = format!("{c} COLLATE BINARY >= CAST(? AS TEXT)");
                    let mut values = vec![SqlValue::Blob(prefix.as_bytes().to_vec())];
                    if let Some(upper) = successor {
                        sql.push_str(&format!(" AND {c} COLLATE BINARY < CAST(? AS TEXT)"));
                        values.push(SqlValue::Blob(upper));
                    }
                    Ok((sql, values))
                } else {
                    Ok((
                        format!("substr(CAST({c} AS BLOB),1,?) = ?"),
                        vec![
                            (prefix.len() as i64).into(),
                            SqlValue::Blob(prefix.as_bytes().to_vec()),
                        ],
                    ))
                }
            }
            _ => Err(schema("unsupported filter")),
        }
    }
    /// Runs a candidate query that yields `(rowid, vector)` and reranks the candidates on their
    /// full vectors with the contract's distance, ties by id.
    fn rerank(
        &self,
        sql: &str,
        values: Vec<SqlValue>,
        def: &VectorIndexDef,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Hit>> {
        let mut stmt = self.db.prepare_cached(sql).map_err(backend)?;
        let mut rows = stmt.query(params_from_iter(values)).map_err(backend)?;
        let mut hits = Vec::new();
        while let Some(r) = rows.next().map_err(backend)? {
            let id: i64 = r.get(0).map_err(backend)?;
            let blob = r.get_ref(1).map_err(backend)?.as_blob().map_err(backend)?;
            let vector = floats(blob, def.dim)?;
            hits.push(Hit {
                id: id as u64,
                distance: distance(def.metric, query, &vector),
            });
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        hits.dedup_by_key(|h| h.id);
        hits.truncate(k);
        Ok(hits)
    }
    fn search(
        &self,
        index: &str,
        query: &[f32],
        k: usize,
        ef: usize,
        restriction: Option<(&str, &Filter)>,
    ) -> Result<Vec<Hit>> {
        let def = self.validate_query(index, query)?;
        let (join, filter_values) = if let Some((table, filter)) = restriction {
            let (predicate, values) = self.filter(self.table(table)?, filter)?;
            (
                format!(" CROSS JOIN \"t_{table}\" AS p ON p.id = v.rowid WHERE {predicate}"),
                values,
            )
        } else {
            (String::new(), vec![])
        };
        if k == 0 {
            return Ok(vec![]);
        }
        let blob = SqlValue::Blob(bytes(query));
        let zero_cosine = def.metric == Metric::Cosine && query.iter().all(|x| *x == 0.0);
        if !self.trained.contains_key(index) {
            // Flat mode: vec1 scans every vector; asking for all of them keeps ties exact.
            let n: i64 = self
                .db
                .query_row(&format!("SELECT count(*) FROM \"v_{index}\""), [], |r| {
                    r.get(0)
                })
                .map_err(backend)?;
            if n == 0 {
                return Ok(vec![]);
            }
            let sql = format!("SELECT v.rowid, v.vector FROM \"v_{index}\"(?, ?) AS v{join}");
            let mut values = vec![blob, SqlValue::Text(format!("{{\"K\":{}}}", (n as usize).min(MAX_K)))];
            values.extend(filter_values);
            return self.rerank(&sql, values, def, query, k);
        }
        if zero_cosine {
            // Every distance is 1, so the answer is the smallest ids; no index can rank that.
            let sql = format!("SELECT v.rowid, v.vector FROM \"v_{index}\" AS v{join} ORDER BY v.rowid LIMIT ?");
            let mut values = filter_values;
            values.push(limit_value(k));
            return self.rerank(&sql, values, def, query, k);
        }
        let candidates = k.max(ef.saturating_mul(self.rerank_factor)).min(MAX_K);
        let nprobe = (ef / self.nprobe_div).max(1);
        let (sql, mut values) = if restriction.is_some() {
            // Streaming query (vec1 manual section 3.1): vec1 keeps producing neighbours in
            // roughly ascending distance until `candidates` rows have passed the join filter.
            (
                format!("SELECT v.rowid, v.vector FROM \"v_{index}\"(?, ?) AS v{join} LIMIT ?"),
                vec![
                    blob,
                    SqlValue::Text(format!(
                        "{{\"K\":{candidates},\"nprobe\":{nprobe},\"streaming\":1}}"
                    )),
                ],
            )
        } else {
            (
                format!("SELECT v.rowid, v.vector FROM \"v_{index}\"(?, ?) AS v"),
                vec![
                    blob,
                    SqlValue::Text(format!("{{\"K\":{candidates},\"nprobe\":{nprobe}}}")),
                ],
            )
        };
        if restriction.is_some() {
            values.extend(filter_values);
            values.push(limit_value(candidates));
        }
        self.rerank(&sql, values, def, query, k)
    }
    /// Trains a model and rebuilds the index for every vector index that reached `TRAIN_AT`
    /// vectors without a model, or doubled in size since its model was trained.
    fn train_due(&mut self) -> Result<()> {
        let mut names: Vec<String> = self.definitions.keys().cloned().collect();
        names.sort();
        for name in names {
            let count: i64 = self
                .db
                .query_row(&format!("SELECT count(*) FROM \"v_{name}\""), [], |r| {
                    r.get(0)
                })
                .map_err(backend)?;
            let due = match self.trained.get(&name) {
                None => count >= TRAIN_AT,
                Some(&at) => count >= at.saturating_mul(2),
            };
            if !due {
                continue;
            }
            let params = model_params(&self.definitions[&name], self.threads);
            let tx = self.db.transaction().map_err(backend)?;
            let model: Vec<u8> = if count as usize <= TRAIN_SAMPLE {
                tx.query_row(
                    &format!("SELECT vec1_train(vector, ?) FROM \"v_{name}\""),
                    [&params],
                    |r| r.get(0),
                )
                .map_err(backend)?
            } else {
                // A deterministic, evenly spaced sample over the ids, so that runs repeat.
                let ids: Vec<i64> = {
                    let mut stmt = tx
                        .prepare(&format!("SELECT rowid FROM \"v_{name}\" ORDER BY rowid"))
                        .map_err(backend)?;
                    let ids = stmt
                        .query_map([], |r| r.get(0))
                        .map_err(backend)?
                        .collect::<rusqlite::Result<_>>()
                        .map_err(backend)?;
                    ids
                };
                tx.execute_batch(
                    "CREATE TEMP TABLE IF NOT EXISTS vec1_sample(id INTEGER PRIMARY KEY); DELETE FROM temp.vec1_sample;",
                )
                .map_err(backend)?;
                {
                    let mut insert = tx
                        .prepare("INSERT OR IGNORE INTO temp.vec1_sample VALUES(?)")
                        .map_err(backend)?;
                    for i in 0..TRAIN_SAMPLE {
                        insert
                            .execute([ids[i * ids.len() / TRAIN_SAMPLE]])
                            .map_err(backend)?;
                    }
                }
                let model = tx
                    .query_row(
                        &format!("SELECT vec1_train(v.vector, ?) FROM temp.vec1_sample AS s CROSS JOIN \"v_{name}\" AS v ON v.rowid = s.id"),
                        [&params],
                        |r| r.get(0),
                    )
                    .map_err(backend)?;
                tx.execute_batch("DELETE FROM temp.vec1_sample")
                    .map_err(backend)?;
                model
            };
            tx.execute(
                &format!("INSERT INTO \"v_{name}\"(cmd, arg) VALUES('rebuild', ?)"),
                [&model],
            )
            .map_err(backend)?;
            tx.execute(
                "INSERT OR REPLACE INTO vec1_models VALUES(?,?,?)",
                params![name, count, params],
            )
            .map_err(backend)?;
            tx.commit().map_err(backend)?;
            self.trained.insert(name, count);
        }
        Ok(())
    }
}

impl Store for Vec1Store {
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self> {
        if !opts.create && !path.join("vec1.sqlite").is_file() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "store missing").into());
        }
        if opts.create {
            fs::create_dir_all(path)?;
        }
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.join("vec1.lock"))?;
        match lock.try_lock() {
            Ok(()) => (),
            Err(fs::TryLockError::WouldBlock) => {
                return Err(backend("store already open for writing"))
            }
            Err(fs::TryLockError::Error(e)) => return Err(e.into()),
        }
        register_extension()?;
        let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
            | if opts.create {
                rusqlite::OpenFlags::SQLITE_OPEN_CREATE
            } else {
                rusqlite::OpenFlags::empty()
            };
        let db =
            Connection::open_with_flags(path.join("vec1.sqlite"), flags).map_err(backend)?;
        db.busy_timeout(std::time::Duration::from_secs(30))
            .map_err(backend)?;
        db.set_prepared_statement_cache_capacity(128);
        let journal: String = db
            .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
            .map_err(backend)?;
        if !journal.eq_ignore_ascii_case("wal") {
            return Err(backend(format!("SQLite refused WAL mode: {journal}")));
        }
        db.execute_batch("PRAGMA synchronous=FULL; PRAGMA temp_store=MEMORY;
            CREATE TABLE IF NOT EXISTS table_defs(name TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS column_defs(tbl TEXT NOT NULL,pos INTEGER NOT NULL,name TEXT NOT NULL,ty INTEGER NOT NULL,indexed INTEGER NOT NULL,PRIMARY KEY(tbl,pos));
            CREATE TABLE IF NOT EXISTS vector_defs(name TEXT PRIMARY KEY,dim INTEGER NOT NULL,metric INTEGER NOT NULL,m INTEGER NOT NULL,ef INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS vec1_models(name TEXT PRIMARY KEY,trained_count INTEGER NOT NULL,params TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS edges(src INTEGER NOT NULL,rel TEXT NOT NULL,dst INTEGER NOT NULL,weight REAL NOT NULL,PRIMARY KEY(src,rel,dst));
            CREATE INDEX IF NOT EXISTS edges_in ON edges(dst,rel,src);").map_err(backend)?;
        if !opts.sync_each_write {
            db.execute_batch("PRAGMA synchronous=NORMAL")
                .map_err(backend)?;
        }
        let threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        // Per connection: the thread count vec1 uses for training and for 'rebuild'.
        db.query_row(
            "SELECT vec1_config('nthread', ?)",
            [threads as i64],
            |r| r.get::<_, rusqlite::types::Value>(0),
        )
        .map_err(backend)?;
        let mut store = Self {
            db,
            path: path.to_path_buf(),
            tables: HashMap::new(),
            definitions: HashMap::new(),
            trained: HashMap::new(),
            sync_each_write: opts.sync_each_write,
            threads,
            nprobe_div: env_factor("QUICKDB_VEC1_NPROBE_DIV", 4),
            rerank_factor: env_factor("QUICKDB_VEC1_RERANK", 1),
            _lock: lock,
        };
        store.load_definitions()?;
        Ok(store)
    }
    fn backend_name(&self) -> &'static str {
        "vec1"
    }
    fn define_table(&mut self, def: &TableDef) -> Result<()> {
        check_name(&def.name)?;
        let mut names = HashSet::new();
        for col in &def.columns {
            check_name(&col.name)?;
            if col.name == "id" || !names.insert(&col.name) {
                return Err(schema("duplicate or reserved column"));
            }
        }
        if let Some(old) = self.tables.get(&def.name) {
            return if old == def {
                Ok(())
            } else {
                Err(schema("table redefinition"))
            };
        }
        let mut sql = format!("CREATE TABLE \"t_{}\"(id INTEGER PRIMARY KEY", def.name);
        for col in &def.columns {
            let ty = match col.ty {
                ColumnType::Int => "INTEGER",
                ColumnType::Float => "REAL",
                ColumnType::Text => "TEXT COLLATE BINARY",
                ColumnType::Bytes => "BLOB",
                _ => return Err(schema("unsupported column type")),
            };
            sql.push_str(&format!(",\"c_{}\" {ty}", col.name));
        }
        sql.push(')');
        let tx = self.db.transaction().map_err(backend)?;
        tx.execute_batch(&sql).map_err(backend)?;
        tx.execute("INSERT INTO table_defs VALUES(?)", [&def.name])
            .map_err(backend)?;
        for (pos, col) in def.columns.iter().enumerate() {
            tx.execute(
                "INSERT INTO column_defs VALUES(?,?,?,?,?)",
                params![def.name, pos as i64, col.name, ty_code(col.ty), col.indexed],
            )
            .map_err(backend)?;
            if col.indexed {
                tx.execute_batch(&format!(
                    "CREATE INDEX \"idx_{}_{}_{}\" ON \"t_{}\"(\"c_{}\")",
                    def.name.len(),
                    def.name,
                    col.name,
                    def.name,
                    col.name
                ))
                .map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        self.tables.insert(def.name.clone(), def.clone());
        self.finish_write()
    }
    fn upsert_rows(&mut self, table: &str, rows: &[Row]) -> Result<()> {
        let def = self.table(table)?;
        for row in rows {
            check_id(row.id)?;
            if row.values.len() != def.columns.len()
                || !row
                    .values
                    .iter()
                    .zip(&def.columns)
                    .all(|(v, c)| compatible(v, c.ty))
            {
                return Err(schema("row shape/type mismatch"));
            }
        }
        let sql = format!(
            "INSERT OR REPLACE INTO \"t_{table}\"({}) VALUES({})",
            projection(def),
            vec!["?"; def.columns.len() + 1].join(",")
        );
        let tx = self.db.transaction().map_err(backend)?;
        {
            let mut stmt = tx.prepare_cached(&sql).map_err(backend)?;
            for row in rows {
                let values = std::iter::once(SqlValue::Integer(row.id as i64))
                    .chain(row.values.iter().map(sql_value));
                stmt.execute(params_from_iter(values)).map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        self.finish_write()
    }
    fn get_row(&self, table: &str, id: NodeId) -> Result<Option<Row>> {
        let def = self.table(table)?;
        check_id(id)?;
        self.db
            .prepare_cached(&format!(
                "SELECT {} FROM \"t_{table}\" WHERE id=?",
                projection(def)
            ))
            .map_err(backend)?
            .query_row([id as i64], |r| read_row(r, def))
            .optional()
            .map_err(backend)
    }
    fn select(&self, table: &str, filter: &Filter, limit: usize) -> Result<Vec<Row>> {
        let def = self.table(table)?;
        let (condition, mut values) = self.filter(def, filter)?;
        values.push(limit_value(limit));
        let sql = format!("SELECT {} FROM \"t_{table}\" WHERE id IN (SELECT id FROM \"t_{table}\" WHERE {condition} ORDER BY id LIMIT CAST(? AS INTEGER)) ORDER BY id", projection(def));
        let mut stmt = self.db.prepare_cached(&sql).map_err(backend)?;
        let rows = stmt
            .query_map(params_from_iter(values), |r| read_row(r, def))
            .map_err(backend)?;
        rows.collect::<rusqlite::Result<_>>().map_err(backend)
    }
    fn count_rows(&self, table: &str) -> Result<u64> {
        self.table(table)?;
        self.db
            .query_row(&format!("SELECT count(*) FROM \"t_{table}\""), [], |r| {
                r.get(0)
            })
            .map_err(backend)
    }
    fn delete_rows(&mut self, table: &str, ids: &[NodeId]) -> Result<u64> {
        self.table(table)?;
        for &id in ids {
            check_id(id)?;
        }
        let tx = self.db.transaction().map_err(backend)?;
        let mut count = 0;
        {
            let mut stmt = tx
                .prepare_cached(&format!("DELETE FROM \"t_{table}\" WHERE id=?"))
                .map_err(backend)?;
            for id in ids {
                count += stmt.execute([*id as i64]).map_err(backend)? as u64;
            }
        }
        tx.commit().map_err(backend)?;
        self.finish_write()?;
        Ok(count)
    }
    fn define_vector_index(&mut self, def: &VectorIndexDef) -> Result<()> {
        check_name(&def.name)?;
        // vec1 accepts 2..=4096 float32 components (VEC1_VECSIZE_MIN/MAX); HNSW settings are
        // stored for the redefinition check only.
        if def.dim < 2
            || def.dim > 4096
            || def.m > i64::MAX as usize
            || def.ef_construction > i64::MAX as usize
        {
            return Err(schema("invalid vector definition: vec1 requires 2..=4096 dimensions and signed-64-bit settings"));
        }
        if let Some(old) = self.definitions.get(&def.name) {
            return if old == def {
                Ok(())
            } else {
                Err(schema("vector index redefinition"))
            };
        }
        let metric_code = match def.metric {
            Metric::Cosine => 0,
            Metric::L2 => 1,
            _ => return Err(schema("unsupported metric")),
        };
        let tx = self.db.transaction().map_err(backend)?;
        let metric = if metric_code == 0 { "cos" } else { "l2" };
        tx.execute_batch(&format!(
            "CREATE VIRTUAL TABLE \"v_{}\" USING vec1(vector)",
            def.name
        ))
        .map_err(backend)?;
        // Exact flat mode until `flush` trains a model (see `train_due`).
        tx.execute(
            &format!("INSERT INTO \"v_{}\"(cmd, arg) VALUES('rebuild', ?)", def.name),
            [format!(r#"{{"index":"flat","distance":"{metric}"}}"#)],
        )
        .map_err(backend)?;
        tx.execute(
            "INSERT INTO vector_defs VALUES(?,?,?,?,?)",
            params![
                def.name,
                def.dim as i64,
                metric_code,
                def.m as i64,
                def.ef_construction as i64
            ],
        )
        .map_err(backend)?;
        tx.commit().map_err(backend)?;
        self.definitions.insert(def.name.clone(), def.clone());
        self.finish_write()
    }
    fn upsert_vectors(&mut self, index: &str, items: &[(NodeId, &[f32])]) -> Result<()> {
        self.index(index)?;
        for &(id, vector) in items {
            check_id(id)?;
            self.validate_query(index, vector)?;
        }
        let tx = self.db.transaction().map_err(backend)?;
        {
            // Replacement as delete + insert in one transaction, which also makes the batch
            // atomic; vec1 assigns the new vector to its bucket when a model exists.
            let mut delete = tx
                .prepare_cached(&format!("DELETE FROM \"v_{index}\" WHERE rowid=?"))
                .map_err(backend)?;
            let mut insert = tx
                .prepare_cached(&format!(
                    "INSERT INTO \"v_{index}\"(rowid,vector) VALUES(?,?)"
                ))
                .map_err(backend)?;
            for &(id, vector) in items {
                delete.execute([id as i64]).map_err(backend)?;
                insert
                    .execute(params![id as i64, bytes(vector)])
                    .map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        self.finish_write()
    }
    fn knn(&self, index: &str, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<Hit>> {
        self.search(index, query, k, ef_search, None)
    }
    fn knn_filtered(
        &self,
        index: &str,
        query: &[f32],
        k: usize,
        ef_search: usize,
        table: &str,
        filter: &Filter,
    ) -> Result<Vec<Hit>> {
        self.search(index, query, k, ef_search, Some((table, filter)))
    }
    fn count_vectors(&self, index: &str) -> Result<u64> {
        self.index(index)?;
        self.db
            .query_row(&format!("SELECT count(*) FROM \"v_{index}\""), [], |r| {
                r.get(0)
            })
            .map_err(backend)
    }
    fn upsert_edges(&mut self, edges: &[Edge]) -> Result<()> {
        for edge in edges {
            check_id(edge.src)?;
            check_id(edge.dst)?;
            if !edge.weight.is_finite() {
                return Err(schema("non-finite edge weight"));
            }
        }
        let tx = self.db.transaction().map_err(backend)?;
        {
            let mut stmt = tx
                .prepare_cached("INSERT OR REPLACE INTO edges VALUES(?,?,?,?)")
                .map_err(backend)?;
            for edge in edges {
                stmt.execute(params![
                    edge.src as i64,
                    edge.rel,
                    edge.dst as i64,
                    edge.weight
                ])
                .map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        self.finish_write()
    }
    fn neighbors(
        &self,
        node: NodeId,
        rel: Option<&str>,
        dir: Direction,
        limit: usize,
    ) -> Result<Vec<Edge>> {
        check_id(node)?;
        let condition = match dir {
            Direction::Out => "src=?1",
            Direction::In => "dst=?1",
            Direction::Both => "(src=?1 OR dst=?1)",
            _ => return Err(schema("unsupported direction")),
        };
        let sql = format!("SELECT src,rel,dst,weight FROM edges WHERE {condition} AND (?2 IS NULL OR rel=?2) ORDER BY rel,src,dst LIMIT CAST(?3 AS INTEGER)");
        let mut stmt = self.db.prepare_cached(&sql).map_err(backend)?;
        let rows = stmt
            .query_map(
                params![node as i64, rel, limit.min(i64::MAX as usize) as i64],
                |r| {
                    Ok(Edge {
                        src: r.get(0)?,
                        rel: r.get(1)?,
                        dst: r.get(2)?,
                        weight: r.get(3)?,
                    })
                },
            )
            .map_err(backend)?;
        rows.collect::<rusqlite::Result<_>>().map_err(backend)
    }
    fn k_hop(
        &self,
        start: NodeId,
        rel: Option<&str>,
        dir: Direction,
        max_depth: u32,
        limit: usize,
    ) -> Result<Vec<Reached>> {
        check_id(start)?;
        let mut seen = HashSet::from([start]);
        let mut frontier = vec![start];
        let mut result = Vec::new();
        for depth in 1..=max_depth {
            if frontier.is_empty() || result.len() >= limit {
                break;
            }
            let mut next = Vec::new();
            for node in frontier {
                for edge in self.neighbors(node, rel, dir, usize::MAX)? {
                    let other = match dir {
                        Direction::Out => edge.dst,
                        Direction::In => edge.src,
                        Direction::Both => {
                            if edge.src == node {
                                edge.dst
                            } else {
                                edge.src
                            }
                        }
                        _ => return Err(schema("unsupported direction")),
                    };
                    if seen.insert(other) {
                        next.push(other);
                    }
                }
            }
            next.sort_unstable();
            result.extend(
                next.iter()
                    .map(|&id| Reached { id, depth })
                    .take(limit - result.len()),
            );
            frontier = next;
        }
        Ok(result)
    }
    fn count_edges(&self) -> Result<u64> {
        self.db
            .query_row("SELECT count(*) FROM edges", [], |r| r.get(0))
            .map_err(backend)
    }
    fn flush(&mut self) -> Result<()> {
        // Training and index building run here, inside the caller's timed load.
        self.train_due()?;
        self.db
            .execute_batch("PRAGMA synchronous=FULL")
            .map_err(backend)?;
        let busy: i64 = self
            .db
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))
            .map_err(backend)?;
        if busy != 0 {
            return Err(backend("flush checkpoint blocked by another reader"));
        }
        #[cfg(unix)]
        File::open(&self.path)?.sync_all()?;
        if !self.sync_each_write {
            self.db
                .execute_batch("PRAGMA synchronous=NORMAL")
                .map_err(backend)?;
        }
        Ok(())
    }
}
