//! SQLite rows and edges plus sqlite-vec's exact, chunked vec0 vector search.
//!
//! See the crate README for upstream sources, durability and filtering details.
use quickdb_api::*;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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
    // sqlite-vec interprets BLOBs as native float32 arrays.
    vector.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn limit_value(limit: usize) -> SqlValue {
    SqlValue::Integer(limit.min(i64::MAX as usize) as i64)
}
fn register_extension() -> Result<()> {
    static REGISTERED: OnceLock<i32> = OnceLock::new();
    let rc = *REGISTERED.get_or_init(|| {
        // SAFETY: This is the registration pattern documented by sqlite-vec. The
        // statically linked entrypoint has SQLite's extension-init ABI and remains
        // valid for the process lifetime. SQLite serializes auto-extension setup.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::ffi::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::ffi::c_int,
            >(
                sqlite_vec::sqlite3_vec_init as *const ()
            )))
        }
    });
    if rc == rusqlite::ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(backend(format!("sqlite-vec registration failed: {rc}")))
    }
}

/// An independent SQLite/sqlite-vec implementation of the frozen Store contract.
/// A directory admits one writable adapter handle, enforced by an OS file lock.
pub struct SqliteVecStore {
    db: Connection,
    path: PathBuf,
    tables: HashMap<String, TableDef>,
    definitions: HashMap<String, VectorIndexDef>,
    sync_each_write: bool,
    _lock: File,
}

impl SqliteVecStore {
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
    fn query_hits(&self, sql: &str, values: Vec<SqlValue>, metric: Metric) -> Result<Vec<Hit>> {
        let mut stmt = self.db.prepare_cached(sql).map_err(backend)?;
        let rows = stmt
            .query_map(params_from_iter(values), |r| {
                let distance: f32 = r.get(1)?;
                Ok(Hit {
                    id: r.get(0)?,
                    distance: if metric == Metric::L2 {
                        distance * distance
                    } else {
                        distance
                    },
                })
            })
            .map_err(backend)?;
        rows.collect::<rusqlite::Result<_>>().map_err(backend)
    }
    fn search(
        &self,
        index: &str,
        query: &[f32],
        k: usize,
        restriction: Option<(&str, &Filter)>,
    ) -> Result<Vec<Hit>> {
        let metric = self.validate_query(index, query)?.metric;
        let (condition, filter_values) = if let Some((table, filter)) = restriction {
            let (predicate, values) = self.filter(self.table(table)?, filter)?;
            (
                format!("rowid IN (SELECT id FROM \"t_{table}\" WHERE {predicate})"),
                values,
            )
        } else {
            ("1".into(), vec![])
        };
        if k == 0 {
            return Ok(vec![]);
        }
        let blob = bytes(query);
        // vec0 has a hard k <= 4096 limit. Probe one extra result to detect a
        // boundary tie, since vec0 does not promise ID ordering among equal scores.
        let has_zero: bool = if metric == Metric::Cosine {
            self.db
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM zero_vectors WHERE name=?)",
                    [index],
                    |r| r.get(0),
                )
                .map_err(backend)?
        } else {
            false
        };
        if k < 4096 && !has_zero && (metric != Metric::Cosine || query.iter().any(|x| *x != 0.0)) {
            let sql = format!("SELECT rowid,distance FROM \"v_{index}\" WHERE embedding MATCH ? AND k = ? AND {condition} ORDER BY distance");
            let mut values = vec![SqlValue::Blob(blob.clone()), limit_value(k + 1)];
            values.extend(filter_values.clone());
            let mut hits = self.query_hits(&sql, values, metric)?;
            hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
            if hits.len() <= k || hits[k - 1].distance != hits[k].distance {
                hits.truncate(k);
                return Ok(hits);
            }
        }
        // Documented scalar-distance SQL over vec0 handles boundary ties, large k
        // and the contract's zero-norm cosine convention without ANN or truncation.
        let score = if metric == Metric::L2 {
            "vec_distance_L2(embedding,?)"
        } else if query.iter().all(|x| *x == 0.0) {
            "1.0 + 0 * length(?)"
        } else {
            "coalesce(vec_distance_cosine(embedding,?),1.0)"
        };
        let sql = format!("SELECT rowid,{score} AS score FROM \"v_{index}\" WHERE {condition} ORDER BY score,rowid LIMIT CAST(? AS INTEGER)");
        let mut values = vec![SqlValue::Blob(blob)];
        values.extend(filter_values);
        values.push(limit_value(k));
        let mut hits = self.query_hits(&sql, values, metric)?;
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        Ok(hits)
    }
}

impl Store for SqliteVecStore {
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self> {
        if !opts.create && !path.join("sqlitevec.sqlite").is_file() {
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
            .open(path.join("sqlitevec.lock"))?;
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
            Connection::open_with_flags(path.join("sqlitevec.sqlite"), flags).map_err(backend)?;
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
            CREATE TABLE IF NOT EXISTS zero_vectors(name TEXT NOT NULL,id INTEGER NOT NULL,PRIMARY KEY(name,id));
            CREATE TABLE IF NOT EXISTS edges(src INTEGER NOT NULL,rel TEXT NOT NULL,dst INTEGER NOT NULL,weight REAL NOT NULL,PRIMARY KEY(src,rel,dst));
            CREATE INDEX IF NOT EXISTS edges_in ON edges(dst,rel,src);").map_err(backend)?;
        if !opts.sync_each_write {
            db.execute_batch("PRAGMA synchronous=NORMAL")
                .map_err(backend)?;
        }
        let mut store = Self {
            db,
            path: path.to_path_buf(),
            tables: HashMap::new(),
            definitions: HashMap::new(),
            sync_each_write: opts.sync_each_write,
            _lock: lock,
        };
        store.load_definitions()?;
        Ok(store)
    }
    fn backend_name(&self) -> &'static str {
        "sqlite-vec"
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
        // vec0's float32 dimension limit, independent of ignored HNSW settings.
        if def.dim == 0
            || def.dim > 8192
            || def.m > i64::MAX as usize
            || def.ef_construction > i64::MAX as usize
        {
            return Err(schema("invalid vector definition: vec0 requires 1..=8192 dimensions and signed-64-bit settings"));
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
        let metric = if metric_code == 0 { "cosine" } else { "l2" };
        tx.execute_batch(&format!("CREATE VIRTUAL TABLE \"v_{}\" USING vec0(embedding float[{}] distance_metric={metric})", def.name, def.dim)).map_err(backend)?;
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
            // vec0 does not implement SQLite's UPSERT syntax. Delete and insert
            // in one transaction provides replacement and batch atomicity.
            let mut delete = tx
                .prepare_cached(&format!("DELETE FROM \"v_{index}\" WHERE rowid=?"))
                .map_err(backend)?;
            let mut insert = tx
                .prepare_cached(&format!(
                    "INSERT INTO \"v_{index}\"(rowid,embedding) VALUES(?,?)"
                ))
                .map_err(backend)?;
            let mut zero_delete = tx
                .prepare_cached("DELETE FROM zero_vectors WHERE name=? AND id=?")
                .map_err(backend)?;
            let mut zero_insert = tx
                .prepare_cached("INSERT INTO zero_vectors VALUES(?,?)")
                .map_err(backend)?;
            for &(id, vector) in items {
                delete.execute([id as i64]).map_err(backend)?;
                insert
                    .execute(params![id as i64, bytes(vector)])
                    .map_err(backend)?;
                zero_delete
                    .execute(params![index, id as i64])
                    .map_err(backend)?;
                if vector.iter().all(|x| *x == 0.0) {
                    zero_insert
                        .execute(params![index, id as i64])
                        .map_err(backend)?;
                }
            }
        }
        tx.commit().map_err(backend)?;
        self.finish_write()
    }
    fn knn(&self, index: &str, query: &[f32], k: usize, _ef_search: usize) -> Result<Vec<Hit>> {
        self.search(index, query, k, None)
    }
    fn knn_filtered(
        &self,
        index: &str,
        query: &[f32],
        k: usize,
        _ef_search: usize,
        table: &str,
        filter: &Filter,
    ) -> Result<Vec<Hit>> {
        self.search(index, query, k, Some((table, filter)))
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
