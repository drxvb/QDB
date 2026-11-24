//! The QDB engine: [`NativeStore`] implements [`quickdb_api::Store`] with bundled SQLite for rows,
//! edges and authoritative vectors, and a persisted, pure-Rust HNSW index for vector search.
//!
//! A store is one directory holding `quickdb.sqlite` (WAL mode), a lock file and one
//! checksummed `<index>.hnsw` snapshot per vector index. A missing, damaged or stale snapshot is
//! rebuilt from SQLite on open. One writer per directory: a second [`Store::open`] fails at once,
//! while [`NativeStore::open_read_only`] takes no lock. Keep stores on local disk.
//!
//! Most users depend on the `quickdb` crate, which re-exports this engine and the contract and
//! has a quick start.
//!
//! Full-text search (SQLite FTS5, BM25, with Arabic folding) and hybrid text + vector search are
//! inherent methods of [`NativeStore`]; see [`fts`].
pub mod fts;
mod hnsw;

pub use fts::TextIndexDef;

use hnsw::{snapshot, Hnsw};
use quickdb_api::*;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const LOCK_FILE: &str = "quickdb.lock";

/// Identifies the exact SQLite vector state a snapshot was taken from. The nonce is a
/// fresh random value per vector write, so a restored backup whose revision counter
/// happens to equal the snapshot's still does not match it.
/// (store id, revision, nonce)
type SnapshotHeader = ([u8; 16], u64, i64);

struct ChecksumWriter<W> {
    inner: W,
    hash: crc32fast::Hasher,
}
impl<W: Write> Write for ChecksumWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hash.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn load_snapshot(path: &Path) -> Option<(SnapshotHeader, Hnsw)> {
    let mut input = BufReader::new(File::open(path).ok()?);
    let length = input.get_ref().metadata().ok()?.len();
    let payload = length.checked_sub(4)?;
    if payload > 8 * 1024 * 1024 * 1024 {
        return None;
    }
    // Verify before deserializing lengths or graph pointers from damaged bytes.
    let mut checksum = ChecksumWriter {
        inner: std::io::sink(),
        hash: crc32fast::Hasher::new(),
    };
    std::io::copy(&mut input.by_ref().take(payload), &mut checksum).ok()?;
    let mut stored = [0; 4];
    input.read_exact(&mut stored).ok()?;
    if checksum.hash.finalize() != u32::from_le_bytes(stored) {
        return None;
    }
    input.seek(SeekFrom::Start(0)).ok()?;
    let (header, graph) = snapshot::decode(input.by_ref().take(payload), payload).ok()?;
    let h = Hnsw::from_snapshot(graph).ok()?;
    if input.stream_position().ok()? != payload {
        return None;
    }
    Some((header, h))
}

fn has_table(db: &Connection, name: &str) -> Result<bool> {
    db.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
        [name],
        |r| r.get::<_, i64>(0),
    )
    .map(|n| n > 0)
    .map_err(backend)
}

fn store_id(db: &Connection) -> Result<Option<[u8; 16]>> {
    if !has_table(db, "store_meta")? {
        return Ok(None);
    }
    let raw: Option<Vec<u8>> = db
        .query_row(
            "SELECT value FROM store_meta WHERE key='store_id'",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(backend)?;
    Ok(raw.and_then(|v| <[u8; 16]>::try_from(v.as_slice()).ok()))
}

/// Removes snapshot temps left by an interrupted flush. Only a lock holder calls this.
fn remove_snapshot_temps(dir: &Path, name: &str) -> Result<()> {
    let prefix = format!("{name}.hnsw.tmp");
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            remove_snapshot_temp(&entry.path())?;
        }
    }
    Ok(())
}

fn take_lock(dir: &Path) -> Result<File> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(LOCK_FILE))?;
    // std's File::try_lock is flock(LOCK_EX|LOCK_NB) on Unix and LockFileEx on Windows.
    // The OS drops it when the descriptor closes, including on kill -9.
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(fs::TryLockError::WouldBlock) => Err(backend(format!(
            "store is locked by another process (or another handle in this process): {}; open it with NativeStore::open_read_only to read",
            dir.display()
        ))),
        Err(fs::TryLockError::Error(e)) => Err(Error::Io(e)),
    }
}

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
fn sql_value(v: &Value) -> rusqlite::types::Value {
    use rusqlite::types::Value as S;
    match v {
        Value::Null => S::Null,
        Value::Int(v) => S::Integer(*v),
        Value::Float(v) => S::Real(*v),
        Value::Text(v) => S::Text(v.clone()),
        Value::Bytes(v) => S::Blob(v.clone()),
        // Unreachable in practice: `compatible` rejects an unknown variant first.
        _ => S::Null,
    }
}
fn compatible(v: &Value, ty: ColumnType) -> bool {
    matches!(
        (v, ty),
        (Value::Null, _)
            | (Value::Int(_), ColumnType::Int)
            | (Value::Float(_), ColumnType::Float)
            | (Value::Text(_), ColumnType::Text)
            | (Value::Bytes(_), ColumnType::Bytes)
    ) && !matches!(v, Value::Float(x) if !x.is_finite())
}
fn ty_code(ty: ColumnType) -> i64 {
    match ty {
        ColumnType::Int => 0,
        ColumnType::Float => 1,
        ColumnType::Text => 2,
        ColumnType::Bytes => 3,
        // Unreachable in practice: `define_table` rejects an unknown type before this is called.
        _ => -1,
    }
}
fn code_ty(n: i64) -> Result<ColumnType> {
    match n {
        0 => Ok(ColumnType::Int),
        1 => Ok(ColumnType::Float),
        2 => Ok(ColumnType::Text),
        3 => Ok(ColumnType::Bytes),
        _ => Err(backend("invalid stored column type")),
    }
}
fn read_row(row: &rusqlite::Row<'_>, def: &TableDef) -> rusqlite::Result<Row> {
    let id = row.get::<_, u64>(0)?;
    let mut values = Vec::with_capacity(def.columns.len());
    for (i, col) in def.columns.iter().enumerate() {
        let pos = i + 1;
        values.push(
            if row.get_ref(pos)?.data_type() == rusqlite::types::Type::Null {
                Value::Null
            } else {
                match col.ty {
                    ColumnType::Int => Value::Int(row.get(pos)?),
                    ColumnType::Float => Value::Float(row.get(pos)?),
                    ColumnType::Text => Value::Text(row.get(pos)?),
                    ColumnType::Bytes => Value::Bytes(row.get(pos)?),
                    _ => return Err(rusqlite::Error::InvalidQuery),
                }
            },
        );
    }
    Ok(Row { id, values })
}
fn projection(def: &TableDef) -> String {
    let mut p = String::from("id");
    for c in &def.columns {
        p.push_str(&format!(",\"c_{}\"", c.name));
    }
    p
}

fn prefix_upper(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut upper = prefix.to_vec();
    while let Some(last) = upper.pop() {
        if last != u8::MAX {
            upper.push(last + 1);
            return Some(upper);
        }
    }
    None
}

fn prefix_range(column: &str, prefix: &[u8]) -> (String, Vec<rusqlite::types::Value>) {
    // The successor can be invalid UTF-8. SQLite's CAST preserves its bytes while
    // giving both bounds TEXT storage class, so BINARY compares text with text.
    let mut values = vec![rusqlite::types::Value::Blob(prefix.to_vec())];
    let mut sql = format!("{column} COLLATE BINARY >= CAST(? AS TEXT)");
    if let Some(upper) = prefix_upper(prefix) {
        sql.push_str(&format!(" AND {column} COLLATE BINARY < CAST(? AS TEXT)"));
        values.push(rusqlite::types::Value::Blob(upper));
    }
    (sql, values)
}

fn remove_snapshot_temp(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn enter_wal(db: &Connection) -> Result<()> {
    let journal: String = db
        .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
        .map_err(backend)?;
    if !journal.eq_ignore_ascii_case("wal") {
        return Err(backend(format!("SQLite refused WAL mode: {journal}")));
    }
    Ok(())
}

/// Positions of `columns` in `table`; each must be a distinct `Text` column.
fn text_positions(table: &TableDef, columns: &[String]) -> Result<Vec<usize>> {
    if columns.is_empty() {
        return Err(schema("text index needs at least one column"));
    }
    let mut positions = Vec::with_capacity(columns.len());
    for name in columns {
        check_name(name)?;
        let pos = table
            .columns
            .iter()
            .position(|c| &c.name == name)
            .ok_or_else(|| Error::NotFound(name.clone()))?;
        if table.columns[pos].ty != ColumnType::Text {
            return Err(schema(format!("text index column {name} is not Text")));
        }
        if positions.contains(&pos) {
            return Err(schema(format!("duplicate text index column {name}")));
        }
        positions.push(pos);
    }
    Ok(positions)
}

/// `INSERT` into an FTS5 table: the rowid, then the indexed values in columns `c0, c1, ...`.
fn fts_insert(fts: &str, columns: usize) -> String {
    let mut names = String::from("rowid");
    let mut slots = String::from("?");
    for i in 0..columns {
        names.push_str(&format!(",c{i}"));
        slots.push_str(",?");
    }
    format!("INSERT INTO \"{fts}\"({names}) VALUES({slots})")
}

/// Inserts the folded form of one row's indexed values into an FTS5 table.
fn insert_text<'a>(
    insert: &mut rusqlite::Statement<'_>,
    id: NodeId,
    values: impl Iterator<Item = &'a Value>,
) -> Result<()> {
    let folded = values.map(|v| match v {
        Value::Text(text) => rusqlite::types::Value::Text(fts::fold(text)),
        _ => rusqlite::types::Value::Null,
    });
    insert
        .execute(params_from_iter(
            std::iter::once(rusqlite::types::Value::Integer(id as i64)).chain(folded),
        ))
        .map_err(backend)?;
    Ok(())
}

/// A QuickDB store on bundled SQLite (rows and edges) with persisted pure-Rust HNSW vector indexes.
/// One writable handle per directory is enforced by a file lock; see [`NativeStore::open_read_only`]
/// for readers.
pub struct NativeStore {
    db: Connection,
    path: PathBuf,
    tables: HashMap<String, TableDef>,
    definitions: HashMap<String, VectorIndexDef>,
    indexes: HashMap<String, Hnsw>,
    sync_each_write: bool,
    rebuilds: usize,
    stale_indexes: HashSet<String>,
    /// (revision, nonce) per index, as committed by this handle.
    states: HashMap<String, (u64, i64)>,
    store_id: Option<[u8; 16]>,
    /// Held for the life of a writable store; `None` for `open_read_only`.
    lock: Option<File>,
    text_indexes: HashMap<String, TextIndexDef>,
}
impl NativeStore {
    /// Opens an existing store for reading from any number of processes, beside at most
    /// one writer. Never writes files in the directory except SQLite's own `-wal`/`-shm`;
    /// a stale or foreign snapshot is rebuilt in memory only. Every write method and
    /// `flush` return `Error::Backend`. Rows and edges are read live from SQLite; vectors
    /// are those committed when this handle opened.
    pub fn open_read_only(path: &Path) -> Result<Self> {
        if !path.join("quickdb.sqlite").is_file() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "store missing",
            )));
        }
        let db = Connection::open_with_flags(
            path.join("quickdb.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .map_err(backend)?;
        Self::finish_open(db, path, false, None)
    }
    fn writable(&self) -> Result<()> {
        if self.lock.is_none() {
            return Err(backend("store was opened read-only"));
        }
        Ok(())
    }
    fn finish_open(
        db: Connection,
        path: &Path,
        sync_each_write: bool,
        lock: Option<File>,
    ) -> Result<Self> {
        db.busy_timeout(std::time::Duration::from_secs(30))
            .map_err(backend)?;
        db.set_prepared_statement_cache_capacity(128);
        // The bundled SQLite (libsqlite3-sys 0.39+) is built with SQLITE_ENABLE_STAT4, which makes
        // plans depend on bound values: every cached statement then expires when the cache clears
        // its bindings and is re-prepared on the next run. The store's statements are written so
        // that one plan fits every value, so the query planner stability guarantee loses nothing.
        db.set_db_config(
            rusqlite::config::DbConfig::SQLITE_DBCONFIG_ENABLE_QPSG,
            true,
        )
        .map_err(backend)?;
        if lock.is_some() {
            enter_wal(&db)?;
            db.execute_batch("PRAGMA synchronous=FULL; PRAGMA temp_store=MEMORY; CREATE TABLE IF NOT EXISTS table_defs(name TEXT PRIMARY KEY); CREATE TABLE IF NOT EXISTS column_defs(tbl TEXT NOT NULL,pos INTEGER NOT NULL,name TEXT NOT NULL,ty INTEGER NOT NULL,indexed INTEGER NOT NULL,PRIMARY KEY(tbl,pos)); CREATE TABLE IF NOT EXISTS vector_defs(name TEXT PRIMARY KEY,dim INTEGER NOT NULL,metric INTEGER NOT NULL,m INTEGER NOT NULL,ef INTEGER NOT NULL,revision INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS vectors(name TEXT NOT NULL,id INTEGER NOT NULL,data BLOB NOT NULL,PRIMARY KEY(name,id)); CREATE TABLE IF NOT EXISTS edges(src INTEGER NOT NULL,rel TEXT NOT NULL,dst INTEGER NOT NULL,weight REAL NOT NULL,PRIMARY KEY(src,rel,dst)); CREATE INDEX IF NOT EXISTS edges_in ON edges(dst,rel,src); CREATE TABLE IF NOT EXISTS store_meta(key TEXT PRIMARY KEY,value BLOB NOT NULL); INSERT OR IGNORE INTO store_meta VALUES('store_id',randomblob(16)); CREATE TABLE IF NOT EXISTS vector_state(name TEXT PRIMARY KEY,nonce INTEGER NOT NULL); INSERT OR IGNORE INTO vector_state SELECT name,random() FROM vector_defs;").map_err(backend)?;
            db.execute_batch("CREATE TABLE IF NOT EXISTS text_defs(name TEXT PRIMARY KEY,tbl TEXT NOT NULL,columns TEXT NOT NULL);")
                .map_err(backend)?;
            if !sync_each_write {
                db.execute_batch("PRAGMA synchronous=NORMAL;")
                    .map_err(backend)?;
            }
        } else {
            db.execute_batch("PRAGMA temp_store=MEMORY; PRAGMA query_only=1;")
                .map_err(backend)?;
        }
        let tables = Self::load_tables(&db)?;
        let text_indexes = Self::load_text_indexes(&db, &tables)?;
        let mut store = Self {
            store_id: store_id(&db)?,
            db,
            path: path.to_path_buf(),
            tables,
            definitions: HashMap::new(),
            indexes: HashMap::new(),
            sync_each_write,
            rebuilds: 0,
            stale_indexes: HashSet::new(),
            states: HashMap::new(),
            lock,
            text_indexes,
        };
        store.load_indexes()?;
        Ok(store)
    }
    fn load_tables(db: &Connection) -> Result<HashMap<String, TableDef>> {
        let mut tables = HashMap::new();
        let mut stmt = db
            .prepare("SELECT name FROM table_defs ORDER BY name")
            .map_err(backend)?;
        let names = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(backend)?;
        for name in names {
            let name = name.map_err(backend)?;
            check_name(&name)?;
            let mut columns = Vec::new();
            let mut stmt = db
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
            for c in cols {
                let (name, ty, indexed) = c.map_err(backend)?;
                check_name(&name)?;
                columns.push(ColumnDef {
                    name,
                    ty: code_ty(ty)?,
                    indexed,
                });
            }
            tables.insert(name.clone(), TableDef { name, columns });
        }
        Ok(tables)
    }
    fn update_index_after_commit(
        &mut self,
        name: &str,
        state: (u64, i64),
        update: impl FnOnce(&mut Hnsw),
    ) -> Result<()> {
        let revision = state.0;
        self.states.insert(name.to_owned(), state);
        // Poison before touching HNSW. An unwind must never expose a partially
        // updated index or let flush stamp a stale snapshot with a new revision.
        self.stale_indexes.insert(name.to_owned());
        let h = self
            .indexes
            .get_mut(name)
            .ok_or_else(|| backend(format!("missing committed vector index {name}")))?;
        update(h);
        h.revision = revision;
        self.stale_indexes.remove(name);
        Ok(())
    }
    /// Number of indexes recovered from SQL during this open, zero for a valid snapshot.
    pub fn index_rebuilds(&self) -> usize {
        self.rebuilds
    }
    fn table(&self, name: &str) -> Result<&TableDef> {
        self.tables
            .get(name)
            .ok_or_else(|| Error::NotFound(name.into()))
    }
    fn index(&self, name: &str) -> Result<&Hnsw> {
        if self.stale_indexes.contains(name) {
            return Err(backend(format!(
                "vector index {name} is stale after a failed update; reopen the store to rebuild"
            )));
        }
        self.indexes
            .get(name)
            .ok_or_else(|| Error::NotFound(name.into()))
    }
    fn filter(
        &self,
        def: &TableDef,
        filter: &Filter,
    ) -> Result<(String, Vec<rusqlite::types::Value>)> {
        let column = match filter {
            Filter::All => return Ok(("1".into(), Vec::new())),
            Filter::Eq { column, .. }
            | Filter::IntRange { column, .. }
            | Filter::TextPrefix { column, .. } => column,
            _ => return Err(schema("unsupported filter")),
        };
        let col = def
            .columns
            .iter()
            .find(|c| &c.name == column)
            .ok_or_else(|| Error::NotFound(column.clone()))?;
        let c = format!("\"c_{column}\"");
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
                if col.indexed {
                    return Ok(prefix_range(&c, prefix.as_bytes()));
                }
                if prefix.is_empty() {
                    return Ok((format!("{c} IS NOT NULL"), Vec::new()));
                }
                // Blob substr is byte-wise and does not interpret '%' or '_' as wildcards.
                Ok((
                    format!("substr(CAST({c} AS BLOB),1,?) = ?"),
                    vec![
                        (prefix.len() as i64).into(),
                        rusqlite::types::Value::Blob(prefix.as_bytes().to_vec()),
                    ],
                ))
            }
            _ => Err(schema("unsupported filter")),
        }
    }
    fn validate_query(&self, index: &str, query: &[f32]) -> Result<&Hnsw> {
        let idx = self.index(index)?;
        let def = &self.definitions[index];
        if query.len() != def.dim || query.iter().any(|x| !x.is_finite()) {
            return Err(schema("invalid vector dimension or non-finite coordinate"));
        }
        Ok(idx)
    }
    fn finish_write(&mut self) -> Result<()> {
        self.writable()?;
        if self.sync_each_write {
            self.flush()?;
        }
        Ok(())
    }
    fn rebuild_index(db: &Connection, name: &str, def: &VectorIndexDef) -> Result<Hnsw> {
        let mut h = Hnsw::new(def.dim, def.metric, def.m, def.ef_construction);
        let mut vectors = db
            .prepare_cached("SELECT id,data FROM vectors WHERE name=? ORDER BY id")
            .map_err(backend)?;
        let mut rows = vectors.query([name]).map_err(backend)?;
        let mut batch = Vec::with_capacity(1000);
        while let Some(row) = rows.next().map_err(backend)? {
            let raw_id: i64 = row
                .get(0)
                .map_err(|e| backend(format!("invalid stored vector id in index {name}: {e}")))?;
            let id = u64::try_from(raw_id).map_err(|_| {
                backend(format!(
                    "invalid stored vector id in index {name}: {raw_id}"
                ))
            })?;
            let bytes: Vec<u8> = row.get(1).map_err(|e| {
                backend(format!(
                    "corrupt stored vector in index {name}, id {id}: {e}"
                ))
            })?;
            if bytes.len()
                != def
                    .dim
                    .checked_mul(4)
                    .ok_or_else(|| backend("dimension overflow"))?
            {
                return Err(backend(format!(
                    "corrupt stored vector in index {name}, id {id}: wrong byte length"
                )));
            }
            let vector: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                .collect();
            if vector.iter().any(|x| !x.is_finite()) {
                return Err(backend(format!(
                    "non-finite stored vector in index {name}, id {id}"
                )));
            }
            batch.push((id, vector));
            if batch.len() == 1000 {
                let items: Vec<_> = batch.iter().map(|(id, v)| (*id, v.as_slice())).collect();
                h.insert_many(&items);
                batch.clear();
            }
        }
        let items: Vec<_> = batch.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        h.insert_many(&items);
        Ok(h)
    }
    fn load_indexes(&mut self) -> Result<()> {
        // A legacy store opened read-only has no vector_state; its snapshots are rebuilt.
        let sql = if has_table(&self.db, "vector_state")? {
            "SELECT d.name,d.dim,d.metric,d.m,d.ef,d.revision,s.nonce FROM vector_defs d LEFT JOIN vector_state s ON s.name=d.name"
        } else {
            "SELECT name,dim,metric,m,ef,revision,NULL FROM vector_defs"
        };
        let mut stmt = self.db.prepare(sql).map_err(backend)?;
        let defs = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, usize>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, usize>(3)?,
                    r.get::<_, usize>(4)?,
                    r.get::<_, u64>(5)?,
                    r.get::<_, Option<i64>>(6)?,
                ))
            })
            .map_err(backend)?;
        for row in defs {
            let (name, dim, metric, m, ef, revision, nonce) = row.map_err(backend)?;
            check_name(&name)?;
            if self.lock.is_some() {
                remove_snapshot_temps(&self.path, &name)?;
            }
            if dim == 0 || !matches!(metric, 0 | 1) {
                return Err(backend("invalid stored vector definition"));
            }
            let metric = if metric == 0 {
                Metric::Cosine
            } else {
                Metric::L2
            };
            let def = VectorIndexDef {
                name: name.clone(),
                dim,
                metric,
                m,
                ef_construction: ef,
            };
            let expected = self
                .store_id
                .zip(nonce)
                .map(|(id, nonce)| (id, revision, nonce));
            let snapshot = expected
                .and_then(|want| {
                    load_snapshot(&self.path.join(format!("{name}.hnsw")))
                        .filter(|(header, _)| *header == want)
                })
                .map(|(_, h)| h)
                .filter(|h| h.revision == revision && h.valid(dim, metric, m, ef));
            let index = if let Some(h) = snapshot {
                h
            } else {
                let mut h = Self::rebuild_index(&self.db, &name, &def)?;
                h.revision = revision;
                self.rebuilds += 1;
                h
            };
            self.definitions.insert(name.clone(), def);
            self.states
                .insert(name.clone(), (revision, nonce.unwrap_or(0)));
            self.indexes.insert(name, index);
        }
        Ok(())
    }
    fn load_text_indexes(
        db: &Connection,
        tables: &HashMap<String, TableDef>,
    ) -> Result<HashMap<String, TextIndexDef>> {
        let mut indexes = HashMap::new();
        if !has_table(db, "text_defs")? {
            return Ok(indexes);
        }
        let mut stmt = db
            .prepare("SELECT name,tbl,columns FROM text_defs ORDER BY name")
            .map_err(backend)?;
        let defs = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(backend)?;
        for row in defs {
            let (name, table, columns) = row.map_err(backend)?;
            let def = TextIndexDef {
                name,
                table,
                columns: columns.split(',').map(str::to_owned).collect(),
            };
            check_name(&def.name)?;
            let table = tables
                .get(&def.table)
                .ok_or_else(|| backend(format!("text index {} has no table", def.name)))?;
            text_positions(table, &def.columns)
                .map_err(|e| backend(format!("invalid stored text index {}: {e}", def.name)))?;
            indexes.insert(def.name.clone(), def);
        }
        Ok(indexes)
    }
    /// FTS5 table and indexed column positions of every text index on `table`.
    fn text_targets(&self, table: &str) -> Result<Vec<(String, Vec<usize>)>> {
        let def = self.table(table)?;
        let mut targets = Vec::new();
        for index in self.text_indexes.values().filter(|i| i.table == table) {
            targets.push((
                fts::table_name(&index.name),
                text_positions(def, &index.columns)?,
            ));
        }
        Ok(targets)
    }
    fn index_text(tx: &Connection, fts: &str, positions: &[usize], rows: &[Row]) -> Result<()> {
        let mut delete = tx
            .prepare_cached(&format!("DELETE FROM \"{fts}\" WHERE rowid=?"))
            .map_err(backend)?;
        let mut insert = tx
            .prepare_cached(&fts_insert(fts, positions.len()))
            .map_err(backend)?;
        for row in rows {
            delete.execute([row.id as i64]).map_err(backend)?;
            insert_text(
                &mut insert,
                row.id,
                positions.iter().map(|&p| &row.values[p]),
            )?;
        }
        Ok(())
    }
    /// Defines a full-text index over `Text` columns of an existing table. Idempotent for an
    /// identical definition; a different definition under the same name is `Error::Schema`.
    /// Rows already in the table are indexed in the same transaction that records the index,
    /// and later `upsert_rows` and `delete_rows` on the table update it in their own
    /// transactions. Indexed text is folded with [`fts::fold`]; stored rows are unchanged.
    pub fn define_text_index(&mut self, def: &TextIndexDef) -> Result<()> {
        self.writable()?;
        check_name(&def.name)?;
        let table = self.table(&def.table)?;
        let positions = text_positions(table, &def.columns)?;
        if let Some(old) = self.text_indexes.get(&def.name) {
            return if old == def {
                Ok(())
            } else {
                Err(schema("text index redefinition"))
            };
        }
        let fts = fts::table_name(&def.name);
        let columns: Vec<String> = positions
            .iter()
            .map(|&p| format!("\"c_{}\"", table.columns[p].name))
            .collect();
        let select = format!(
            "SELECT id,{} FROM \"t_{}\" ORDER BY id",
            columns.join(","),
            def.table
        );
        let fts_columns: Vec<String> = (0..positions.len()).map(|i| format!("c{i}")).collect();
        let create = format!(
            "CREATE VIRTUAL TABLE \"{fts}\" USING fts5({}, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2')",
            fts_columns.join(",")
        );
        let insert = fts_insert(&fts, positions.len());
        let tx = self.db.transaction().map_err(backend)?;
        tx.execute_batch(&create).map_err(backend)?;
        tx.execute(
            "INSERT INTO text_defs VALUES(?,?,?)",
            params![def.name, def.table, def.columns.join(",")],
        )
        .map_err(backend)?;
        {
            let mut read = tx.prepare(&select).map_err(backend)?;
            let mut insert = tx.prepare(&insert).map_err(backend)?;
            let mut rows = read.query([]).map_err(backend)?;
            let mut values = Vec::with_capacity(positions.len());
            while let Some(row) = rows.next().map_err(backend)? {
                let id: u64 = row.get(0).map_err(backend)?;
                values.clear();
                for i in 1..=positions.len() {
                    values.push(match row.get::<_, Option<String>>(i).map_err(backend)? {
                        Some(text) => Value::Text(text),
                        None => Value::Null,
                    });
                }
                insert_text(&mut insert, id, values.iter())?;
            }
        }
        tx.commit().map_err(backend)?;
        self.text_indexes.insert(def.name.clone(), def.clone());
        self.finish_write()
    }
    /// Text index definitions known to this handle, sorted by name.
    pub fn text_indexes(&self) -> Vec<TextIndexDef> {
        let mut defs: Vec<_> = self.text_indexes.values().cloned().collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }
    /// Full-text search with BM25 ranking. The query is folded like indexed text and split
    /// into terms on every non-alphanumeric character; each term is quoted, so the query is
    /// never read as FTS5 syntax, and a row matches when it contains every term (the first
    /// [`fts::MAX_QUERY_TERMS`] distinct terms are used). Returns up to `k` `(id, score)`
    /// pairs, best first, where the score is the negated FTS5 `bm25()` value (higher is
    /// better); ties are broken by ascending id. A query without terms returns no hits.
    pub fn text_search(&self, index: &str, query: &str, k: usize) -> Result<Vec<(NodeId, f64)>> {
        self.text_search_mode(index, query, k, false)
    }

    /// Like [`NativeStore::text_search`], but a row matches when it contains **any** term of
    /// the query; rows containing more (and rarer) terms rank higher under BM25. Suited to
    /// natural-language questions, where few rows contain every word.
    pub fn text_search_any(
        &self,
        index: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f64)>> {
        self.text_search_mode(index, query, k, true)
    }

    fn text_search_mode(
        &self,
        index: &str,
        query: &str,
        k: usize,
        any: bool,
    ) -> Result<Vec<(NodeId, f64)>> {
        let def = self
            .text_indexes
            .get(index)
            .ok_or_else(|| Error::NotFound(index.into()))?;
        let Some(expression) = fts::match_terms(query, any) else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }
        let fts = fts::table_name(&def.name);
        let mut stmt = self
            .db
            .prepare_cached(&format!(
                "SELECT rowid,bm25(\"{fts}\") AS score FROM \"{fts}\" WHERE \"{fts}\" MATCH ? ORDER BY score,rowid LIMIT CAST(? AS INTEGER)"
            ))
            .map_err(backend)?;
        let rows = stmt
            .query_map(params![expression, k.min(i64::MAX as usize) as i64], |r| {
                Ok((r.get::<_, u64>(0)?, -r.get::<_, f64>(1)?))
            })
            .map_err(backend)?;
        rows.collect::<rusqlite::Result<_>>().map_err(backend)
    }
    /// Hybrid text and vector search fused with Reciprocal Rank Fusion: an id scores
    /// `1 / (RRF_K + rank)` for each ranking it appears in, with [`fts::RRF_K`] `= 60` and
    /// 1-based ranks. Each ranking contributes its top
    /// `max(k, min(ef_search, fts::MAX_FUSION_DEPTH))` candidates, from
    /// [`NativeStore::text_search_any`] (a row matches on any query term) and from `knn` asked
    /// for that many results. Returns up
    /// to `k` `(id, fused score)` pairs, best first, ties broken by ascending id. Rows and
    /// vectors are joined by id. A query text without terms ranks by vector only.
    pub fn hybrid_search(
        &self,
        text_index: &str,
        vector_index: &str,
        query_text: &str,
        query_vector: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(NodeId, f64)>> {
        if !self.text_indexes.contains_key(text_index) {
            return Err(Error::NotFound(text_index.into()));
        }
        self.validate_query(vector_index, query_vector)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let depth = k.max(ef_search.min(fts::MAX_FUSION_DEPTH));
        let text = self.text_search_any(text_index, query_text, depth)?;
        let vector = self.knn(vector_index, query_vector, depth, ef_search.max(depth))?;
        Ok(fts::fuse(&text, &vector, k))
    }
    /// Breadth-first traversal over neighbor IDs from SQLite's covering indexes.
    pub fn k_hop_bfs(
        &self,
        start: NodeId,
        rel: Option<&str>,
        dir: Direction,
        max_depth: u32,
        limit: usize,
    ) -> Result<Vec<Reached>> {
        check_id(start)?;
        let relation = if rel.is_some() { " AND rel=?2" } else { "" };
        let sql = match dir {
            Direction::Out => format!("SELECT dst FROM edges WHERE src=?1{relation}"),
            Direction::In => format!("SELECT src FROM edges WHERE dst=?1{relation}"),
            // Duplicates, including self-loops, are removed by the visited set.
            Direction::Both => format!("SELECT dst FROM edges WHERE src=?1{relation} UNION ALL SELECT src FROM edges WHERE dst=?1{relation}"),
            _ => return Err(schema("unsupported direction")),
        };
        if max_depth == 0 || limit == 0 {
            return Ok(Vec::new());
        }
        let mut stmt = self.db.prepare_cached(&sql).map_err(backend)?;
        let mut seen = HashSet::from([start]);
        let mut frontier = vec![start];
        let mut reached = Vec::new();
        for depth in 1..=max_depth {
            if frontier.is_empty() || reached.len() >= limit {
                break;
            }
            let mut next = Vec::new();
            let remaining = limit - reached.len();
            let mut smallest: Option<std::collections::BinaryHeap<NodeId>> = None;
            // A full depth fills the answer, so no deeper frontier can contribute.
            // Until then every new node fits and must be kept for the next depth.
            // Once full, retain only its smallest remaining ids, evicting no node
            // from an earlier depth. Discarded ids cannot beat a decreasing maximum.
            for node in frontier {
                let mut rows = if let Some(rel) = rel {
                    stmt.query(params![node as i64, rel])
                } else {
                    stmt.query([node as i64])
                }
                .map_err(backend)?;
                while let Some(row) = rows.next().map_err(backend)? {
                    let id: u64 = row.get(0).map_err(backend)?;
                    if next.len() < remaining && smallest.is_none() {
                        if seen.insert(id) {
                            next.push(id);
                        }
                    } else if !seen.contains(&id) {
                        let heap = smallest.get_or_insert_with(|| {
                            std::collections::BinaryHeap::from(std::mem::take(&mut next))
                        });
                        if let Some(mut largest) = heap.peek_mut() {
                            if id < *largest {
                                seen.remove(&*largest);
                                seen.insert(id);
                                *largest = id;
                            }
                        }
                    }
                }
            }
            if let Some(heap) = smallest {
                next = heap.into_vec();
            }
            next.sort_unstable();
            reached.extend(
                next.iter()
                    .take(limit - reached.len())
                    .map(|&id| Reached { id, depth }),
            );
            frontier = next;
        }
        Ok(reached)
    }
    /// Recursive SQL alternative retained for reproducible traversal comparisons.
    pub fn k_hop_sql(
        &self,
        start: NodeId,
        rel: Option<&str>,
        dir: Direction,
        max_depth: u32,
        limit: usize,
    ) -> Result<Vec<Reached>> {
        check_id(start)?;
        let (join, next) = match dir {
            Direction::Out => ("e.src=w.id", "e.dst"),
            Direction::In => ("e.dst=w.id", "e.src"),
            Direction::Both => (
                "(e.src=w.id OR e.dst=w.id)",
                "CASE WHEN e.src=w.id THEN e.dst ELSE e.src END",
            ),
            _ => return Err(schema("unsupported direction")),
        };
        // A bare LIMIT binding makes bundled SQLite reprepare on every call.
        let sql = format!("WITH RECURSIVE walk(id,depth) AS (SELECT ?1,0 UNION SELECT {next},w.depth+1 FROM walk w JOIN edges e ON {join} WHERE w.depth<?2 AND (?3 IS NULL OR e.rel=?3)) SELECT id,min(depth) AS d FROM walk WHERE id<>?1 GROUP BY id ORDER BY d,id LIMIT CAST(?4 AS INTEGER)");
        let mut stmt = self.db.prepare_cached(&sql).map_err(backend)?;
        let rows = stmt
            .query_map(
                params![
                    start as i64,
                    max_depth,
                    rel,
                    limit.min(i64::MAX as usize) as i64
                ],
                |r| {
                    Ok(Reached {
                        id: r.get::<_, u64>(0)?,
                        depth: r.get(1)?,
                    })
                },
            )
            .map_err(backend)?;
        rows.collect::<rusqlite::Result<_>>().map_err(backend)
    }
}

impl Store for NativeStore {
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self> {
        if !opts.create && !path.join("quickdb.sqlite").is_file() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "store missing",
            )));
        }
        if opts.create {
            fs::create_dir_all(path)?;
        }
        // The lock comes first: nothing below may touch a directory another writer owns.
        let lock = take_lock(path)?;
        let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
            | if opts.create {
                rusqlite::OpenFlags::SQLITE_OPEN_CREATE
            } else {
                rusqlite::OpenFlags::empty()
            };
        let db =
            Connection::open_with_flags(path.join("quickdb.sqlite"), flags).map_err(backend)?;
        Self::finish_open(db, path, opts.sync_each_write, Some(lock))
    }
    fn backend_name(&self) -> &'static str {
        "native"
    }
    fn define_table(&mut self, def: &TableDef) -> Result<()> {
        self.writable()?;
        check_name(&def.name)?;
        let mut names = HashSet::new();
        for c in &def.columns {
            check_name(&c.name)?;
            if c.name == "id" || !names.insert(&c.name) {
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
        for c in &def.columns {
            sql.push_str(&format!(
                ",\"c_{}\" {}",
                c.name,
                match c.ty {
                    ColumnType::Int => "INTEGER",
                    ColumnType::Float => "REAL",
                    ColumnType::Text => "TEXT COLLATE BINARY",
                    ColumnType::Bytes => "BLOB",
                    _ => return Err(schema("unsupported column type")),
                }
            ));
        }
        sql.push(')');
        let tx = self.db.transaction().map_err(backend)?;
        tx.execute_batch(&sql).map_err(backend)?;
        tx.execute("INSERT INTO table_defs VALUES(?)", [&def.name])
            .map_err(backend)?;
        for (pos, c) in def.columns.iter().enumerate() {
            tx.execute(
                "INSERT INTO column_defs VALUES(?,?,?,?,?)",
                params![def.name, pos as i64, c.name, ty_code(c.ty), c.indexed],
            )
            .map_err(backend)?;
            if c.indexed {
                tx.execute_batch(&format!(
                    "CREATE INDEX \"idx_{}_{}_{}\" ON \"t_{}\"(\"c_{}\")",
                    def.name.len(),
                    def.name,
                    c.name,
                    def.name,
                    c.name
                ))
                .map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        self.tables.insert(def.name.clone(), def.clone());
        self.finish_write()
    }
    fn upsert_rows(&mut self, table: &str, rows: &[Row]) -> Result<()> {
        self.writable()?;
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
        let text = self.text_targets(table)?;
        let tx = self.db.transaction().map_err(backend)?;
        {
            let mut stmt = tx.prepare_cached(&sql).map_err(backend)?;
            for row in rows {
                let values = std::iter::once(rusqlite::types::Value::Integer(row.id as i64))
                    .chain(row.values.iter().map(sql_value));
                stmt.execute(params_from_iter(values)).map_err(backend)?;
            }
            // Text indexes change in the same transaction as the rows they mirror.
            for (fts, positions) in &text {
                Self::index_text(&tx, fts, positions, rows)?;
            }
        }
        tx.commit().map_err(backend)?;
        self.finish_write()
    }
    fn get_row(&self, table: &str, id: NodeId) -> Result<Option<Row>> {
        let def = self.table(table)?;
        check_id(id)?;
        let mut stmt = self
            .db
            .prepare_cached(&format!(
                "SELECT {} FROM \"t_{table}\" WHERE id=?",
                projection(def)
            ))
            .map_err(backend)?;
        stmt.query_row([id as i64], |r| read_row(r, def))
            .optional()
            .map_err(backend)
    }
    fn select(&self, table: &str, filter: &Filter, limit: usize) -> Result<Vec<Row>> {
        let def = self.table(table)?;
        let (condition, mut values) = self.filter(def, filter)?;
        values.push((limit.min(i64::MAX as usize) as i64).into());
        // A bare LIMIT parameter triggers a reprepare on every execution in
        // bundled SQLite, even for cached statements. The cast keeps the same
        // integer bound without making the plan depend on its current value.
        let sql = if matches!(filter, Filter::IntRange { column, .. } | Filter::TextPrefix { column, .. }
            if def.columns.iter().any(|c| c.name == *column && c.indexed))
        {
            // Range order differs from id order. Sort covering-index IDs first,
            // not potentially thousands of full text/blob payloads before LIMIT.
            format!(
                "SELECT {} FROM \"t_{table}\" WHERE id IN (SELECT id FROM \"t_{table}\" WHERE {condition} ORDER BY id LIMIT CAST(? AS INTEGER)) ORDER BY id",
                projection(def)
            )
        } else {
            format!(
                "SELECT {} FROM \"t_{table}\" WHERE {condition} ORDER BY id LIMIT CAST(? AS INTEGER)",
                projection(def)
            )
        };
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
        self.writable()?;
        self.table(table)?;
        for &id in ids {
            check_id(id)?;
        }
        let text = self.text_targets(table)?;
        let tx = self.db.transaction().map_err(backend)?;
        let mut count = 0;
        {
            let mut stmt = tx
                .prepare_cached(&format!("DELETE FROM \"t_{table}\" WHERE id=?"))
                .map_err(backend)?;
            for id in ids {
                count += stmt.execute([*id as i64]).map_err(backend)? as u64;
            }
            for (fts, _) in &text {
                let mut stmt = tx
                    .prepare_cached(&format!("DELETE FROM \"{fts}\" WHERE rowid=?"))
                    .map_err(backend)?;
                for id in ids {
                    stmt.execute([*id as i64]).map_err(backend)?;
                }
            }
        }
        tx.commit().map_err(backend)?;
        self.finish_write()?;
        Ok(count)
    }
    fn define_vector_index(&mut self, def: &VectorIndexDef) -> Result<()> {
        self.writable()?;
        check_name(&def.name)?;
        if def.dim == 0
            || def.dim > i32::MAX as usize / 4
            || def.m == 0
            || def.m > 4096
            || def.ef_construction == 0
            || def.ef_construction > i32::MAX as usize
        {
            return Err(schema("invalid vector index parameters"));
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
        let nonce: i64 = {
            let tx = self.db.transaction().map_err(backend)?;
            tx.execute(
                "INSERT INTO vector_defs VALUES(?,?,?,?,?,0)",
                params![
                    def.name,
                    def.dim as i64,
                    metric_code,
                    def.m as i64,
                    def.ef_construction as i64
                ],
            )
            .map_err(backend)?;
            let nonce = tx
                .query_row(
                    "INSERT OR REPLACE INTO vector_state VALUES(?,random()) RETURNING nonce",
                    [&def.name],
                    |r| r.get(0),
                )
                .map_err(backend)?;
            tx.commit().map_err(backend)?;
            nonce
        };
        self.states.insert(def.name.clone(), (0, nonce));
        self.indexes.insert(
            def.name.clone(),
            Hnsw::new(def.dim, def.metric, def.m, def.ef_construction),
        );
        self.definitions.insert(def.name.clone(), def.clone());
        self.finish_write()
    }
    fn upsert_vectors(&mut self, index: &str, items: &[(NodeId, &[f32])]) -> Result<()> {
        self.writable()?;
        self.index(index)?;
        for &(id, v) in items {
            check_id(id)?;
            self.validate_query(index, v)?;
        }
        if items.is_empty() {
            return Ok(());
        }
        let tx = self.db.transaction().map_err(backend)?;
        {
            let mut stmt = tx
                .prepare_cached("INSERT OR REPLACE INTO vectors VALUES(?,?,?)")
                .map_err(backend)?;
            for &(id, v) in items {
                let bytes: Vec<_> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                stmt.execute(params![index, id as i64, bytes])
                    .map_err(backend)?;
            }
        }
        // Revision and nonce come from SQLite inside this transaction, not from memory,
        // so no other handle's stale counter can stamp a snapshot that looks current.
        let revision: u64 = tx
            .query_row(
                "UPDATE vector_defs SET revision=revision+1 WHERE name=? RETURNING revision",
                [index],
                |r| r.get(0),
            )
            .map_err(backend)?;
        let nonce: i64 = tx
            .query_row(
                "INSERT OR REPLACE INTO vector_state VALUES(?,random()) RETURNING nonce",
                [index],
                |r| r.get(0),
            )
            .map_err(backend)?;
        tx.commit().map_err(backend)?;
        self.update_index_after_commit(index, (revision, nonce), |h| h.insert_many(items))?;
        self.finish_write()
    }
    fn knn(&self, index: &str, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<Hit>> {
        Ok(self
            .validate_query(index, query)?
            .search(query, k, ef_search))
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
        let h = self.validate_query(index, query)?;
        let def = self.table(table)?;
        let (condition, values) = self.filter(def, filter)?;
        if k == 0 || h.len() == 0 {
            return Ok(Vec::new());
        }
        const PROBE_IDS: usize = 128;
        let ef = ef_search.max(k);
        let cap = h.capacity();
        {
            // One statement answers a whole expansion: PROBE_IDS primary-key seeks with
            // the filter, padded with -1, which is never a stored id.
            let slots = vec!["?"; PROBE_IDS].join(",");
            let mut probe = self
                .db
                .prepare_cached(&format!(
                "SELECT id FROM \"t_{table}\" NOT INDEXED WHERE id IN ({slots}) AND ({condition})"
            ))
                .map_err(backend)?;
            let failure = std::cell::RefCell::new(None);
            // Filter values are bound once; SQLite keeps bindings across resets, so
            // each lookup rebinds only the ids.
            for (i, v) in values.iter().enumerate() {
                probe
                    .raw_bind_parameter(i + PROBE_IDS + 1, v)
                    .map_err(backend)?;
            }
            let mut matches = |nodes: &[usize], out: &mut [bool]| {
                for (chunk, flags) in nodes.chunks(PROBE_IDS).zip(out.chunks_mut(PROBE_IDS)) {
                    let mut run =
                        |probe: &mut rusqlite::CachedStatement<'_>| -> rusqlite::Result<()> {
                            for slot in 0..PROBE_IDS {
                                let id = chunk.get(slot).map_or(-1, |&n| h.id_at(n) as i64);
                                probe.raw_bind_parameter(slot + 1, id)?;
                            }
                            let mut rows = probe.raw_query();
                            while let Some(row) = rows.next()? {
                                let id: u64 = row.get(0)?;
                                if let Some(i) = chunk.iter().position(|&n| h.id_at(n) == id) {
                                    flags[i] = true;
                                }
                            }
                            Ok(())
                        };
                    if let Err(e) = run(&mut probe) {
                        failure.borrow_mut().get_or_insert(backend(e));
                    }
                }
            };
            // Selectivity estimate from lookups on evenly spread live positions.
            let samples = 256.min(cap);
            let spread: Vec<usize> = (0..samples)
                .map(|i| i * cap / samples)
                .filter(|&n| h.node_of(h.id_at(n)) == Some(n))
                .collect();
            let mut seen = vec![false; spread.len()];
            if !spread.is_empty() {
                matches(&spread, &mut seen);
            }
            let hits_in_sample = seen.iter().filter(|&&b| b).count();
            if let Some(e) = failure.borrow_mut().take() {
                return Err(e);
            }
            let estimate = if spread.is_empty() {
                0.0
            } else {
                hits_in_sample as f64 / spread.len() as f64
            };
            // Lazy predicate: about ef/s lookups at ~1.5 us each, cheaper than
            // fetching s*N ids at ~0.1 us each once s exceeds sqrt(15*ef/N).
            let lazy_from = (15.0 * ef as f64 / h.len() as f64).sqrt().max(0.02);
            if !spread.is_empty() && estimate >= lazy_from {
                // Measured at 1M x 128, ef 64: the walk also scores non-matching nodes,
                // so fewer accepted candidates keep recall >= 0.96 (0.6 ef) and, for
                // sparse matches, >= 0.99 (0.4 ef). Eight expansions per lookup batch
                // were fastest among 1, 2, 4, 8 and 16.
                let width = 8;
                let share: u128 = if estimate < 0.1 { 40 } else { 60 };
                let mut e = ((ef as u128 * share / 100) as usize).max(k);
                loop {
                    let (hits, _) = h.search_filtered(query, k, e, usize::MAX, width, &mut matches);
                    if let Some(err) = failure.borrow_mut().take() {
                        return Err(err);
                    }
                    if hits.len() >= k || e >= cap {
                        if hits.len() >= k {
                            return Ok(hits);
                        }
                        break;
                    }
                    e = e.saturating_mul(4).min(cap);
                }
            }
        }
        let mut stmt = self
            .db
            .prepare_cached(&format!("SELECT id FROM \"t_{table}\" WHERE {condition}"))
            .map_err(backend)?;
        let mut rows = stmt
            .query(params_from_iter(values.iter()))
            .map_err(backend)?;
        // Exact costs |allowed| distances, graph search about ef/s = ef*N/|allowed|
        // plus graph overhead; they meet near |allowed| = 2*sqrt(ef*N).
        let exact_up_to = (2.0 * (ef as f64 * h.len() as f64).sqrt()).max(1024.0) as usize;
        let mut nodes = Vec::with_capacity(exact_up_to.min(h.len()));
        let mut mask = vec![0u64; cap.div_ceil(64)];
        let mut allowed = 0usize;
        while let Some(row) = rows.next().map_err(backend)? {
            if let Some(n) = h.node_of(row.get::<_, u64>(0).map_err(backend)?) {
                mask[n / 64] |= 1 << (n % 64);
                allowed += 1;
                if allowed <= exact_up_to {
                    nodes.push(n);
                } else if allowed == exact_up_to + 1 {
                    // The decision can no longer be exact. Release, rather than clear,
                    // the prefix so the graph walk retains only the mask.
                    nodes = Vec::new();
                }
            }
        }
        if allowed <= exact_up_to {
            return Ok(h.exact_nodes(query, k, &nodes));
        }
        let mut e = ef;
        loop {
            let (hits, _) = h.search_filtered(query, k, e, usize::MAX, 1, |nodes, out| {
                for (&n, flag) in nodes.iter().zip(out) {
                    *flag = (mask[n / 64] >> (n % 64)) & 1 == 1;
                }
            });
            if hits.len() >= k.min(allowed) {
                return Ok(hits);
            }
            if e >= cap {
                // Preserve the exact fallback without rebuilding the unbounded id list.
                // The global top k is contained in the union of each batch's top k.
                let mut hits = Vec::new();
                nodes.reserve_exact(exact_up_to);
                for (word, &bits) in mask.iter().enumerate() {
                    let mut bits = bits;
                    while bits != 0 {
                        nodes.push(word * 64 + bits.trailing_zeros() as usize);
                        bits &= bits - 1;
                        if nodes.len() == exact_up_to {
                            hits.extend(h.exact_nodes(query, k, &nodes));
                            hnsw::sort_hits(&mut hits, k);
                            nodes.clear();
                        }
                    }
                }
                hits.extend(h.exact_nodes(query, k, &nodes));
                hnsw::sort_hits(&mut hits, k);
                return Ok(hits);
            }
            e = e.saturating_mul(4).min(cap);
        }
    }
    fn count_vectors(&self, index: &str) -> Result<u64> {
        Ok(self.index(index)?.len() as u64)
    }
    fn upsert_edges(&mut self, edges: &[Edge]) -> Result<()> {
        self.writable()?;
        for e in edges {
            check_id(e.src)?;
            check_id(e.dst)?;
            if !e.weight.is_finite() {
                return Err(schema("non-finite edge weight"));
            }
        }
        let tx = self.db.transaction().map_err(backend)?;
        {
            let mut stmt = tx
                .prepare_cached("INSERT OR REPLACE INTO edges VALUES(?,?,?,?)")
                .map_err(backend)?;
            for e in edges {
                stmt.execute(params![e.src as i64, e.rel, e.dst as i64, e.weight])
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
        // As in select, do not let LIMIT bindings invalidate the cached plan.
        let sql = format!("SELECT src,rel,dst,weight FROM edges WHERE {condition}{} ORDER BY rel,src,dst LIMIT CAST(?2 AS INTEGER)", if rel.is_some() { " AND rel=?3" } else { "" });
        let mut stmt = self.db.prepare_cached(&sql).map_err(backend)?;
        let mut values = vec![
            rusqlite::types::Value::Integer(node as i64),
            (limit.min(i64::MAX as usize) as i64).into(),
        ];
        if let Some(rel) = rel {
            values.push(rel.to_string().into());
        }
        let rows = stmt
            .query_map(params_from_iter(values), |r| {
                Ok(Edge {
                    src: r.get::<_, u64>(0)?,
                    rel: r.get(1)?,
                    dst: r.get::<_, u64>(2)?,
                    weight: r.get(3)?,
                })
            })
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
        // Covering-ID BFS won depths 1..=3 on both the corpus-like and million-node
        // probes, without the recursive SQL's per-depth duplicate state.
        self.k_hop_bfs(start, rel, dir, max_depth, limit)
    }
    fn count_edges(&self) -> Result<u64> {
        self.db
            .query_row("SELECT count(*) FROM edges", [], |r| r.get(0))
            .map_err(backend)
    }
    fn flush(&mut self) -> Result<()> {
        self.writable()?;
        let store_id = self
            .store_id
            .ok_or_else(|| backend("store id missing from quickdb.sqlite"))?;
        if !self.stale_indexes.is_empty() {
            return Err(backend(
                "cannot flush stale vector indexes; reopen the store to rebuild",
            ));
        }
        for (name, h) in &mut self.indexes {
            if h.needs_compaction() {
                let mut compact = Self::rebuild_index(&self.db, name, &self.definitions[name])?;
                compact.revision = h.revision;
                *h = compact;
            }
        }
        // A per-process temp name: even a second writer that bypassed the lock
        // cannot rename this file away or publish a half-written one.
        let pid = std::process::id();
        for (name, h) in &self.indexes {
            let (revision, nonce) = self.states[name];
            let header: SnapshotHeader = (store_id, revision, nonce);
            let temp = self.path.join(format!("{name}.hnsw.tmp.{pid}"));
            let result = (|| -> Result<()> {
                let checked = ChecksumWriter {
                    inner: File::create(&temp)?,
                    hash: crc32fast::Hasher::new(),
                };
                let mut out = BufWriter::new(checked);
                snapshot::encode_header(&mut out, header).map_err(backend)?;
                h.encode_snapshot(&mut out).map_err(backend)?;
                out.flush()?;
                let checked = out.into_inner().map_err(|e| Error::Io(e.into_error()))?;
                let checksum = checked.hash.finalize();
                let mut file = checked.inner;
                file.write_all(&checksum.to_le_bytes())?;
                file.sync_all()?;
                fs::rename(&temp, self.path.join(format!("{name}.hnsw")))?;
                Ok(())
            })();
            if result.is_err() {
                let _ = remove_snapshot_temp(&temp);
            }
            result?;
        }
        // NORMAL bulk writes become durable at the explicit FULL checkpoint, which
        // fsyncs the database file itself before it truncates the WAL.
        // Never open and close another descriptor on quickdb.sqlite here: POSIX
        // close() drops every fcntl lock this process holds on that inode,
        // including SQLite's WAL locks, and readers then see a torn database.
        self.db
            .execute_batch("PRAGMA synchronous=FULL;")
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
                .execute_batch("PRAGMA synchronous=NORMAL;")
                .map_err(backend)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "store_repair_tests.rs"]
mod store_repair_tests;

#[cfg(test)]
mod sql_tests {
    use super::*;

    #[test]
    fn wal_refusal_is_an_error() {
        let db = Connection::open_in_memory().unwrap();
        let error = enter_wal(&db).unwrap_err();
        assert!(matches!(error, Error::Backend(_)));
        assert!(error.to_string().contains("memory"));
        println!("SQL_HARDENING item=13 refused_wal_is_error=true");
    }

    #[test]
    fn prefix_byte_ranges_match_old_predicate_for_2000_prefixes() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE words(id INTEGER PRIMARY KEY,c TEXT COLLATE BINARY); CREATE INDEX words_c ON words(c)").unwrap();
        let alphabet = [
            'a',
            'b',
            'z',
            'A',
            '%',
            '_',
            '\0',
            '\u{7f}',
            '\u{7ff}',
            '\u{ffff}',
            '\u{10ffff}',
            'ك',
            'ت',
        ];
        let mut rng = quickdb_api::conformance::Lcg(8127);
        let mut words = vec![
            String::new(),
            "ك".into(),
            "a\u{7f}".into(),
            "a\u{10ffff}".into(),
        ];
        for _ in 0..1000 {
            let n = 2 + rng.next_u32() as usize % 11;
            words.push(
                (0..n)
                    .map(|_| alphabet[rng.next_u32() as usize % alphabet.len()])
                    .collect(),
            );
        }
        for (id, word) in words.iter().enumerate() {
            db.execute("INSERT INTO words VALUES(?,?)", params![id as i64, word])
                .unwrap();
        }
        db.execute("INSERT INTO words VALUES(?,NULL)", [words.len() as i64])
            .unwrap();
        let mut prefixes = vec![
            vec![],
            vec![0xff],
            vec![0xff, 0xff],
            vec![0x7f],
            vec![0xd9],
            vec![b'a', 0xff],
            vec![0xf4, 0x8f, 0xbf, 0xbf],
            b"no match".to_vec(),
        ];
        while prefixes.len() < 2000 {
            let w = &words[rng.next_u32() as usize % words.len()];
            let n = rng.next_u32() as usize % (w.len() + 1);
            prefixes.push(w.as_bytes()[..n].to_vec());
        }
        for prefix in &prefixes {
            for limit in [0, 1, 17, 2000] {
                let mut old = db.prepare_cached("SELECT id FROM words WHERE CASE WHEN ?1=0 THEN c IS NOT NULL ELSE substr(CAST(c AS BLOB),1,?1)=?2 END ORDER BY id LIMIT ?3").unwrap();
                let expected: Vec<i64> = old
                    .query_map(params![prefix.len() as i64, prefix, limit], |r| r.get(0))
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();
                let (condition, mut values) = prefix_range("c", prefix);
                values.push(limit.into());
                let mut new = db.prepare_cached(&format!("SELECT id FROM words WHERE id IN (SELECT id FROM words WHERE {condition} ORDER BY id LIMIT CAST(? AS INTEGER)) ORDER BY id")).unwrap();
                let actual: Vec<i64> = new
                    .query_map(params_from_iter(values), |r| r.get(0))
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();
                assert_eq!(actual, expected, "prefix={prefix:?}, limit={limit}");
            }
        }
        assert_eq!(prefix_upper(&[0x61, 0xff, 0xff]), Some(vec![0x62]));
        assert_eq!(prefix_upper(&[0xff, 0xff]), None);
        assert_eq!(prefix_upper(&[]), None);
        println!("SQL_PREFIX_EQUIVALENCE prefixes=2000 limits=4 raw_utf8_cuts=true high_bytes=true ordered_equal=true");
    }

    #[test]
    fn failed_post_commit_update_is_not_silently_searchable() {
        let root = std::env::var_os("QUICKDB_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let dir = tempfile::tempdir_in(root).unwrap();
        let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        let def = VectorIndexDef {
            name: "emb".into(),
            dim: 2,
            metric: Metric::L2,
            m: 8,
            ef_construction: 32,
        };
        s.define_vector_index(&def).unwrap();
        s.upsert_vectors("emb", &[(1, &[0.0, 0.0])]).unwrap();
        s.flush().unwrap();
        // Simulate the SQL half of upsert succeeding, then inject an unwind in
        // the same guarded update helper used by the production path.
        let tx = s.db.transaction().unwrap();
        tx.execute(
            "UPDATE vectors SET data=? WHERE name='emb' AND id=1",
            [10.0f32.to_le_bytes().repeat(2)],
        )
        .unwrap();
        tx.execute("UPDATE vector_defs SET revision=2 WHERE name='emb'", [])
            .unwrap();
        tx.commit().unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = s.update_index_after_commit("emb", (2, 0), |h| {
                h.insert_many(&[(2, &[1.0, 1.0])]);
                panic!("injected index update failure");
            });
        }));
        assert!(panic.is_err());
        assert!(matches!(
            s.knn("emb", &[0.0, 0.0], 1, 32),
            Err(Error::Backend(_))
        ));
        assert!(matches!(s.count_vectors("emb"), Err(Error::Backend(_))));
        assert!(matches!(s.flush(), Err(Error::Backend(_))));
        drop(s);
        let s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        assert_eq!(s.index_rebuilds(), 1);
        let hits = s.knn("emb", &[10.0, 10.0], 1, 32).unwrap();
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[0].distance, 0.0);
        println!("SQL_POST_COMMIT_PANIC stale_reads_refused=true stale_flush_refused=true reopen_rebuilt=true");
    }
}
