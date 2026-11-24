//! LanceDB (embedded, local directory) behind the QuickDB contract.
//!
//! Layout inside the store directory: one Lance table per row table (`t_<name>`), one per vector
//! index (`v_<name>`, columns `id`, `vector`), one edge table (`edges`), and `quickdb-meta.txt`
//! holding the definitions (written with tmp + rename).
//!
//! Every write is one Lance commit (a new manifest version), so a returned write is durable and
//! each write call is all-or-nothing. `flush` builds missing indexes and runs `optimize(All)`
//! (compaction, incremental index update, prune of versions older than Lance's default 7 days).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::index::vector::{IvfHnswFlatIndexBuilder, IvfHnswSqIndexBuilder};
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::OptimizeAction;
use lancedb::{Connection, DistanceType, Table};
use quickdb_api::{
    valid_name, ColumnDef, ColumnType, Direction, Edge, Error, Filter, Hit, Metric, NodeId, OpenOptions, Reached,
    Result, Row, Store, TableDef, Value, VectorIndexDef,
};

const META: &str = "quickdb-meta.txt";
const EDGES: &str = "edges";
const VCOL: &str = "vector";
/// Worker threads of the store's own Tokio runtime.
pub const RUNTIME_THREADS: usize = 4;

fn be(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}

fn check_id(id: NodeId) -> Result<i64> {
    i64::try_from(id).map_err(|_| Error::Schema(format!("id {id} exceeds i64::MAX")))
}

/// Which HNSW variant LanceDB builds; `QUICKDB_LANCE_INDEX=sq` selects IVF_HNSW_SQ. FLAT is the
/// default: at 100k x 128 it reached recall@10 0.975 at ef 64 where SQ stalled near 0.91, at equal latency.
#[derive(Clone, Copy, Debug, PartialEq)]
enum IndexKind {
    HnswSq,
    HnswFlat,
}

fn index_kind() -> IndexKind {
    match std::env::var("QUICKDB_LANCE_INDEX").as_deref() {
        Ok("sq") => IndexKind::HnswSq,
        _ => IndexKind::HnswFlat,
    }
}

/// Below this many vectors no ANN index is trained; the flat scan is exact and cheap.
fn min_index_rows() -> usize {
    std::env::var("QUICKDB_LANCE_MIN_INDEX").ok().and_then(|s| s.parse().ok()).unwrap_or(1000)
}

fn ty_code(t: ColumnType) -> &'static str {
    match t {
        ColumnType::Int => "i",
        ColumnType::Float => "f",
        ColumnType::Text => "t",
        ColumnType::Bytes => "b",
        // Unreachable: `define_table` rejects an unknown type before this is called.
        _ => "?",
    }
}

fn arrow_ty(t: ColumnType) -> DataType {
    match t {
        ColumnType::Int => DataType::Int64,
        ColumnType::Float => DataType::Float64,
        ColumnType::Text => DataType::Utf8,
        ColumnType::Bytes => DataType::Binary,
        // Unreachable: `define_table` rejects an unknown type before this is called.
        _ => DataType::Null,
    }
}

fn row_schema(def: &TableDef) -> SchemaRef {
    let mut f = vec![Field::new("id", DataType::Int64, false)];
    f.extend(def.columns.iter().map(|c| Field::new(&c.name, arrow_ty(c.ty), true)));
    Arc::new(Schema::new(f))
}

fn vec_schema(dim: usize) -> Result<SchemaRef> {
    let d = i32::try_from(dim).map_err(|_| Error::Schema("dim too large".into()))?;
    Ok(Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(VCOL, DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), d), false),
    ])))
}

fn edge_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("rel", DataType::Utf8, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float32, false),
    ]))
}

fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn id_list(ids: &[i64]) -> String {
    ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
}

fn col<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T> {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| Error::Backend(format!("column {name} missing or of unexpected type")))
}

#[derive(Clone, Debug, PartialEq)]
struct VecMeta {
    def: VectorIndexDef,
    indexed: bool,
}

pub struct LanceStore {
    rt: tokio::runtime::Runtime,
    conn: Connection,
    dir: PathBuf,
    tables: BTreeMap<String, (TableDef, Table)>,
    vectors: BTreeMap<String, (VecMeta, Table)>,
    edges: Table,
}

impl LanceStore {
    fn save_meta(&self) -> Result<()> {
        let mut s = String::new();
        for (def, _) in self.tables.values() {
            s.push_str("T ");
            s.push_str(&def.name);
            for c in &def.columns {
                s.push_str(&format!(" {}:{}:{}", c.name, ty_code(c.ty), u8::from(c.indexed)));
            }
            s.push('\n');
        }
        for (m, _) in self.vectors.values() {
            let d = &m.def;
            let metric = if d.metric == Metric::Cosine { "cos" } else { "l2" };
            s.push_str(&format!("V {} {} {} {} {} {}\n", d.name, d.dim, metric, d.m, d.ef_construction, u8::from(m.indexed)));
        }
        let tmp = self.dir.join(format!("{META}.tmp"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(s.as_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, self.dir.join(META))?;
        Ok(())
    }

    fn parse_meta(text: &str) -> Result<(Vec<TableDef>, Vec<VecMeta>)> {
        let bad = || Error::Backend(format!("corrupt {META}"));
        let (mut t, mut v) = (Vec::new(), Vec::new());
        for line in text.lines() {
            let p: Vec<&str> = line.split(' ').collect();
            match p.first() {
                Some(&"T") if p.len() >= 2 => {
                    let mut columns = Vec::new();
                    for c in &p[2..] {
                        let q: Vec<&str> = c.split(':').collect();
                        if q.len() != 3 {
                            return Err(bad());
                        }
                        let ty = match q[1] {
                            "i" => ColumnType::Int,
                            "f" => ColumnType::Float,
                            "t" => ColumnType::Text,
                            "b" => ColumnType::Bytes,
                            _ => return Err(bad()),
                        };
                        columns.push(ColumnDef { name: q[0].to_string(), ty, indexed: q[2] == "1" });
                    }
                    t.push(TableDef { name: p[1].to_string(), columns });
                }
                Some(&"V") if p.len() == 7 => {
                    let n = |s: &str| s.parse::<usize>().map_err(|_| bad());
                    let metric = if p[3] == "cos" { Metric::Cosine } else { Metric::L2 };
                    let def = VectorIndexDef { name: p[1].to_string(), dim: n(p[2])?, metric, m: n(p[4])?, ef_construction: n(p[5])? };
                    v.push(VecMeta { def, indexed: p[6] == "1" });
                }
                Some(&"") | None => {}
                _ => return Err(bad()),
            }
        }
        Ok((t, v))
    }

    fn table(&self, name: &str) -> Result<&(TableDef, Table)> {
        self.tables.get(name).ok_or_else(|| Error::NotFound(format!("table {name}")))
    }

    fn vindex(&self, name: &str) -> Result<&(VecMeta, Table)> {
        self.vectors.get(name).ok_or_else(|| Error::NotFound(format!("vector index {name}")))
    }

    async fn open_or_create(conn: &Connection, name: &str, schema: SchemaRef) -> Result<Table> {
        match conn.open_table(name).execute().await {
            Ok(t) => Ok(t),
            Err(lancedb::Error::TableNotFound { .. }) => conn.create_empty_table(name, schema).execute().await.map_err(be),
            Err(e) => Err(be(e)),
        }
    }

    async fn collect(q: impl ExecutableQuery) -> Result<Vec<RecordBatch>> {
        q.execute().await.map_err(be)?.try_collect::<Vec<_>>().await.map_err(be)
    }

    async fn merge(t: &Table, on: &[&str], batch: RecordBatch) -> Result<()> {
        let schema = batch.schema();
        let mut b = t.merge_insert(on);
        b.when_matched_update_all(None).when_not_matched_insert_all();
        b.execute(Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema))).await.map_err(be)?;
        Ok(())
    }

    fn predicate(def: &TableDef, f: &Filter) -> Result<Option<String>> {
        let find = |c: &str| {
            def.columns.iter().find(|x| x.name == c).ok_or_else(|| Error::NotFound(format!("column {c}")))
        };
        Ok(match f {
            Filter::All => None,
            Filter::Eq { column, value } => {
                let c = find(column)?;
                let lit = match (c.ty, value) {
                    (_, Value::Null) => return Ok(Some("false".into())),
                    (ColumnType::Int, Value::Int(i)) => i.to_string(),
                    (ColumnType::Float, Value::Float(x)) if x.is_finite() => format!("{x:?}"),
                    (ColumnType::Float, Value::Int(i)) => format!("{i}.0"),
                    (ColumnType::Text, Value::Text(s)) => sql_str(s),
                    (ColumnType::Bytes, Value::Bytes(b)) => {
                        format!("X'{}'", b.iter().map(|x| format!("{x:02x}")).collect::<String>())
                    }
                    _ => return Err(Error::Schema(format!("value does not match column {column}"))),
                };
                Some(format!("{} = {lit}", c.name))
            }
            Filter::IntRange { column, min, max } => {
                let c = find(column)?;
                if c.ty != ColumnType::Int {
                    return Err(Error::Schema(format!("{column} is not an Int column")));
                }
                Some(format!("{} >= {min} AND {} <= {max}", c.name, c.name))
            }
            Filter::TextPrefix { column, prefix } => {
                let c = find(column)?;
                if c.ty != ColumnType::Text {
                    return Err(Error::Schema(format!("{column} is not a Text column")));
                }
                Some(format!("starts_with({}, {})", c.name, sql_str(prefix)))
            }
            _ => return Err(Error::Schema("unsupported filter".into())),
        })
    }

    fn decode_rows(def: &TableDef, batches: &[RecordBatch]) -> Result<Vec<Row>> {
        let mut out = Vec::new();
        for b in batches {
            let ids = col::<Int64Array>(b, "id")?;
            for r in 0..b.num_rows() {
                let mut values = Vec::with_capacity(def.columns.len());
                for c in &def.columns {
                    let a = b.column_by_name(&c.name).ok_or_else(|| Error::Backend(format!("missing {}", c.name)))?;
                    if a.is_null(r) {
                        values.push(Value::Null);
                        continue;
                    }
                    let any = a.as_any();
                    let bad = || Error::Backend(format!("column {} has unexpected type", c.name));
                    values.push(match c.ty {
                        ColumnType::Int => Value::Int(any.downcast_ref::<Int64Array>().ok_or_else(bad)?.value(r)),
                        ColumnType::Float => Value::Float(any.downcast_ref::<Float64Array>().ok_or_else(bad)?.value(r)),
                        ColumnType::Text => Value::Text(any.downcast_ref::<StringArray>().ok_or_else(bad)?.value(r).to_string()),
                        ColumnType::Bytes => Value::Bytes(any.downcast_ref::<BinaryArray>().ok_or_else(bad)?.value(r).to_vec()),
                        _ => return Err(bad()),
                    });
                }
                out.push(Row { id: ids.value(r) as NodeId, values });
            }
        }
        Ok(out)
    }

    fn select_ids(&self, table: &str, filter: &Filter) -> Result<Vec<i64>> {
        let (def, t) = self.table(table)?;
        let pred = Self::predicate(def, filter)?;
        self.rt.block_on(async {
            let mut q = t.query().select(lancedb::query::Select::columns(&["id"]));
            if let Some(p) = pred {
                q = q.only_if(p);
            }
            let batches = Self::collect(q).await?;
            let mut ids = Vec::new();
            for b in &batches {
                ids.extend(col::<Int64Array>(b, "id")?.values().iter().copied());
            }
            Ok(ids)
        })
    }

    fn search(&self, index: &str, query: &[f32], k: usize, ef: usize, only: Option<String>) -> Result<Vec<Hit>> {
        let (m, t) = self.vindex(index)?;
        if query.len() != m.def.dim {
            return Err(Error::Schema(format!("query has {} values, index dim is {}", query.len(), m.def.dim)));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let dt = if m.def.metric == Metric::Cosine { DistanceType::Cosine } else { DistanceType::L2 };
        self.rt.block_on(async {
            let mut q = t
                .vector_search(query.to_vec())
                .map_err(be)?
                .column(VCOL)
                .distance_type(dt)
                .limit(k)
                .select(lancedb::query::Select::columns(&["id"]));
            if m.indexed {
                // One IVF partition, so nprobes 1 searches the whole HNSW graph. refine_factor(1)
                // re-ranks with the stored float32 vectors, so reported distances are exact.
                q = q.ef(ef.max(k)).nprobes(1).refine_factor(1);
            }
            if let Some(p) = only {
                q = q.only_if(p);
            }
            let batches = Self::collect(q).await?;
            let mut hits = Vec::new();
            for b in &batches {
                let ids = col::<Int64Array>(b, "id")?;
                let d = col::<Float32Array>(b, "_distance")?;
                for r in 0..b.num_rows() {
                    hits.push(Hit { id: ids.value(r) as NodeId, distance: d.value(r) });
                }
            }
            hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
            hits.truncate(k);
            Ok(hits)
        })
    }

    fn decode_edges(batches: &[RecordBatch]) -> Result<Vec<Edge>> {
        let mut out = Vec::new();
        for b in batches {
            let (s, r, d, w) = (
                col::<Int64Array>(b, "src")?,
                col::<StringArray>(b, "rel")?,
                col::<Int64Array>(b, "dst")?,
                col::<Float32Array>(b, "weight")?,
            );
            for i in 0..b.num_rows() {
                out.push(Edge { src: s.value(i) as NodeId, rel: r.value(i).to_string(), dst: d.value(i) as NodeId, weight: w.value(i) });
            }
        }
        Ok(out)
    }

    fn edge_query(&self, nodes: &[i64], rel: Option<&str>, dir: Direction) -> Result<Vec<Edge>> {
        let l = id_list(nodes);
        let mut p = match dir {
            Direction::Out => format!("src IN ({l})"),
            Direction::In => format!("dst IN ({l})"),
            Direction::Both => format!("(src IN ({l}) OR dst IN ({l}))"),
            _ => return Err(Error::Schema("unsupported direction".into())),
        };
        if let Some(r) = rel {
            p = format!("{p} AND rel = {}", sql_str(r));
        }
        self.rt.block_on(async { Self::decode_edges(&Self::collect(self.edges.query().only_if(p)).await?) })
    }

    async fn ensure_scalar(t: &Table, cols: &[&str]) -> Result<bool> {
        if t.count_rows(None).await.map_err(be)? == 0 {
            return Ok(false);
        }
        let have: HashSet<String> = t.list_indices().await.map_err(be)?.into_iter().flat_map(|i| i.columns).collect();
        let mut made = false;
        for c in cols {
            if !have.contains(*c) {
                t.create_index(&[*c], Index::BTree(BTreeIndexBuilder::default())).execute().await.map_err(be)?;
                made = true;
            }
        }
        Ok(made)
    }
}

impl Store for LanceStore {
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self> {
        if !path.is_dir() {
            if !opts.create {
                return Err(Error::NotFound(format!("store directory {}", path.display())));
            }
            std::fs::create_dir_all(path)?;
        }
        let uri = path.to_str().ok_or_else(|| Error::Schema("store path is not UTF-8".into()))?.to_string();
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(RUNTIME_THREADS).enable_all().build()?;
        let meta = match std::fs::read_to_string(path.join(META)) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e.into()),
        };
        let (tdefs, vdefs) = Self::parse_meta(&meta)?;
        let (conn, tables, vectors, edges) = rt.block_on(async {
            let conn = lancedb::connect(&uri).execute().await.map_err(be)?;
            let mut tables = BTreeMap::new();
            for d in tdefs {
                let t = Self::open_or_create(&conn, &format!("t_{}", d.name), row_schema(&d)).await?;
                tables.insert(d.name.clone(), (d, t));
            }
            let mut vectors = BTreeMap::new();
            for m in vdefs {
                let t = Self::open_or_create(&conn, &format!("v_{}", m.def.name), vec_schema(m.def.dim)?).await?;
                vectors.insert(m.def.name.clone(), (m, t));
            }
            let edges = Self::open_or_create(&conn, EDGES, edge_schema()).await?;
            Ok::<_, Error>((conn, tables, vectors, edges))
        })?;
        Ok(LanceStore { rt, conn, dir: path.to_path_buf(), tables, vectors, edges })
    }

    fn backend_name(&self) -> &'static str {
        "lancedb"
    }

    fn define_table(&mut self, def: &TableDef) -> Result<()> {
        if !valid_name(&def.name) || def.columns.iter().any(|c| !valid_name(&c.name) || c.name == "id") {
            return Err(Error::Schema(format!("illegal name in table {}", def.name)));
        }
        let mut seen = HashSet::new();
        if !def.columns.iter().all(|c| seen.insert(c.name.as_str())) {
            return Err(Error::Schema("duplicate column".into()));
        }
        if def.columns.iter().any(|c| !matches!(c.ty, ColumnType::Int | ColumnType::Float | ColumnType::Text | ColumnType::Bytes)) {
            return Err(Error::Schema("unsupported column type".into()));
        }
        if let Some((old, _)) = self.tables.get(&def.name) {
            return if old == def { Ok(()) } else { Err(Error::Schema(format!("table {} redefined", def.name))) };
        }
        let t = self.rt.block_on(Self::open_or_create(&self.conn, &format!("t_{}", def.name), row_schema(def)))?;
        self.tables.insert(def.name.clone(), (def.clone(), t));
        self.save_meta()
    }

    fn upsert_rows(&mut self, table: &str, rows: &[Row]) -> Result<()> {
        let (def, t) = self.table(table)?;
        // Merge-insert rejects duplicate keys in one source batch; the last row for an id wins.
        let mut last: HashMap<NodeId, usize> = HashMap::new();
        for (i, r) in rows.iter().enumerate() {
            check_id(r.id)?;
            if r.values.len() != def.columns.len() {
                return Err(Error::Schema(format!("row {} has {} values, table has {} columns", r.id, r.values.len(), def.columns.len())));
            }
            for (v, c) in r.values.iter().zip(&def.columns) {
                let ok = matches!(
                    (c.ty, v),
                    (_, Value::Null)
                        | (ColumnType::Int, Value::Int(_))
                        | (ColumnType::Float, Value::Float(_))
                        | (ColumnType::Text, Value::Text(_))
                        | (ColumnType::Bytes, Value::Bytes(_))
                );
                if !ok {
                    return Err(Error::Schema(format!("row {}: value for {} has the wrong type", r.id, c.name)));
                }
            }
            last.insert(r.id, i);
        }
        if rows.is_empty() {
            return Ok(());
        }
        let mut keep: Vec<usize> = last.into_values().collect();
        keep.sort_unstable();
        let rows: Vec<&Row> = keep.iter().map(|&i| &rows[i]).collect();
        let mut cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.id as i64)))];
        for (ci, c) in def.columns.iter().enumerate() {
            let vals = rows.iter().map(|r| &r.values[ci]);
            let a: ArrayRef = match c.ty {
                ColumnType::Int => Arc::new(vals.map(|v| if let Value::Int(i) = v { Some(*i) } else { None }).collect::<Int64Array>()),
                ColumnType::Float => Arc::new(vals.map(|v| if let Value::Float(x) = v { Some(*x) } else { None }).collect::<Float64Array>()),
                ColumnType::Text => Arc::new(vals.map(|v| if let Value::Text(s) = v { Some(s.as_str()) } else { None }).collect::<StringArray>()),
                ColumnType::Bytes => Arc::new(vals.map(|v| if let Value::Bytes(b) = v { Some(b.as_slice()) } else { None }).collect::<BinaryArray>()),
                _ => return Err(Error::Schema("unsupported column type".into())),
            };
            cols.push(a);
        }
        let batch = RecordBatch::try_new(row_schema(def), cols).map_err(be)?;
        self.rt.block_on(Self::merge(t, &["id"], batch))
    }

    fn get_row(&self, table: &str, id: NodeId) -> Result<Option<Row>> {
        let (def, t) = self.table(table)?;
        let Ok(id) = i64::try_from(id) else { return Ok(None) };
        let batches = self.rt.block_on(Self::collect(t.query().only_if(format!("id = {id}")).limit(1)))?;
        Ok(Self::decode_rows(def, &batches)?.into_iter().next())
    }

    fn select(&self, table: &str, filter: &Filter, limit: usize) -> Result<Vec<Row>> {
        let (def, t) = self.table(table)?;
        let pred = Self::predicate(def, filter)?;
        // Lance queries have no ORDER BY: fetch every match, then sort by id.
        let batches = self.rt.block_on(async {
            let mut q = t.query();
            if let Some(p) = pred {
                q = q.only_if(p);
            }
            Self::collect(q).await
        })?;
        let mut rows = Self::decode_rows(def, &batches)?;
        rows.sort_by_key(|r| r.id);
        rows.truncate(limit);
        Ok(rows)
    }

    fn count_rows(&self, table: &str) -> Result<u64> {
        let (_, t) = self.table(table)?;
        Ok(self.rt.block_on(t.count_rows(None)).map_err(be)? as u64)
    }

    fn delete_rows(&mut self, table: &str, ids: &[NodeId]) -> Result<u64> {
        let (_, t) = self.table(table)?;
        let mut v: Vec<i64> = ids.iter().filter_map(|&i| i64::try_from(i).ok()).collect();
        v.sort_unstable();
        v.dedup();
        if v.is_empty() {
            return Ok(0);
        }
        let p = format!("id IN ({})", id_list(&v));
        self.rt.block_on(async {
            let n = t.count_rows(Some(p.clone())).await.map_err(be)?;
            if n > 0 {
                t.delete(p.as_str()).await.map_err(be)?;
            }
            Ok(n as u64)
        })
    }

    fn define_vector_index(&mut self, def: &VectorIndexDef) -> Result<()> {
        if !valid_name(&def.name) || def.dim == 0 {
            return Err(Error::Schema(format!("bad vector index {}", def.name)));
        }
        if !matches!(def.metric, Metric::Cosine | Metric::L2) {
            return Err(Error::Schema("unsupported metric".into()));
        }
        if let Some((old, _)) = self.vectors.get(&def.name) {
            return if &old.def == def { Ok(()) } else { Err(Error::Schema(format!("vector index {} redefined", def.name))) };
        }
        let t = self.rt.block_on(Self::open_or_create(&self.conn, &format!("v_{}", def.name), vec_schema(def.dim)?))?;
        self.vectors.insert(def.name.clone(), (VecMeta { def: def.clone(), indexed: false }, t));
        self.save_meta()
    }

    fn upsert_vectors(&mut self, index: &str, items: &[(NodeId, &[f32])]) -> Result<()> {
        let (m, t) = self.vindex(index)?;
        let dim = m.def.dim;
        let mut last: HashMap<NodeId, usize> = HashMap::new();
        for (i, (id, v)) in items.iter().enumerate() {
            check_id(*id)?;
            if v.len() != dim {
                return Err(Error::Schema(format!("vector {id} has {} values, dim is {dim}", v.len())));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(Error::Schema(format!("vector {id} has a non-finite value")));
            }
            last.insert(*id, i);
        }
        if items.is_empty() {
            return Ok(());
        }
        let mut keep: Vec<usize> = last.into_values().collect();
        keep.sort_unstable();
        let ids = Int64Array::from_iter_values(keep.iter().map(|&i| items[i].0 as i64));
        let flat = Float32Array::from_iter_values(keep.iter().flat_map(|&i| items[i].1.iter().copied()));
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let list = FixedSizeListArray::try_new(field, dim as i32, Arc::new(flat), None).map_err(be)?;
        let batch = RecordBatch::try_new(vec_schema(dim)?, vec![Arc::new(ids), Arc::new(list)]).map_err(be)?;
        self.rt.block_on(Self::merge(t, &["id"], batch))
    }

    fn knn(&self, index: &str, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<Hit>> {
        self.search(index, query, k, ef_search, None)
    }

    fn knn_filtered(&self, index: &str, query: &[f32], k: usize, ef_search: usize, table: &str, filter: &Filter) -> Result<Vec<Hit>> {
        self.vindex(index)?;
        let ids = self.select_ids(table, filter)?;
        if ids.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        // Prefilter (LanceDB's default): the id set restricts the search before ranking.
        self.search(index, query, k, ef_search, Some(format!("id IN ({})", id_list(&ids))))
    }

    fn count_vectors(&self, index: &str) -> Result<u64> {
        let (_, t) = self.vindex(index)?;
        Ok(self.rt.block_on(t.count_rows(None)).map_err(be)? as u64)
    }

    fn upsert_edges(&mut self, edges: &[Edge]) -> Result<()> {
        let mut last: HashMap<(NodeId, &str, NodeId), usize> = HashMap::new();
        for (i, e) in edges.iter().enumerate() {
            check_id(e.src)?;
            check_id(e.dst)?;
            last.insert((e.src, e.rel.as_str(), e.dst), i);
        }
        if edges.is_empty() {
            return Ok(());
        }
        let mut keep: Vec<usize> = last.into_values().collect();
        keep.sort_unstable();
        let es: Vec<&Edge> = keep.iter().map(|&i| &edges[i]).collect();
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(es.iter().map(|e| e.src as i64))),
            Arc::new(es.iter().map(|e| Some(e.rel.as_str())).collect::<StringArray>()),
            Arc::new(Int64Array::from_iter_values(es.iter().map(|e| e.dst as i64))),
            Arc::new(Float32Array::from_iter_values(es.iter().map(|e| e.weight))),
        ];
        let batch = RecordBatch::try_new(edge_schema(), cols).map_err(be)?;
        self.rt.block_on(Self::merge(&self.edges, &["src", "rel", "dst"], batch))
    }

    fn neighbors(&self, node: NodeId, rel: Option<&str>, dir: Direction, limit: usize) -> Result<Vec<Edge>> {
        let Ok(n) = i64::try_from(node) else { return Ok(Vec::new()) };
        if !matches!(dir, Direction::Out | Direction::In | Direction::Both) {
            return Err(Error::Schema("unsupported direction".into()));
        }
        let mut es = self.edge_query(&[n], rel, dir)?;
        es.sort_by(|a, b| (&a.rel, a.src, a.dst).cmp(&(&b.rel, b.src, b.dst)));
        es.truncate(limit);
        Ok(es)
    }

    fn k_hop(&self, start: NodeId, rel: Option<&str>, dir: Direction, max_depth: u32, limit: usize) -> Result<Vec<Reached>> {
        let Ok(s) = i64::try_from(start) else { return Ok(Vec::new()) };
        if !matches!(dir, Direction::Out | Direction::In | Direction::Both) {
            return Err(Error::Schema("unsupported direction".into()));
        }
        let mut seen: HashSet<i64> = HashSet::from([s]);
        let mut frontier = vec![s];
        let mut out = Vec::new();
        for depth in 1..=max_depth {
            if frontier.is_empty() || out.len() >= limit {
                break;
            }
            let mut next = Vec::new();
            for chunk in frontier.chunks(4096) {
                let set: HashSet<i64> = chunk.iter().copied().collect();
                for e in self.edge_query(chunk, rel, dir)? {
                    let (src, dst) = (e.src as i64, e.dst as i64);
                    if matches!(dir, Direction::Out | Direction::Both) && set.contains(&src) && seen.insert(dst) {
                        next.push(dst);
                    }
                    if matches!(dir, Direction::In | Direction::Both) && set.contains(&dst) && seen.insert(src) {
                        next.push(src);
                    }
                }
            }
            next.sort_unstable();
            out.extend(next.iter().map(|&id| Reached { id: id as NodeId, depth }));
            frontier = next;
        }
        out.truncate(limit);
        Ok(out)
    }

    fn count_edges(&self) -> Result<u64> {
        Ok(self.rt.block_on(self.edges.count_rows(None)).map_err(be)? as u64)
    }

    fn flush(&mut self) -> Result<()> {
        let kind = index_kind();
        let min_rows = min_index_rows();
        let mut meta_changed = false;
        let tables: Vec<(Vec<String>, Table)> = self
            .tables
            .values()
            .map(|(d, t)| {
                let mut c = vec!["id".to_string()];
                c.extend(d.columns.iter().filter(|c| c.indexed && c.ty != ColumnType::Bytes).map(|c| c.name.clone()));
                (c, t.clone())
            })
            .collect();
        let rt = &self.rt;
        rt.block_on(async {
            for (cols, t) in &tables {
                let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
                Self::ensure_scalar(t, &refs).await?;
                t.optimize(OptimizeAction::All).await.map_err(be)?;
            }
            Self::ensure_scalar(&self.edges, &["src", "dst"]).await?;
            self.edges.optimize(OptimizeAction::All).await.map_err(be)?;
            Ok::<_, Error>(())
        })?;
        for (m, t) in self.vectors.values_mut() {
            let n = rt.block_on(t.count_rows(None)).map_err(be)?;
            rt.block_on(async {
                if !m.indexed && n >= min_rows {
                    let d = &m.def;
                    let dt = if d.metric == Metric::Cosine { DistanceType::Cosine } else { DistanceType::L2 };
                    let idx = match kind {
                        IndexKind::HnswSq => Index::IvfHnswSq(
                            IvfHnswSqIndexBuilder::default().distance_type(dt).num_partitions(1).num_edges(d.m as u32).ef_construction(d.ef_construction as u32),
                        ),
                        IndexKind::HnswFlat => Index::IvfHnswFlat(
                            IvfHnswFlatIndexBuilder::default().distance_type(dt).num_partitions(1).num_edges(d.m as u32).ef_construction(d.ef_construction as u32),
                        ),
                    };
                    t.create_index(&[VCOL], idx).execute().await.map_err(be)?;
                    m.indexed = true;
                    meta_changed = true;
                }
                Self::ensure_scalar(t, &["id"]).await?;
                t.optimize(OptimizeAction::All).await.map_err(be)?;
                Ok::<_, Error>(())
            })?;
        }
        if meta_changed {
            self.save_meta()?;
        }
        Ok(())
    }
}
