use quickdb_api::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

type EdgeKey = (String, NodeId, NodeId);

/// Exact reference store. Secondary indexes are intentionally not used for rows.
pub struct MemStore {
    path: PathBuf,
    sync: bool,
    tables: BTreeMap<String, (TableDef, BTreeMap<NodeId, Row>)>,
    vectors: BTreeMap<String, (VectorIndexDef, BTreeMap<NodeId, Vec<f32>>)>,
    edges: BTreeMap<EdgeKey, Edge>,
    outgoing: BTreeMap<NodeId, BTreeSet<EdgeKey>>,
    incoming: BTreeMap<NodeId, BTreeSet<EdgeKey>>,
}

fn schema(s: &str) -> Error {
    Error::Schema(s.into())
}
fn id_ok(id: NodeId) -> Result<()> {
    if id > i64::MAX as u64 {
        Err(schema("id exceeds i64::MAX"))
    } else {
        Ok(())
    }
}
fn typed(v: &Value, t: ColumnType) -> bool {
    matches!(
        (v, t),
        (Value::Null, _)
            | (Value::Int(_), ColumnType::Int)
            | (Value::Float(_), ColumnType::Float)
            | (Value::Text(_), ColumnType::Text)
            | (Value::Bytes(_), ColumnType::Bytes)
    )
}
fn known_type(t: ColumnType) -> bool {
    matches!(
        t,
        ColumnType::Int | ColumnType::Float | ColumnType::Text | ColumnType::Bytes
    )
}
fn column(def: &TableDef, f: &Filter) -> Result<Option<usize>> {
    let name = match f {
        Filter::All => return Ok(None),
        Filter::Eq { column, .. }
        | Filter::IntRange { column, .. }
        | Filter::TextPrefix { column, .. } => column,
        _ => return Err(schema("unsupported filter")),
    };
    let i = def
        .columns
        .iter()
        .position(|c| &c.name == name)
        .ok_or_else(|| Error::NotFound(name.clone()))?;
    let valid = match f {
        Filter::Eq { value, .. } => typed(value, def.columns[i].ty),
        Filter::IntRange { .. } => def.columns[i].ty == ColumnType::Int,
        Filter::TextPrefix { .. } => def.columns[i].ty == ColumnType::Text,
        Filter::All => true,
        _ => false,
    };
    if valid {
        Ok(Some(i))
    } else {
        Err(schema("filter type does not match column"))
    }
}
fn matches(row: &Row, f: &Filter, i: Option<usize>) -> bool {
    let Some(i) = i else { return true };
    match (f, &row.values[i]) {
        (Filter::Eq { value, .. }, v) => !matches!(v, Value::Null) && v == value,
        (Filter::IntRange { min, max, .. }, Value::Int(v)) => min <= v && v <= max,
        (Filter::TextPrefix { prefix, .. }, Value::Text(v)) => v.starts_with(prefix),
        _ => false,
    }
}
fn valid_vector(v: &[f32], dim: usize) -> Result<()> {
    if v.len() != dim || v.iter().any(|x| !x.is_finite()) {
        Err(schema("wrong dimension or non-finite vector"))
    } else {
        Ok(())
    }
}

impl MemStore {
    fn after_write(&mut self) -> Result<()> {
        if self.sync {
            self.flush()
        } else {
            Ok(())
        }
    }
    fn table(&self, name: &str) -> Result<&(TableDef, BTreeMap<NodeId, Row>)> {
        self.tables
            .get(name)
            .ok_or_else(|| Error::NotFound(name.into()))
    }
    fn index(&self, name: &str) -> Result<&(VectorIndexDef, BTreeMap<NodeId, Vec<f32>>)> {
        self.vectors
            .get(name)
            .ok_or_else(|| Error::NotFound(name.into()))
    }
    fn search(
        &self,
        index: &str,
        q: &[f32],
        k: usize,
        allowed: Option<&BTreeSet<NodeId>>,
    ) -> Result<Vec<Hit>> {
        let (def, data) = self.index(index)?;
        valid_vector(q, def.dim)?;
        let mut hits: Vec<_> = data
            .iter()
            .filter(|(id, _)| allowed.is_none_or(|set| set.contains(id)))
            .map(|(id, v)| Hit {
                id: *id,
                distance: conformance::exact_distance(def.metric, q, v),
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        Ok(hits)
    }
}

impl Store for MemStore {
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self> {
        if !path.is_dir() {
            if !opts.create {
                return Err(Error::NotFound(path.display().to_string()));
            }
            fs::create_dir_all(path)?;
        }
        let mut s = Self {
            path: path.into(),
            sync: false,
            tables: BTreeMap::new(),
            vectors: BTreeMap::new(),
            edges: BTreeMap::new(),
            outgoing: BTreeMap::new(),
            incoming: BTreeMap::new(),
        };
        let p = path.join("mem.json");
        if p.exists() {
            let snapshot: Snapshot = serde_json::from_reader(BufReader::new(File::open(p)?))
                .map_err(|e| Error::Backend(format!("invalid mem snapshot: {e}")))?;
            if snapshot.version != 1 {
                return Err(schema("unsupported snapshot version"));
            }
            for t in snapshot.tables {
                let def = TableDef {
                    name: t.name,
                    columns: t
                        .columns
                        .into_iter()
                        .map(|(name, ty, indexed)| {
                            Ok(ColumnDef {
                                name,
                                ty: match ty {
                                    0 => ColumnType::Int,
                                    1 => ColumnType::Float,
                                    2 => ColumnType::Text,
                                    3 => ColumnType::Bytes,
                                    _ => return Err(schema("invalid column type")),
                                },
                                indexed,
                            })
                        })
                        .collect::<Result<_>>()?,
                };
                s.define_table(&def)?;
                let rows = t
                    .rows
                    .into_iter()
                    .map(|(id, values)| Row {
                        id,
                        values: values.into_iter().map(Value::from).collect(),
                    })
                    .collect::<Vec<_>>();
                s.upsert_rows(&def.name, &rows)?;
            }
            for v in snapshot.vectors {
                let def = VectorIndexDef {
                    name: v.name,
                    dim: v.dim,
                    metric: if v.cosine { Metric::Cosine } else { Metric::L2 },
                    m: v.m,
                    ef_construction: v.ef,
                };
                s.define_vector_index(&def)?;
                let items = v
                    .items
                    .iter()
                    .map(|(id, v)| (*id, v.as_slice()))
                    .collect::<Vec<_>>();
                s.upsert_vectors(&def.name, &items)?;
            }
            s.upsert_edges(
                &snapshot
                    .edges
                    .into_iter()
                    .map(|(src, rel, dst, weight)| Edge {
                        src,
                        rel,
                        dst,
                        weight,
                    })
                    .collect::<Vec<_>>(),
            )?;
        } else if !opts.create {
            return Err(Error::NotFound("mem.json".into()));
        }
        s.sync = opts.sync_each_write;
        Ok(s)
    }
    fn backend_name(&self) -> &'static str {
        "mem"
    }
    fn define_table(&mut self, def: &TableDef) -> Result<()> {
        let mut names = BTreeSet::new();
        if !valid_name(&def.name)
            || def.columns.iter().any(|c| {
                !valid_name(&c.name)
                    || c.name == "id"
                    || !names.insert(c.name.clone())
                    || !known_type(c.ty)
            })
        {
            return Err(schema(
                "invalid or duplicate name, or unsupported column type",
            ));
        }
        if let Some((old, _)) = self.tables.get(&def.name) {
            return if old == def {
                Ok(())
            } else {
                Err(schema("table redefinition"))
            };
        }
        self.tables
            .insert(def.name.clone(), (def.clone(), BTreeMap::new()));
        self.after_write()
    }
    fn upsert_rows(&mut self, table: &str, rows: &[Row]) -> Result<()> {
        let (def, _) = self.table(table)?;
        for r in rows {
            id_ok(r.id)?;
            if r.values.len() != def.columns.len()
                || r.values
                    .iter()
                    .zip(&def.columns)
                    .any(|(v, c)| !typed(v, c.ty) || matches!(v, Value::Float(f) if !f.is_finite()))
            {
                return Err(schema("row does not match schema"));
            }
        }
        if let Some((_, data)) = self.tables.get_mut(table) {
            for r in rows {
                data.insert(r.id, r.clone());
            }
        }
        self.after_write()
    }
    fn get_row(&self, table: &str, id: NodeId) -> Result<Option<Row>> {
        id_ok(id)?;
        Ok(self.table(table)?.1.get(&id).cloned())
    }
    fn select(&self, table: &str, filter: &Filter, limit: usize) -> Result<Vec<Row>> {
        let (def, data) = self.table(table)?;
        let i = column(def, filter)?;
        Ok(data
            .values()
            .filter(|r| matches(r, filter, i))
            .take(limit)
            .cloned()
            .collect())
    }
    fn count_rows(&self, table: &str) -> Result<u64> {
        Ok(self.table(table)?.1.len() as u64)
    }
    fn delete_rows(&mut self, table: &str, ids: &[NodeId]) -> Result<u64> {
        self.table(table)?;
        for id in ids {
            id_ok(*id)?;
        }
        let mut n = 0;
        if let Some((_, data)) = self.tables.get_mut(table) {
            for id in ids {
                n += u64::from(data.remove(id).is_some());
            }
        }
        self.after_write()?;
        Ok(n)
    }
    fn define_vector_index(&mut self, def: &VectorIndexDef) -> Result<()> {
        if !valid_name(&def.name) || def.dim == 0 {
            return Err(schema("invalid vector index"));
        }
        if let Some((old, _)) = self.vectors.get(&def.name) {
            return if old == def {
                Ok(())
            } else {
                Err(schema("index redefinition"))
            };
        }
        self.vectors
            .insert(def.name.clone(), (def.clone(), BTreeMap::new()));
        self.after_write()
    }
    fn upsert_vectors(&mut self, index: &str, items: &[(NodeId, &[f32])]) -> Result<()> {
        let dim = self.index(index)?.0.dim;
        for (id, v) in items {
            id_ok(*id)?;
            valid_vector(v, dim)?;
        }
        if let Some((_, data)) = self.vectors.get_mut(index) {
            for (id, v) in items {
                data.insert(*id, v.to_vec());
            }
        }
        self.after_write()
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
        let allowed = self
            .select(table, filter, usize::MAX)?
            .iter()
            .map(|r| r.id)
            .collect();
        self.search(index, query, k, Some(&allowed))
    }
    fn count_vectors(&self, index: &str) -> Result<u64> {
        Ok(self.index(index)?.1.len() as u64)
    }
    fn upsert_edges(&mut self, edges: &[Edge]) -> Result<()> {
        for e in edges {
            id_ok(e.src)?;
            id_ok(e.dst)?;
            if !e.weight.is_finite() {
                return Err(schema("invalid edge"));
            }
        }
        for e in edges {
            let key = (e.rel.clone(), e.src, e.dst);
            self.outgoing.entry(e.src).or_default().insert(key.clone());
            self.incoming.entry(e.dst).or_default().insert(key.clone());
            self.edges.insert(key, e.clone());
        }
        self.after_write()
    }
    fn neighbors(
        &self,
        node: NodeId,
        rel: Option<&str>,
        dir: Direction,
        limit: usize,
    ) -> Result<Vec<Edge>> {
        id_ok(node)?;
        let mut keys = BTreeSet::new();
        if dir != Direction::In {
            if let Some(k) = self.outgoing.get(&node) {
                keys.extend(k.iter());
            }
        }
        if dir != Direction::Out {
            if let Some(k) = self.incoming.get(&node) {
                keys.extend(k.iter());
            }
        }
        Ok(keys
            .into_iter()
            .filter(|k| rel.is_none_or(|r| r == k.0))
            .take(limit)
            .filter_map(|k| self.edges.get(k).cloned())
            .collect())
    }
    fn k_hop(
        &self,
        start: NodeId,
        rel: Option<&str>,
        dir: Direction,
        max_depth: u32,
        limit: usize,
    ) -> Result<Vec<Reached>> {
        id_ok(start)?;
        let mut seen = BTreeSet::from([start]);
        let mut frontier = BTreeSet::from([start]);
        let mut result = Vec::new();
        for depth in 1..=max_depth {
            if result.len() >= limit || frontier.is_empty() {
                break;
            }
            let mut next = BTreeSet::new();
            for node in frontier {
                for e in self.neighbors(node, rel, dir, usize::MAX)? {
                    let other = if e.src == node && dir != Direction::In {
                        e.dst
                    } else {
                        e.src
                    };
                    if seen.insert(other) {
                        next.insert(other);
                    }
                }
            }
            result.extend(
                next.iter()
                    .take(limit - result.len())
                    .map(|id| Reached { id: *id, depth }),
            );
            frontier = next;
        }
        Ok(result)
    }
    fn count_edges(&self) -> Result<u64> {
        Ok(self.edges.len() as u64)
    }
    fn flush(&mut self) -> Result<()> {
        let snapshot = Snapshot {
            version: 1,
            tables: self
                .tables
                .values()
                .map(|(d, rows)| DiskTable {
                    name: d.name.clone(),
                    columns: d
                        .columns
                        .iter()
                        .map(|c| {
                            (
                                c.name.clone(),
                                match c.ty {
                                    ColumnType::Int => 0,
                                    ColumnType::Float => 1,
                                    ColumnType::Text => 2,
                                    ColumnType::Bytes => 3,
                                    // Unreachable: `define_table` rejects an unknown type.
                                    _ => u8::MAX,
                                },
                                c.indexed,
                            )
                        })
                        .collect(),
                    rows: rows
                        .values()
                        .map(|r| (r.id, r.values.iter().map(DiskValue::from).collect()))
                        .collect(),
                })
                .collect(),
            vectors: self
                .vectors
                .values()
                .map(|(d, data)| DiskVectors {
                    name: d.name.clone(),
                    dim: d.dim,
                    cosine: d.metric == Metric::Cosine,
                    m: d.m,
                    ef: d.ef_construction,
                    items: data.iter().map(|(id, v)| (*id, v.clone())).collect(),
                })
                .collect(),
            edges: self
                .edges
                .values()
                .map(|e| (e.src, e.rel.clone(), e.dst, e.weight))
                .collect(),
        };
        let tmp = self.path.join("mem.json.tmp");
        let mut writer = BufWriter::new(File::create(&tmp)?);
        serde_json::to_writer(&mut writer, &snapshot).map_err(|e| Error::Backend(e.to_string()))?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(tmp, self.path.join("mem.json"))?;
        #[cfg(unix)]
        File::open(&self.path)?.sync_all()?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
enum DiskValue {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
    Bytes(Vec<u8>),
}
impl From<&Value> for DiskValue {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => Self::Null,
            Value::Int(v) => Self::Int(*v),
            Value::Float(v) => Self::Float(*v),
            Value::Text(v) => Self::Text(v.clone()),
            Value::Bytes(v) => Self::Bytes(v.clone()),
            // Unreachable: `typed` rejects an unknown variant.
            _ => Self::Null,
        }
    }
}
impl From<DiskValue> for Value {
    fn from(v: DiskValue) -> Self {
        match v {
            DiskValue::Null => Self::Null,
            DiskValue::Int(v) => Self::Int(v),
            DiskValue::Float(v) => Self::Float(v),
            DiskValue::Text(v) => Self::Text(v),
            DiskValue::Bytes(v) => Self::Bytes(v),
        }
    }
}
#[derive(Serialize, Deserialize)]
struct DiskTable {
    name: String,
    columns: Vec<(String, u8, bool)>,
    rows: Vec<(NodeId, Vec<DiskValue>)>,
}
#[derive(Serialize, Deserialize)]
struct DiskVectors {
    name: String,
    dim: usize,
    cosine: bool,
    m: usize,
    ef: usize,
    items: Vec<(NodeId, Vec<f32>)>,
}
#[derive(Serialize, Deserialize)]
struct Snapshot {
    version: u32,
    tables: Vec<DiskTable>,
    vectors: Vec<DiskVectors>,
    edges: Vec<(NodeId, String, NodeId, f32)>,
}
