//! QuickDB backend on CozoDB (embedded SQLite or opt-in RocksDB, HNSW, Datalog recursion).
//!
//! Layout inside the store directory: `cozo.db` (SQLite file or RocksDB directory). Cozo relations:
//! - `qdb_meta {kind, name => def}`: the QuickDB definitions, so they survive reopen.
//! - `t_<table> {id: Int => <columns, all nullable>}` with an index `t_<table>:<col>` per indexed column.
//! - `v_<index> {id: Int => v: <F32; dim>}` with an HNSW index `v_<index>:hnsw`.
//! - `edges {src: Int, rel: String, dst: Int => weight: Float}` with an index `edges:by_dst {dst, rel, src}`.
//!
//! Every value reaches Cozo as a query parameter; query text only contains names that passed
//! `valid_name` and integers chosen by this crate.

use cozo::{DataValue, DbInstance, NamedRows, Num, ScriptMutability, Vector};
use quickdb_api::{
    valid_name, ColumnDef, ColumnType, Direction, Edge, Error, Filter, Hit, Metric, NodeId, OpenOptions, Reached,
    Result, Row, Store, TableDef, Value, VectorIndexDef,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

/// Strategy for `k_hop`. Recursive Datalog is the default: it measured faster on a 100k-node graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HopStrategy {
    Frontier,
    Datalog,
}

/// Feature-selected engine name shared with the benchmark entry point.
pub const BACKEND_NAME: &str = if cfg!(feature = "rocksdb") { "cozo-rocksdb" } else { "cozo-sqlite" };

pub struct CozoStore {
    db: DbInstance,
    tables: HashMap<String, TableDef>,
    indexes: HashMap<String, VectorIndexDef>,
    pub hop_strategy: HopStrategy,
}

fn id_val(id: NodeId) -> Result<DataValue> {
    i64::try_from(id).map(DataValue::from).map_err(|_| Error::Schema(format!("id {id} is larger than i64::MAX")))
}

fn to_id(v: &DataValue) -> Result<NodeId> {
    match v {
        DataValue::Num(Num::Int(i)) if *i >= 0 => Ok(*i as NodeId),
        other => Err(Error::Backend(format!("expected a non-negative Int id, got {other:?}"))),
    }
}

fn ty_name(t: ColumnType) -> &'static str {
    match t {
        ColumnType::Int => "Int",
        ColumnType::Float => "Float",
        ColumnType::Text => "String",
        ColumnType::Bytes => "Bytes",
        // Unreachable: `define_table` rejects an unknown type before this is called.
        _ => "Unsupported",
    }
}

fn ty_from(s: &str) -> Option<ColumnType> {
    Some(match s {
        "Int" => ColumnType::Int,
        "Float" => ColumnType::Float,
        "String" => ColumnType::Text,
        "Bytes" => ColumnType::Bytes,
        _ => return None,
    })
}

fn encode_table(def: &TableDef) -> String {
    def.columns.iter().map(|c| format!("{}:{}:{}", c.name, ty_name(c.ty), u8::from(c.indexed))).collect::<Vec<_>>().join(",")
}

fn decode_table(name: &str, s: &str) -> Option<TableDef> {
    let mut columns = Vec::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        let mut it = part.split(':');
        let (n, t, i) = (it.next()?, it.next()?, it.next()?);
        columns.push(ColumnDef { name: n.to_string(), ty: ty_from(t)?, indexed: i == "1" });
    }
    Some(TableDef { name: name.to_string(), columns })
}

fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "Cosine",
        Metric::L2 => "L2",
        // Unreachable: `define_vector_index` rejects an unknown metric before this is called.
        _ => "Unsupported",
    }
}

fn encode_index(d: &VectorIndexDef) -> String {
    format!("{}:{}:{}:{}", d.dim, metric_name(d.metric), d.m, d.ef_construction)
}

fn decode_index(name: &str, s: &str) -> Option<VectorIndexDef> {
    let mut it = s.split(':');
    let dim = it.next()?.parse().ok()?;
    let metric = match it.next()? {
        "Cosine" => Metric::Cosine,
        "L2" => Metric::L2,
        _ => return None,
    };
    Some(VectorIndexDef { name: name.to_string(), dim, metric, m: it.next()?.parse().ok()?, ef_construction: it.next()?.parse().ok()? })
}

fn value_to_dv(v: &Value, ty: ColumnType) -> Option<DataValue> {
    Some(match (v, ty) {
        (Value::Null, _) => DataValue::Null,
        (Value::Int(i), ColumnType::Int) => DataValue::from(*i),
        (Value::Float(f), ColumnType::Float) => DataValue::from(*f),
        (Value::Text(s), ColumnType::Text) => DataValue::from(s.as_str()),
        (Value::Bytes(b), ColumnType::Bytes) => DataValue::Bytes(b.clone()),
        _ => return None,
    })
}

fn dv_to_value(v: &DataValue) -> Result<Value> {
    Ok(match v {
        DataValue::Null => Value::Null,
        DataValue::Num(Num::Int(i)) => Value::Int(*i),
        DataValue::Num(Num::Float(f)) => Value::Float(*f),
        DataValue::Str(s) => Value::Text(s.to_string()),
        DataValue::Bytes(b) => Value::Bytes(b.clone()),
        other => return Err(Error::Backend(format!("unexpected value from cozo: {other:?}"))),
    })
}

fn f32_vec(v: &[f32]) -> DataValue {
    DataValue::Vec(Vector::F32(ndarray::Array1::from(v.to_vec())))
}

fn exact(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    quickdb_api::conformance::exact_distance(metric, a, b)
}

fn count_of(rows: &NamedRows) -> Result<u64> {
    match rows.rows.first().and_then(|r| r.first()) {
        None => Ok(0),
        Some(DataValue::Num(Num::Int(i))) => Ok(*i as u64),
        Some(DataValue::Num(Num::Float(f))) => Ok(*f as u64),
        Some(other) => Err(Error::Backend(format!("unexpected count {other:?}"))),
    }
}

fn params<const N: usize>(p: [(&str, DataValue); N]) -> BTreeMap<String, DataValue> {
    p.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

impl CozoStore {
    fn run(&self, script: &str, p: BTreeMap<String, DataValue>, mutable: bool) -> Result<NamedRows> {
        let m = if mutable { ScriptMutability::Mutable } else { ScriptMutability::Immutable };
        self.db.run_script(script, p, m).map_err(|e| Error::Backend(format!("{e:?}")))
    }

    fn table(&self, name: &str) -> Result<&TableDef> {
        self.tables.get(name).ok_or_else(|| Error::NotFound(format!("table {name}")))
    }

    fn index(&self, name: &str) -> Result<&VectorIndexDef> {
        self.indexes.get(name).ok_or_else(|| Error::NotFound(format!("vector index {name}")))
    }

    fn col_list(def: &TableDef) -> String {
        let mut s = String::from("id");
        for c in &def.columns {
            s.push_str(", ");
            s.push_str(&c.name);
        }
        s
    }

    fn rows_from(def: &TableDef, rows: &NamedRows) -> Result<Vec<Row>> {
        rows.rows
            .iter()
            .map(|r| {
                if r.len() != def.columns.len() + 1 {
                    return Err(Error::Backend("row width differs from the table definition".into()));
                }
                Ok(Row { id: to_id(&r[0])?, values: r[1..].iter().map(dv_to_value).collect::<Result<_>>()? })
            })
            .collect()
    }

    fn put_meta(&self, kind: &str, name: &str, def: String) -> Result<()> {
        self.run(
            "?[kind, name, def] <- [[$kind, $name, $def]] :put qdb_meta {kind, name => def}",
            params([("kind", kind.into()), ("name", name.into()), ("def", DataValue::from(def))]),
            true,
        )
        .map(|_| ())
    }

    /// Up to `k` exact-distance hits among HNSW candidates; `keep` filters ids.
    fn hnsw_search(&self, def: &VectorIndexDef, query: &[f32], k: usize, ef: usize, keep: Option<&HashSet<NodeId>>) -> Result<Vec<Hit>> {
        let script = format!(
            "?[id, v] := q = $q, ~v_{n}:hnsw{{ id, v | query: q, k: {k}, ef: {ef} }}",
            n = def.name
        );
        let rows = self.run(&script, params([("q", f32_vec(query))]), false)?;
        let mut hits = Vec::with_capacity(rows.rows.len());
        for r in &rows.rows {
            let id = to_id(&r[0])?;
            if keep.is_some_and(|s| !s.contains(&id)) {
                continue;
            }
            let distance = match &r[1] {
                DataValue::Vec(Vector::F32(a)) => exact(def.metric, query, a.as_slice().unwrap_or(&a.to_vec())),
                other => return Err(Error::Backend(format!("unexpected vector {other:?}"))),
            };
            hits.push(Hit { id, distance });
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        Ok(hits)
    }

    fn check_query(def: &VectorIndexDef, query: &[f32]) -> Result<()> {
        if query.len() != def.dim {
            return Err(Error::Schema(format!("query has {} dimensions, index {} has {}", query.len(), def.name, def.dim)));
        }
        Ok(())
    }

    fn edge_atoms(dir: Direction) -> &'static [&'static str] {
        // (x = known end, y = other end)
        match dir {
            Direction::Out => &["*edges{src: x, rel, dst: y}"],
            Direction::In => &["*edges:by_dst{dst: x, rel, src: y}"],
            Direction::Both => &["*edges{src: x, rel, dst: y}", "*edges:by_dst{dst: x, rel, src: y}"],
            // Unreachable: `k_hop` rejects an unknown direction before this is called.
            _ => &[],
        }
    }

    fn hop_frontier(&self, start: NodeId, rel: Option<&str>, dir: Direction, max_depth: u32) -> Result<Vec<Reached>> {
        let rel_filter = if rel.is_some() { ", rel == $rel" } else { "" };
        let script = Self::edge_atoms(dir)
            .iter()
            .map(|a| format!("?[y] := x in $f, {a}{rel_filter}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut seen: HashSet<NodeId> = HashSet::from([start]);
        let mut frontier = vec![start];
        let mut out = Vec::new();
        for depth in 1..=max_depth {
            if frontier.is_empty() {
                break;
            }
            let f = frontier.iter().map(|&n| id_val(n)).collect::<Result<Vec<_>>>()?;
            let mut p = params([("f", DataValue::List(f))]);
            if let Some(r) = rel {
                p.insert("rel".into(), r.into());
            }
            let rows = self.run(&script, p, false)?;
            let mut next = Vec::new();
            for r in &rows.rows {
                let y = to_id(&r[0])?;
                if seen.insert(y) {
                    next.push(y);
                    out.push(Reached { id: y, depth });
                }
            }
            frontier = next;
        }
        Ok(out)
    }

    fn hop_datalog(&self, start: NodeId, rel: Option<&str>, dir: Direction, max_depth: u32) -> Result<Vec<Reached>> {
        let rel_filter = if rel.is_some() { ", rel == $rel" } else { "" };
        let mut script = String::from("reach[x, min(d)] := x = $start, d = 0\n");
        for a in Self::edge_atoms(dir) {
            script.push_str(&format!("reach[y, min(d1)] := reach[x, d], d < $max, {a}{rel_filter}, d1 = d + 1\n"));
        }
        script.push_str("?[y, d] := reach[y, d], y != $start");
        let mut p = params([("start", id_val(start)?), ("max", DataValue::from(max_depth as i64))]);
        if let Some(r) = rel {
            p.insert("rel".into(), r.into());
        }
        let rows = self.run(&script, p, false)?;
        rows.rows
            .iter()
            .map(|r| match &r[1] {
                DataValue::Num(Num::Int(d)) => Ok(Reached { id: to_id(&r[0])?, depth: *d as u32 }),
                other => Err(Error::Backend(format!("unexpected depth {other:?}"))),
            })
            .collect()
    }
}

impl Store for CozoStore {
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self> {
        if !path.is_dir() {
            if !opts.create {
                return Err(Error::NotFound(format!("store directory {}", path.display())));
            }
            std::fs::create_dir_all(path)?;
        }
        let file = path.join("cozo.db");
        let file_str = file.to_str().ok_or_else(|| Error::Schema("store path is not UTF-8".into()))?;
        if !opts.create && !file.exists() {
            return Err(Error::NotFound(format!("store file {}", file.display())));
        }
        let engine = if cfg!(feature = "rocksdb") { "rocksdb" } else { "sqlite" };
        let db = DbInstance::new(engine, file_str, "").map_err(|e| Error::Backend(format!("{e:?}")))?;
        let mut s = CozoStore { db, tables: HashMap::new(), indexes: HashMap::new(), hop_strategy: HopStrategy::Datalog };
        let rels = s.run("::relations", BTreeMap::new(), false)?;
        let names: HashSet<String> =
            rels.rows.iter().filter_map(|r| r.first().and_then(|v| v.get_str()).map(str::to_string)).collect();
        if !names.contains("qdb_meta") {
            s.run(":create qdb_meta {kind: String, name: String => def: String}", BTreeMap::new(), true)?;
        }
        if !names.contains("edges") {
            s.run(":create edges {src: Int, rel: String, dst: Int => weight: Float}", BTreeMap::new(), true)?;
            s.run("::index create edges:by_dst {dst, rel, src}", BTreeMap::new(), true)?;
        }
        let meta = s.run("?[kind, name, def] := *qdb_meta{kind, name, def}", BTreeMap::new(), false)?;
        for r in &meta.rows {
            let (Some(kind), Some(name), Some(def)) = (r[0].get_str(), r[1].get_str(), r[2].get_str()) else {
                return Err(Error::Backend("corrupt qdb_meta row".into()));
            };
            match kind {
                "table" => {
                    let d = decode_table(name, def).ok_or_else(|| Error::Backend(format!("corrupt table def {name}")))?;
                    s.tables.insert(name.to_string(), d);
                }
                "vector" => {
                    let d = decode_index(name, def).ok_or_else(|| Error::Backend(format!("corrupt index def {name}")))?;
                    s.indexes.insert(name.to_string(), d);
                }
                _ => return Err(Error::Backend(format!("unknown meta kind {kind}"))),
            }
        }
        Ok(s)
    }

    fn backend_name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn define_table(&mut self, def: &TableDef) -> Result<()> {
        if !valid_name(&def.name) {
            return Err(Error::Schema(format!("illegal table name {:?}", def.name)));
        }
        let mut seen = HashSet::new();
        for c in &def.columns {
            if !valid_name(&c.name) || c.name == "id" || !seen.insert(c.name.as_str()) {
                return Err(Error::Schema(format!("illegal or duplicate column name {:?}", c.name)));
            }
            if !matches!(c.ty, ColumnType::Int | ColumnType::Float | ColumnType::Text | ColumnType::Bytes) {
                return Err(Error::Schema("unsupported column type".into()));
            }
        }
        if let Some(old) = self.tables.get(&def.name) {
            return if old == def { Ok(()) } else { Err(Error::Schema(format!("table {} exists with another shape", def.name))) };
        }
        let cols: Vec<String> = def.columns.iter().map(|c| format!("{}: {}?", c.name, ty_name(c.ty))).collect();
        let script = if cols.is_empty() {
            format!(":create t_{} {{id: Int}}", def.name)
        } else {
            format!(":create t_{} {{id: Int => {}}}", def.name, cols.join(", "))
        };
        self.run(&script, BTreeMap::new(), true)?;
        for c in def.columns.iter().filter(|c| c.indexed) {
            self.run(&format!("::index create t_{}:{} {{{}}}", def.name, c.name, c.name), BTreeMap::new(), true)?;
        }
        self.put_meta("table", &def.name, encode_table(def))?;
        self.tables.insert(def.name.clone(), def.clone());
        Ok(())
    }

    fn upsert_rows(&mut self, table: &str, rows: &[Row]) -> Result<()> {
        let def = self.table(table)?;
        let mut data = Vec::with_capacity(rows.len());
        for r in rows {
            if r.values.len() != def.columns.len() {
                return Err(Error::Schema(format!("row {} has {} values, table has {} columns", r.id, r.values.len(), def.columns.len())));
            }
            let mut t = Vec::with_capacity(r.values.len() + 1);
            t.push(id_val(r.id)?);
            for (v, c) in r.values.iter().zip(&def.columns) {
                t.push(value_to_dv(v, c.ty).ok_or_else(|| Error::Schema(format!("row {}: column {} wants {:?}", r.id, c.name, c.ty)))?);
            }
            data.push(DataValue::List(t));
        }
        if data.is_empty() {
            return Ok(());
        }
        let cols = Self::col_list(def);
        let rest: Vec<&str> = def.columns.iter().map(|c| c.name.as_str()).collect();
        let spec = if rest.is_empty() { "id".to_string() } else { format!("id => {}", rest.join(", ")) };
        let script = format!("?[{cols}] <- $rows :put t_{table} {{{spec}}}");
        self.run(&script, params([("rows", DataValue::List(data))]), true).map(|_| ())
    }

    fn get_row(&self, table: &str, id: NodeId) -> Result<Option<Row>> {
        let def = self.table(table)?;
        let cols = Self::col_list(def);
        let script = format!("?[{cols}] := id = $id, *t_{table}{{{cols}}}");
        let rows = self.run(&script, params([("id", id_val(id)?)]), false)?;
        Ok(Self::rows_from(def, &rows)?.into_iter().next())
    }

    fn select(&self, table: &str, filter: &Filter, limit: usize) -> Result<Vec<Row>> {
        let def = self.table(table)?;
        let cols = Self::col_list(def);
        let col = |name: &str| -> Result<&ColumnDef> {
            def.columns.iter().find(|c| c.name == name).ok_or_else(|| Error::NotFound(format!("column {table}.{name}")))
        };
        let main = format!("*t_{table}{{{cols}}}");
        let mut p = BTreeMap::new();
        let body = match filter {
            Filter::All => main,
            Filter::Eq { column, value } => {
                let c = col(column)?;
                let Some(v) = value_to_dv(value, c.ty).filter(|v| *v != DataValue::Null) else {
                    // Null or a value of another type never equals a stored value.
                    return Ok(Vec::new());
                };
                p.insert("v".to_string(), v);
                if c.indexed {
                    format!("{n} = $v, *t_{table}:{n}{{{n}, id}}, {main}", n = c.name)
                } else {
                    format!("{main}, {n} == $v", n = c.name)
                }
            }
            Filter::IntRange { column, min, max } => {
                let c = col(column)?;
                if c.ty != ColumnType::Int {
                    return Err(Error::Schema(format!("IntRange on non-Int column {column}")));
                }
                p.insert("min".to_string(), DataValue::from(*min));
                p.insert("max".to_string(), DataValue::from(*max));
                if c.indexed {
                    format!("*t_{table}:{n}{{{n}, id}}, is_int({n}), {n} >= $min, {n} <= $max, {main}", n = c.name)
                } else {
                    format!("{main}, is_int({n}), {n} >= $min, {n} <= $max", n = c.name)
                }
            }
            Filter::TextPrefix { column, prefix } => {
                let c = col(column)?;
                if c.ty != ColumnType::Text {
                    return Err(Error::Schema(format!("TextPrefix on non-Text column {column}")));
                }
                p.insert("p".to_string(), DataValue::from(prefix.as_str()));
                format!("{main}, is_string({n}), starts_with({n}, $p)", n = c.name)
            }
            _ => return Err(Error::Schema("unsupported filter".into())),
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        let script = format!("?[{cols}] := {body}\n:order id\n:limit {limit}");
        let rows = self.run(&script, p, false)?;
        Self::rows_from(def, &rows)
    }

    fn count_rows(&self, table: &str) -> Result<u64> {
        self.table(table)?;
        count_of(&self.run(&format!("?[count(id)] := *t_{table}{{id}}"), BTreeMap::new(), false)?)
    }

    fn delete_rows(&mut self, table: &str, ids: &[NodeId]) -> Result<u64> {
        self.table(table)?;
        let list = DataValue::List(ids.iter().map(|&i| id_val(i)).collect::<Result<_>>()?);
        // `_del` is a temp relation that lives only for this script, so count and delete are one transaction.
        let script = format!(
            "{{?[id] := id in $ids, *t_{table}{{id}} :replace _del {{id}}}}\n\
             {{?[id] := *_del{{id}} :rm t_{table} {{id}}}}\n\
             {{?[count(id)] := *_del{{id}}}}"
        );
        count_of(&self.run(&script, params([("ids", list)]), true)?)
    }

    fn define_vector_index(&mut self, def: &VectorIndexDef) -> Result<()> {
        if !valid_name(&def.name) {
            return Err(Error::Schema(format!("illegal index name {:?}", def.name)));
        }
        if def.dim == 0 || def.m == 0 || def.ef_construction == 0 {
            return Err(Error::Schema("dim, m and ef_construction must be positive".into()));
        }
        if !matches!(def.metric, Metric::Cosine | Metric::L2) {
            return Err(Error::Schema("unsupported metric".into()));
        }
        if let Some(old) = self.indexes.get(&def.name) {
            return if old == def { Ok(()) } else { Err(Error::Schema(format!("index {} exists with another shape", def.name))) };
        }
        self.run(&format!(":create v_{} {{id: Int => v: <F32; {}>}}", def.name, def.dim), BTreeMap::new(), true)?;
        self.run(
            &format!(
                "::hnsw create v_{n}:hnsw {{ dim: {d}, m: {m}, dtype: F32, fields: [v], distance: {dist}, ef_construction: {ef} }}",
                n = def.name,
                d = def.dim,
                m = def.m,
                dist = metric_name(def.metric),
                ef = def.ef_construction
            ),
            BTreeMap::new(),
            true,
        )?;
        self.put_meta("vector", &def.name, encode_index(def))?;
        self.indexes.insert(def.name.clone(), def.clone());
        Ok(())
    }

    fn upsert_vectors(&mut self, index: &str, items: &[(NodeId, &[f32])]) -> Result<()> {
        let def = self.index(index)?;
        let mut data = Vec::with_capacity(items.len());
        for (id, v) in items {
            if v.len() != def.dim {
                return Err(Error::Schema(format!("vector {id} has {} dimensions, index {index} has {}", v.len(), def.dim)));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(Error::Schema(format!("vector {id} has a non-finite component")));
            }
            data.push(DataValue::List(vec![id_val(*id)?, f32_vec(v)]));
        }
        if data.is_empty() {
            return Ok(());
        }
        self.run(&format!("?[id, v] <- $items :put v_{index} {{id => v}}"), params([("items", DataValue::List(data))]), true)
            .map(|_| ())
    }

    fn knn(&self, index: &str, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<Hit>> {
        let def = self.index(index)?;
        Self::check_query(def, query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut hits = self.hnsw_search(def, query, k, ef_search.max(k), None)?;
        hits.truncate(k);
        Ok(hits)
    }

    fn knn_filtered(&self, index: &str, query: &[f32], k: usize, ef_search: usize, table: &str, filter: &Filter) -> Result<Vec<Hit>> {
        let def = self.index(index)?;
        Self::check_query(def, query)?;
        // Cozo's HNSW `filter:` sees only the vector relation's own fields, so the row filter is
        // applied to a widening candidate list instead.
        let keep: HashSet<NodeId> = self.select(table, filter, usize::MAX >> 1)?.into_iter().map(|r| r.id).collect();
        if k == 0 || keep.is_empty() {
            return Ok(Vec::new());
        }
        let total = self.count_vectors(index)? as usize;
        let mut kk = k.max(ef_search).min(total.max(1));
        loop {
            let mut hits = self.hnsw_search(def, query, kk, ef_search.max(kk), Some(&keep))?;
            if hits.len() >= k || kk >= total {
                hits.truncate(k);
                return Ok(hits);
            }
            kk = (kk * 4).min(total);
        }
    }

    fn count_vectors(&self, index: &str) -> Result<u64> {
        self.index(index)?;
        count_of(&self.run(&format!("?[count(id)] := *v_{index}{{id}}"), BTreeMap::new(), false)?)
    }

    fn upsert_edges(&mut self, edges: &[Edge]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let data = edges
            .iter()
            .map(|e| {
                Ok(DataValue::List(vec![id_val(e.src)?, e.rel.as_str().into(), id_val(e.dst)?, DataValue::from(e.weight as f64)]))
            })
            .collect::<Result<Vec<_>>>()?;
        self.run("?[src, rel, dst, weight] <- $e :put edges {src, rel, dst => weight}", params([("e", DataValue::List(data))]), true)
            .map(|_| ())
    }

    fn neighbors(&self, node: NodeId, rel: Option<&str>, dir: Direction, limit: usize) -> Result<Vec<Edge>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let rel_filter = if rel.is_some() { ", rel == $rel" } else { "" };
        let out = format!("?[rel, src, dst, w] := src = $n, *edges{{src, rel, dst, weight: w}}{rel_filter}");
        let inn = format!("?[rel, src, dst, w] := dst = $n, *edges:by_dst{{dst, rel, src}}, *edges{{src, rel, dst, weight: w}}{rel_filter}");
        let body = match dir {
            Direction::Out => out,
            Direction::In => inn,
            Direction::Both => format!("{out}\n{inn}"),
            _ => return Err(Error::Schema("unsupported direction".into())),
        };
        let mut p = params([("n", id_val(node)?)]);
        if let Some(r) = rel {
            p.insert("rel".into(), r.into());
        }
        let rows = self.run(&format!("{body}\n:order rel, src, dst\n:limit {limit}"), p, false)?;
        rows.rows
            .iter()
            .map(|r| {
                let rel = r[0].get_str().ok_or_else(|| Error::Backend("rel is not a string".into()))?.to_string();
                let weight = match &r[3] {
                    DataValue::Num(Num::Float(f)) => *f as f32,
                    DataValue::Num(Num::Int(i)) => *i as f32,
                    other => return Err(Error::Backend(format!("unexpected weight {other:?}"))),
                };
                Ok(Edge { src: to_id(&r[1])?, rel, dst: to_id(&r[2])?, weight })
            })
            .collect()
    }

    fn k_hop(&self, start: NodeId, rel: Option<&str>, dir: Direction, max_depth: u32, limit: usize) -> Result<Vec<Reached>> {
        id_val(start)?;
        if !matches!(dir, Direction::Out | Direction::In | Direction::Both) {
            return Err(Error::Schema("unsupported direction".into()));
        }
        if max_depth == 0 || limit == 0 {
            return Ok(Vec::new());
        }
        let mut out = match self.hop_strategy {
            HopStrategy::Frontier => self.hop_frontier(start, rel, dir, max_depth)?,
            HopStrategy::Datalog => self.hop_datalog(start, rel, dir, max_depth)?,
        };
        out.sort_by_key(|r| (r.depth, r.id));
        out.truncate(limit);
        Ok(out)
    }

    fn count_edges(&self) -> Result<u64> {
        count_of(&self.run("?[count(src)] := *edges{src}", BTreeMap::new(), false)?)
    }

    fn flush(&mut self) -> Result<()> {
        // SQLite commits every script. RocksDB keeps Cozo's default WAL policy.
        // Cozo 0.7.6 exposes no explicit WAL sync through DbInstance, so this is
        // not a power-loss durability barrier for RocksDB (reported as a limitation).
        Ok(())
    }
}
