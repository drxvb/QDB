//! Embedded SurrealDB comparison backend using SurrealKV, native records and HNSW.
//! Query values are bound parameters. Only validated identifiers and integer index
//! settings are interpolated into SurrealQL.

use quickdb_api::{
    valid_name, ColumnDef, ColumnType, Direction, Edge, Error, Filter, Hit, Metric,
    NodeId, OpenOptions, Reached, Result, Row, Store, TableDef, Value, VectorIndexDef,
};
use serde_json::{json, Value as Json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use surrealdb::engine::local::{Db, SurrealKv};
use surrealdb::types::{Bytes, SurrealValue, Value as SqlValue};
use surrealdb::{IndexedResults, Surreal};
use tokio::runtime::Runtime;

/// The two engine-native representations measured by `examples/graph_layout.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphLayout {
    Plain,
    Relation,
}

pub struct SurrealStore {
    // Field order drops the client before shutting down and joining the runtime.
    db: Surreal<Db>,
    runtime: Runtime,
    tables: HashMap<String, TableDef>,
    indexes: HashMap<String, VectorIndexDef>,
    graph_layout: GraphLayout,
}

fn backend(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}

fn schema(message: impl Into<String>) -> Error {
    Error::Schema(message.into())
}

fn checked_id(id: NodeId) -> Result<i64> {
    i64::try_from(id).map_err(|_| schema("NodeId exceeds i64::MAX"))
}

fn name_ok(name: &str) -> Result<()> {
    if valid_name(name) { Ok(()) } else { Err(schema(format!("invalid name: {name}"))) }
}

fn params<const N: usize>(items: [(&str, SqlValue); N]) -> BTreeMap<String, SqlValue> {
    items.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
}

fn value(v: &Value, ty: ColumnType) -> Result<SqlValue> {
    Ok(match (v, ty) {
        (Value::Null, _) => SqlValue::Null,
        (Value::Int(n), ColumnType::Int) => n.into_value(),
        (Value::Float(n), ColumnType::Float) if n.is_finite() => n.into_value(),
        (Value::Text(s), ColumnType::Text) => s.clone().into_value(),
        (Value::Bytes(b), ColumnType::Bytes) => SqlValue::Bytes(Bytes(b.clone().into())),
        _ => return Err(schema("value does not match its column type")),
    })
}

fn from_value(v: &Json, ty: ColumnType) -> Result<Value> {
    if v.is_null() { return Ok(Value::Null); }
    Ok(match ty {
        ColumnType::Int => Value::Int(v.as_i64().ok_or_else(|| backend("expected integer"))?),
        ColumnType::Float => Value::Float(v.as_f64().ok_or_else(|| backend("expected float"))?),
        ColumnType::Text => Value::Text(v.as_str().ok_or_else(|| backend("expected string"))?.to_owned()),
        ColumnType::Bytes => Value::Bytes(serde_json::from_value(v.clone()).map_err(backend)?),
        _ => return Err(backend("unsupported column type")),
    })
}

fn table_json(def: &TableDef) -> Result<Json> {
    let columns = def.columns.iter().map(|c| {
        let ty = match c.ty { ColumnType::Int=>"int",ColumnType::Float=>"float",ColumnType::Text=>"text",ColumnType::Bytes=>"bytes", _=>return Err(schema("unsupported column type")) };
        Ok(json!({"name":c.name,"ty":ty,"indexed":c.indexed}))
    }).collect::<Result<Vec<_>>>()?;
    Ok(json!({"name":def.name,"columns":columns}))
}

fn table_from(v: &Json) -> Result<TableDef> {
    let name = text(&v["name"])?;
    let columns = array(&v["columns"])? .iter().map(|c| {
        Ok(ColumnDef { name: text(&c["name"])?, ty: match c["ty"].as_str() {
            Some("int") => ColumnType::Int, Some("float") => ColumnType::Float,
            Some("text") => ColumnType::Text, Some("bytes") => ColumnType::Bytes,
            _ => return Err(backend("invalid persisted column type")),
        }, indexed: c["indexed"].as_bool().ok_or_else(|| backend("invalid persisted index flag"))? })
    }).collect::<Result<Vec<_>>>()?;
    Ok(TableDef { name, columns })
}

fn index_json(def: &VectorIndexDef) -> Result<Json> {
    let metric = match def.metric {Metric::Cosine=>"cosine",Metric::L2=>"l2", _=>return Err(schema("unsupported metric"))};
    Ok(json!({"name":def.name,"dim":def.dim,"metric":metric,"m":def.m,"efc":def.ef_construction}))
}

fn index_from(v: &Json) -> Result<VectorIndexDef> {
    Ok(VectorIndexDef { name: text(&v["name"])?, dim: number(&v["dim"])? as usize,
        metric: match v["metric"].as_str() { Some("cosine")=>Metric::Cosine, Some("l2")=>Metric::L2, _=>return Err(backend("invalid persisted metric")) },
        m: number(&v["m"])? as usize, ef_construction: number(&v["efc"])? as usize })
}

fn text(v: &Json) -> Result<String> {
    v.as_str().map(str::to_owned).ok_or_else(|| backend("expected string"))
}

fn number(v: &Json) -> Result<u64> {
    v.as_u64().ok_or_else(|| backend("expected unsigned integer"))
}

fn array(v: &Json) -> Result<&Vec<Json>> {
    v.as_array().ok_or_else(|| backend("expected array"))
}

impl SurrealStore {
    fn run(&self, sql: &str, p: BTreeMap<String, SqlValue>) -> Result<IndexedResults> {
        self.runtime.block_on(async { self.db.query(sql).bind(p).await?.check() }).map_err(backend)
    }

    fn query(&self, sql: &str, p: BTreeMap<String, SqlValue>) -> Result<Json> {
        self.run(sql, p)?.take(0).map_err(backend)
    }

    fn execute(&self, sql: &str, p: BTreeMap<String, SqlValue>) -> Result<()> {
        self.run(sql, p).map(|_| ())
    }

    fn table(&self, name: &str) -> Result<&TableDef> {
        self.tables.get(name).ok_or_else(|| Error::NotFound(format!("table {name}")))
    }

    fn index(&self, name: &str) -> Result<&VectorIndexDef> {
        self.indexes.get(name).ok_or_else(|| Error::NotFound(format!("index {name}")))
    }

    fn meta_statement() -> &'static str {
        "UPSERT type::record('qdbmeta', $key) SET kind = $kind, payload = $payload;"
    }

    fn meta_params(kind: &str, name: &str, payload: Json) -> BTreeMap<String, SqlValue> {
        params([("key",format!("{kind}_{name}").into_value()),("kind",kind.into_value()),("payload",payload.into_value())])
    }

    /// Only empty graphs can change representation. The choice survives reopen.
    pub fn set_graph_layout(&mut self, layout: GraphLayout) -> Result<()> {
        if layout == self.graph_layout { return Ok(()); }
        if self.count_edges()? != 0 { return Err(schema("cannot change layout of a non-empty graph")); }
        self.execute(Self::meta_statement(), Self::meta_params("graph", "layout", json!(match layout {GraphLayout::Plain=>"plain",GraphLayout::Relation=>"relation"})))?;
        self.graph_layout = layout;
        Ok(())
    }

    fn edge_table(&self) -> &'static str {
        match self.graph_layout { GraphLayout::Plain=>"qdbplain", GraphLayout::Relation=>"qdbrel" }
    }

    fn projection(def: &TableDef) -> String {
        let values = def.columns.iter().map(|c| format!("`{}`", c.name)).collect::<Vec<_>>().join(",");
        format!("record::id(id) AS qid, [{values}] AS vals")
    }

    fn decode_rows(def: &TableDef, result: &Json) -> Result<Vec<Row>> {
        array(result)?.iter().map(|r| {
            let vals = array(&r["vals"])?;
            if vals.len() != def.columns.len() { return Err(backend("unexpected row width")); }
            Ok(Row { id: number(&r["qid"])?, values: vals.iter().zip(&def.columns).map(|(v,c)| from_value(v,c.ty)).collect::<Result<_>>()? })
        }).collect()
    }

    fn condition(def: &TableDef, filter: &Filter) -> Result<(String, BTreeMap<String, SqlValue>)> {
        let column = match filter {
            Filter::All => return Ok(("true".into(), BTreeMap::new())),
            Filter::Eq {column,..} | Filter::IntRange {column,..} | Filter::TextPrefix {column,..} => column,
            _ => return Err(schema("unsupported filter")),
        };
        let col = def.columns.iter().find(|c| &c.name == column).ok_or_else(|| Error::NotFound(format!("column {column}")))?;
        let c = format!("`{column}`");
        Ok(match filter {
            Filter::Eq {value: Value::Null,..} => ("false".into(), BTreeMap::new()),
            Filter::Eq {value:v,..} => (format!("{c} = $val"), params([("val",value(v,col.ty)?)])),
            Filter::IntRange {min,max,..} if col.ty == ColumnType::Int => (format!("{c} >= $min AND {c} <= $max"),params([("min",min.into_value()),("max",max.into_value())])),
            Filter::TextPrefix {prefix,..} if col.ty == ColumnType::Text => (format!("{c} != NONE AND {c} != NULL AND string::starts_with({c}, $prefix)"),params([("prefix",prefix.clone().into_value())])),
            _ => return Err(schema("filter does not match column type")),
        })
    }

    fn count(&self, table: &str) -> Result<u64> {
        let result = self.query(&format!("SELECT count() AS n FROM {table} GROUP ALL"), BTreeMap::new())?;
        match array(&result)?.first() { Some(v)=>number(&v["n"]),None=>Ok(0) }
    }

    fn check_vector(def: &VectorIndexDef, v: &[f32]) -> Result<()> {
        if v.len() != def.dim || v.iter().any(|n| !n.is_finite()) { Err(schema("wrong vector dimension or non-finite component")) } else { Ok(()) }
    }

    fn search(&self, def: &VectorIndexDef, q: &[f32], k: usize, ef: usize) -> Result<Vec<Hit>> {
        if k == 0 { return Ok(Vec::new()); }
        let sql = format!("SELECT record::id(id) AS qid, v FROM v_{} WHERE v <|{},{}|> $q",def.name,k,ef.max(k));
        let result = self.query(&sql,params([("q",q.to_vec().into_value())]))?;
        let mut hits = array(&result)?.iter().map(|r| {
            let v: Vec<f32> = serde_json::from_value(r["v"].clone()).map_err(backend)?;
            if v.len() != q.len() { return Err(backend("stored vector dimension differs")); }
            Ok(Hit { id: number(&r["qid"])?, distance: quickdb_api::conformance::exact_distance(def.metric,q,&v) })
        }).collect::<Result<Vec<_>>>()?;
        hits.sort_by(|a,b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        Ok(hits)
    }

    fn edge_rows(result: &Json) -> Result<Vec<Edge>> {
        array(result)?.iter().map(|r| Ok(Edge {src:number(&r["src"])?,rel:text(&r["rel"])?,dst:number(&r["dst"])?,weight:r["weight"].as_f64().ok_or_else(||backend("expected edge weight"))? as f32})).collect()
    }

    fn frontier(&self, nodes: &[NodeId], rel: Option<&str>, dir: Direction) -> Result<Vec<NodeId>> {
        let mut p = params([("frontier",nodes.to_vec().into_value())]);
        if let Some(r) = rel { p.insert("rel".into(),r.into_value()); }
        let relation = if rel.is_some() { " AND rel = $rel" } else { "" };
        let (sql, p) = match self.graph_layout {
            GraphLayout::Plain => {
                let cond = match dir { Direction::Out=>"src IN $frontier", Direction::In=>"dst IN $frontier", Direction::Both=>"(src IN $frontier OR dst IN $frontier)", _=>return Err(schema("unsupported direction")) };
                (format!("SELECT src, dst FROM qdbplain WHERE {cond}{relation}"),p)
            },
            GraphLayout::Relation => {
                // A graph arrow walks the engine's native adjacency keys, not a table scan.
                let edge = if rel.is_some() { "(qdbrel WHERE rel = $rel)" } else { "qdbrel" };
                let path = match dir { Direction::Out=>format!("->{}",edge),Direction::In=>format!("<-{}",edge),Direction::Both=>format!("<->{}",edge), _=>return Err(schema("unsupported direction")) };
                p.insert("nodes".into(),nodes.iter().map(|&id| surrealdb::types::RecordId::new("qdbnode",id as i64)).collect::<Vec<_>>().into_value());
                (format!("SELECT VALUE {path}.* FROM $nodes"),p)
            },
        };
        let result = self.query(&sql,p)?;
        let mut endpoints = Vec::new();
        let known: HashSet<_> = nodes.iter().copied().collect();
        fn flatten<'a>(v: &'a Json, out: &mut Vec<&'a Json>) {
            match v { Json::Array(a)=>for r in a {flatten(r,out)},Json::Object(_)=>out.push(v),_=>{} }
        }
        let mut records=Vec::new();
        flatten(&result,&mut records);
        for r in records {
            let src=number(&r["src"])?;
            let dst=number(&r["dst"])?;
            if dir != Direction::In && known.contains(&src) { endpoints.push(dst); }
            if dir != Direction::Out && known.contains(&dst) { endpoints.push(src); }
        }
        Ok(endpoints)
    }
}

impl Store for SurrealStore {
    fn open(path: &Path, opts: &OpenOptions) -> Result<Self> {
        if !opts.create && !path.join("surrealkv").exists() { return Err(Error::NotFound("store directory".into())); }
        if opts.create { std::fs::create_dir_all(path)?; }
        let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(6).thread_stack_size(10*1024*1024).enable_all().build()?;
        let endpoint = path.join("surrealkv").to_str().ok_or_else(||schema("store path is not UTF-8"))?.to_owned();
        let db = runtime.block_on(async {
            let db=Surreal::new::<SurrealKv>(endpoint).await?;
            db.use_ns("quickdb").use_db("quickdb").await?;
            Ok::<_,surrealdb::Error>(db)
        }).map_err(backend)?;
        let mut s=Self {db,runtime,tables:HashMap::new(),indexes:HashMap::new(),graph_layout:GraphLayout::Plain};
        s.execute("DEFINE TABLE IF NOT EXISTS qdbmeta SCHEMALESS; DEFINE TABLE IF NOT EXISTS qdbplain SCHEMALESS; DEFINE INDEX IF NOT EXISTS src ON qdbplain FIELDS src; DEFINE INDEX IF NOT EXISTS dst ON qdbplain FIELDS dst; DEFINE TABLE IF NOT EXISTS qdbrel TYPE RELATION SCHEMALESS; DEFINE INDEX IF NOT EXISTS src ON qdbrel FIELDS src; DEFINE INDEX IF NOT EXISTS dst ON qdbrel FIELDS dst;",BTreeMap::new())?;
        let metadata=s.query("SELECT kind, payload FROM qdbmeta",BTreeMap::new())?;
        for row in array(&metadata)? {
            match row["kind"].as_str() {
                Some("table")=>{let d=table_from(&row["payload"])?;s.tables.insert(d.name.clone(),d);},
                Some("index")=>{let d=index_from(&row["payload"])?;s.indexes.insert(d.name.clone(),d);},
                Some("graph")=>s.graph_layout=match row["payload"].as_str() {Some("plain")=>GraphLayout::Plain,Some("relation")=>GraphLayout::Relation,_=>return Err(backend("invalid graph layout"))},
                _=>return Err(backend("invalid metadata kind")),
            }
        }
        Ok(s)
    }

    fn backend_name(&self) -> &'static str { "surrealdb" }

    fn define_table(&mut self, def: &TableDef) -> Result<()> {
        name_ok(&def.name)?;
        let mut seen=HashSet::new();
        for c in &def.columns { name_ok(&c.name)?; if c.name=="id" || !seen.insert(&c.name) {return Err(schema("duplicate or reserved column"));} }
        if let Some(old)=self.tables.get(&def.name) {return if old==def {Ok(())} else {Err(schema("table definition differs"))};}
        let mut sql=format!("BEGIN TRANSACTION; DEFINE TABLE t_{} SCHEMALESS;",def.name);
        for c in &def.columns {
            if c.indexed {sql.push_str(&format!("DEFINE INDEX c_{} ON t_{} FIELDS `{}`;",c.name,def.name,c.name));}
        }
        sql.push_str(Self::meta_statement());
        sql.push_str("COMMIT TRANSACTION;");
        self.execute(&sql,Self::meta_params("table",&def.name,table_json(def)?))?;
        self.tables.insert(def.name.clone(),def.clone());
        Ok(())
    }

    fn upsert_rows(&mut self, table: &str, rows: &[Row]) -> Result<()> {
        let def=self.table(table)?;
        let mut items=Vec::with_capacity(rows.len());
        for row in rows {
            let id=checked_id(row.id)?;
            if row.values.len()!=def.columns.len() {return Err(schema("wrong row width"));}
            let fields=row.values.iter().zip(&def.columns).map(|(v,c)|Ok((c.name.clone(),value(v,c.ty)?))).collect::<Result<BTreeMap<_,_>>>()?;
            items.push(params([("n",id.into_value()),("fields",fields.into_value())]).into_value());
        }
        if items.is_empty() {return Ok(());}
        self.execute("BEGIN TRANSACTION; FOR $r IN $items { UPSERT type::record($table, $r.n) CONTENT $r.fields; }; COMMIT TRANSACTION;",params([("table",format!("t_{table}").into_value()),("items",items.into_value())]))
    }

    fn get_row(&self, table: &str, id: NodeId) -> Result<Option<Row>> {
        checked_id(id)?;
        let def=self.table(table)?;
        let result=self.query(&format!("SELECT {} FROM $record",Self::projection(def)),params([("record",surrealdb::types::RecordId::new(format!("t_{table}"),id as i64).into_value())]))?;
        Ok(Self::decode_rows(def,&result)?.into_iter().next())
    }

    fn select(&self, table: &str, filter: &Filter, limit: usize) -> Result<Vec<Row>> {
        let def=self.table(table)?;
        let (cond,mut p)=Self::condition(def,filter)?;
        if limit==0 {return Ok(Vec::new());}
        p.insert("limit".into(),limit.min(i64::MAX as usize).into_value());
        let result=self.query(&format!("SELECT {} FROM t_{table} WHERE {cond} ORDER BY qid LIMIT $limit",Self::projection(def)),p)?;
        Self::decode_rows(def,&result)
    }

    fn count_rows(&self, table: &str) -> Result<u64> {self.table(table)?;self.count(&format!("t_{table}"))}

    fn delete_rows(&mut self, table: &str, ids: &[NodeId]) -> Result<u64> {
        self.table(table)?;
        for &id in ids {checked_id(id)?;}
        if ids.is_empty() {return Ok(0);}
        let result=self.query(&format!("DELETE FROM t_{table} WHERE record::id(id) IN $ids RETURN BEFORE"),params([("ids",ids.to_vec().into_value())]))?;
        Ok(array(&result)?.len() as u64)
    }

    fn define_vector_index(&mut self, def: &VectorIndexDef) -> Result<()> {
        name_ok(&def.name)?;
        if def.dim==0 || def.dim>65536 || def.m<2 || def.ef_construction<def.m {return Err(schema("invalid HNSW dimensions, M or EFC"));}
        if let Some(old)=self.indexes.get(&def.name) {return if old==def {Ok(())} else {Err(schema("vector index definition differs"))};}
        let dist=match def.metric {Metric::Cosine=>"COSINE",Metric::L2=>"EUCLIDEAN", _=>return Err(schema("unsupported metric"))};
        let sql=format!("BEGIN TRANSACTION; DEFINE TABLE v_{n} SCHEMALESS; DEFINE INDEX hnsw ON v_{n} FIELDS v HNSW DIMENSION {dim} TYPE F32 DIST {dist} EFC {efc} M {m}; {meta} COMMIT TRANSACTION;",n=def.name,dim=def.dim,efc=def.ef_construction,m=def.m,meta=Self::meta_statement());
        self.execute(&sql,Self::meta_params("index",&def.name,index_json(def)?))?;
        self.indexes.insert(def.name.clone(),def.clone());
        Ok(())
    }

    fn upsert_vectors(&mut self, index: &str, items: &[(NodeId, &[f32])]) -> Result<()> {
        let def=self.index(index)?;
        let mut rows=Vec::with_capacity(items.len());
        for &(id,v) in items {checked_id(id)?;Self::check_vector(def,v)?;rows.push(json!({"n":id,"v":v}));}
        if rows.is_empty() {return Ok(());}
        self.execute("BEGIN TRANSACTION; FOR $r IN $items { UPSERT type::record($table, $r.n) SET v = $r.v; }; COMMIT TRANSACTION;",params([("table",format!("v_{index}").into_value()),("items",json!(rows).into_value())]))
    }

    fn knn(&self, index: &str, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<Hit>> {
        let def=self.index(index)?;Self::check_vector(def,query)?;
        self.search(def,query,k,ef_search)
    }

    fn knn_filtered(&self, index: &str, query: &[f32], k: usize, ef_search: usize, table: &str, filter: &Filter) -> Result<Vec<Hit>> {
        let def=self.index(index)?;Self::check_vector(def,query)?;
        let keep: HashSet<_>=self.select(table,filter,usize::MAX)?.into_iter().map(|r|r.id).collect();
        if k==0 || keep.is_empty() {return Ok(Vec::new());}
        let total=self.count_vectors(index)? as usize;
        let mut candidates=k.saturating_mul(4).max(ef_search).min(total);
        loop {
            let mut hits=self.search(def,query,candidates,ef_search.max(candidates))?;
            hits.retain(|h|keep.contains(&h.id));
            if hits.len()>=k || candidates==total {hits.truncate(k);return Ok(hits);}
            candidates=candidates.saturating_mul(2).max(1).min(total);
        }
    }

    fn count_vectors(&self, index: &str) -> Result<u64> {self.index(index)?;self.count(&format!("v_{index}"))}

    fn upsert_edges(&mut self, edges: &[Edge]) -> Result<()> {
        let mut rows=Vec::with_capacity(edges.len());
        for e in edges {
            checked_id(e.src)?;checked_id(e.dst)?;
            if !e.weight.is_finite() {return Err(schema("non-finite edge weight"));}
            // Length-prefixing keeps the composite key injective even with arbitrary relation text.
            let key=format!("{}:{}:{}:{}",e.src,e.rel.len(),e.rel,e.dst);
            rows.push(json!({"key":key,"src":e.src,"dst":e.dst,"rel":e.rel,"weight":e.weight}));
        }
        if rows.is_empty() {return Ok(());}
        let sql=match self.graph_layout {
            GraphLayout::Plain=>"BEGIN TRANSACTION; FOR $e IN $edges { UPSERT type::record('qdbplain', $e.key) SET src=$e.src, dst=$e.dst, rel=$e.rel, weight=$e.weight; }; COMMIT TRANSACTION;",
            GraphLayout::Relation=>"BEGIN TRANSACTION; FOR $e IN $edges { UPSERT type::record('qdbnode', $e.src); UPSERT type::record('qdbnode', $e.dst); RELATE type::record('qdbnode', $e.src)->type::record('qdbrel', $e.key)->type::record('qdbnode', $e.dst) SET src=$e.src, dst=$e.dst, rel=$e.rel, weight=$e.weight; }; COMMIT TRANSACTION;",
        };
        self.execute(sql,params([("edges",json!(rows).into_value())]))
    }

    fn neighbors(&self, node: NodeId, rel: Option<&str>, dir: Direction, limit: usize) -> Result<Vec<Edge>> {
        checked_id(node)?;
        if limit==0 {return Ok(Vec::new());}
        let cond=match dir {Direction::Out=>"src=$node",Direction::In=>"dst=$node",Direction::Both=>"(src=$node OR dst=$node)", _=>return Err(schema("unsupported direction"))};
        let mut p=params([("node",node.into_value()),("limit",limit.min(i64::MAX as usize).into_value())]);
        if let Some(r)=rel {p.insert("rel".into(),r.into_value());}
        let rel_cond=if rel.is_some() {" AND rel=$rel"} else {""};
        let sql=format!("SELECT src, rel, dst, weight FROM {} WHERE {cond}{rel_cond} ORDER BY rel, src, dst LIMIT $limit",self.edge_table());
        Self::edge_rows(&self.query(&sql,p)?)
    }

    fn k_hop(&self, start: NodeId, rel: Option<&str>, dir: Direction, max_depth: u32, limit: usize) -> Result<Vec<Reached>> {
        checked_id(start)?;
        if limit==0 {return Ok(Vec::new());}
        let mut seen=HashSet::from([start]);
        let mut frontier=vec![start];
        let mut reached=Vec::new();
        for depth in 1..=max_depth {
            if frontier.is_empty() {break;}
            let mut next=Vec::new();
            for id in self.frontier(&frontier,rel,dir)? {if seen.insert(id) {next.push(id);}}
            next.sort_unstable();
            for &id in &next {reached.push(Reached{id,depth});}
            if reached.len()>=limit {break;}
            frontier=next;
        }
        reached.truncate(limit);
        Ok(reached)
    }

    fn count_edges(&self) -> Result<u64> {self.count(self.edge_table())}

    fn flush(&mut self) -> Result<()> {
        // SurrealKV's default sync=every makes every committed write durable.
        // A successful read is a barrier after all synchronous Store write calls.
        self.query("RETURN true",BTreeMap::new()).map(|_|())
    }
}
