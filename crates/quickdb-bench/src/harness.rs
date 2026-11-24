use crate::questions::QuestionSet;
use quickdb_api::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// Fully materialized input is shared by every backend and retained for exact truth.
pub struct Dataset {
    pub description: serde_json::Value,
    pub dim: usize,
    pub metric: Metric,
    pub query_vectors: Option<Vec<Vec<f32>>>,
    pub truth: Option<Vec<Vec<NodeId>>>,
    pub rows: Vec<Row>,
    pub vectors: Vec<(NodeId, Vec<f32>)>,
    pub edges: Vec<Edge>,
}

fn bad(s: impl Into<String>) -> Error {
    Error::Schema(s.into())
}
fn json_error(e: serde_json::Error) -> Error {
    Error::Backend(e.to_string())
}

pub fn benchmark_table() -> TableDef {
    TableDef {
        name: "docs".into(),
        columns: vec![
            ColumnDef {
                name: "kind".into(),
                ty: ColumnType::Text,
                indexed: true,
            },
            ColumnDef {
                name: "title".into(),
                ty: ColumnType::Text,
                indexed: false,
            },
            ColumnDef {
                name: "size".into(),
                ty: ColumnType::Int,
                indexed: true,
            },
            ColumnDef {
                name: "score".into(),
                ty: ColumnType::Float,
                indexed: false,
            },
            ColumnDef {
                name: "body".into(),
                ty: ColumnType::Text,
                indexed: false,
            },
        ],
    }
}

pub fn synthetic(rows: usize, dim: usize) -> Result<Dataset> {
    if rows == 0 || dim == 0 || rows as u128 > i64::MAX as u128 {
        return Err(bad("rows and dim must be positive, ids must fit i64"));
    }
    let mut rng = conformance::Lcg(0x514442);
    let centers: Vec<_> = (0..20).map(|_| rng.vector(dim)).collect();
    let mut data = Dataset {
        description: serde_json::json!({"type":"synthetic", "rows":rows, "dim":dim, "seed":0x514442u64, "clusters":20}),
        dim,
        metric: Metric::Cosine,
        query_vectors: None,
        truth: None,
        rows: Vec::with_capacity(rows),
        vectors: Vec::with_capacity(rows),
        edges: Vec::with_capacity(rows.saturating_mul(5)),
    };
    for i in 0..rows {
        let id = i as u64 + 1;
        data.rows.push(Row {
            id,
            values: vec![
                Value::Text(format!("kind{:02}", i % 20)),
                Value::Text(format!("Document {id} عنوان")),
                Value::Int((i % 100000) as i64),
                Value::Float(rng.next_f32() as f64),
                Value::Text(
                    "Deterministic research paragraph with useful words and citations. ".repeat(3),
                ),
            ],
        });
        data.vectors.push((
            id,
            centers[i % 20]
                .iter()
                .map(|c| c + 0.12 * rng.next_f32())
                .collect(),
        ));
        for j in 0..5 {
            let dst = if j == 0 {
                (i + 1) % rows
            } else {
                let u = rng.next_u32() as f64 / 2147483648.0;
                ((u * u * u * rows as f64) as usize).min(rows - 1)
            };
            data.edges.push(Edge {
                src: id,
                rel: ["next", "cites", "links", "cites", "links"][j].into(),
                dst: dst as u64 + 1,
                weight: 1.0,
            });
        }
    }
    Ok(data)
}

#[derive(Deserialize)]
struct Chunk {
    id: NodeId,
    doc_id: NodeId,
    corpus: String,
    path: String,
    seq: usize,
    chars: usize,
    text: String,
}
#[derive(Deserialize)]
struct CorpusEdge {
    src: NodeId,
    rel: String,
    dst: NodeId,
    weight: f32,
}
fn json_lines<T: serde::de::DeserializeOwned>(p: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(p)?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(&line?).map_err(|e| bad(format!("{}:{}: {e}", p.display(), i + 1)))
        })
        .collect()
}

pub fn corpus(path: &Path) -> Result<Dataset> {
    let meta: serde_json::Value =
        serde_json::from_reader(File::open(path.join("meta.json"))?).map_err(json_error)?;
    let dim = meta["dim"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| bad("meta.dim must be positive"))?;
    let chunks: Vec<Chunk> = json_lines(&path.join("chunks.jsonl"))?;
    if chunks.is_empty() || meta["chunks"].as_u64() != Some(chunks.len() as u64) {
        return Err(bad("empty corpus or chunk count mismatch"));
    }
    let docs: Vec<serde_json::Value> = json_lines(&path.join("docs.jsonl"))?;
    if meta["docs"].as_u64() != Some(docs.len() as u64) {
        return Err(bad("document count mismatch"));
    }
    let edges: Vec<CorpusEdge> = json_lines(&path.join("edges.jsonl"))?;
    if meta["edges"].as_u64() != Some(edges.len() as u64) {
        return Err(bad("edge count mismatch"));
    }
    let mut input = BufReader::new(File::open(path.join("vectors.f32"))?);
    let expected = chunks
        .len()
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| bad("vector size overflow"))?;
    if input.get_ref().metadata()?.len() != expected as u64 {
        return Err(bad("vectors.f32 byte length mismatch"));
    }
    let mut data = Dataset {
        description: serde_json::json!({"type":"corpus", "meta":meta}),
        dim,
        metric: Metric::Cosine,
        query_vectors: None,
        truth: None,
        rows: Vec::new(),
        vectors: Vec::new(),
        edges: edges
            .into_iter()
            .map(|e| Edge {
                src: e.src,
                rel: e.rel,
                dst: e.dst,
                weight: e.weight,
            })
            .collect(),
    };
    let mut ids = BTreeSet::new();
    for c in chunks {
        if c.id > i64::MAX as u64 || !ids.insert(c.id) || c.chars != c.text.chars().count() {
            return Err(bad("invalid/duplicate chunk id or char count"));
        }
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut bytes = [0; 4];
            input.read_exact(&mut bytes)?;
            let v = f32::from_le_bytes(bytes);
            if !v.is_finite() {
                return Err(bad("non-finite embedding"));
            }
            vector.push(v);
        }
        data.rows.push(Row {
            id: c.id,
            values: vec![
                Value::Text(c.corpus),
                Value::Text(format!("{}#{} (doc {})", c.path, c.seq, c.doc_id)),
                Value::Int(c.chars as i64),
                Value::Float(c.seq as f64),
                Value::Text(c.text),
            ],
        });
        data.vectors.push((c.id, vector));
    }
    Ok(data)
}

#[derive(Debug, Serialize)]
pub struct Latency {
    pub samples: usize,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub mean_results: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_10: Option<f64>,
}
fn latency(mut times: Vec<f64>, results: usize, recall: Option<f64>) -> Latency {
    times.sort_by(f64::total_cmp);
    let n = times.len();
    let p = |percent: usize| {
        times
            .get((n * percent).div_ceil(100).saturating_sub(1))
            .copied()
            .unwrap_or(0.0)
    };
    Latency {
        samples: n,
        p50_us: p(50),
        p95_us: p(95),
        p99_us: p(99),
        mean_results: if n == 0 {
            0.0
        } else {
            results as f64 / n as f64
        },
        recall_at_10: recall,
    }
}
#[derive(Debug, Serialize)]
pub struct Memory {
    pub vm_hwm_bytes: Option<u64>,
    pub vm_rss_bytes: Option<u64>,
}
pub fn memory() -> Memory {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |key: &str| {
        s.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    };
    Memory {
        vm_hwm_bytes: field("VmHWM:"),
        vm_rss_bytes: field("VmRSS:"),
    }
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub backend: String,
    pub dataset: serde_json::Value,
    pub hostname: String,
    pub cpu_count: usize,
    pub available_parallelism: usize,
    pub phases_seconds: BTreeMap<String, f64>,
    pub size_bytes: u64,
    pub memory: BTreeMap<String, Memory>,
    pub queries: BTreeMap<String, Latency>,
    pub rows: u64,
    pub vectors: u64,
    pub edges: u64,
    pub metric: &'static str,
    pub query_seed: u64,
    /// Present only when the run was given written questions; absent otherwise, so old reports keep their keys.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub questions: Option<serde_json::Value>,
}
fn record<T>(
    phases: &mut BTreeMap<String, f64>,
    name: &str,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let start = Instant::now();
    let value = f()?;
    phases.insert(name.into(), start.elapsed().as_secs_f64());
    Ok(value)
}
fn measured<T>(times: &mut Vec<f64>, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let start = Instant::now();
    let value = std::hint::black_box(f()?);
    times.push(start.elapsed().as_secs_f64() * 1e6);
    Ok(value)
}

/// Independent brute-force truth, never asks a backend for the expected answer.
pub fn exact_top10(
    data: &[(NodeId, Vec<f32>)],
    q: &[f32],
    allowed: Option<&BTreeSet<NodeId>>,
) -> Vec<NodeId> {
    exact_top10_metric(data, q, allowed, Metric::Cosine)
}

pub fn exact_top10_metric(
    data: &[(NodeId, Vec<f32>)],
    q: &[f32],
    allowed: Option<&BTreeSet<NodeId>>,
    metric: Metric,
) -> Vec<NodeId> {
    let mut best: Vec<(f32, NodeId)> = Vec::with_capacity(11);
    let qnorm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    for (id, v) in data {
        if allowed.is_some_and(|a| !a.contains(id)) {
            continue;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dot = v.iter().zip(q).map(|(x, y)| x * y).sum::<f32>();
        let distance = if metric == Metric::L2 {
            v.iter().zip(q).map(|(x, y)| (x - y) * (x - y)).sum()
        } else if norm == 0.0 || qnorm == 0.0 {
            1.0
        } else {
            1.0 - dot / (norm * qnorm)
        };
        let at = best.partition_point(|(d, i)| d.total_cmp(&distance).then(i.cmp(id)).is_lt());
        if at < 10 {
            best.insert(at, (distance, *id));
            best.truncate(10);
        }
    }
    best.into_iter().map(|(_, id)| id).collect()
}
fn recall(hits: &[Hit], truth: &[NodeId]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let unique: BTreeSet<_> = hits.iter().take(10).map(|h| h.id).collect();
    truth.iter().filter(|id| unique.contains(id)).count() as f64 / truth.len() as f64
}
fn make_queries(data: &Dataset) -> Result<Vec<Vec<f32>>> {
    let mut rng = conformance::Lcg(0x5155455259);
    let mut queries = Vec::with_capacity(200);
    for _ in 0..200 {
        let base = &data.vectors[rng.next_u32() as usize % data.vectors.len()].1;
        let mut q: Vec<_> = base.iter().map(|v| v + rng.next_f32() * 0.02).collect();
        // A query equal to a stored vector would make recall artificially easy.
        let mut tries = 0;
        while data.vectors.iter().any(|(_, v)| v == &q) {
            q[0] += 0.0001;
            tries += 1;
            if tries > 100 || !q[0].is_finite() {
                return Err(bad("cannot construct held-out finite queries"));
            }
        }
        queries.push(q);
    }
    Ok(queries)
}

/// Identical load, lifecycle and query calls for every backend. Refuses nonempty output stores.
pub fn run<S: Store>(data: &Dataset, path: &Path) -> Result<Report> {
    run_with_questions::<S>(data, path, None)
}

/// `run`, plus a `questions` block (document-level relevance next to exact recall) when `questions` is given.
/// Every other measure is taken exactly as in `run`.
pub fn run_with_questions<S: Store>(
    data: &Dataset,
    path: &Path,
    questions: Option<&QuestionSet>,
) -> Result<Report> {
    if data.rows.is_empty() || data.vectors.is_empty() || data.dim == 0 {
        return Err(bad("benchmark requires rows, vectors, positive dim"));
    }
    if path.exists() && fs::read_dir(path)?.next().is_some() {
        return Err(bad(
            "data-dir must be empty to avoid contaminating measurements",
        ));
    }
    let total = Instant::now();
    let mut phases = BTreeMap::new();
    let mut memories = BTreeMap::new();
    memories.insert("before_open".into(), memory());
    let mut store = record(&mut phases, "open", || {
        S::open(path, &OpenOptions::default())
    })?;
    record(&mut phases, "define_schema", || {
        store.define_table(&benchmark_table())?;
        store.define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim: data.dim,
            metric: data.metric,
            m: 16,
            ef_construction: 200,
        })
    })?;
    record(&mut phases, "load_rows", || {
        for batch in data.rows.chunks(1000) {
            store.upsert_rows("docs", batch)?;
        }
        Ok(())
    })?;
    record(&mut phases, "load_vectors", || {
        for batch in data.vectors.chunks(1000) {
            let items: Vec<_> = batch.iter().map(|(id, v)| (*id, v.as_slice())).collect();
            store.upsert_vectors("emb", &items)?;
        }
        Ok(())
    })?;
    record(&mut phases, "load_edges", || {
        for batch in data.edges.chunks(5000) {
            store.upsert_edges(batch)?;
        }
        Ok(())
    })?;
    record(&mut phases, "flush", || store.flush())?;
    let size_bytes = record(&mut phases, "size_on_disk", || Ok(dir_size(path)?))?;
    memories.insert("after_load".into(), memory());
    record(&mut phases, "drop", || {
        drop(store);
        Ok(())
    })?;
    let store = record(&mut phases, "reopen", || {
        S::open(
            path,
            &OpenOptions {
                create: false,
                sync_each_write: false,
            },
        )
    })?;
    memories.insert("after_reopen".into(), memory());
    let mut queries = BTreeMap::new();
    let mut rng = conformance::Lcg(0x5142494453);
    record(&mut phases, "get_row", || {
        let mut times = Vec::new();
        let mut n = 0;
        for _ in 0..10000 {
            let id = data.rows[rng.next_u32() as usize % data.rows.len()].id;
            n += usize::from(measured(&mut times, || store.get_row("docs", id))?.is_some());
        }
        queries.insert("get_row".into(), latency(times, n, None));
        Ok(())
    })?;
    let kinds: Vec<_> = data
        .rows
        .iter()
        .filter_map(|r| {
            if let Value::Text(v) = &r.values[0] {
                Some(v.clone())
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if kinds.is_empty() {
        return Err(bad("no text kinds"));
    }
    record(&mut phases, "select_eq", || {
        let mut times = Vec::new();
        let mut n = 0;
        for i in 0..500 {
            let f = Filter::Eq {
                column: "kind".into(),
                value: Value::Text(kinds[i % kinds.len()].clone()),
            };
            n += measured(&mut times, || store.select("docs", &f, 100))?.len();
        }
        queries.insert("select_eq".into(), latency(times, n, None));
        Ok(())
    })?;
    record(&mut phases, "select_int_range", || {
        let mut times = Vec::new();
        let mut n = 0;
        for _ in 0..500 {
            let row = &data.rows[rng.next_u32() as usize % data.rows.len()];
            let min = if let Value::Int(n) = row.values[2] {
                n
            } else {
                0
            };
            let f = Filter::IntRange {
                column: "size".into(),
                min,
                max: min.saturating_add(100),
            };
            n += measured(&mut times, || store.select("docs", &f, 100))?.len();
        }
        queries.insert("select_int_range".into(), latency(times, n, None));
        Ok(())
    })?;
    let qs = record(&mut phases, "prepare_queries", || {
        data.query_vectors
            .clone()
            .map_or_else(|| make_queries(data), Ok)
    })?;
    let truth = record(&mut phases, "exact_truth", || {
        Ok(data.truth.clone().unwrap_or_else(|| {
            qs.iter()
                .map(|q| exact_top10_metric(&data.vectors, q, None, data.metric))
                .collect::<Vec<_>>()
        }))
    })?;
    for ef in [32, 64, 128, 256] {
        let name = format!("knn_ef{ef}");
        record(&mut phases, &name, || {
            let mut times = Vec::new();
            let mut n = 0;
            let mut r = 0.0;
            for (q, t) in qs.iter().zip(&truth) {
                let hits = measured(&mut times, || store.knn("emb", q, 10, ef))?;
                n += hits.len();
                r += recall(&hits, t);
            }
            queries.insert(name.clone(), latency(times, n, Some(r / qs.len() as f64)));
            Ok(())
        })?;
    }
    let filter = Filter::Eq {
        column: "kind".into(),
        value: Value::Text(kinds[0].clone()),
    };
    let allowed: BTreeSet<_> = data
        .rows
        .iter()
        .filter(|r| r.values[0] == Value::Text(kinds[0].clone()))
        .map(|r| r.id)
        .collect();
    let filtered_truth = record(&mut phases, "exact_filtered_truth", || {
        Ok(qs
            .iter()
            .map(|q| exact_top10_metric(&data.vectors, q, Some(&allowed), data.metric))
            .collect::<Vec<_>>())
    })?;
    record(&mut phases, "knn_filtered_ef64", || {
        let mut times = Vec::new();
        let mut n = 0;
        let mut r = 0.0;
        for (q, t) in qs.iter().zip(&filtered_truth) {
            let hits = measured(&mut times, || {
                store.knn_filtered("emb", q, 10, 64, "docs", &filter)
            })?;
            n += hits.len();
            r += recall(&hits, t);
        }
        queries.insert(
            "knn_filtered_ef64".into(),
            latency(times, n, Some(r / qs.len() as f64)),
        );
        Ok(())
    })?;
    let starts: Vec<_> = (0..500)
        .map(|_| data.rows[rng.next_u32() as usize % data.rows.len()].id)
        .collect();
    for depth in 1..=3 {
        let name = format!("k_hop_depth{depth}");
        record(&mut phases, &name, || {
            let mut times = Vec::new();
            let mut n = 0;
            for start in &starts {
                n += measured(&mut times, || {
                    store.k_hop(*start, None, Direction::Out, depth, 1000)
                })?
                .len();
            }
            queries.insert(name.clone(), latency(times, n, None));
            Ok(())
        })?;
    }
    record(&mut phases, "mixed", || {
        let mut times = Vec::new();
        let mut n = 0;
        for q in &qs {
            n += measured(&mut times, || {
                let hits = store.knn("emb", q, 10, 64)?;
                let mut reached = BTreeSet::new();
                for hit in hits {
                    for r in store.k_hop(hit.id, None, Direction::Out, 2, 50)? {
                        reached.insert(r.id);
                    }
                }
                let mut found = 0;
                for id in reached.into_iter().take(20) {
                    found += usize::from(store.get_row("docs", id)?.is_some());
                }
                Ok(found)
            })?;
        }
        queries.insert("mixed".into(), latency(times, n, None));
        Ok(())
    })?;
    // After every existing measure, so the added work cannot disturb them.
    let question_block = match questions {
        Some(set) => Some(record(&mut phases, "questions", || {
            crate::questions::evaluate(&store, data, set, &[32, 64, 128, 256])
        })?),
        None => None,
    };
    let rows = store.count_rows("docs")?;
    let vectors = store.count_vectors("emb")?;
    let edges = store.count_edges()?;
    if rows != data.rows.len() as u64 || vectors != data.vectors.len() as u64 {
        return Err(bad("reopened counts differ from input"));
    }
    memories.insert("after_queries".into(), memory());
    phases.insert("total".into(), total.elapsed().as_secs_f64());
    let cpu_count = fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count();
    Ok(Report {
        schema_version: 1,
        backend: store.backend_name().into(),
        dataset: data.description.clone(),
        hostname: std::env::var("QUICKDB_HOST")
            .or_else(|_| fs::read_to_string("/etc/hostname"))
            .unwrap_or_else(|_| "unknown".into())
            .trim()
            .into(),
        cpu_count,
        available_parallelism: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        phases_seconds: phases,
        size_bytes,
        memory: memories,
        queries,
        rows,
        vectors,
        edges,
        metric: match data.metric {
            Metric::Cosine => "cosine",
            Metric::L2 => "l2",
            _ => return Err(bad("unsupported metric")),
        },
        query_seed: 0x5155455259,
        questions: question_block,
    })
}
