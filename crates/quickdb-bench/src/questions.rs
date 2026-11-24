//! Written questions against a corpus: document-level relevance reported next to exact recall.
//!
//! Question text is never read here. Only ids, language, corpus folder, expected document ids and the
//! precomputed question vectors are used, and error messages carry file names and line numbers only.
use crate::{exact_top10_metric, Dataset};
use quickdb_api::{Error, NodeId, Result, Store};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Instant;

fn bad(s: impl Into<String>) -> Error {
    Error::Schema(s.into())
}

#[derive(Deserialize)]
struct Record {
    id: String,
    lang: String,
    corpus: String,
    expected_doc_ids: Vec<NodeId>,
}
#[derive(Deserialize)]
struct DocRef {
    id: NodeId,
}
#[derive(Deserialize)]
struct ChunkRef {
    id: NodeId,
    doc_id: NodeId,
}

/// Calls `each` for every non-blank line. A malformed line is reported by number, never by content.
fn scan<T: serde::de::DeserializeOwned>(
    path: &Path,
    mut each: impl FnMut(T) -> Result<()>,
) -> Result<()> {
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    for (i, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let item = serde_json::from_str(&line)
            .map_err(|_| bad(format!("{name}:{}: malformed record", i + 1)))?;
        each(item)?;
    }
    Ok(())
}

pub struct Question {
    pub id: String,
    pub lang: String,
    pub corpus: String,
    /// Expected document ids that exist in `docs.jsonl`. Empty means the question is unresolved.
    pub expected: BTreeSet<NodeId>,
    pub vector: Vec<f32>,
}

pub struct QuestionSet {
    pub items: Vec<Question>,
    chunk_doc: BTreeMap<NodeId, NodeId>,
}

impl QuestionSet {
    /// `corpus_dir` holds `docs.jsonl` and `chunks.jsonl`; `questions_dir` holds `questions.jsonl`,
    /// `qvectors.f32` (little-endian, one row per question in file order) and optionally `qmeta.json`.
    pub fn load(corpus_dir: &Path, questions_dir: &Path, dim: usize) -> Result<Self> {
        let mut docs = BTreeSet::new();
        scan(&corpus_dir.join("docs.jsonl"), |d: DocRef| {
            docs.insert(d.id);
            Ok(())
        })?;
        let mut chunk_doc = BTreeMap::new();
        scan(&corpus_dir.join("chunks.jsonl"), |c: ChunkRef| {
            chunk_doc.insert(c.id, c.doc_id);
            Ok(())
        })?;
        let mut items = Vec::new();
        let mut seen = BTreeSet::new();
        scan(&questions_dir.join("questions.jsonl"), |r: Record| {
            if !seen.insert(r.id.clone()) {
                return Err(bad("duplicate question id"));
            }
            items.push(Question {
                id: r.id,
                lang: r.lang,
                corpus: r.corpus,
                expected: r
                    .expected_doc_ids
                    .into_iter()
                    .filter(|d| docs.contains(d))
                    .collect(),
                vector: Vec::new(),
            });
            Ok(())
        })?;
        if items.is_empty() {
            return Err(bad("questions.jsonl has no questions"));
        }
        if let Ok(file) = File::open(questions_dir.join("qmeta.json")) {
            let meta: serde_json::Value =
                serde_json::from_reader(file).map_err(|_| bad("qmeta.json is not valid JSON"))?;
            if meta["count"].as_u64() != Some(items.len() as u64)
                || meta["dim"].as_u64() != Some(dim as u64)
            {
                return Err(bad(
                    "qmeta.json count or dim differs from questions and corpus",
                ));
            }
        }
        let path = questions_dir.join("qvectors.f32");
        let want = items
            .len()
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| bad("question vector size overflow"))?;
        let mut bytes = Vec::with_capacity(want);
        File::open(&path)?
            .take(want as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() != want {
            return Err(bad(
                "qvectors.f32 byte length differs from question count times dim",
            ));
        }
        for (q, row) in items.iter_mut().zip(bytes.chunks_exact(dim * 4)) {
            q.vector = row
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            if q.vector.iter().any(|v| !v.is_finite()) {
                return Err(bad("non-finite question vector"));
            }
        }
        Ok(Self { items, chunk_doc })
    }
}

/// `(any expected document among the first ten chunks, reciprocal rank of the first expected document)`.
/// Rank counts distinct documents in order of first appearance, so a document's second chunk does not push
/// the next document down.
fn score(ids: &[NodeId], q: &Question, chunk_doc: &BTreeMap<NodeId, NodeId>) -> (bool, f64) {
    let mut seen = BTreeSet::new();
    for id in ids.iter().take(10) {
        let Some(doc) = chunk_doc.get(id) else {
            continue;
        };
        if !seen.insert(*doc) {
            continue;
        }
        if q.expected.contains(doc) {
            return (true, 1.0 / seen.len() as f64);
        }
    }
    (false, 0.0)
}

#[derive(Default)]
struct Tally {
    count: usize,
    hits: usize,
    rr: f64,
    recall: f64,
}
impl Tally {
    fn add(&mut self, hit: bool, rr: f64, recall: f64) {
        self.count += 1;
        self.hits += usize::from(hit);
        self.rr += rr;
        self.recall += recall;
    }
    fn json(&self) -> serde_json::Value {
        let rate = |sum: f64| (self.count > 0).then(|| sum / self.count as f64);
        serde_json::json!({
            "count": self.count,
            "doc_hit_at_10": rate(self.hits as f64),
            "mrr_at_10": rate(self.rr),
            "recall_at_10_vs_exact": rate(self.recall),
        })
    }
}

fn p50(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times
        .get((times.len() * 50).div_ceil(100).saturating_sub(1))
        .copied()
        .unwrap_or(0.0)
}

fn recall(ids: &[NodeId], truth: &[NodeId]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let got: BTreeSet<_> = ids.iter().take(10).collect();
    truth.iter().filter(|id| got.contains(id)).count() as f64 / truth.len() as f64
}

/// Scores every resolved question against `store.knn` at each `ef` and against brute-force exact search.
/// The exact entry is the reference, so its `recall_at_10_vs_exact` is 1 by construction.
pub fn evaluate<S: Store>(
    store: &S,
    data: &Dataset,
    set: &QuestionSet,
    efs: &[usize],
) -> Result<serde_json::Value> {
    let resolved: Vec<&Question> = set
        .items
        .iter()
        .filter(|q| !q.expected.is_empty())
        .collect();
    let mut truth = Vec::with_capacity(resolved.len());
    let mut exact_times = Vec::with_capacity(resolved.len());
    for q in &resolved {
        let start = Instant::now();
        let ids = std::hint::black_box(exact_top10_metric(
            &data.vectors,
            &q.vector,
            None,
            data.metric,
        ));
        exact_times.push(start.elapsed().as_secs_f64() * 1e6);
        truth.push(ids);
    }
    let mut configs: Vec<(String, Vec<Vec<NodeId>>, Vec<f64>)> =
        vec![("exact".into(), truth.clone(), exact_times)];
    for &ef in efs {
        let mut times = Vec::with_capacity(resolved.len());
        let mut all = Vec::with_capacity(resolved.len());
        for q in &resolved {
            let start = Instant::now();
            let hits = std::hint::black_box(store.knn("emb", &q.vector, 10, ef)?);
            times.push(start.elapsed().as_secs_f64() * 1e6);
            all.push(hits.iter().map(|h| h.id).collect());
        }
        configs.push((format!("knn_ef{ef}"), all, times));
    }
    let mut results = BTreeMap::new();
    for (name, found, times) in configs {
        let mut total = Tally::default();
        let mut by_lang: BTreeMap<&str, Tally> = BTreeMap::new();
        let mut by_corpus: BTreeMap<&str, Tally> = BTreeMap::new();
        for ((q, ids), t) in resolved.iter().zip(&found).zip(&truth) {
            let (hit, rr) = score(ids, q, &set.chunk_doc);
            let r = recall(ids, t);
            total.add(hit, rr, r);
            by_lang.entry(&q.lang).or_default().add(hit, rr, r);
            by_corpus.entry(&q.corpus).or_default().add(hit, rr, r);
        }
        let group = |m: &BTreeMap<&str, Tally>| -> serde_json::Value {
            m.iter()
                .map(|(k, v)| ((*k).to_string(), v.json()))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        let mut entry = total.json();
        entry["p50_us"] = p50(times).into();
        entry["by_lang"] = group(&by_lang);
        entry["by_corpus"] = group(&by_corpus);
        results.insert(name, entry);
    }
    let mut lang: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut corpus: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for q in &set.items {
        let unresolved = usize::from(q.expected.is_empty());
        for (map, key) in [(&mut lang, &q.lang), (&mut corpus, &q.corpus)] {
            let e = map.entry(key.as_str()).or_default();
            e.0 += 1 - unresolved;
            e.1 += unresolved;
        }
    }
    let counts = |m: &BTreeMap<&str, (usize, usize)>| -> serde_json::Value {
        m.iter()
            .map(|(k, (n, u))| {
                (
                    (*k).to_string(),
                    serde_json::json!({"count": n, "unresolved": u}),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    Ok(serde_json::json!({
        "schema_version": 1,
        "count": resolved.len(),
        "unresolved": set.items.len() - resolved.len(),
        "by_lang": counts(&lang),
        "by_corpus": counts(&corpus),
        "results": results,
    }))
}
