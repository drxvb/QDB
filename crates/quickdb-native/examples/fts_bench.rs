//! Latency of `text_search` and `hybrid_search` over synthetic short texts.
//!
//! Run with `cargo run --release --example fts_bench -p quickdb-native [docs] [queries]`
//! (defaults: 22,000 texts, 1,000 queries). Texts are 6 to 15 words drawn from a
//! 3,000-word synthetic vocabulary with a skewed distribution; each text also has a
//! 64-dimension vector. Every query is two words taken from one stored text, so it has at
//! least one hit. The store lives in a temporary directory under `QUICKDB_DATA` (or the
//! system temporary directory).

use quickdb_api::conformance::Lcg;
use quickdb_api::VectorIndexDef;
use quickdb_api::{ColumnDef, ColumnType, Metric, OpenOptions, Row, Store, TableDef, Value};
use quickdb_native::{NativeStore, TextIndexDef};
use std::time::Instant;

const VOCABULARY: usize = 3_000;
const DIM: usize = 64;

fn word(rng: &mut Lcg) -> String {
    let a = rng.next_u32() as usize % VOCABULARY;
    let b = rng.next_u32() as usize % VOCABULARY;
    let syllables = ["ka", "lo", "mi", "ne", "su", "ta", "ri", "vo", "de", "pa"];
    let n = a.min(b);
    format!(
        "{}{}{}{n}",
        syllables[n % 10],
        syllables[n / 10 % 10],
        syllables[n / 100 % 10]
    )
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[((samples.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let docs: u64 = args.next().map_or(Ok(22_000), |a| a.parse())?;
    let queries: usize = args.next().map_or(Ok(1_000), |a| a.parse())?;
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::tempdir_in(root)?;
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default())?;
    s.define_table(&TableDef {
        name: "notes".into(),
        columns: vec![ColumnDef {
            name: "body".into(),
            ty: ColumnType::Text,
            indexed: false,
        }],
    })?;
    s.define_text_index(&TextIndexDef {
        name: "notes_text".into(),
        table: "notes".into(),
        columns: vec!["body".into()],
    })?;
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: DIM,
        metric: Metric::Cosine,
        m: 16,
        ef_construction: 100,
    })?;
    let mut rng = Lcg(0x05EE_DF75);
    let mut texts = Vec::with_capacity(docs as usize);
    let rows: Vec<Row> = (0..docs)
        .map(|id| {
            let n = 6 + rng.next_u32() as usize % 10;
            let body: Vec<String> = (0..n).map(|_| word(&mut rng)).collect();
            texts.push(body.clone());
            Row {
                id,
                values: vec![Value::Text(body.join(" "))],
            }
        })
        .collect();
    let vectors: Vec<Vec<f32>> = (0..docs)
        .map(|_| (0..DIM).map(|_| rng.next_f32()).collect())
        .collect();
    let started = Instant::now();
    for chunk in rows.chunks(1_000) {
        s.upsert_rows("notes", chunk)?;
    }
    let rows_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let items: Vec<_> = vectors
        .iter()
        .enumerate()
        .map(|(id, v)| (id as u64, v.as_slice()))
        .collect();
    s.upsert_vectors("emb", &items)?;
    s.flush()?;
    let vectors_s = started.elapsed().as_secs_f64();

    let mut cases = Vec::with_capacity(queries);
    for _ in 0..queries {
        let doc = rng.next_u32() as usize % texts.len();
        let words = &texts[doc];
        let a = &words[rng.next_u32() as usize % words.len()];
        let b = &words[rng.next_u32() as usize % words.len()];
        let vector: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        cases.push((format!("{a} {b}"), vector));
    }
    let mut text_us = Vec::with_capacity(queries);
    let mut hybrid_us = Vec::with_capacity(queries);
    let mut text_hits = 0;
    for (text, vector) in &cases {
        let t = Instant::now();
        let hits = s.text_search("notes_text", text, 10)?;
        text_us.push(t.elapsed().as_secs_f64() * 1e6);
        assert!(!hits.is_empty(), "query {text:?} found nothing");
        text_hits += hits.len();
        let t = Instant::now();
        let fused = s.hybrid_search("notes_text", "emb", text, vector, 10, 64)?;
        hybrid_us.push(t.elapsed().as_secs_f64() * 1e6);
        assert_eq!(fused.len(), 10);
    }
    println!(
        "FTS_BENCH docs={docs} queries={queries} k=10 ef_search=64 dim={DIM} rows_and_text_index_s={rows_s:.2} vectors_and_flush_s={vectors_s:.2} mean_text_hits={:.2}",
        text_hits as f64 / queries as f64
    );
    println!(
        "FTS_BENCH text_search_p50_us={:.1} p95_us={:.1} hybrid_search_p50_us={:.1} p95_us={:.1}",
        percentile(&mut text_us, 0.5),
        percentile(&mut text_us, 0.95),
        percentile(&mut hybrid_us, 0.5),
        percentile(&mut hybrid_us, 0.95)
    );
    Ok(())
}
