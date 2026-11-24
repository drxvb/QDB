//! Benchmark: 20,000 rows with 256-dimension vectors and 100,000 edges, then a k_hop
//! strategy comparison on a 100,000-node graph. Run: `cargo run -p quickdb-cozo --release --example bench`.

use quickdb_api::conformance::{docs_table, exact_distance, Lcg};
use quickdb_api::{Direction, Edge, Metric, NodeId, OpenOptions, Row, Store, Value, VectorIndexDef};
use quickdb_cozo::{CozoStore, HopStrategy};
use std::time::{Duration, Instant};

const ROWS: u64 = 20_000;
const DIM: usize = 256;
const EDGES: u64 = 100_000;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn row(id: NodeId) -> Row {
    Row {
        id,
        values: vec![
            Value::Text(if id.is_multiple_of(3) { "paper" } else { "note" }.to_string()),
            Value::Text(format!("title {id}")),
            Value::Int(id as i64),
            Value::Float(id as f64 / 7.0),
            Value::Bytes(vec![1, 2, 3]),
        ],
    }
}

fn hop_compare(base: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let nodes = 100_000u64;
    let mut s = CozoStore::open(&base.join("graph100k"), &OpenOptions::default())?;
    let mut rng = Lcg(5);
    let edges: Vec<Edge> = (1..=nodes)
        .flat_map(|src| {
            let a = 1 + rng.next_u32() as u64 % nodes;
            let b = 1 + rng.next_u32() as u64 % nodes;
            [a, b].map(|dst| Edge { src, rel: "r".to_string(), dst, weight: 1.0 })
        })
        .collect();
    let t = Instant::now();
    for c in edges.chunks(20_000) {
        s.upsert_edges(c)?;
    }
    println!("graph100k: {} nodes, {} edges loaded in {:.1} s", nodes, s.count_edges()?, t.elapsed().as_secs_f64());
    let starts: Vec<NodeId> = (0..30).map(|_| 1 + rng.next_u32() as u64 % nodes).collect();
    for strat in [HopStrategy::Frontier, HopStrategy::Datalog] {
        s.hop_strategy = strat;
        let mut times = Vec::new();
        let mut reached = 0usize;
        for &st in &starts {
            let t = Instant::now();
            reached += s.k_hop(st, None, Direction::Out, 3, usize::MAX)?.len();
            times.push(t.elapsed());
        }
        println!("graph100k k_hop depth 3 {strat:?}: median {:.3} ms, mean reached {}", ms(median(times)), reached / starts.len());
    }
    s.hop_strategy = HopStrategy::Frontier;
    for &st in &starts[..5] {
        let a = s.k_hop(st, None, Direction::Both, 3, usize::MAX)?;
        s.hop_strategy = HopStrategy::Datalog;
        let b = s.k_hop(st, None, Direction::Both, 3, usize::MAX)?;
        s.hop_strategy = HopStrategy::Frontier;
        assert_eq!(a, b, "strategies disagree");
    }
    println!("graph100k: both strategies return identical results (5 starts, Both, depth 3)");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::var_os("QUICKDB_DATA").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir).join("cozo-bench");
    let _ = std::fs::remove_dir_all(&base);
    let dir = base.join("main");
    let mut rng = Lcg(42);
    let vecs: Vec<Vec<f32>> = (0..ROWS).map(|_| rng.vector(DIM)).collect();
    let edges: Vec<Edge> = (0..EDGES)
        .map(|i| Edge { src: 1 + i % ROWS, rel: if i % 2 == 0 { "a" } else { "b" }.to_string(), dst: 1 + rng.next_u32() as u64 % ROWS, weight: 1.0 })
        .collect();

    let def = VectorIndexDef { name: "emb".to_string(), dim: DIM, metric: Metric::Cosine, m: 16, ef_construction: 200 };
    let t0 = Instant::now();
    {
        let mut s = CozoStore::open(&dir, &OpenOptions::default())?;
        s.define_table(&docs_table())?;
        s.define_vector_index(&def)?;
        let t = Instant::now();
        let rows: Vec<Row> = (1..=ROWS).map(row).collect();
        for c in rows.chunks(5_000) {
            s.upsert_rows("docs", c)?;
        }
        let tr = t.elapsed();
        let t = Instant::now();
        let items: Vec<(NodeId, &[f32])> = vecs.iter().enumerate().map(|(i, v)| (i as NodeId + 1, v.as_slice())).collect();
        for c in items.chunks(1_000) {
            s.upsert_vectors("emb", c)?;
        }
        let tv = t.elapsed();
        let t = Instant::now();
        for c in edges.chunks(20_000) {
            s.upsert_edges(c)?;
        }
        let te = t.elapsed();
        s.flush()?;
        println!(
            "load: rows {:.2} s, vectors+HNSW {:.2} s, edges {:.2} s, total {:.2} s",
            tr.as_secs_f64(), tv.as_secs_f64(), te.as_secs_f64(), t0.elapsed().as_secs_f64()
        );
        println!("counts: rows {} vectors {} edges {}", s.count_rows("docs")?, s.count_vectors("emb")?, s.count_edges()?);
    }
    println!("size on disk: {:.1} MB", quickdb_api::dir_size(&dir)? as f64 / 1e6);

    let t = Instant::now();
    let s = CozoStore::open(&dir, &OpenOptions { create: false, sync_each_write: false })?;
    let reopen = t.elapsed();
    let t = Instant::now();
    s.knn("emb", &vecs[0], 1, 64)?;
    println!("reopen: {:.2} ms (first knn after reopen {:.2} ms)", ms(reopen), ms(t.elapsed()));

    let queries: Vec<Vec<f32>> = (0..100).map(|_| rng.vector(DIM)).collect();
    let mut times = Vec::new();
    let (mut found, mut wanted) = (0usize, 0usize);
    for q in &queries {
        let t = Instant::now();
        let hits = s.knn("emb", q, 10, 64)?;
        times.push(t.elapsed());
        let mut truth: Vec<(f32, NodeId)> = vecs.iter().enumerate().map(|(i, v)| (exact_distance(Metric::Cosine, q, v), i as NodeId + 1)).collect();
        truth.sort_by(|a, b| a.0.total_cmp(&b.0));
        wanted += 10;
        found += truth[..10].iter().filter(|(_, id)| hits.iter().any(|h| h.id == *id)).count();
    }
    println!("knn k=10 ef_search=64: median {:.3} ms, recall@10 {:.3}", ms(median(times)), found as f64 / wanted as f64);
    for ef in [128usize, 256, 512] {
        let mut times = Vec::new();
        let (mut found, mut wanted) = (0usize, 0usize);
        for q in queries.iter().take(30) {
            let t = Instant::now();
            let hits = s.knn("emb", q, 10, ef)?;
            times.push(t.elapsed());
            let mut truth: Vec<(f32, NodeId)> = vecs.iter().enumerate().map(|(i, v)| (exact_distance(Metric::Cosine, q, v), i as NodeId + 1)).collect();
            truth.sort_by(|a, b| a.0.total_cmp(&b.0));
            wanted += 10;
            found += truth[..10].iter().filter(|(_, id)| hits.iter().any(|h| h.id == *id)).count();
        }
        println!("knn k=10 ef_search={ef}: median {:.3} ms, recall@10 {:.3} (30 queries)", ms(median(times)), found as f64 / wanted as f64);
    }

    let mut times = Vec::new();
    for i in 0..1000u64 {
        let id = 1 + (i * 7919) % ROWS;
        let t = Instant::now();
        let r = s.get_row("docs", id)?;
        times.push(t.elapsed());
        assert!(r.is_some());
    }
    println!("get_row: median {:.3} ms", ms(median(times)));

    let mut times = Vec::new();
    let mut reached = 0;
    for i in 0..50u64 {
        let t = Instant::now();
        reached += s.k_hop(1 + (i * 397) % ROWS, None, Direction::Out, 3, usize::MAX)?.len();
        times.push(t.elapsed());
    }
    println!("k_hop depth 3 (Out, default Datalog): median {:.3} ms, mean reached {}", ms(median(times)), reached / 50);
    drop(s);

    hop_compare(&base)?;
    let _ = std::fs::remove_dir_all(&base);
    Ok(())
}
