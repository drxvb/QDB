//! Compare embedded graph layouts on identical data and deterministic queries.
use quickdb_api::{Direction, Edge, OpenOptions, Reached, Store};
use quickdb_surreal::{GraphLayout, SurrealStore};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const NODES: usize = 20_000;
const FANOUT: usize = 5;
const EDGE_COUNT: usize = NODES * FANOUT;
const BATCH_SIZE: usize = 1_000;
const REPEATS: usize = 9;
const STARTS: [u64; 12] = [0, 1, 17, 101, 997, 2048, 4093, 8191, 12007, 16001, 19001, 19999];

struct RunDir(PathBuf);

impl RunDir {
    fn new() -> std::io::Result<Self> {
        let base = std::env::var_os("QUICKDB_DATA").map(PathBuf::from).unwrap_or_else(|| {
            if Path::new("/data").is_dir() { PathBuf::from("/data") } else { std::env::temp_dir() }
        });
        std::fs::create_dir_all(&base)?;
        loop {
            let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
            let path = base.join(format!("quickdb-surreal-graph-layout-{}-{stamp}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for RunDir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            eprintln!("cleanup own graph directory {}: {e}", self.0.display());
        }
    }
}

fn edges() -> Vec<Edge> {
    // Distinct offsets guarantee exactly five unique edges per source, including wraparound cycles.
    let offsets = [1, 7, 31, 127, 509];
    (0..NODES).flat_map(|src| {
        offsets.into_iter().map(move |offset| Edge {
            src: src as u64,
            rel: "link".into(),
            dst: ((src + offset) % NODES) as u64,
            weight: 1.0,
        })
    }).collect()
}

fn expected(start: u64, adjacency: &[Vec<u64>]) -> Vec<Reached> {
    let mut depths = vec![u32::MAX; NODES];
    let mut queue = VecDeque::from([start]);
    depths[start as usize] = 0;
    while let Some(node) = queue.pop_front() {
        let depth = depths[node as usize];
        if depth == 3 { continue; }
        for &next in &adjacency[node as usize] {
            if depths[next as usize] == u32::MAX {
                depths[next as usize] = depth + 1;
                queue.push_back(next);
            }
        }
    }
    let mut reached: Vec<_> = depths.into_iter().enumerate()
        .filter(|(_, depth)| (1..=3).contains(depth))
        .map(|(id, depth)| Reached { id: id as u64, depth }).collect();
    reached.sort_by_key(|r| (r.depth, r.id));
    reached
}

fn query(store: &SurrealStore, start: u64) -> quickdb_api::Result<Vec<Reached>> {
    store.k_hop(start, Some("link"), Direction::Out, 3, NODES)
}

fn sample(store: &SurrealStore) -> quickdb_api::Result<Duration> {
    let began = Instant::now();
    for start in STARTS {
        std::hint::black_box(query(store, start)?);
    }
    Ok(began.elapsed())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = RunDir::new()?;
    let edges = edges();
    assert_eq!(edges.len(), EDGE_COUNT);
    let mut adjacency = vec![Vec::new(); NODES];
    for edge in &edges { adjacency[edge.src as usize].push(edge.dst); }
    let mut plain = SurrealStore::open(&dir.0.join("plain"), &OpenOptions::default())?;
    let mut relation = SurrealStore::open(&dir.0.join("relation"), &OpenOptions::default())?;
    plain.set_graph_layout(GraphLayout::Plain)?;
    relation.set_graph_layout(GraphLayout::Relation)?;
    for (name, store) in [("plain", &mut plain), ("relation", &mut relation)] {
        let began = Instant::now();
        for batch in edges.chunks(BATCH_SIZE) { store.upsert_edges(batch)?; }
        store.flush()?;
        assert_eq!(store.count_edges()?, EDGE_COUNT as u64, "{name} edge count");
        println!("layout={name} edges={EDGE_COUNT} batch_size={BATCH_SIZE} load_flush_ms={:.3}",
            began.elapsed().as_secs_f64() * 1000.0);
    }
    // Both layouts must agree with independent BFS before any traversal timing is collected.
    for start in STARTS {
        let oracle = expected(start, &adjacency);
        let a = query(&plain, start)?;
        let b = query(&relation, start)?;
        assert_eq!(a, oracle, "plain three-hop correctness, start={start}");
        assert_eq!(b, oracle, "relation three-hop correctness, start={start}");
        assert_eq!(a, b, "layout agreement, start={start}");
    }
    println!("correctness=passed starts={} depth=3 direction=out relation=link", STARTS.len());
    let mut plain_times = Vec::with_capacity(REPEATS);
    let mut relation_times = Vec::with_capacity(REPEATS);
    // Alternate order to reduce systematic first/second-run bias, with identical starts every time.
    for repeat in 0..REPEATS {
        if repeat.is_multiple_of(2) {
            plain_times.push(sample(&plain)?);
            relation_times.push(sample(&relation)?);
        } else {
            relation_times.push(sample(&relation)?);
            plain_times.push(sample(&plain)?);
        }
    }
    plain_times.sort_unstable();
    relation_times.sort_unstable();
    let plain_median = plain_times[REPEATS / 2];
    let relation_median = relation_times[REPEATS / 2];
    for (name, median) in [("plain", plain_median), ("relation", relation_median)] {
        println!("layout={name} repeats={REPEATS} starts={} median_batch_ms={:.3} median_per_start_us={:.3}",
            STARTS.len(), median.as_secs_f64() * 1000.0,
            median.as_secs_f64() * 1_000_000.0 / STARTS.len() as f64);
    }
    let winner = match plain_median.cmp(&relation_median) {
        std::cmp::Ordering::Less => "plain",
        std::cmp::Ordering::Greater => "relation",
        std::cmp::Ordering::Equal => "tie",
    };
    println!("winner={winner} basis=median_three_hop_batch no_performance_assertion=true");
    Ok(())
}
