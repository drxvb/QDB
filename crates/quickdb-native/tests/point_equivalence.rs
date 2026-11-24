//! Deterministic differential coverage for traversal dispatch and both engines.
use quickdb_api::conformance::Lcg;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;

struct Fixture {
    store: NativeStore,
    path: PathBuf,
    edges: Vec<Edge>,
}

impl Fixture {
    fn new(nodes: u64, edge_count: usize, corpus: bool) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "quickdb-point-equivalence-{}-{nodes}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| Error::Backend(e.to_string()))?
                .as_nanos()
        ));
        let mut store = NativeStore::open(&path, &OpenOptions::default())?;
        let mut unique = BTreeMap::new();
        // Explicit self-loop, cycle, parallel relations and replacement writes.
        for (src, rel, dst) in [
            (0, "link", 0),
            (0, "link", 1),
            (1, "link", 2),
            (2, "link", 0),
            (0, "cite", 1),
        ] {
            unique.insert((src, rel.to_string(), dst), 1.0f32);
        }
        let mut rng = Lcg(if corpus { 9271 } else { 12881 });
        while unique.len() < edge_count {
            let src = u64::from(rng.next_u32()) % nodes;
            // Corpus-like documents have local links inside small clusters.
            let dst = if corpus {
                let base = src / 4 * 4;
                base + u64::from(rng.next_u32()) % (nodes - base).min(4)
            } else {
                u64::from(rng.next_u32()) % nodes
            };
            let rel = ["link", "cite", "related"][rng.next_u32() as usize % 3];
            unique.insert((src, rel.to_string(), dst), rng.next_f32());
        }
        let edges: Vec<_> = unique
            .into_iter()
            .map(|((src, rel, dst), weight)| Edge {
                src,
                rel,
                dst,
                weight,
            })
            .collect();
        for chunk in edges.chunks(5000) {
            store.upsert_edges(chunk)?;
        }
        let replacements: Vec<_> = edges
            .iter()
            .take(32)
            .map(|e| Edge {
                weight: 0.25,
                ..e.clone()
            })
            .collect();
        store.upsert_edges(&replacements)?;
        assert_eq!(store.count_edges()?, edge_count as u64);
        Ok(Self { store, path, edges })
    }
}

fn oracle(
    edges: &[Edge],
    start: u64,
    rel: Option<&str>,
    dir: Direction,
    depth: u32,
    limit: usize,
) -> Vec<Reached> {
    let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
    for e in edges {
        if rel.is_some_and(|r| r != e.rel) {
            continue;
        }
        if matches!(dir, Direction::Out | Direction::Both) {
            adjacency.entry(e.src).or_default().push(e.dst);
        }
        if matches!(dir, Direction::In | Direction::Both) {
            adjacency.entry(e.dst).or_default().push(e.src);
        }
    }
    let mut seen = HashSet::from([start]);
    let mut queue = VecDeque::from([(start, 0)]);
    let mut result = Vec::new();
    while let Some((node, hops)) = queue.pop_front() {
        if hops == depth {
            continue;
        }
        if let Some(next) = adjacency.get(&node) {
            for &id in next {
                if seen.insert(id) {
                    result.push(Reached {
                        id,
                        depth: hops + 1,
                    });
                    queue.push_back((id, hops + 1));
                }
            }
        }
    }
    result.sort_unstable_by_key(|r| (r.depth, r.id));
    result.truncate(limit);
    result
}

fn check_graph(nodes: u64, edges: usize, corpus: bool) -> Result<()> {
    let fixture = Fixture::new(nodes, edges, corpus)?;
    let store = &fixture.store;
    let mut rng = Lcg(0x4551554956);
    let limits = [0, 1, 2, 5, 17, 64, usize::MAX];
    for sample in 0..2000 {
        let start = u64::from(rng.next_u32()) % nodes;
        for dir in [Direction::Out, Direction::In, Direction::Both] {
            for rel in [None, Some("link")] {
                for depth in 1..=3 {
                    // Rotate limits without multiplying an already large matrix.
                    let limit = limits[(sample + depth as usize) % limits.len()];
                    let sql = store.k_hop_sql(start, rel, dir, depth, limit)?;
                    let bfs = store.k_hop_bfs(start, rel, dir, depth, limit)?;
                    let dispatched = store.k_hop(start, rel, dir, depth, limit)?;
                    assert_eq!(sql, bfs, "nodes={nodes} sample={sample} start={start} dir={dir:?} rel={rel:?} depth={depth} limit={limit}");
                    assert_eq!(sql, dispatched, "dispatch nodes={nodes} sample={sample} start={start} dir={dir:?} rel={rel:?} depth={depth} limit={limit}");
                    if sample < 2 {
                        assert_eq!(sql, oracle(&fixture.edges, start, rel, dir, depth, limit));
                    }
                }
            }
        }
    }
    for start in [0, nodes - 1, nodes + 17, i64::MAX as u64] {
        for dir in [Direction::Out, Direction::In, Direction::Both] {
            for rel in [None, Some("link"), Some("missing_relation")] {
                for (depth, limit) in [(0, 0), (0, 10), (3, 0), (1, 1), (3, 64)] {
                    let sql = store.k_hop_sql(start, rel, dir, depth, limit)?;
                    assert_eq!(sql, store.k_hop_bfs(start, rel, dir, depth, limit)?);
                    assert_eq!(sql, store.k_hop(start, rel, dir, depth, limit)?);
                    assert_eq!(sql, oracle(&fixture.edges, start, rel, dir, depth, limit));
                }
            }
        }
    }
    for invalid in [i64::MAX as u64 + 1, u64::MAX] {
        for (depth, limit) in [(0, 0), (0, 10), (3, 0), (3, 10)] {
            for dir in [Direction::Out, Direction::In, Direction::Both] {
                for rel in [None, Some("link")] {
                    assert!(matches!(
                        store.k_hop_sql(invalid, rel, dir, depth, limit),
                        Err(Error::Schema(_))
                    ));
                    assert!(matches!(
                        store.k_hop_bfs(invalid, rel, dir, depth, limit),
                        Err(Error::Schema(_))
                    ));
                    assert!(matches!(
                        store.k_hop(invalid, rel, dir, depth, limit),
                        Err(Error::Schema(_))
                    ));
                }
            }
        }
    }
    // Explicitly release SQLite handles before cleanup on Windows as well.
    let Fixture {
        store,
        path,
        edges: _,
    } = fixture;
    drop(store);
    std::fs::remove_dir_all(path)?;
    println!("POINT_EQUIVALENCE nodes={nodes} edges={edges} starts=2000 directions=3 relations=2 depths=3 cases=36000 varying_limits=7 identical=true");
    Ok(())
}

#[test]
fn corpus_traversal_equivalence_2000_starts() -> Result<()> {
    check_graph(22_666, 45_000, true)
}

#[test]
fn dense_traversal_equivalence_2000_starts() -> Result<()> {
    check_graph(10_000, 50_000, false)
}
