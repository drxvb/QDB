//! Frozen collect-all BFS oracle versus the public bounded traversal.
use quickdb_api::{Direction, Edge, NodeId, OpenOptions, Reached, Store};
use quickdb_native::NativeStore;
use std::collections::{HashMap, HashSet};

fn old_bfs(
    edges: &HashMap<NodeId, Vec<Edge>>,
    start: NodeId,
    rel: Option<&str>,
    dir: Direction,
    max_depth: u32,
    limit: usize,
) -> Vec<Reached> {
    let mut seen = HashSet::from([start]);
    let mut frontier = vec![start];
    let mut reached = Vec::new();
    for depth in 1..=max_depth {
        if frontier.is_empty() || reached.len() >= limit {
            break;
        }
        let mut next = Vec::new();
        for node in frontier {
            for edge in edges.get(&node).into_iter().flatten() {
                if rel.is_some_and(|r| r != edge.rel) {
                    continue;
                }
                let mut visit = |id| {
                    if seen.insert(id) {
                        next.push(id);
                    }
                };
                if edge.src == node && matches!(dir, Direction::Out | Direction::Both) {
                    visit(edge.dst);
                }
                if edge.dst == node && matches!(dir, Direction::In | Direction::Both) {
                    visit(edge.src);
                }
            }
        }
        next.sort_unstable();
        reached.extend(
            next.iter()
                .take(limit - reached.len())
                .map(|&id| Reached { id, depth }),
        );
        frontier = next;
    }
    reached
}

#[test]
fn bfs_matches_collect_all_on_6000_random_starts() {
    let mut rng = quickdb_api::conformance::Lcg(0x79abcd);
    let mut checks = 0;
    for degree in [3, 18] {
        let dir = tempfile::tempdir().expect("store directory");
        let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
        let mut edges = Vec::new();
        for src in 0..80 {
            for _ in 0..degree {
                edges.push(Edge {
                    src,
                    dst: u64::from(rng.next_u32() % 80),
                    rel: if rng.next_u32().is_multiple_of(2) {
                        "a"
                    } else {
                        "b"
                    }
                    .into(),
                    weight: 1.0,
                });
            }
        }
        s.upsert_edges(&edges).expect("edges");
        let mut adjacency: HashMap<NodeId, Vec<Edge>> = HashMap::new();
        for edge in edges {
            adjacency.entry(edge.src).or_default().push(edge.clone());
            adjacency.entry(edge.dst).or_default().push(edge);
        }
        for _ in 0..3000 {
            let start = u64::from(rng.next_u32() % 100);
            for direction in [Direction::Out, Direction::In, Direction::Both] {
                for rel in [None, Some("a")] {
                    for depth in 1..=4 {
                        for limit in [1, 2, 7, 1000] {
                            let expected = old_bfs(&adjacency, start, rel, direction, depth, limit);
                            let actual = s.k_hop(start, rel, direction, depth, limit).expect("BFS");
                            assert_eq!(actual, expected, "degree={degree} start={start} dir={direction:?} rel={rel:?} depth={depth} limit={limit}");
                            checks += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checks, 576_000);
    println!("BOUND BFS starts=6000 graphs=2 checks={checks}");
}

#[test]
fn bfs_handles_late_smaller_ids_duplicates_and_terminal_frontiers() {
    let dir = tempfile::tempdir().expect("directory");
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
    let triples = [
        (0, "a", 90),
        (0, "b", 2),
        (0, "c", 2),
        (0, "d", 1),
        (1, "a", 0),
        (1, "b", 3),
        (2, "a", 4),
        (90, "a", 5),
        (3, "a", 3),
    ];
    let edges: Vec<_> = triples
        .into_iter()
        .map(|(src, rel, dst)| Edge {
            src,
            rel: rel.into(),
            dst,
            weight: 1.0,
        })
        .collect();
    s.upsert_edges(&edges).expect("edges");
    for limit in [0, 1, 2, 3, 4, 5, 20] {
        let actual = s.k_hop(0, None, Direction::Both, 4, limit).expect("BFS");
        let all = [
            Reached { id: 1, depth: 1 },
            Reached { id: 2, depth: 1 },
            Reached { id: 90, depth: 1 },
            Reached { id: 3, depth: 2 },
            Reached { id: 4, depth: 2 },
            Reached { id: 5, depth: 2 },
        ];
        assert_eq!(actual, all[..limit.min(all.len())]);
    }
    assert!(s
        .k_hop(0, None, Direction::Out, 0, 10)
        .expect("zero depth")
        .is_empty());
    assert!(s.k_hop(u64::MAX, None, Direction::Out, 1, 1).is_err());
}
