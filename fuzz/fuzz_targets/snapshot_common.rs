// Shared by the snapshot targets through `include!`. The including file declares `mod hnsw`
// from the crate source, so decode, validation and graph code are the exact production code.
use hnsw::{snapshot, Hnsw};
use quickdb_api::Metric;

/// Largest parameters the store itself accepts for an index definition; a snapshot is only
/// loaded when it matches a stored definition, so larger values never reach search or insert.
const MAX_M: usize = 4096;
const MAX_DIM: usize = 1024;
const MAX_EF: usize = 4096;

fn check_hits(hits: &[quickdb_api::Hit], k: usize) {
    assert!(hits.len() <= k, "more hits than k");
    assert!(
        hits.windows(2)
            .all(|w| w[0].distance.total_cmp(&w[1].distance).is_le()),
        "hits not sorted by distance"
    );
}

/// Round-trips an accepted graph, loads it and runs searches and an insert on it.
fn exercise(graph: snapshot::Graph, seed: u8) {
    let mut encoded = Vec::new();
    snapshot::encode_graph(&mut encoded, &graph.parts(), graph.active.iter().copied())
        .expect("encode to memory");
    let again =
        snapshot::decode_graph(encoded.as_slice(), encoded.len() as u64).expect("re-decode");
    assert_eq!(again, graph, "encode/decode round trip changed an accepted graph");

    let (dim, cosine, m, ef) = (graph.dim, graph.cosine, graph.m, graph.ef);
    let metric = if cosine { Metric::Cosine } else { Metric::L2 };
    let Ok(mut h) = Hnsw::from_snapshot(graph) else {
        return;
    };
    // The store accepts a loaded graph only when it matches the stored definition.
    if !h.valid(dim, metric, m, ef) || m > MAX_M || dim > MAX_DIM || ef > MAX_EF {
        return;
    }
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 + seed as f32) * 0.25).collect();
    let k = 1 + seed as usize % 8;
    let search_ef = 1 + seed as usize % 64;
    check_hits(&h.search(&query, k, search_ef), k);
    let (hits, _) = h.search_filtered(&query, k, search_ef, 10_000, 8, |_, accept| {
        accept.iter_mut().for_each(|a| *a = true)
    });
    check_hits(&hits, k);
    let id = (0..=h.capacity() as u64)
        .find(|id| h.node_of(*id).is_none())
        .expect("free id");
    h.insert(id, &query);
    assert!(h.node_of(id).is_some(), "inserted id missing");
    check_hits(&h.search(&query, k, search_ef), k);
    let mut out = Vec::new();
    h.encode_snapshot(&mut out).expect("encode loaded graph");
    snapshot::decode_graph(out.as_slice(), out.len() as u64)
        .expect("a graph written after insert must decode");
}
