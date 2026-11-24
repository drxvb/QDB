//! Arbitrary bytes into the snapshot decoder and validator, as both a full payload
//! (magic, header, graph) and a graph-only payload, then into the loaded graph.
#![no_main]
use libfuzzer_sys::fuzz_target;

#[allow(dead_code, unused_imports)]
#[path = "../../crates/quickdb-native/src/hnsw.rs"]
mod hnsw;

include!("snapshot_common.rs");

fuzz_target!(|data: &[u8]| {
    let seed = data.first().copied().unwrap_or(0);
    if let Ok((_, graph)) = snapshot::decode(data, data.len() as u64) {
        exercise(graph, seed);
    }
    if let Ok(graph) = snapshot::decode_graph(data, data.len() as u64) {
        exercise(graph, seed);
    }
    // A declared length shorter or longer than the bytes present must also fail cleanly.
    let short = data.len().saturating_sub(1 + seed as usize % 16) as u64;
    let _ = snapshot::decode(data, short);
    let _ = snapshot::decode(data, data.len() as u64 + 8);
});
