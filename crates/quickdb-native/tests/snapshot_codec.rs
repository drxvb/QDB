use quickdb_api::{Metric, OpenOptions, Store, VectorIndexDef};
use quickdb_native::NativeStore;
use sha2::{Digest, Sha256};
use std::io;

#[allow(dead_code)]
#[path = "../src/snapshot.rs"]
mod snapshot;

const FIXTURES: [(&str, &[u8], &str); 4] = [
    (
        "small",
        include_bytes!("fixtures/snap/small.bin"),
        "effb181c432c7cabcc3641a4326e0ce26951278a5f8f8128303bfd165f779f0f",
    ),
    (
        "deletions",
        include_bytes!("fixtures/snap/deletions.bin"),
        "6dad153b57372b3043ea3297a2e900a6676434614d5843d8732a98c5099a6dac",
    ),
    (
        "empty",
        include_bytes!("fixtures/snap/empty.bin"),
        "78e29220d95a2f4d865373cdc0fc5dfb83612075691a313a9571114c1f11121b",
    ),
    (
        "cosine",
        include_bytes!("fixtures/snap/cosine.bin"),
        "72dd134bb3f4fb8948316666b3f054c9a278f90c2259f597e8ee7f3c641ebbfb",
    ),
];

// The fixtures are the old-format oracle: their SHA-256 values were pinned from bytes written
// by the original bincode 1 encoder before the codec existed (commit 716bee7).
fn checked_decode(bytes: &[u8]) -> io::Result<(snapshot::Header, snapshot::Graph)> {
    let length = bytes
        .len()
        .checked_sub(4)
        .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?;
    let expected = u32::from_le_bytes(bytes[length..].try_into().expect("four checksum bytes"));
    if crc32fast::hash(&bytes[..length]) != expected {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "checksum"));
    }
    snapshot::decode(&bytes[..length], length as u64)
}
fn checksum(mut payload: Vec<u8>) -> Vec<u8> {
    payload.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    payload
}
fn encode(header: snapshot::Header, graph: &snapshot::Graph) -> Vec<u8> {
    let mut payload = Vec::new();
    snapshot::encode_header(&mut payload, header).expect("encode header");
    snapshot::encode_graph(&mut payload, &graph.parts(), graph.active.iter().copied())
        .expect("encode graph");
    checksum(payload)
}

#[test]
fn original_fixture_digest_structure_and_byte_identity() {
    for (name, bytes, expected) in FIXTURES {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
        let (header, graph) = checked_decode(bytes).expect("original fixture decodes");
        // Fixed offsets of the original layout, read without the codec.
        let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("word"));
        assert_eq!(word(0), snapshot::MAGIC);
        assert_eq!(header.0[..], bytes[8..24]);
        assert_eq!(header.1, word(24));
        assert_eq!(header.2, word(32) as i64);
        assert_eq!(graph.revision, word(40));
        assert_eq!(graph.dim as u64, word(48));
        assert_eq!(u8::from(graph.cosine), bytes[56]);
        assert_eq!((graph.m as u64, graph.ef as u64), (word(57), word(65)));
        assert_eq!(graph.data.len() as u64, word(73));
        assert_eq!(graph.top as u64, word(bytes.len() - 12));
        let encoded = encode(header, &graph);
        if let Some(offset) = bytes.iter().zip(&encoded).position(|(a, b)| a != b) {
            panic!("first differing offset={offset} fixture={name}");
        }
        assert_eq!(encoded, bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&encoded)), expected);
        let expected_nodes = match name {
            "small" => 6,
            "deletions" => 9,
            "empty" => 0,
            "cosine" => 5,
            _ => unreachable!(),
        };
        assert_eq!(graph.ids.len(), expected_nodes);
        assert_eq!(
            graph.active.len(),
            if name == "deletions" {
                8
            } else {
                expected_nodes
            }
        );
        assert_eq!(graph.cosine, name == "cosine");
        println!(
            "SNAP_IDENTITY fixture={name} sha256={expected} bytes={} identical=true structure=true",
            bytes.len()
        );
    }
}

fn corrupted(bytes: &[u8], kind: &str) -> Vec<u8> {
    let mut payload = bytes[..bytes.len() - 4].to_vec();
    match kind {
        "truncation" => {
            payload.pop();
        }
        "huge_length" => payload[73..81].copy_from_slice(&u64::MAX.to_le_bytes()),
        "unknown_tag" => payload[56] = 2,
        "trailing" => payload.push(0),
        "level" => {
            let length = payload.len();
            payload[length - 8..].copy_from_slice(&25u64.to_le_bytes());
        }
        "neighbor" | "duplicate_key" => {
            let (header, mut graph) = checked_decode(bytes).expect("valid seed");
            if kind == "neighbor" {
                graph.links[0][0].push(graph.ids.len());
            } else {
                graph.active[1] = graph.active[0];
            }
            return encode(header, &graph);
        }
        _ => unreachable!(),
    }
    checksum(payload)
}

#[test]
fn recomputed_checksum_corruptions_reach_decoder_and_rebuild_cleanly() {
    for kind in [
        "truncation",
        "huge_length",
        "unknown_tag",
        "trailing",
        "level",
        "neighbor",
        "duplicate_key",
    ] {
        let dir = tempfile::tempdir().expect("directory");
        let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).expect("store");
        store
            .define_vector_index(&VectorIndexDef {
                name: "emb".into(),
                dim: 3,
                metric: Metric::L2,
                m: 4,
                ef_construction: 24,
            })
            .expect("index");
        store
            .upsert_vectors("emb", &[(1, &[0.0, 1.0, 2.0]), (2, &[3.0, 2.0, 1.0])])
            .expect("vectors");
        store.flush().expect("flush");
        drop(store);
        let path = dir.path().join("emb.hnsw");
        let original = std::fs::read(&path).expect("snapshot");
        let mutation = corrupted(&original, kind);
        let n = mutation.len() - 4;
        assert_eq!(crc32fast::hash(&mutation[..n]).to_le_bytes(), mutation[n..]);
        assert!(std::panic::catch_unwind(|| checked_decode(&mutation))
            .expect("no decoder panic")
            .is_err());
        std::fs::write(&path, mutation).expect("install corruption");
        let store = NativeStore::open(dir.path(), &OpenOptions::default()).expect("clean rebuild");
        assert_eq!(store.index_rebuilds(), 1);
        assert_eq!(
            store.knn("emb", &[0.0, 1.0, 2.0], 1, 24).expect("search")[0].id,
            1
        );
        println!("SNAP_CORRUPTION kind={kind} checksum_valid=true decoder_error=true rebuilt=1 panic=false");
    }
}

#[test]
fn every_fixture_truncation_and_invalid_option_tag_is_an_error() {
    for (_, bytes, _) in FIXTURES {
        for n in 0..bytes.len() - 4 {
            let mutation = checksum(bytes[..n].to_vec());
            assert!(checked_decode(&mutation).is_err());
        }
        let mut payload = bytes[..bytes.len() - 4].to_vec();
        let (_, graph) = checked_decode(bytes).expect("fixture");
        let option_offset = payload.len() - 8 - if graph.entry.is_some() { 9 } else { 1 };
        payload[option_offset] = 2;
        assert!(checked_decode(&checksum(payload)).is_err());
    }
}

#[test]
fn seeded_decoder_mutation_fuzz() {
    let Some(seconds) = std::env::var("QUICKDB_SNAPSHOT_FUZZ_SECONDS").ok() else {
        return;
    };
    let seconds: u64 = seconds.parse().expect("numeric fuzz duration");
    let start = std::time::Instant::now();
    let mut rng = quickdb_api::conformance::Lcg(0x736e6170666d7433);
    let mut iterations = 0u64;
    let mut accepted = 0u64;
    while start.elapsed().as_secs() < seconds {
        let seed = FIXTURES[rng.next_u32() as usize % FIXTURES.len()].1;
        let mut payload = seed[..seed.len() - 4].to_vec();
        match rng.next_u32() % 5 {
            0 => {
                let length = rng.next_u32() as usize % (payload.len() + 1);
                payload.truncate(length);
            }
            1 => {
                for _ in 0..1 + rng.next_u32() % 16 {
                    payload.push(rng.next_u32() as u8);
                }
            }
            2 if payload.len() >= 8 => {
                let offset = rng.next_u32() as usize % (payload.len() - 7);
                payload[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            }
            _ => {
                for _ in 0..1 + rng.next_u32() % 8 {
                    let offset = rng.next_u32() as usize % payload.len();
                    payload[offset] ^= 1 << (rng.next_u32() % 8);
                }
            }
        }
        let mutation = checksum(payload);
        let result = std::panic::catch_unwind(|| checked_decode(&mutation))
            .expect("decoder panic during seeded fuzz");
        if result.is_ok() {
            accepted += 1;
        }
        iterations += 1;
    }
    assert!(iterations > 0);
    println!("SNAP_FUZZ seconds={} iterations={iterations} accepted={accepted} rejected={} panics=0 seed=0x736e6170666d7433", start.elapsed().as_secs(), iterations - accepted);
}
