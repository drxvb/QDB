//! Force the rare exhausted-graph fallback without changing production visibility.
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::collections::HashSet;

// The crate's own version 3 codec, used only to disconnect a valid test fixture.
#[allow(dead_code)]
#[path = "../src/snapshot.rs"]
mod snapshot;

#[test]
fn masked_exhaustion_exact_fallback_merges_multiple_batches_and_distance_ties() {
    const N: usize = 4096;
    let dir = tempfile::tempdir().expect("directory");
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).expect("open");
    s.define_table(&TableDef {
        name: "docs".into(),
        columns: vec![ColumnDef {
            name: "p".into(),
            ty: ColumnType::Int,
            indexed: true,
        }],
    })
    .expect("table");
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 1,
        metric: Metric::L2,
        m: 4,
        ef_construction: 16,
    })
    .expect("index");
    let data: Vec<_> = (0..N)
        .map(|id| {
            let value = if id < 3000 {
                10.0 + (id % 11) as f32
            } else {
                (id % 3) as f32
            };
            (id as u64, [value])
        })
        .collect();
    let items: Vec<_> = data.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    s.upsert_vectors("emb", &items).expect("vectors");
    s.flush().expect("snapshot");
    let path = dir.path().join("emb.hnsw");
    let bytes = std::fs::read(&path).expect("snapshot bytes");
    let payload = &bytes[..bytes.len() - 4];
    let (header, mut graph) =
        snapshot::decode(payload, payload.len() as u64).expect("snapshot schema");
    let entry = graph.ids[graph.entry.expect("entry")];
    let sampled: HashSet<_> = (0..256).map(|i| graph.ids[i * N / 256]).collect();
    let allowed: HashSet<_> = (0..N as u64)
        .filter(|id| id.is_multiple_of(2) && *id != entry && !sampled.contains(id))
        .collect();
    assert!(allowed.len() > 1024);
    assert!(!allowed.contains(&entry));
    let rows: Vec<_> = (0..N as u64)
        .map(|id| Row {
            id,
            values: vec![Value::Int(i64::from(allowed.contains(&id)))],
        })
        .collect();
    s.upsert_rows("docs", &rows).expect("rows");
    s.flush().expect("flush rows");
    drop(s);
    // Preserve every node's required layers and all store/revision bindings.
    // Only connectivity changes. The loader must accept it, not rebuild it.
    for layers in &mut graph.links {
        for links in layers {
            links.clear();
        }
    }
    let mut bytes = Vec::new();
    snapshot::encode_header(&mut bytes, header).expect("snapshot header");
    snapshot::encode_graph(&mut bytes, &graph.parts(), graph.active.iter().copied())
        .expect("snapshot encoding");
    bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
    std::fs::write(path, bytes).expect("install disconnected graph");
    let s = NativeStore::open(dir.path(), &OpenOptions::default()).expect("reopen");
    assert_eq!(s.index_rebuilds(), 0, "the disconnected graph must be used");
    assert_eq!(
        s.knn("emb", &[0.0], 2, 32).expect("unfiltered walk").len(),
        1,
        "only the isolated entry is reachable"
    );
    let filter = Filter::Eq {
        column: "p".into(),
        value: Value::Int(1),
    };
    let mut checks = 0;
    for query in [0.0f32, 1.0, 2.0] {
        let mut expected: Vec<_> = data
            .iter()
            .filter(|(id, _)| allowed.contains(id))
            .map(|(id, v)| Hit {
                id: *id,
                distance: (v[0] - query) * (v[0] - query),
            })
            .collect();
        expected.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        for k in [1, 7, 31] {
            // Zero sampled matches prevents lazy search; >1024 matches prevents
            // the initial exact branch. The isolated disallowed entry yields no
            // masked hits at any breadth, forcing exact fallback after widening.
            let actual = s
                .knn_filtered("emb", &[query], k, 32, "docs", &filter)
                .expect("fallback");
            let bits: Vec<_> = actual
                .iter()
                .map(|h| (h.id, h.distance.to_bits()))
                .collect();
            let expected_bits: Vec<_> = expected[..k]
                .iter()
                .map(|h| (h.id, h.distance.to_bits()))
                .collect();
            assert_eq!(bits, expected_bits);
            assert!(actual.iter().all(|hit| hit.id >= 3000));
            checks += 1;
        }
    }
    println!("BOUND fallback checks={checks} allowed={} exact_up_to=1024 batches=2 isolated_entry=true identical_ids_and_distance_bits=true", allowed.len());
}
