//! Opens a store whose `<index>.hnsw` snapshot was replaced by fuzzed bytes, next to a valid
//! `quickdb.sqlite`. Open must return Ok (rebuilding the index if needed) or Err, never panic,
//! abort or hang; an opened store must answer queries, accept a write and reopen.
//!
//! The first input byte picks how the snapshot file is made from the rest:
//! - mode 0: the bytes as they are;
//! - mode 1: the bytes followed by their correct CRC32;
//! - mode 2: the template snapshot's own magic and header, then the bytes as the graph, sealed
//!   with a correct CRC32, so the fuzzed graph passes the checksum and the freshness check;
//! - mode 3: the template snapshot damaged by the bytes as edit operations, re-sealed when the
//!   control byte's top bit is set.
#![no_main]
use libfuzzer_sys::fuzz_target;
use quickdb_api::*;
use quickdb_native::NativeStore;
use std::path::Path;
use std::sync::OnceLock;

const HEADER_LEN: usize = 8 + 16 + 8 + 8;

struct Template {
    files: Vec<(String, Vec<u8>)>,
    snapshots: [Vec<u8>; 2],
}

const INDEXES: [(&str, usize, Metric); 2] = [("v", 4, Metric::L2), ("c", 3, Metric::Cosine)];

fn template() -> &'static Template {
    static TEMPLATE: OnceLock<Template> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let dir = tempfile::tempdir().expect("template dir");
        {
            let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).expect("create");
            s.define_table(&TableDef {
                name: "t".into(),
                columns: vec![ColumnDef {
                    name: "x".into(),
                    ty: ColumnType::Int,
                    indexed: true,
                }],
            })
            .expect("table");
            let rows: Vec<_> = (0..40)
                .map(|i| Row {
                    id: i,
                    values: vec![Value::Int(i as i64 % 3)],
                })
                .collect();
            s.upsert_rows("t", &rows).expect("rows");
            for (name, dim, metric) in INDEXES {
                s.define_vector_index(&VectorIndexDef {
                    name: name.into(),
                    dim,
                    metric,
                    m: 4,
                    ef_construction: 16,
                })
                .expect("index");
                let vectors: Vec<Vec<f32>> = (0..40u64)
                    .map(|i| (0..dim).map(|d| ((i * 7 + d as u64) % 11) as f32).collect())
                    .collect();
                let items: Vec<_> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as u64, v.as_slice()))
                    .collect();
                s.upsert_vectors(name, &items).expect("vectors");
            }
            // Replace some vectors so the snapshot carries dead nodes too.
            s.upsert_vectors("v", &[(3, &[1.0, 2.0, 3.0, 4.0][..])])
                .expect("replace");
            s.flush().expect("flush");
        }
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir.path()).expect("list") {
            let entry = entry.expect("entry");
            let name = entry.file_name().into_string().expect("utf-8 name");
            if name == "quickdb.lock" || name.ends_with(".hnsw") {
                continue;
            }
            files.push((name, std::fs::read(entry.path()).expect("read")));
        }
        let snapshot = |name: &str| std::fs::read(dir.path().join(format!("{name}.hnsw"))).unwrap();
        Template {
            files,
            snapshots: [snapshot("v"), snapshot("c")],
        }
    })
}

fn seal(mut payload: Vec<u8>) -> Vec<u8> {
    let crc = crc32fast::hash(&payload);
    payload.extend_from_slice(&crc.to_le_bytes());
    payload
}

fn damage(mut bytes: Vec<u8>, ops: &[u8]) -> Vec<u8> {
    for op in ops.chunks_exact(4) {
        let at = u16::from_le_bytes([op[1], op[2]]) as usize;
        let len = bytes.len();
        match op[0] % 4 {
            0 if len > 0 => bytes[at % len] ^= 1 << (op[3] % 8),
            1 if len > 0 => bytes[at % len] = op[3],
            2 => bytes.truncate(at % (len + 1)),
            _ => bytes.insert(at % (len + 1), op[3]),
        }
    }
    bytes
}

fn check_store(s: &NativeStore) {
    for (name, dim, _) in INDEXES {
        let query: Vec<f32> = (0..dim).map(|d| d as f32 + 0.5).collect();
        for k in [1, 5, 50] {
            let hits = s.knn(name, &query, k, 32).expect("knn on an opened store");
            assert!(hits.len() <= k);
            assert!(hits
                .windows(2)
                .all(|w| w[0].distance.total_cmp(&w[1].distance).is_le()));
            let filter = Filter::Eq {
                column: "x".into(),
                value: Value::Int(1),
            };
            let hits = s
                .knn_filtered(name, &query, k, 32, "t", &filter)
                .expect("filtered knn on an opened store");
            assert!(hits.len() <= k);
        }
    }
}

fn run(dir: &Path, file: &str, snapshot: &[u8]) {
    let t = template();
    for (name, bytes) in &t.files {
        std::fs::write(dir.join(name), bytes).expect("write template file");
    }
    std::fs::write(dir.join(file), snapshot).expect("write snapshot");
    if let Ok(s) = NativeStore::open_read_only(dir) {
        check_store(&s);
    }
    let Ok(mut s) = NativeStore::open(dir, &OpenOptions::default()) else {
        return;
    };
    check_store(&s);
    s.upsert_vectors("v", &[(41, &[0.5, 0.5, 0.5, 0.5][..])])
        .expect("write after open");
    s.flush().expect("flush after open");
    drop(s);
    let s = NativeStore::open(dir, &OpenOptions::default()).expect("reopen after flush");
    check_store(&s);
}

fuzz_target!(|data: &[u8]| {
    let Some((&control, rest)) = data.split_first() else {
        return;
    };
    let t = template();
    let which = usize::from(control & 0x40 != 0);
    let original = &t.snapshots[which];
    let snapshot = match control % 4 {
        0 => rest.to_vec(),
        1 => seal(rest.to_vec()),
        2 => {
            let mut payload = original[..HEADER_LEN].to_vec();
            payload.extend_from_slice(rest);
            seal(payload)
        }
        _ => {
            let damaged = damage(original.clone(), rest);
            if control & 0x80 != 0 && damaged.len() >= 4 {
                seal(damaged[..damaged.len() - 4].to_vec())
            } else {
                damaged
            }
        }
    };
    let dir = tempfile::tempdir().expect("temp dir");
    run(dir.path(), &format!("{}.hnsw", INDEXES[which].0), &snapshot);
});
