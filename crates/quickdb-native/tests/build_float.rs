use quickdb_api::{conformance::Lcg, Metric, OpenOptions, Store, VectorIndexDef};
use quickdb_native::NativeStore;
use std::path::Path;
use std::process::Command;

#[allow(dead_code)]
#[path = "../src/snapshot.rs"]
mod snapshot;

fn graph(path: &Path) -> snapshot::Graph {
    let bytes = std::fs::read(path.join("emb.hnsw")).expect("snapshot");
    let payload = &bytes[..bytes.len().checked_sub(4).expect("checksum trailer")];
    let (_, mut graph) =
        snapshot::decode(payload, payload.len() as u64).expect("unchanged snapshot tuple");
    // The writer emits the active map in HashMap order, which differs between processes;
    // the former map comparison ignored that order, so the sorted sequence keeps its meaning.
    graph.active.sort_unstable();
    graph
}

#[test]
fn float_build_child() {
    let Some(path) = std::env::var_os("QUICKDB_FLOAT_TEST_DIR") else {
        return;
    };
    let dim: usize = std::env::var("QUICKDB_FLOAT_TEST_DIM")
        .expect("dimension")
        .parse()
        .expect("numeric dimension");
    let cosine = std::env::var_os("QUICKDB_FLOAT_TEST_COSINE").is_some();
    let mut rng = Lcg(0x666c6f6174);
    let data: Vec<_> = (1..=768)
        .map(|id| {
            let vector = if dim == 128 {
                (0..dim).map(|_| (rng.next_u32() % 256) as f32).collect()
            } else {
                rng.vector(dim)
            };
            (id, vector)
        })
        .collect();
    let mut store = NativeStore::open(Path::new(&path), &OpenOptions::default()).expect("open");
    store
        .define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim,
            metric: if cosine { Metric::Cosine } else { Metric::L2 },
            m: 16,
            ef_construction: 200,
        })
        .expect("index");
    for batch in data.chunks(257) {
        let items: Vec<_> = batch.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        store.upsert_vectors("emb", &items).expect("batch");
    }
    let before: Vec<_> = data
        .iter()
        .step_by(97)
        .map(|(_, q)| store.knn("emb", q, 10, 64).expect("search"))
        .collect();
    store.flush().expect("flush");
    drop(store);
    let store = NativeStore::open(Path::new(&path), &OpenOptions::default()).expect("reopen");
    assert_eq!(store.index_rebuilds(), 0);
    let after: Vec<_> = data
        .iter()
        .step_by(97)
        .map(|(_, q)| store.knn("emb", q, 10, 64).expect("reopened search"))
        .collect();
    assert_eq!(before, after);
}

#[test]
fn float_switch_preserves_graph_and_snapshot_for_l2_cosine_and_sift() {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    for (dim, cosine) in [(960, false), (960, true), (128, false)] {
        let old = tempfile::tempdir_in(&root).expect("old directory");
        let new = tempfile::tempdir_in(&root).expect("new directory");
        for (off, dir) in [(true, &old), (false, &new)] {
            let mut child = Command::new(std::env::current_exe().expect("test binary"));
            child
                .args(["--exact", "float_build_child", "--nocapture"])
                .env("QUICKDB_FLOAT_TEST_DIR", dir.path())
                .env("QUICKDB_FLOAT_TEST_DIM", dim.to_string())
                .env("QUICKDB_BUILD_WORKERS", "4");
            if off {
                child.env("QUICKDB_BUILD_FLOAT_OFF", "1");
            } else {
                child.env("QUICKDB_BUILD_FLOAT_OFF", "0");
            }
            if cosine {
                child.env("QUICKDB_FLOAT_TEST_COSINE", "1");
            } else {
                child.env_remove("QUICKDB_FLOAT_TEST_COSINE");
            }
            assert!(child.status().expect("run child").success());
        }
        assert_eq!(graph(old.path()), graph(new.path()));
        println!("BUILD_FLOAT dim={dim} cosine={cosine} graph_identical=true snapshot_rebuilds=0");
    }
}
