//! Reachability repair (`QUICKDB_REACH_REPAIR`). The environment is process-wide, so every test
//! holds `ENV` while it builds.
use quickdb_api::conformance::Lcg;
use quickdb_api::{Metric, OpenOptions, Store, VectorIndexDef};
use quickdb_native::NativeStore;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

const VARS: [&str; 3] = [
    "QUICKDB_REACH_REPAIR",
    "QUICKDB_REACH_PLANT",
    "QUICKDB_REACH_REPORT",
];

fn set_env(vars: &[(&str, &str)]) {
    for var in VARS {
        std::env::remove_var(var);
    }
    for (key, value) in vars {
        std::env::set_var(key, value);
    }
}

fn scratch() -> tempfile::TempDir {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    tempfile::tempdir_in(root).expect("test directory")
}

fn build(
    dir: &std::path::Path,
    seed: u64,
    n: u64,
    dim: usize,
    batch: usize,
) -> (NativeStore, Vec<Vec<f32>>) {
    let mut rng = Lcg(seed);
    let data: Vec<Vec<f32>> = (0..n).map(|_| rng.vector(dim)).collect();
    let mut store = NativeStore::open(dir, &OpenOptions::default()).expect("store");
    store
        .define_vector_index(&VectorIndexDef {
            name: "emb".into(),
            dim,
            metric: Metric::L2,
            m: 16,
            ef_construction: 200,
        })
        .expect("index");
    for (batch_no, chunk) in data.chunks(batch).enumerate() {
        let items: Vec<_> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| ((batch_no * batch + i + 1) as u64, v.as_slice()))
            .collect();
        store.upsert_vectors("emb", &items).expect("vectors");
    }
    (store, data)
}

fn knn_digest(store: &NativeStore, data: &[Vec<f32>], seed: u64) -> u64 {
    let mut rng = Lcg(seed);
    let mut digest = 14695981039346656037u64;
    let fresh: Vec<Vec<f32>> = (0..200).map(|_| rng.vector(data[0].len())).collect();
    for q in fresh.iter().chain(data.iter().step_by(101)) {
        for hit in store.knn("emb", q, 10, 64).expect("knn") {
            for word in [hit.id, u64::from(hit.distance.to_bits())] {
                digest = (digest ^ word).wrapping_mul(1099511628211);
            }
        }
    }
    digest
}

/// One JSON object per finished build: (key, value) pairs of its integer fields and the policy.
fn report_lines(path: &std::path::Path) -> Vec<(String, std::collections::HashMap<String, u64>)> {
    let text = std::fs::read_to_string(path).expect("reach report");
    text.lines()
        .map(|line| {
            let body = line.trim_start_matches('{').trim_end_matches('}');
            let mut policy = String::new();
            let mut numbers = std::collections::HashMap::new();
            for field in body.split(',') {
                let (key, value) = field.split_once(':').expect("field");
                let key = key.trim_matches('"').to_owned();
                if key == "policy" {
                    policy = value.trim_matches('"').to_owned();
                } else {
                    numbers.insert(key, value.parse().expect("integer"));
                }
            }
            (policy, numbers)
        })
        .collect()
}

/// knn digest of this build and these queries on 05e4927, before the reach code existed
/// (Linux benchmark host, run that printed `REACH_OFF_DIGEST 7b8bb7be7504c841`).
const GOLDEN_OFF_DIGEST: u64 = 0x7b8b_b7be_7504_c841;

#[test]
fn off_matches_build_without_reach_code() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    for setting in [None, Some("off"), Some("unknown")] {
        match setting {
            Some(value) => set_env(&[("QUICKDB_REACH_REPAIR", value)]),
            None => set_env(&[]),
        }
        let dir = scratch();
        let (mut store, data) = build(dir.path(), 0x5eac4, 20_000, 32, 1000);
        let digest = knn_digest(&store, &data, 0x9e11);
        store.flush().expect("flush");
        let flushed = knn_digest(&store, &data, 0x9e11);
        drop(store);
        let reopened = NativeStore::open(dir.path(), &OpenOptions::default()).expect("reopen");
        assert_eq!(reopened.index_rebuilds(), 0);
        let loaded = knn_digest(&reopened, &data, 0x9e11);
        println!(
            "REACH_OFF setting={setting:?} digest={digest:016x} after_flush={flushed:016x} after_reopen={loaded:016x}"
        );
        assert_eq!(digest, GOLDEN_OFF_DIGEST);
        assert_eq!(flushed, GOLDEN_OFF_DIGEST);
        assert_eq!(loaded, GOLDEN_OFF_DIGEST);
    }
    set_env(&[]);
}

#[test]
fn planted_unreachable_node_is_repaired_and_found_at_rank_one() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    const PLANT: u64 = 2_500;
    for policy in ["off", "reference", "selective"] {
        let dir = scratch();
        let report = dir.path().join("reach.jsonl");
        let report_text = report.to_str().expect("utf-8 path");
        let plant = PLANT.to_string();
        set_env(&[
            ("QUICKDB_REACH_REPAIR", policy),
            ("QUICKDB_REACH_PLANT", &plant),
            ("QUICKDB_REACH_REPORT", report_text),
        ]);
        let (store, data) = build(dir.path(), 0x91a7, 5_000, 16, 5_000);
        set_env(&[]);
        let lines = report_lines(&report);
        assert_eq!(lines.len(), 1, "one build, one report line");
        let (name, counts) = &lines[0];
        assert_eq!(name, policy);
        let own = &data[(PLANT - 1) as usize];
        let hits = store.knn("emb", own, 10, 64).expect("knn");
        println!(
            "REACH_PLANT policy={policy} unreachable_before={} unreachable_after={} zero_inbound={} bridges={} rank1={} rank1_distance={}",
            counts["unreachable_before"],
            counts["unreachable_after"],
            counts["zero_inbound"],
            counts["bridges"],
            hits[0].id,
            hits[0].distance
        );
        assert!(
            counts["unreachable_before"] >= 1,
            "the cut must strand the node"
        );
        assert!(counts["zero_inbound"] >= 1);
        if policy == "off" {
            assert_eq!(counts["unreachable_after"], counts["unreachable_before"]);
            assert!(
                hits.iter().all(|h| h.id != PLANT),
                "an unreachable node is never returned"
            );
        } else {
            assert_eq!(counts["unreachable_after"], 0);
            assert_eq!(counts["failed"], 0);
            assert!(counts["bridges"] >= 1);
            assert_eq!(hits[0].id, PLANT);
            assert_eq!(hits[0].distance, 0.0);
        }
    }
}

#[test]
fn both_policies_leave_every_node_reachable_on_random_20k() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    // Unplanted, this build strands nothing; the planted pass cuts id 500 again after every
    // batch so that each of the 20 repairs has work to do.
    for (policy, plant) in [
        ("reference", None),
        ("selective", None),
        ("reference", Some("500")),
        ("selective", Some("500")),
    ] {
        let dir = scratch();
        let report = dir.path().join("reach.jsonl");
        let mut vars = vec![
            ("QUICKDB_REACH_REPAIR", policy),
            ("QUICKDB_REACH_REPORT", report.to_str().expect("utf-8 path")),
        ];
        if let Some(id) = plant {
            vars.push(("QUICKDB_REACH_PLANT", id));
        }
        set_env(&vars);
        let (store, data) = build(dir.path(), 0x5eac4, 20_000, 32, 1000);
        set_env(&[]);
        let lines = report_lines(&report);
        assert_eq!(lines.len(), 20, "one report per upsert batch");
        let stranded: u64 = lines.iter().map(|(_, c)| c["unreachable_before"]).sum();
        let bridges: u64 = lines.iter().map(|(_, c)| c["bridges"]).sum();
        for (name, counts) in &lines {
            assert_eq!(name, policy);
            assert_eq!(counts["unreachable_after"], 0);
            assert_eq!(counts["failed"], 0);
        }
        let last = &lines[lines.len() - 1].1;
        assert_eq!(last["live"], 20_000);
        assert_eq!(store.count_vectors("emb").expect("count"), 20_000);
        if plant.is_some() {
            assert!(stranded >= 20, "the cut strands id 500 after every batch");
            let hits = store.knn("emb", &data[499], 1, 64).expect("knn");
            assert_eq!(hits[0].id, 500);
        }
        println!(
            "REACH_RANDOM policy={policy} plant={plant:?} batches={} stranded_over_batches={stranded} bridges_added={bridges} bridges_total={} final_unreachable=0",
            lines.len(),
            last["bridges_total"]
        );
    }
}
