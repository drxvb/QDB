use quickdb_lance::LanceStore;

#[test]
fn conformance() {
    let base = std::env::var_os("QUICKDB_DATA").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!("lance-conformance-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let failures = quickdb_api::conformance::run_all::<LanceStore>(&dir);
    for (name, msg) in &failures {
        eprintln!("FAIL {name}: {msg}");
    }
    assert!(failures.is_empty(), "{} conformance checks failed", failures.len());
    println!("conformance: all checks passed");
}

/// The conformance sets (300 vectors) stay below the ANN threshold; this forces the HNSW index.
#[test]
fn hnsw_index_used_after_flush() {
    use quickdb_api::{Metric, OpenOptions, Store, VectorIndexDef};
    let base = std::env::var_os("QUICKDB_DATA").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!("lance-hnsw-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut s = LanceStore::open(&dir, &OpenOptions::default()).expect("open");
    let def = VectorIndexDef { name: "e".into(), dim: 16, metric: Metric::L2, m: 16, ef_construction: 200 };
    s.define_vector_index(&def).expect("define");
    let mut rng = quickdb_api::conformance::Lcg(3);
    let data: Vec<(u64, Vec<f32>)> = (1..=3000u64).map(|i| (i, rng.vector(16))).collect();
    let items: Vec<(u64, &[f32])> = data.iter().map(|(i, v)| (*i, v.as_slice())).collect();
    s.upsert_vectors("e", &items).expect("upsert");
    s.flush().expect("flush");
    let late = vec![9.0f32; 16];
    s.upsert_vectors("e", &[(5000, late.as_slice())]).expect("late upsert");
    let mut found = 0;
    for (id, v) in data.iter().step_by(150) {
        let h = s.knn("e", v, 10, 200).expect("knn");
        if h.first().map(|h| h.id) == Some(*id) {
            found += 1;
        }
    }
    assert!(found >= 19, "self-recall {found}/20");
    let h = s.knn("e", &late, 1, 200).expect("knn late");
    assert_eq!(h[0].id, 5000, "an unindexed vector must still be found");
    let h = s.knn("e", &data[0].1, 1, 200).expect("knn");
    println!("hnsw ok; self-recall {found}/20, top distance {}", h[0].distance);
}
