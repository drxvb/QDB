use quickdb_api::{Error, Metric};
use quickdb_bench::{ab, synthetic, MemStore};

#[test]
fn paired_run_reopens_one_graph_balances_order_and_refuses_bad_inputs() {
    let path = std::env::temp_dir().join(format!("ab-fixture-{}", std::process::id()));
    std::fs::create_dir(&path).expect("fixture");
    let mut data = synthetic(32, 4).expect("dataset");
    data.metric = Metric::L2;
    data.query_vectors = Some(vec![vec![0.1; 4], vec![0.2; 4]]);
    let store = path.join("store");
    assert!(matches!(
        ab::run::<MemStore>(&data, &store, 9, 1, false, 64),
        Err(Error::Schema(_))
    ));
    assert!(matches!(
        ab::run::<MemStore>(&data, &store, 10, 0, false, 64),
        Err(Error::Schema(_))
    ));
    assert!(matches!(
        ab::run::<MemStore>(&data, &store, 10, 1, false, 0),
        Err(Error::Schema(_))
    ));
    let prior = std::env::var_os("QUICKDB_BUILD_SCALAR");
    let report = ab::run::<MemStore>(&data, &store, 10, 2, true, 64).expect("paired run");
    assert_eq!(std::env::var_os("QUICKDB_BUILD_SCALAR"), prior);
    assert!(report["truth_seconds"].as_f64().expect("time") >= 0.0);
    let trials = report["trials"].as_array().expect("trials");
    assert_eq!(trials.len(), 2);
    for (trial, result) in trials.iter().enumerate() {
        assert_eq!(result["same_results"], true);
        assert_eq!(result["same_counts"], serde_json::Value::Null);
        assert_eq!(result["recall_at_10"], 1.0);
        let blocks = result["blocks"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 20);
        for pair in 0..10 {
            assert_eq!(blocks[pair * 2]["arm"], (pair + trial) % 2);
            assert_eq!(blocks[pair * 2 + 1]["arm"], (pair + trial + 1) % 2);
            assert_eq!(blocks[pair * 2]["distances"], serde_json::Value::Null);
        }
    }
    assert!(matches!(
        ab::run::<MemStore>(&data, &store, 10, 1, false, 64),
        Err(Error::Schema(_))
    ));
    data.query_vectors = None;
    assert!(matches!(
        ab::run::<MemStore>(&data, &path.join("missing"), 10, 1, false, 64),
        Err(Error::Schema(_))
    ));
    std::fs::remove_dir_all(path).expect("cleanup");
}
