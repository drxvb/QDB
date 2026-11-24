use super::*;
use quickdb_api::conformance::{exact_distance, Lcg};

include!("range_repair_tests.rs");

#[test]
fn range_limits_nulls_and_extreme_bounds_preserve_id_order() {
    for indexed in [false, true] {
        let dir = directory();
        let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        store
            .define_table(&TableDef {
                name: "docs".into(),
                columns: vec![ColumnDef {
                    name: "size".into(),
                    ty: ColumnType::Int,
                    indexed,
                }],
            })
            .unwrap();
        let values = [
            Value::Int(i64::MAX),
            Value::Int(10),
            Value::Null,
            Value::Int(i64::MIN),
            Value::Int(10),
            Value::Int(9),
        ];
        let rows: Vec<_> = values
            .iter()
            .enumerate()
            .rev()
            .map(|(id, value)| Row {
                id: id as u64,
                values: vec![value.clone()],
            })
            .collect();
        store.upsert_rows("docs", &rows).unwrap();
        for (min, max) in [(i64::MIN, i64::MAX), (10, 10), (11, 9), (100, 101)] {
            for limit in [0, 1, 3, usize::MAX] {
                let mut expected: Vec<_> = rows.iter().filter(|row|
                    matches!(row.values[0], Value::Int(value) if value >= min && value <= max))
                    .cloned().collect();
                expected.sort_by_key(|row| row.id);
                expected.truncate(limit);
                assert_eq!(
                    store
                        .select(
                            "docs",
                            &Filter::IntRange {
                                column: "size".into(),
                                min,
                                max,
                            },
                            limit
                        )
                        .unwrap(),
                    expected
                );
            }
        }
    }
}

fn directory() -> tempfile::TempDir {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    tempfile::Builder::new()
        .prefix("native-repair-")
        .tempdir_in(root)
        .unwrap()
}
fn definition(metric: Metric) -> VectorIndexDef {
    VectorIndexDef {
        name: "emb".into(),
        dim: 16,
        metric,
        m: 16,
        ef_construction: 200,
    }
}
fn load(s: &mut NativeStore, data: &[(u64, Vec<f32>)]) {
    for batch in data.chunks(1000) {
        let items: Vec<_> = batch.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        s.upsert_vectors("emb", &items).unwrap();
    }
}

#[test]
fn replacements_compact_to_fresh_snapshot_and_retain_recall() {
    for metric in [Metric::Cosine, Metric::L2] {
        let dir = directory();
        let fresh_dir = directory();
        let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        store.define_vector_index(&definition(metric)).unwrap();
        let mut rng = Lcg(7654);
        for _ in 0..2 {
            let data: Vec<_> = (0..2000).map(|id| (id, rng.vector(16))).collect();
            load(&mut store, &data);
        }
        let data: Vec<_> = (0..2000).map(|id| (id, rng.vector(16))).collect();
        load(&mut store, &data);
        assert_eq!(store.indexes["emb"].node_count(), 6000);
        assert!(store.indexes["emb"].needs_compaction());
        store.flush().unwrap();
        assert_eq!(store.indexes["emb"].node_count(), 2000);
        let mut fresh = NativeStore::open(fresh_dir.path(), &OpenOptions::default()).unwrap();
        fresh.define_vector_index(&definition(metric)).unwrap();
        load(&mut fresh, &data);
        fresh.flush().unwrap();
        let size = fs::metadata(dir.path().join("emb.hnsw")).unwrap().len();
        let fresh_size = fs::metadata(fresh_dir.path().join("emb.hnsw"))
            .unwrap()
            .len();
        assert_eq!(size, fresh_size);
        assert_eq!(
            store.indexes["emb"].node_count(),
            fresh.indexes["emb"].node_count()
        );
        drop(store);
        let store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        assert_eq!(store.index_rebuilds(), 0);
        assert_eq!(store.indexes["emb"].node_count(), 2000);
        let mut found = 0;
        for _ in 0..30 {
            let q = rng.vector(16);
            let mut truth: Vec<_> = data
                .iter()
                .map(|(id, v)| (exact_distance(metric, &q, v), *id))
                .collect();
            truth.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let hits = store.knn("emb", &q, 10, 128).unwrap();
            assert_eq!(hits, fresh.knn("emb", &q, 10, 128).unwrap());
            found += hits
                .iter()
                .filter(|h| truth[..10].iter().any(|t| t.1 == h.id))
                .count();
        }
        let recall = found as f64 / 300.0;
        assert!(recall >= 0.95, "recall={recall}");
        println!("COMPACTION metric={metric:?} before_nodes=6000 after_nodes=2000 fresh_nodes=2000 snapshot_bytes={size} fresh_bytes={fresh_size} recall_at_10={recall:.4} rebuilds=0");
    }
}

#[test]
fn checksum_rejects_single_byte_damage_and_rebuilds_from_vectors() {
    let dir = directory();
    let mut store = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    store
        .define_vector_index(&definition(Metric::Cosine))
        .unwrap();
    let mut rng = Lcg(987);
    let data: Vec<_> = (0..2000).map(|id| (id, rng.vector(16))).collect();
    load(&mut store, &data);
    store.flush().unwrap();
    let q = rng.vector(16);
    let before = store.knn("emb", &q, 10, 128).unwrap();
    drop(store);
    let path = dir.path().join("emb.hnsw");
    let bytes = fs::read(&path).unwrap();
    for offset in [0, 49, bytes.len() / 2, bytes.len() - 1] {
        let mut damaged = bytes.clone();
        damaged[offset] ^= 1;
        fs::write(&path, damaged).unwrap();
        let recovered = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
        assert_eq!(recovered.index_rebuilds(), 1, "offset={offset}");
        assert_eq!(recovered.count_vectors("emb").unwrap(), 2000);
        assert_eq!(recovered.knn("emb", &q, 10, 128).unwrap(), before);
        println!("SNAPSHOT_BYTE_FLIP offset={offset} rebuilt=1 vectors=2000 hits_unchanged=true");
    }
    fs::write(&path, &bytes).unwrap();
    let recovered = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    assert_eq!(recovered.index_rebuilds(), 0);
}
