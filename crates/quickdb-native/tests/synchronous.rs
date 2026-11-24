use quickdb_api::{conformance::Lcg, *};
use quickdb_native::NativeStore;

#[test]
fn synchronous_writes_and_large_index_replacement_survive_without_explicit_flush() {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::Builder::new()
        .prefix("native-sync-")
        .tempdir_in(root)
        .unwrap();
    let mut s = NativeStore::open(dir.path(), &OpenOptions::default()).unwrap();
    s.define_vector_index(&VectorIndexDef {
        name: "emb".into(),
        dim: 8,
        metric: Metric::L2,
        m: 32,
        ef_construction: 400,
    })
    .unwrap();
    let mut rng = Lcg(832);
    let data: Vec<_> = (0..2000).map(|id| (id, rng.vector(8))).collect();
    s.upsert_vectors(
        "emb",
        &data
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    s.flush().unwrap();
    drop(s);
    let mut s = NativeStore::open(
        dir.path(),
        &OpenOptions {
            create: false,
            sync_each_write: true,
        },
    )
    .unwrap();
    assert_eq!(s.index_rebuilds(), 0);
    s.upsert_vectors("emb", &[(0, &[50.; 8]), (0, &[60.; 8]), (1999, &[70.; 8])])
        .unwrap();
    assert_eq!(s.count_vectors("emb").unwrap(), 2000);
    assert_eq!(s.knn("emb", &[60.; 8], 1, 128).unwrap()[0].id, 0);
    assert_eq!(s.knn("emb", &[70.; 8], 1, 128).unwrap()[0].id, 1999);
    drop(s);
    let s = NativeStore::open(
        dir.path(),
        &OpenOptions {
            create: false,
            sync_each_write: false,
        },
    )
    .unwrap();
    assert_eq!(s.index_rebuilds(), 0);
    assert_eq!(s.count_vectors("emb").unwrap(), 2000);
    assert_eq!(s.knn("emb", &[60.; 8], 1, 128).unwrap()[0].id, 0);
}
