use quickdb_api::conformance;
use quickdb_native::NativeStore;

#[test]
fn conformance() {
    let root = std::env::var_os("QUICKDB_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::Builder::new()
        .prefix("native-conformance-")
        .tempdir_in(root)
        .expect("fresh directory");
    let failures = conformance::run_all::<NativeStore>(dir.path());
    assert!(failures.is_empty(), "{failures:#?}");
}
