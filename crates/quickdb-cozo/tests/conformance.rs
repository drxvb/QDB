use quickdb_cozo::CozoStore;

#[test]
fn conformance() {
    let base = std::env::var_os("QUICKDB_DATA").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!("cozo-conformance-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let failures = quickdb_api::conformance::run_all::<CozoStore>(&dir);
    for (name, msg) in &failures {
        eprintln!("FAIL {name}: {msg}");
    }
    assert!(failures.is_empty(), "{} conformance checks failed", failures.len());
    println!("conformance: all checks passed");
}
