//! Compiles the vendored vec1 0.8 against the headers of the SQLite that rusqlite bundles.
use sha2::{Digest, Sha256};

/// SHA-256 of `vendor/vec1.c` as fetched from sqlite.org/src on 2026-11-22.
const VEC1_SHA256: &str = "b5c7a38510708133b66b3949df5d2f93220fd6a96e990f31be6977a41a9ca2df";

fn main() {
    println!("cargo:rerun-if-changed=vendor/vec1.c");
    let source = std::fs::read("vendor/vec1.c").expect("read vendor/vec1.c");
    let digest: String = Sha256::digest(&source)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest, VEC1_SHA256, "vendor/vec1.c is not the pinned vec1 0.8 source");
    // libsqlite3-sys (bundled) exports its include directory to direct dependents.
    let include = std::env::var("DEP_SQLITE3_INCLUDE").expect("DEP_SQLITE3_INCLUDE from libsqlite3-sys");
    cc::Build::new()
        .file("vendor/vec1.c")
        .include(include)
        .flag("-O3")
        .flag("-mavx2")
        .flag("-mfma")
        .define("NDEBUG", None)
        // Loadable-extension build (sqlite3ext.h API table) with unique global names, so that
        // the generic entry point and API pointer cannot clash with another extension.
        .define("sqlite3_extension_init", "sqlite3_vec1_init")
        .define("sqlite3_api", "quickdb_vec1_sqlite3_api")
        .warnings(false)
        .compile("quickdb_vec1");
}
