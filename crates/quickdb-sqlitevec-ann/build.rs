//! Compiles the vendored sqlite-vec v0.1.10-alpha.4 sources (with DiskANN and rescore, IVF off,
//! as upstream ships it) against the headers of the SQLite that rusqlite bundles.
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const DIR: &str = "vendor/sqlite-vec";
/// SHA-256 of every vendored file that reaches the build, from the tag v0.1.10-alpha.4.
const FILES: [(&str, &str); 7] = [
    ("sqlite-vec.c", "905a4bc025a63553aff1da9af0b9efcd5c18beec6e6ab2c25428f24a696aa19b"),
    ("sqlite-vec-diskann.c", "eba84318f6a0c43ecacc9769bf967af345791760e1eebb8ad3b200ea0a7afbc7"),
    ("sqlite-vec-rescore.c", "06f26bb3c51ea4c51fbae2011a47c06440f2c1d43820a6eb65732476abc3450c"),
    ("sqlite-vec-ivf.c", "0506a3c493990eb80466ae8c0fdcf1e3fa4b63955713c053e7350b7451ec7c39"),
    ("sqlite-vec-ivf-kmeans.c", "50cd3ecc0fd0b5b75900821e2b9d1a4d1157da589c39bd1faa8749e047d8fc14"),
    ("sqlite-vec.h.tmpl", "259d7a7f7f4b2ada97def925c4d249856a62f073514dbe4aef838ecb284a2a25"),
    ("VERSION", "02405190643237dcef69016c14beef1530fa5ce971ad3adba21bedfcb74b727d"),
];

fn main() {
    for (name, want) in FILES {
        let path = format!("{DIR}/{name}");
        println!("cargo:rerun-if-changed={path}");
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let got: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(got, want, "{path} is not the pinned v0.1.10-alpha.4 file");
    }
    // The Makefile rule `sqlite-vec.h: sqlite-vec.h.tmpl VERSION`. DATE and SOURCE come from git
    // there; the vendored copy has no history, so they name the tag instead.
    let version = std::fs::read_to_string(format!("{DIR}/VERSION")).expect("VERSION");
    let version = version.trim();
    let mut parts = version.split('.');
    let major = parts.next().unwrap_or("0");
    let minor = parts.next().unwrap_or("0");
    let patch = parts.next().unwrap_or("0").split('-').next().unwrap_or("0");
    let header = std::fs::read_to_string(format!("{DIR}/sqlite-vec.h.tmpl"))
        .expect("sqlite-vec.h.tmpl")
        .replace("${VERSION_MAJOR}", major)
        .replace("${VERSION_MINOR}", minor)
        .replace("${VERSION_PATCH}", patch)
        .replace("${VERSION}", version)
        .replace("${DATE}", "")
        .replace("${SOURCE}", "tag v0.1.10-alpha.4 (vendored)");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(out.join("sqlite-vec.h"), header).expect("write sqlite-vec.h");
    // libsqlite3-sys (bundled) exports its include directory to direct dependents.
    let include = std::env::var("DEP_SQLITE3_INCLUDE").expect("DEP_SQLITE3_INCLUDE from libsqlite3-sys");
    cc::Build::new()
        .file(format!("{DIR}/sqlite-vec.c"))
        .include(&out)
        .include(include)
        .flag("-O3")
        .flag("-mavx")
        .flag("-mavx2")
        .define("SQLITE_VEC_ENABLE_AVX", None)
        .define("NDEBUG", None)
        // Loadable-extension build (sqlite3ext.h API table) with unique global names.
        .define("sqlite3_vec_init", "sqlite3_vecann_init")
        .define("sqlite3_api", "quickdb_vecann_sqlite3_api")
        .warnings(false)
        .compile("quickdb_sqlitevec_ann");
}
