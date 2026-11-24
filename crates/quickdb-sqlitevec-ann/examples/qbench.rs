fn main() {
    quickdb_bench::cli::main_with::<quickdb_sqlitevec_ann::SqliteVecAnnStore>("sqlite-vec-diskann");
}
