fn main() {
    quickdb_bench::cli::main_with::<quickdb_sqlitevec::SqliteVecStore>("sqlite-vec");
}
