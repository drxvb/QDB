fn main() {
    quickdb_bench::cli::main_with::<quickdb_surreal::SurrealStore>("surrealdb");
}
