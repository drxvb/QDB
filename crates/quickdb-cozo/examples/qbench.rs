//! Common comparison harness for the feature-selected Cozo engine.
fn main() {
    quickdb_bench::main_with::<quickdb_cozo::CozoStore>(quickdb_cozo::BACKEND_NAME);
}
