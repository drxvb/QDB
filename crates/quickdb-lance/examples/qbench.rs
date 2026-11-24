//! The common harness with this backend: `cargo run --release --example qbench -- synthetic --backend lancedb ...`
fn main() {
    quickdb_bench::main_with::<quickdb_lance::LanceStore>("lancedb");
}
