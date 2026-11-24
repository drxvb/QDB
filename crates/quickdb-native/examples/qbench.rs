//! The common harness with this backend: `cargo run --release --example qbench -- synthetic --backend native …`
fn main() {
    quickdb_bench::main_with::<quickdb_native::NativeStore>("native");
}
