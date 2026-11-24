# QDB fuzz targets

These are [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer) targets for `quickdb-native`. The `fuzz/` directory is its own Cargo workspace, so the main workspace still builds and tests on stable Rust without it. The fuzz targets need a nightly toolchain.

## Running

```sh
rustup toolchain install nightly
cargo install cargo-fuzz --locked
cd fuzz
cargo +nightly fuzz build -O -a                       # release build with debug assertions
TMPDIR=/dev/shm cargo +nightly fuzz run -O -a store_api -- -max_total_time=900
```

To run all four targets in parallel (one process each), with corpora, findings and logs under one state directory:

```sh
TMPDIR=/dev/shm sh fuzz/run_all.sh 900 /var/tmp/quickdb-fuzz
```

`run_all.sh` seeds the snapshot corpora with the valid fixtures in `crates/quickdb-native/tests/fixtures/snap/`. It exits non-zero if any target reports a crash, timeout or out-of-memory input. The store targets create one temporary store per input, so pointing `TMPDIR` at a RAM file system makes them much faster.

To reproduce a finding:

```sh
cargo +nightly fuzz run -O -a <target> <crash-file>
```

Each finding gets a regression test under `crates/quickdb-native/tests/` that runs with plain `cargo test` on stable Rust. The input that found it is kept in `crates/quickdb-native/tests/fixtures/fuzz/`.

## Findings

| Target | Finding | Fix | Regression test |
|---|---|---|---|
| `store_api` | LeakSanitizer: each parallel insert batch (any `upsert_vectors` or index rebuild past the first 256 vectors) leaked a copy of every adjacency list it touched, about 3.75 KB per inserted 8-dimension vector. Input: `store_api-leak-batch-guard`. | The batch rollback guard in `src/hnsw.rs` was disarmed with `mem::forget`, which also skipped freeing its saved lists. It now sets a `committed` flag and is dropped normally. | `tests/fuzz_leak.rs` |

## Targets

| Target | Input | What it checks |
|---|---|---|
| `snapshot_decode` | Raw bytes | Feeds the bytes to `snapshot::decode` (magic, header and graph) and `snapshot::decode_graph`, using both the true length and wrong declared lengths. Every graph the decoder accepts must re-encode to an identical graph and load with `Hnsw::from_snapshot`. When it also matches the parameter range the store accepts, it must survive `search`, a filtered graph walk, an `insert` and a re-encode. |
| `snapshot_mutate` | Edit operations | Starts from the four valid fixture snapshots and damages them with bit flips, byte overwrites, insertions, deletions, truncation and overwritten length words. The damaged bytes are then optionally re-sealed with a correct CRC32 and read the way the store reads a snapshot: checksum first, then decode, then the same checks as `snapshot_decode`. |
| `store_open` | Mode byte and bytes | Writes the bytes as `<index>.hnsw` next to a valid `quickdb.sqlite` and opens the store read-only and read-write. The mode byte picks one of four inputs: the raw bytes, the bytes with a correct CRC32, a fuzzed graph behind the store's real header (so it passes the checksum and the freshness check), or the store's own snapshot damaged and optionally re-sealed. Open must return `Ok` or `Err`, never panic, abort or hang. An opened store must answer `knn` and `knn_filtered`, accept a write and a `flush`, and reopen. |
| `store_api` | `arbitrary` operation sequence | Runs up to 48 operations on a temporary store: `define_table`, `upsert_rows` with arbitrary values, `get_row`, `select` with an arbitrary `Filter`, `delete_rows`, `define_vector_index` with arbitrary `m` and `ef_construction`, `upsert_vectors` (NaN, infinity, wrong dimensions, and bulk batches past the exact-scan size), `knn` and `knn_filtered` with arbitrary `k` and `ef`, `upsert_edges`, `neighbors`, `k_hop` with arbitrary depth and limit, counts, `flush`, reopen (with or without `sync_each_write`) and a read-only open beside the writer. No operation may panic. `knn` and `knn_filtered` must return at most `k` hits with distinct ids, sorted by distance. `k_hop` must return at most `limit` distinct results, none of them the start node, all within the depth. `select` and `neighbors` must respect `limit`. |

The snapshot targets compile `src/hnsw.rs` (and through it `src/snapshot.rs` and `src/reach.rs`) directly with `#[path]`, as some of the crate's integration tests do. That way they reach the private decoder and graph code without any change to the library's public API.

## Not covered

- Fuzzing does not cover concurrent writers, crashes in the middle of a write, or power loss. The durability tests in `crates/quickdb-native/tests/` cover those.
- `store_open` damages only the snapshot file. A damaged `quickdb.sqlite` is SQLite's own territory.
- Graphs the decoder accepts are checked for memory safety and the invariants listed above, not for search recall.
