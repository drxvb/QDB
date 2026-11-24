# quickdb

QDB (QuickDB): relational rows, vectors and a directed graph in one embedded, in-process Rust store. No server, no
external C or C++ toolchain.

```sh
cargo add quickdb
```

```rust
use quickdb::{NativeStore, OpenOptions, Store};

let mut db = NativeStore::open("my-store".as_ref(), &OpenOptions::default())?;
// define_table / define_vector_index, then upsert_rows, upsert_vectors, upsert_edges, flush;
// then select, knn, knn_filtered and k_hop.
```

The crate documentation has a complete, tested quick start.

## What is inside

This crate re-exports two crates:

- [`quickdb-api`](https://docs.rs/quickdb-api): the `Store` contract (rows, vectors, edges sharing one id space) and the
  `conformance` suite that every backend must pass;
- [`quickdb-native`](https://docs.rs/quickdb-native): `NativeStore`, the engine. Bundled SQLite holds rows, edges and
  authoritative vectors; a persisted, checksummed, pure-Rust HNSW index serves vector search.

A store is one directory with one writer at a time. After `flush()` returns, data survives drop and reopen. Vector
indexes are held in RAM.

## Status

0.1, not yet released. The API may change before 1.0. Minimum Rust version 1.91.

## Licence

MIT OR Apache-2.0, at your option. Copyright Rabie Qotb and Basil.
