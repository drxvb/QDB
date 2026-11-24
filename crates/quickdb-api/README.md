# quickdb-api

The QDB contract: one `Store` trait for relational rows, vectors and a directed graph in one embedded store, and the
`conformance` suite that defines its behaviour.

- Rows, vector ids and edge endpoints share one `NodeId` space (`0..=i64::MAX`).
- Writes are upserts; after `flush()` returns, data survives drop and reopen.
- Bad input is an `Error`, never a panic.

This crate ships no store. For the engine, use [`quickdb`](https://crates.io/crates/quickdb), which re-exports this
contract together with `NativeStore`. To write another backend, implement `Store` and run `conformance::run_all`
against it.

Licence: MIT OR Apache-2.0, at your option.
