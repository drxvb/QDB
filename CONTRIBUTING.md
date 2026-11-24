# Contributing to QuickDB

Thanks for looking. QuickDB is pre-release (version 0.1.0), so the rules below are short and may change.

## Licence of contributions

QuickDB is licensed under MIT OR Apache-2.0 (see `LICENSE-MIT` and `LICENSE-APACHE`). Unless you state otherwise, a contribution you
submit is licensed under the same two licences, with no extra terms.

## Before you open a pull request

Use Rust 1.91.1 (the declared `rust-version` is 1.91). From the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

CI runs the same three commands. Tests run in release mode because several of them build real vector indexes.

## The contract

`crates/quickdb-api/src/lib.rs` defines the `Store` trait and `crates/quickdb-api/src/conformance.rs` is its executable meaning. A backend
is correct when `conformance::run_all` passes. Changes to either file change the contract for every backend. Open an issue first and
explain what is wrong or missing, with a failing case if you have one.

## Dependencies

New dependencies need a permissive licence (MIT, Apache-2.0, BSD, Zlib, Unicode or Unlicense, see `deny.toml`) and must build with a C
compiler and no other system tools. Prefer fewer, well-known crates. Run `cargo deny check licenses bans sources` if you have `cargo-deny`.

## Code

- Rust 2021. No `unsafe` unless a measurement justifies it, with a comment that says why it is sound.
- No panics on bad input: return `quickdb_api::Error`.
- Comments say what the code cannot say.
- Add a test for a fix. Durability and recovery changes need a test that damages or interrupts a store and then reopens it.
- Performance claims need the command that produced the number and the machine it ran on.

## Reporting a security problem

Do not open a public issue. See `SECURITY.md`.
