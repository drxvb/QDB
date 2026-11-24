# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project intends to
follow semantic versioning from the first release. 0.1.0 is not released yet.

## [Unreleased]

### Added

- `quickdb-api`: the `Store` contract for relational rows, vectors and graph edges in one embedded store, and the `conformance` suite that
  defines its behaviour.
- `quickdb-native`: `NativeStore`, a backend on bundled SQLite with a persisted, pure-Rust HNSW vector index.
- `quickdb-bench`: benchmark harness generic over `Store`.
- Dual licence MIT OR Apache-2.0, package metadata, `deny.toml` licence policy, CI workflow for Linux, `CONTRIBUTING.md`, `SECURITY.md`.
- `quickdb-native` example and test `readme_quickstart`.

### Changed

- `quickdb-api`: every public item has a doc comment (meaning, units, order, errors), the crate has an introduction with a compiled example,
  and `#![warn(missing_docs)]` is on. `NativeStore` is documented too.
- `quickdb-api`: `Error`, `Filter`, `Metric`, `ColumnType`, `Direction` and `Value` are `#[non_exhaustive]`, so that a later variant is not a
  breaking change. Callers that match on them need a wildcard arm. Backends answer an unknown `Filter`, `Metric`, `ColumnType` or
  `Direction` with `Error::Schema("unsupported ...")`, never a panic. `NodeId` stays a plain `u64`: its `i64::MAX` limit is documented and
  checked at run time. Public structs stay constructible with literals.
- `quickdb-native` and `quickdb-bench`: the dependency on `quickdb-api` carries `version = "0.1.0"`, so that `cargo package` works.
  `quickdb-bench` is `publish = false`. The `quickdb-native` package leaves out `evidence/`, `*.log` and the 400 kB test data file
  `tests/fixtures/corpus-shapes.tsv` (the unit test that reads it needs a git checkout).
- Adapters `quickdb-sqlitevec`, `quickdb-cozo`, `quickdb-lance`, `quickdb-surreal`: wildcard arms for the enums above.

### Known limits

See `evidence/relprep-gaps.md` for what is still missing before a public release.
