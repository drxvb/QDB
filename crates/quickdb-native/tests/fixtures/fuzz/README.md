# Fuzz finding inputs

Inputs found by the targets in `fuzz/` (see `fuzz/README.md`), kept so a finding can be replayed:

```text
cd fuzz && cargo +nightly fuzz run -O -a <target> ../crates/quickdb-native/tests/fixtures/fuzz/<file>
```

| File | Target | Regression test |
|---|---|---|
| `store_api-leak-batch-guard` | `store_api` | `tests/fuzz_leak.rs` |

The regression tests rebuild each case through the public API, so they do not read these files.
