# QDB — QuickDB by Rabie and Basil

An embedded Rust memory core: relational rows, vectors and a graph in **one in-process store**, with no server.
One id space is shared by rows, vectors and graph nodes; a store is one directory.

QDB started from Rabie Qotb's concept of a sovereign "single Rust binary" whose memory core is embedded rather than
reached over a network (the concept gave that role to CozoDB). We built both ways and measured them against each other;
the native engine won, and CozoDB stays here as a comparison adapter.

> **Status: 0.1.0, pre-release.** Built and measured, not yet reviewed line by line by a person, no fuzzing, no released
> version. Read "Limits" before relying on it.

## What is in the box

| Crate | What it is |
|---|---|
| `crates/quickdb-api` | the contract: one `Store` trait for rows, vectors and edges, plus the executable `conformance` suite (12 check groups) |
| `crates/quickdb-native` | the engine, `NativeStore`: bundled SQLite for rows and edges, a persisted pure-Rust HNSW vector index tied to the store |
| `crates/quickdb-bench` | benchmark harness generic over `Store`, an exact reference store, corpus and question loaders |
| `crates/quickdb-cozo` · `-sqlitevec` · `-sqlitevec-ann` · `-vec1` · `-lance` · `-surreal` | comparison adapters for CozoDB, SQLite + `sqlite-vec` 0.1.9, `sqlite-vec` DiskANN, SQLite `vec1`, LanceDB, SurrealDB (each its own cargo workspace) |
| `scripts/corpus_prepare.py` · `corpus_questions.py` · `reference_ann.py` | corpus chunking and embedding (any OpenAI-compatible `/embeddings` endpoint), written questions, an `hnswlib` reference |

## Use

Full, compiled quickstart: `README-example.md` (it is also a test, `readme_quickstart`).

```rust
use quickdb_api::{Store, OpenOptions, TableDef, ColumnDef, ColumnType, Row, Value,
                  VectorIndexDef, Metric, Edge, Direction, Filter};
use quickdb_native::NativeStore;

let mut db = NativeStore::open("memory.qdb".as_ref(), &OpenOptions::default())?;
db.define_table(&TableDef { name: "notes".into(), columns: vec![
    ColumnDef { name: "kind".into(), ty: ColumnType::Text, indexed: true },
    ColumnDef { name: "text".into(), ty: ColumnType::Text, indexed: false } ] })?;
db.upsert_rows("notes", &[Row { id: 1, values: vec![Value::Text("decision".into()), Value::Text("…".into())] }])?;
db.define_vector_index(&VectorIndexDef { name: "emb".into(), dim: 1024, metric: Metric::Cosine, m: 16, ef_construction: 200 })?;
db.upsert_vectors("emb", &[(1, embedding.as_slice())])?;
db.upsert_edges(&[Edge { src: 1, rel: "follows".into(), dst: 2, weight: 1.0 }])?;
db.flush()?;

let near  = db.knn("emb", &query, 10, 64)?;                       // by meaning
let same  = db.knn_filtered("emb", &query, 10, 64, "notes",
            &Filter::Eq { column: "kind".into(), value: Value::Text("decision".into()) })?;
let reach = db.k_hop(1, None, Direction::Out, 3, 1000)?;          // by relation
let row   = db.get_row("notes", 1)?;                              // by key
```

### One store, several users

- One writable handle per directory (`NativeStore::open`), across all processes; a second one fails at once with
  "store is locked by another process". Other processes and threads read with `NativeStore::open_read_only(path)`.
- `NativeStore` is `Send`, not `Sync`: wrap a writer in a `Mutex`.
- The vector index lives in memory: about 4.25 KB per 1,024-dimension vector. Keep a store on a local disk; the lock may
  not hold on network file systems.

## How data is stored

A store is one directory on local disk. Everything QDB keeps is in these files:

```text
memory.qdb/
├── quickdb.sqlite        SQLite database in WAL mode, the source of truth
├── quickdb.sqlite-wal    write-ahead log (SQLite)
├── quickdb.sqlite-shm    shared-memory index of the log (SQLite)
├── quickdb.lock          exclusive OS lock held by the one writer
└── <index>.hnsw          one vector-index snapshot per vector index (rebuildable cache)
```

Inside `quickdb.sqlite`:

| What | How it is kept |
|---|---|
| Schemas | catalogue tables describing every logical table, column and vector index |
| Rows | one physical SQLite table per logical table; text is `TEXT COLLATE BINARY`; columns marked `indexed` get a secondary index |
| Vectors | little-endian `f32` blobs, one per id per index, with a per-index revision and a random write nonce; this copy is authoritative |
| Edges | `(src, rel, dst, weight)`, with two covering indexes, `(src, rel, dst)` for outgoing and `(dst, rel, src)` for incoming hops |
| Identity | a random 16-byte `store_id` that binds snapshots to this store |

The `<index>.hnsw` snapshot holds the HNSW graph and its vectors: a version marker, a `(store_id, revision, nonce)` header,
the serialized graph and a trailing CRC32. On open, QDB verifies the checksum, the header and the graph's invariants.
A missing, damaged, foreign or stale snapshot is rebuilt from the vectors in SQLite, so losing it costs time, not data.
At run time the graph lives in memory; rows and edges are read from SQLite.

**Durability.** In the default mode every write is committed with `PRAGMA synchronous = FULL`. In bulk mode
(`sync_each_write = false`) writes use `NORMAL` until `flush()`. `flush()` then writes each changed snapshot to a temporary
file, fsyncs it, renames it into place, checkpoints the WAL with `synchronous = FULL`, and on Unix fsyncs the directory.
Ids are one 64-bit space shared by rows, vectors and graph nodes. Full detail: `crates/quickdb-native/README.md`.

## Measured results

Every engine sits behind the same `Store` contract and is driven by the same harness binary (`crates/quickdb-bench`), one
engine at a time, on the same x86_64 Linux virtual machine (Rust 1.91.1, release profile). Cells are medians of 3 runs unless
marked. The columns come from two rounds: **R10** (4 CPUs, 10 GB, SQLite 3.54.1) and **R4** (6 CPUs, 16 GB, SQLite 3.50.2).
The comparison adapters are untuned. Method, datasets and commands: `BENCH.md`.

### SIFT-1M: 1,000,000 × 128, L2, 1,000 queries, published ground truth

| Measure | **QDB native** (R10) | SQLite `vec1` 0.8 (R10) | SQLite + `sqlite-vec` 0.1.9 (R4) | LanceDB 0.40 (R4) |
|---|---:|---:|---:|---:|
| Vector index | HNSW | IVF + product quantisation | exact scan (no index) | IVF_HNSW_FLAT |
| knn p50 at recall ≈ 0.96 | **0.43 ms** (0.963) | 1.33 ms (0.965) | 272 ms (0.999, exact) | 4.24 ms (0.963) |
| knn p50 at recall ≈ 0.99 | **0.80 ms** (0.990) | 2.65 ms (0.992) | 272 ms (0.999, exact) | 5.28 ms (0.988) |
| Filtered knn (ef 64) recall / p50 | 0.992 / 10.1 ms | 0.850 / **7.5 ms** | 1.000 / 271 ms | 1.000 / 342 ms |
| 1-hop neighbours p50 | **9.0 µs** | 40 µs | 26.7 µs | 3,685 µs |
| 3-hop neighbours p50 | **226 µs** ¹ | — | 505 µs | 29,611 µs |
| `get_row` p50 | 12.0 µs | **8.1 µs** | 7.9 µs | 1,398 µs |
| Vector index build (load + flush) | 262 s | 30 s | **13 s** (no index) | 474 s |
| Size on disk | 2,026 MB | 1,299 MB | **1,130 MB** | 2,740 MB |
| Memory | ~3.8 GB added by the store | **~0.06 GB** added | 1.9 GB process ² | 4.2 GB process ² |
| Reopen | 4.0 s | **< 1 ms** | **< 1 ms** | 60 ms |

Not in the table: CozoDB 0.7.6 (RocksDB) and `sqlite-vec` 0.1.10-alpha.4 DiskANN did not finish loading SIFT-1M within the
45-minute limit (one run each).

### Research corpus: 22,666 chunks × 1,024 (BGE-M3), cosine, Arabic and English

| Engine | knn p50 / recall (ef 32) | best recall: knn p50 / recall | Filtered knn p50 / recall | Load | Disk | Reopen |
|---|---:|---:|---:|---:|---:|---:|
| **QDB native** (R10) | **0.24 ms** / 0.990 | **0.70 ms** / 1.000 | **0.47 ms** / 1.000 | 8.5 s | 243 MB | 0.40 s |
| SQLite `vec1` 0.8 (R10) | 1.13 ms / 0.895 | 3.16 ms / 0.994 | 88.6 ms / 0.809 | 56.8 s | 156 MB | < 1 ms |
| `sqlite-vec` DiskANN alpha.4 (R10, 2 runs) | 0.72 ms / 0.762 | 5.68 ms / 0.983 | 387 ms / 0.857 | 74.0 s | 332 MB | < 1 ms |
| SQLite + `sqlite-vec` 0.1.9 (R4) | 63.1 ms / 1.000 (exact) | 63.1 ms / 1.000 | 44.9 ms / 1.000 | 9.5 s | **131 MB** | **< 1 ms** |
| LanceDB 0.40 (R4) | 3.98 ms / 0.991 | 6.78 ms / 0.992 | 10.2 ms / 1.000 | 19.5 s | 402 MB | 3.1 ms |
| CozoDB 0.7.6, RocksDB (R4) | 11.5 ms / 0.983 | 62.2 ms / 0.997 | 2,360 ms / 0.999 | 645 s | 143 MB | 0.52 s |

### Against `hnswlib` 0.8.0 (the reference HNSW library)

| Measure | QDB native | `hnswlib` 0.8.0 |
|---|---:|---:|
| SIFT-1M index build, 6 CPUs | 224 s (3 runs, 223–225 s) | 192–221 s (single runs) |
| GIST-1M (960-d) recall@10 at ef 64 / 256 | 0.755 / 0.930 | 0.755 / 0.926 (single runs) |
| Float build time, GIST (960-d) | 1.20–1.24× `hnswlib` | 1.00× |

### What the matrix says

- **Fastest answers.** At the same recall QDB answers vector queries 3.1–3.3× faster than `vec1` and 6.6–9.9× faster
  than LanceDB on SIFT-1M, and 8–23× faster than the SQLite extensions on the corpus. Filtered queries keep full recall
  where the SQLite extensions drop to 0.81–0.86.
- **The price is build time, memory and reopen time.** QDB keeps an HNSW graph in memory: about 4.25 KB per 1,024-d
  vector, ~3.8 GB at 1M × 128, 4 s to reopen at 1M. `vec1` is the choice when memory and build time matter more than speed.
- **Rows and edges are plain SQLite** in the SQLite-based engines. QDB's covering edge indexes make its hops the fastest.

### Reliability and platforms

| Check | Result |
|---|---|
| Crash safety | 240 `kill -9` runs, 0 failures; 202 damaged stores recovered; emulated Linux power loss, 3,000 images, 0 lost (assumes honest `fsync`) |
| Conformance suite (12 groups) | QDB native, `sqlite-vec` 0.1.9, `vec1`, DiskANN pass; CozoDB and LanceDB fail 6 each |

### Platforms

| Platform | Release tests (passed / failed / ignored) | clippy `-D warnings` |
|---|---:|---|
| Linux x86_64 (Rust 1.91.1, also GitHub CI) | 155 / 0 / 8 | clean |
| macOS 26 on Apple silicon (M1 Max, aarch64) | 148 / 0 / 11 | clean ³ |
| Windows 11 on ARM64 (aarch64 MSVC) | 148 / 0 / 11 | clean ³ |
| Windows x64 (MSVC) and Intel macOS 10.13 | pass (earlier revision) | clean |

ARM64 runs fewer tests: four HNSW unit tests cover x86_64 SIMD kernels only, and the power-loss tests run on Linux only.

### Apple silicon and ARM64: 100,000 × 1,024 synthetic vectors (single run each)

| Measure | macOS, M1 Max (10 cores) | Windows 11 ARM64 VM (4 vCPU) |
|---|---:|---:|
| Load vectors (index build) | 27.8 s | 57.0 s |
| Flush / reopen | 0.53 s / 1.09 s | 0.80 s / 2.60 s |
| knn p50 at ef 32 / 64 / 128 | 376 / 570 / 867 µs | 528 / 856 / 1,349 µs |
| Recall@10 at ef 32 / 64 / 128 | 0.840 / 0.897 / 0.944 | 0.840 / 0.897 / 0.944 |
| Filtered knn (ef 64) p50 / recall | 1.93 ms / 1.000 | 2.29 ms / 1.000 |
| `get_row` p50 | 4.1 µs | 12.2 µs |
| 1-hop / 3-hop p50 | 87.5 / 199 µs | 14.0 / 293 µs |
| Size on disk | 962 MB | 962 MB |

Same graph and same answers on both: recall and size are identical. Synthetic queries are perturbed stored vectors, so recall
here is lower than on SIFT or the corpus and is not comparable with the tables above.

¹ R5 (6 CPUs, same machine), 3 runs. ² Whole-process memory after load, a different measure from the R10 column.
³ After the fix for an x86-only field that clippy flagged on ARM64; tests passed before and after.

## Limits

- Vector indexes stay in RAM (no disk spill). Leave RAM above the index for the OS file cache.
- Float index builds trail `hnswlib` by about 1.2×.
- Lazy filtered search constants were fitted on SIFT; other data may need a larger `ef_search`.
- The snapshot file format is not a stable interchange format yet.
- One writer per store; not a replacement for a multi-writer database server.

## Build and test

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
cd crates/quickdb-cozo && cargo test --release     # each adapter is its own workspace
```

CozoDB and the native engine cannot be linked into one binary: both bundle SQLite under the same link name, which is why
the adapters are separate workspaces.

## Licence

MIT OR Apache-2.0, at your option (`LICENSE-MIT`, `LICENSE-APACHE`). Authors: `AUTHORS.md`. Contributing: `CONTRIBUTING.md`.
Security: `SECURITY.md`.
