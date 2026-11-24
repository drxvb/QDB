# quickdb-bench

One generic `run<S: Store>` loads and measures identical operations for every backend. `MemStore` is the exact reference, not a production storage engine. Each candidate crate supplies its own `qbench` example using `quickdb_bench::main_with`, so engines with incompatible SQLite link dependencies need not share a binary.

## Commands

From the repository root (Linux; tests build real indexes, so use `--release`):

```text
cargo test -p quickdb-bench --release
cargo clippy -p quickdb-bench --all-targets -- -D warnings
cargo run -p quickdb-bench --release -- conformance --backend mem
cargo run -p quickdb-bench --release -- synthetic --backend mem --rows 10000 --dim 64 --data-dir ./mem10k --out ./mem10k.json
cargo run -p quickdb-bench --release -- corpus --backend mem --corpus crates/quickdb-bench/tests/fixture-corpus --data-dir ./corpus --out ./corpus.json
```

Use `cargo run --release -p quickdb-native --example qbench -- ... --backend native` from the root workspace, or run the equivalent example inside another candidate's standalone workspace. The data directory must be missing or empty. Existing result files are not replaced. Errors go to stderr and cause a nonzero exit. Successful benchmark stdout is exactly one JSON object, also written to `--out`. Conformance emits one JSON object with `conformance: pass|fail` and a `failures` array, and fails the process if any check fails. Conformance directories are retained under `QUICKDB_DATA` (otherwise the OS temporary directory), named with backend and process id.

## ANN files

`qbench ann --backend <backend> --base <base.fvecs> --query <query.fvecs> --truth <truth.ivecs> --limit <N> --queries <Q> --metric <l2|cosine> --data-dir <empty-directory> --out <new-report.json>`

The ANN loader uses the first N base vectors and first Q supplied queries, with zero-based vector and row IDs. Every fvecs/ivecs record has a little-endian signed 32-bit dimension followed by that many 32-bit values. The loader validates all record sizes, matching dimensions, finite floats, positive requested counts, and truth IDs. Truth records must contain at least ten neighbors. Truncated or malformed files return `Error::Schema`.

When N is the full base file count, `dataset.truth_source` is `ivecs` and recall uses the first ten published IDs per query. The supplied truth must correspond to the requested metric. With a prefix, `dataset.truth_source` is `exact_prefix` and the harness computes exact top ten over only the loaded vectors using that metric. Filtered truth is always computed independently over eligible loaded vectors. L2 uses squared Euclidean distance. Synthetic and corpus retain cosine distance and their 200 seeded perturbed queries. ANN uses Q unmodified file queries for every vector and mixed phase, while its rows and edges reuse the synthetic distributions with IDs shifted to zero-based. The existing report fields and query-family names are preserved.

Datasets such as SIFT-1M and GIST-1M (http://corpus-texmex.irisa.fr/) are not shipped; point `--base`, `--query` and `--truth` at local copies.

## Result schema, version 1

All fields are required unless marked optional. Timings are elapsed wall-clock time, not CPU time.

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1` |
| `backend` | Backend's stable `Store::backend_name()` |
| `dataset` | Synthetic: `type`, `rows`, `dim`, `seed`, `clusters`. Corpus: `type`, original `meta` object. ANN: `type`, `rows`, `dim`, `base_count`, `queries`, `metric`, file paths, `truth_source` |
| `rows`, `vectors`, `edges` | Counts read from the reopened store. Edge count reflects upsert deduplication |
| `metric` | `cosine` or ANN's `l2`, matching the benchmark vector index |
| `query_seed` | Seed used to create 200 held-out synthetic/corpus queries. Retained for compatibility, not used for ANN file queries |
| `hostname` | `QUICKDB_HOST` if explicitly set, otherwise `/etc/hostname`. In Docker this is the container hostname, not the underlying host. The wrapper command identifies the physical host |
| `cpu_count` | Logical processors visible in `/proc/cpuinfo` |
| `available_parallelism` | Rust's estimate of available CPUs, including container quota |
| `phases_seconds` | Map of phase name to seconds, listed below |
| `size_bytes` | Recursive apparent file size measured after flush, before drop/reopen. Not allocated blocks |
| `memory` | Snapshots `before_open`, `after_load`, `after_reopen`, `after_queries`, each with `vm_hwm_bytes`, `vm_rss_bytes` from `/proc/self/status`. `null` if unavailable |
| `queries` | Query-family map, listed below |

Phases: `prepare_dataset` (CLI only), `open`, `define_schema`, `load_rows`, `load_vectors`, `load_edges`, `flush`, `size_on_disk`, `drop`, `reopen`, `get_row`, `select_eq`, `select_int_range`, `prepare_queries`, `exact_truth`, `knn_ef32`, `knn_ef64`, `knn_ef128`, `knn_ef256`, `exact_filtered_truth`, `knn_filtered_ef64`, `k_hop_depth1`, `k_hop_depth2`, `k_hop_depth3`, `mixed`, `total`. `total` starts just before open and ends after all queries, counts and memory sampling. It excludes dataset preparation and JSON output. Individual query phases include loop overhead, percentile sorting and recall bookkeeping. Exact truth is never included in backend knn latencies.

Each query-family value has `samples`, `p50_us`, `p95_us`, `p99_us`, `mean_results` and optionally `recall_at_10`. Percentiles use sorted nearest-rank (`ceil(p*N)`, one-based), in microseconds. `mean_results` is the average rows/hits/nodes returned, not total nodes explored. For mixed queries it is the mean existing rows fetched.

Query families:

* `get_row`: 10,000 deterministic random existing input row ids.
* `select_eq`: 500 queries, cycling distinct indexed `kind` text values, limit 100.
* `select_int_range`: 500 queries, ranges anchored at random input `size`, inclusive width 100, limit 100.
* `knn_ef32`, `knn_ef64`, `knn_ef128`, `knn_ef256`: same 200 vectors, k=10, with `recall_at_10` against an independent brute-force top-10 computed from input. Queries are input cluster/corpus vectors with deterministic perturbations, explicitly checked not to equal any stored vector.
* `knn_filtered_ef64`: same 200 queries, k=10, first sorted `kind`, independently filtered truth, recall included.
* `k_hop_depth1..3`: same 500 random row ids, Out, all relations, limit 1,000. `mean_results` is mean nodes reached after truncation.
* `mixed`: same 200 query vectors, knn k=10 ef=64, then Out depth-2 limit-50 BFS from each hit. Deduplicate reached ids, fetch up to 20 in ascending id order.

Recall is the mean per-query intersection divided by `min(10, eligible vector count)`, with ties ordered by id. Duplicate backend hits do not inflate recall. The synthetic graph has one ring successor plus four cubic-biased destinations per source, producing hub-heavy in-degrees; duplicate edge identities are upserted. No latency warm-up or concurrency is used. Calls are sequential and phase order is fixed. Comparisons should use the same host, CPU quota, dimensions, datasets and release profile. Memory is whole-process memory: it includes retained input data and exact truth, allocator retention and the backend. It is not an isolated backend footprint. The reference writes a versioned JSON snapshot (`mem.json`) using a synced temporary file and rename, then syncs the directory on Unix. `sync_each_write` flushes each successful mutation. A failed input batch makes no partial changes.

## Written questions (corpus only)

`corpus --questions <dir>` adds a `questions` report block: doc-hit@10, document-level MRR@10, recall against exact
and p50 latency for `exact` and ef 32/64/128/256, split by `lang` and `corpus`, with an `unresolved` count. The
directory holds `questions.jsonl`, `qvectors.f32` and optionally `qmeta.json`, made by `scripts/corpus_questions.py`.
Without the option the report keys are unchanged.

## Corpus preparation

Python 3.9+ standard library only:

```text
python scripts/corpus_prepare.py --roots path/to/corpus --out path/to/new-output --max-chunks 150000
python scripts/corpus_prepare.py --roots crates/quickdb-bench/tests/fixture-source --out path/to/new-fixture --no-embed --dim 8 --batch 2
python crates/quickdb-bench/tests/test_corpus_prepare.py
```

The embedding service is any OpenAI-compatible endpoint: `--embed-url` (default `$QUICKDB_EMBED_URL`, else `http://127.0.0.1:8081/v1`). The model id is read from `/models`. `/embeddings` is called one batch at a time, ordered by returned index, with three retries after the initial attempt. Requests have a 180-second timeout. HTTP and JSON errors are reported without writing `meta.json`. Invalid embedding shapes abort. `--no-embed` uses a seeded Python PRNG and writes the same f32 bytes independent of batch size. The script refuses nonempty output directories and never deletes existing evidence. Partial output has no `meta.json`.

Files and directory queues are sorted in a temporary on-disk SQLite index, so directory width does not require keeping all names in memory. Embedding responses are capped at 32 MiB, dimensions at 65,536 and `batch * dim` at 1,048,576 values. Metadata records process peak memory through Windows process counters or Unix `getrusage`. A maximum-file-size test with 1,024-dimensional no-embed vectors checks the 500 MiB ceiling.

Only the provided roots are walked. Symlinks, sensitive names, excluded directories, files above 2 MiB and invalid UTF-8 are skipped. Skipped-file counters are recorded in metadata. Supported suffixes are case-insensitive. Files are read one at a time, chunked around 1,200 Unicode code points at paragraph boundaries, with oversized paragraphs split at whitespace or the character boundary. UTF-8 and Arabic logical order are preserved, with paragraph whitespace normalized. Documents start at 1, directories at 500,000 and chunks at 1,000,000. Namespace exhaustion aborts instead of overlapping ids. Relative inline Markdown links to included documents create `links` edges, resolved after the walk by a temporary on-disk SQLite index. Remote, absolute, reference-style and image links are not imported. Disk indexes are removed on success or failure. Directory nodes appear as edge endpoints, not document rows.

The five output files are `chunks.jsonl`, `docs.jsonl`, `edges.jsonl`, `vectors.f32` (little-endian, row-major in chunk order) and `meta.json`. Metadata includes counts, directories, dim, model, roots, per-corpus document/chunk counts, UTC start/end times, skipped counts, no-embed flag, vector byte length and process peak memory. Each doc object has `id`, `corpus`, `path`, `chunks` and full-source `chars`. The Rust loader maps chunks to benchmark rows (`kind=corpus`, `title=path#seq`, `size=chars`, `score=seq`, `body=text`) and imports all corpus edges unchanged. It validates JSONL counts, unique chunk ids, Unicode char counts, vector byte length and finite embeddings. Graph starts are chunk ids, matching the vector/row id space. Corpus graph degrees differ intentionally from synthetic graph degrees.

The no-embed fixture under `tests/fixture-corpus` was generated by the command above from the checked-in source folder, with a different output argument. Tests load its real binary vectors rather than synthesizing substitute embeddings. Python tests cover corpus filtering, next/contains/has/links edges, chunk limits, deterministic vectors, retries and malformed embedding responses without starting a server.
