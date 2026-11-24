# quickdb-py

Python bindings for **QDB (QuickDB by Rabie and Basil)**: relational rows, vectors and a graph in one embedded, in-process store,
with no server. One directory holds a table's rows, their embeddings (an HNSW index) and the edges between them, and all three use
the same ids.

The distribution is `quickdb-py`; the import name is `quickdb`. Wheels use the stable ABI (`abi3`), so one wheel per platform
serves CPython 3.9 and later. numpy is optional.

```sh
pip install quickdb-py            # from a built wheel: pip install quickdb_py-*.whl
```

## Example

```python
import quickdb

with quickdb.Store("catalogue.qdb") as db:
    db.define_table("books", [("title", "text", False), ("city", "text", True), ("year", "int", True)])
    db.upsert_rows("books", [
        {"id": 1, "title": "كليلة ودمنة", "city": "بغداد", "year": 750},
        {"id": 2, "title": "مقدمة ابن خلدون", "city": "فاس", "year": 1377},
        (3, ["طوق الحمامة", "قرطبة", 1022]),          # (id, [values]) also works
    ])

    db.define_vector_index("emb", 4, metric="l2")    # metric "cosine" (default) or "l2"
    db.upsert_vectors("emb", [1, 2, 3], [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0]])

    db.upsert_edges([(1, "cites", 2, 1.0), (2, "cites", 3, 1.0)])

    db.knn("emb", [1, 0, 0, 0], k=2)                  # [(1, 0.0), (2, 0.0199...)]
    db.knn_filtered("emb", [1, 0, 0, 0], 2, 64, "books", {"range": ["year", 1000, None]})   # [(2, ...), (3, ...)]
    db.k_hop(1, "cites", depth=2)                     # [(2, 1), (3, 2)]
    db.get_row("books", 2)                            # {'id': 2, 'title': 'مقدمة ابن خلدون', 'city': 'فاس', 'year': 1377}
# leaving the block flushes and closes the store
```

## API

All methods are on `quickdb.Store`. Ids are ints in `0..=2**63-1`. Names of tables, columns and indexes match `[a-z][a-z0-9_]*`.

| Method | Returns |
|---|---|
| `Store(path, sync_each_write=True, create=True)` | a writable store; a context manager |
| `Store.open_read_only(path)` | a read-only store beside the writer |
| `define_table(name, columns)` — `columns`: `[(name, type, indexed)]`, type `"int"`, `"float"`, `"text"`, `"bytes"` (or `int`, `float`, `str`, `bytes`) | `None` |
| `upsert_rows(table, rows)` — dicts with `"id"`, or `(id, [values])` | `None` |
| `get_row(table, id)` | `dict` or `None` |
| `select(table, filter=None, limit=1000)` | `list[dict]`, ascending id |
| `delete_rows(table, ids)` | number of rows removed |
| `define_vector_index(name, dim, metric="cosine", m=16, ef_construction=200)` | `None` |
| `upsert_vectors(index, ids, vectors)` — list of lists or a 2-D float buffer | `None` |
| `knn(index, query, k=10, ef=64)` | `[(id, distance)]`, nearest first |
| `knn_filtered(index, query, k, ef, table, filter)` | `[(id, distance)]` |
| `upsert_edges([(src, rel, dst, weight)])` — weight defaults to 1.0 | `None` |
| `neighbors(node, rel=None, direction="out", limit=1000)` | `[(src, rel, dst, weight)]` |
| `k_hop(start, rel=None, direction="out", depth=1, limit=1000)` | `[(id, hops)]`, by hops then id |
| `count_rows(table)`, `count_vectors(index)`, `count_edges()` | `int` |
| `flush()`, `close()`; properties `path`, `read_only`, `closed` | |

**Values.** `None` is NULL in every column. An `int` is accepted in a float column; nothing else is converted. Text is stored
as UTF-8 and compared byte-wise. A dict row may leave out columns (they become NULL); an unknown key is an error.

**Filters** for `select` and `knn_filtered`: `None` (every row), `{"eq": [column, value]}`, `{"range": [int_column, lo, hi]}`
(inclusive, `None` for an open end) and `{"prefix": [text_column, prefix]}`.

**Distances.** `"l2"` is squared Euclidean distance; `"cosine"` is one minus cosine similarity. `ef` is the HNSW search breadth:
larger is slower and finds more of the true nearest neighbours.

**Vectors and numpy.** `upsert_vectors` takes a list of float lists, a list of 1-D arrays, or any object with the buffer protocol
and two dimensions: a numpy array, for example, of float32 or float64 (narrowed to float32), in any byte order or memory layout.
Queries are a float list or a 1-D buffer. The buffer is copied once into the store's format.

## Durability, readers and threads

- Every write call is atomic. `flush()` makes everything written so far durable. A `with` block flushes and closes on a clean exit;
  after an exception it closes without flushing. `sync_each_write=False` is faster for bulk loads; `flush()` is still durable.
- One writable `Store` per directory, across all processes: a second one raises `BackendError` until the first is closed.
- Any number of `Store.open_read_only(path)` handles may read beside it. They see new rows and edges as they are written;
  vectors, and the set of tables and indexes, are those committed when the handle opened. Reopen a reader to see a table or
  index created later.
- A `Store` may be shared between threads. Calls on one store run one at a time; each call releases the GIL, so other Python
  threads keep running during a search, a bulk upsert or a flush.

## Errors

Every error the store reports is a subclass of `quickdb.QuickDBError`:

| Exception | When |
|---|---|
| `NotFoundError` | unknown table, vector index or column |
| `SchemaError` | wrong row shape or value type, bad name, a different redefinition, id out of range, NaN or infinite value, malformed filter or vector buffer |
| `StorageError` | operating-system error, including opening a missing store with `create=False` |
| `BackendError` | storage-engine error, including a store locked by another writer and a write through a read-only handle |

Arguments of the wrong Python type for a plain parameter (for example `k="ten"`) raise `TypeError` as usual.

## Building from source

Needs Rust 1.91 or later and Python 3.9 or later.

```sh
cd crates/quickdb-py
pip install maturin pytest numpy
maturin build --release              # wheel in target/wheels/ (or $CARGO_TARGET_DIR/wheels/)
pip install target/wheels/quickdb_py-*.whl
pytest
```

This crate is its own cargo workspace, so `cargo` commands run inside `crates/quickdb-py`.

## Licence

MIT OR Apache-2.0.
