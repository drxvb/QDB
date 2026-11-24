"""QDB (QuickDB): relational rows, vectors and a graph in one embedded, in-process store.

Rows, vectors and graph edges share one id space (ints in ``0..=2**63-1``), so a document's row,
its embedding and its links all use the same id::

    import quickdb

    with quickdb.Store("library.qdb") as db:
        db.define_table("books", [("title", "text", False), ("year", "int", True)])
        db.upsert_rows("books", [{"id": 1, "title": "Kalila wa Dimna", "year": 750}])
        db.define_vector_index("emb", 3, metric="l2")
        db.upsert_vectors("emb", [1], [[0.1, 0.2, 0.3]])
        print(db.knn("emb", [0.1, 0.2, 0.3], k=1))   # [(1, 0.0)]
"""

from ._quickdb import (
    BackendError,
    NotFoundError,
    QuickDBError,
    SchemaError,
    StorageError,
    Store,
    __version__,
)

__all__ = [
    "BackendError",
    "NotFoundError",
    "QuickDBError",
    "SchemaError",
    "StorageError",
    "Store",
    "__version__",
]
