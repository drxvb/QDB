"""Type stubs for the native module of :mod:`quickdb`."""

import os
import sys
from types import TracebackType
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

__version__: str

Id = int
"""Row, vector and node id: an int in ``0..=2**63-1``."""

ColumnTypeName = Literal["int", "integer", "i64", "float", "real", "f64", "text", "str", "bytes", "blob"]
ColumnTypeSpec = Union[ColumnTypeName, Type[int], Type[float], Type[str], Type[bytes]]
ColumnSpec = Union[Tuple[str, ColumnTypeSpec, bool], Tuple[str, ColumnTypeSpec]]

Value = Union[None, int, float, str, bytes]
"""A cell. ``None`` is NULL in every column; ``bytearray`` is also accepted on input."""

RowDict = Dict[str, Any]
"""A row read back: ``"id"`` plus one key per column."""

RowInput = Union[Mapping[str, Any], Tuple[Id, Sequence[Any]]]
"""A row to write: a dict with ``"id"`` (missing columns are NULL) or ``(id, [values])``."""

Filter = Optional[Mapping[str, Sequence[Any]]]
"""``None`` (every row), ``{"eq": [col, value]}``, ``{"range": [col, lo, hi]}`` (int column,
inclusive, ``None`` for an open end) or ``{"prefix": [col, text]}``."""

Vector = Union[Sequence[float], Buffer]
"""A float list, or a 1-D float32/float64 buffer such as a numpy array."""

Vectors = Union[Sequence[Sequence[float]], Sequence[Buffer], Buffer]
"""A list of float lists, or a 2-D float32/float64 buffer such as a numpy array."""

EdgeInput = Union[Tuple[Id, str, Id, float], Tuple[Id, str, Id]]
"""``(src, rel, dst, weight)``; ``weight`` defaults to 1.0."""

Direction = Literal["out", "in", "both"]
Metric = Literal["cosine", "l2"]

class QuickDBError(Exception):
    """Base class of every error raised by quickdb."""

class NotFoundError(QuickDBError):
    """Unknown table, vector index or column."""

class SchemaError(QuickDBError):
    """Input that does not fit a definition (row shape, value type, name, id range, non-finite
    value, malformed filter or vector buffer)."""

class StorageError(QuickDBError):
    """Operating-system error on the store directory, including a missing store."""

class BackendError(QuickDBError):
    """Error from the storage engine, including a store locked by another writer and a write
    through a read-only handle."""

class Store:
    """An open QDB store: relational rows, vectors and graph edges in one directory.

    Only one writable handle per directory may exist at a time, in any process; a second one
    raises :class:`BackendError`. Use :meth:`open_read_only` for readers. Every write is atomic;
    :meth:`flush` makes everything written so far durable. Leaving a ``with`` block without an
    exception flushes and closes the store. Store calls release the GIL.
    """

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        sync_each_write: bool = True,
        create: bool = True,
    ) -> None:
        """Open the store in ``path`` for writing, creating it unless ``create=False``."""
    @staticmethod
    def open_read_only(path: Union[str, "os.PathLike[str]"]) -> "Store":
        """Open an existing store for reading. Rows and edges are read live; vectors, and the
        set of tables and indexes, are those committed when this handle opened."""
    @property
    def path(self) -> str: ...
    @property
    def read_only(self) -> bool: ...
    @property
    def closed(self) -> bool: ...
    def __enter__(self) -> "Store": ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        _exc: Optional[BaseException] = None,
        _tb: Optional[TracebackType] = None,
    ) -> bool: ...

    # relational
    def define_table(self, name: str, columns: Iterable[ColumnSpec]) -> None:
        """Create a table; an identical redefinition does nothing, a different one is
        :class:`SchemaError`. Names match ``[a-z][a-z0-9_]*``."""
    def upsert_rows(self, table: str, rows: Iterable[RowInput]) -> None:
        """Insert or replace rows, atomically for the batch."""
    def get_row(self, table: str, id: Id) -> Optional[RowDict]:
        """The row with this id, or None."""
    def select(self, table: str, filter: Filter = None, limit: int = 1000) -> List[RowDict]:
        """Rows matching ``filter``, ascending id, at most ``limit``."""
    def delete_rows(self, table: str, ids: Iterable[Id]) -> int:
        """Delete rows; returns how many existing rows were removed."""
    def count_rows(self, table: str) -> int: ...

    # vectors
    def define_vector_index(
        self,
        name: str,
        dim: int,
        metric: Metric = "cosine",
        m: int = 16,
        ef_construction: int = 200,
    ) -> None:
        """Create an HNSW vector index of ``dim`` float32 components."""
    def upsert_vectors(self, index: str, ids: Iterable[Id], vectors: Vectors) -> None:
        """Insert or replace one vector per id."""
    def knn(self, index: str, query: Vector, k: int = 10, ef: int = 64) -> List[Tuple[Id, float]]:
        """Up to ``k`` nearest ``(id, distance)`` pairs, nearest first."""
    def knn_filtered(
        self,
        index: str,
        query: Vector,
        k: int,
        ef: int,
        table: str,
        filter: Filter,
    ) -> List[Tuple[Id, float]]:
        """Like :meth:`knn`, limited to ids whose row in ``table`` matches ``filter``."""
    def define_text_index(self, name: str, table: str, columns: List[str]) -> None:
        """Full-text index over text ``columns`` of ``table`` (FTS5, BM25, Arabic folding)."""
    def text_search(
        self, index: str, query: str, k: int = 10, match: str = "all"
    ) -> List[Tuple[Id, float]]:
        """Up to ``k`` ``(id, score)`` pairs, best first. ``match`` is "all" (every word) or "any"."""
    def hybrid_search(
        self,
        text_index: str,
        vector_index: str,
        query_text: str,
        query_vector: Vector,
        k: int = 10,
        ef: int = 64,
    ) -> List[Tuple[Id, float]]:
        """Text (any word) and vector results fused by Reciprocal Rank Fusion (k = 60), best first."""
    def count_vectors(self, index: str) -> int: ...

    # graph
    def upsert_edges(self, edges: Iterable[EdgeInput]) -> None:
        """Insert or replace edges; an edge is identified by ``(src, rel, dst)``."""
    def neighbors(
        self,
        node: Id,
        rel: Optional[str] = None,
        direction: Direction = "out",
        limit: int = 1000,
    ) -> List[Tuple[Id, str, Id, float]]:
        """Edges touching ``node``, ordered by ``(rel, src, dst)``."""
    def k_hop(
        self,
        start: Id,
        rel: Optional[str] = None,
        direction: Direction = "out",
        depth: int = 1,
        limit: int = 1000,
    ) -> List[Tuple[Id, int]]:
        """Nodes within ``depth`` hops as ``(id, hops)``, ordered by ``(hops, id)``."""
    def count_edges(self) -> int: ...

    # lifecycle
    def flush(self) -> None:
        """Make everything written so far durable."""
    def close(self) -> None:
        """Flush (writable handle) and close. Closing twice does nothing."""
