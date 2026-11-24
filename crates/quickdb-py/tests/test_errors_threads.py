"""The exception hierarchy, error messages, typing files, and use from several threads."""

import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import quickdb


def test_exception_hierarchy():
    for cls in (quickdb.NotFoundError, quickdb.SchemaError, quickdb.StorageError, quickdb.BackendError):
        assert issubclass(cls, quickdb.QuickDBError)
        assert cls.__module__ == "quickdb"
    assert issubclass(quickdb.QuickDBError, Exception)
    assert quickdb.Store.__module__ == "quickdb"
    assert isinstance(quickdb.__version__, str)


def test_error_messages_name_the_problem(catalogue):
    with pytest.raises(quickdb.NotFoundError, match="nope"):
        catalogue.count_rows("nope")
    with pytest.raises(quickdb.SchemaError, match='column "year"'):
        catalogue.upsert_rows("books", [{"id": 1, "year": "x"}])
    with pytest.raises(quickdb.SchemaError, match="2\\*\\*63-1"):
        catalogue.get_row("books", -1)
    try:
        catalogue.knn("emb", [1.0])
    except quickdb.QuickDBError as e:  # catchable through the base class
        assert "schema" in str(e)
    else:
        pytest.fail("knn with a wrong dimension did not raise")


def test_package_ships_typing_files():
    pkg = pathlib.Path(quickdb.__file__).parent
    assert (pkg / "py.typed").is_file()
    stub = (pkg / "_quickdb.pyi").read_text(encoding="utf-8")
    for name in ("class Store", "def knn_filtered", "def k_hop", "class QuickDBError"):
        assert name in stub
    # Every public method of Store is in the stub.
    public = [n for n in dir(quickdb.Store) if not n.startswith("_")]
    missing = [n for n in public if f"def {n}(" not in stub]
    assert missing == []


def test_parallel_reads_from_threads(catalogue):
    queries = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]] * 40
    serial = [catalogue.knn("emb", q, k=3) for q in queries]
    with ThreadPoolExecutor(max_workers=6) as pool:
        parallel = list(pool.map(lambda q: catalogue.knn("emb", q, k=3), queries))
        rows = list(pool.map(lambda i: catalogue.get_row("books", i), [1, 2, 3, 4, 5, 6] * 20))
    assert parallel == serial
    assert all(r is not None for r in rows)


def test_writes_from_threads_are_serialised(db):
    db.define_table("t", [("x", "int")])

    def write(start):
        for i in range(start, start + 50):
            db.upsert_rows("t", [(i, [i])])

    threads = [threading.Thread(target=write, args=(k * 1000,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert db.count_rows("t") == 200


def test_bulk_upsert_releases_the_gil(db):
    """While one thread runs a long upsert_vectors, the main thread keeps executing Python."""
    np = pytest.importorskip("numpy")
    n, dim = 20000, 64
    data = np.random.default_rng(1).standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n)
    db.define_vector_index("big", dim, metric="l2", m=16, ef_construction=100)
    done = []
    worker = threading.Thread(target=lambda: done.append(db.upsert_vectors("big", ids, data)))
    start = time.perf_counter()
    worker.start()
    last, max_gap, ticks = start, 0.0, 0
    while worker.is_alive():
        now = time.perf_counter()
        max_gap = max(max_gap, now - last)
        last = now
        ticks += 1
    worker.join()
    elapsed = time.perf_counter() - start
    assert done == [None] and db.count_vectors("big") == n
    if elapsed < 0.3:
        pytest.skip(f"upsert took {elapsed:.2f} s, too short to show the GIL release")
    # With the GIL held for the whole call, the main loop would stall for about `elapsed`.
    assert max_gap < elapsed / 3, (max_gap, elapsed, ticks)
