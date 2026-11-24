"""Opening, closing, the context manager, read-only handles, reopen and the single-writer lock."""

import subprocess
import sys

import pytest

import quickdb
from conftest import COLUMNS


def test_context_manager_flushes_and_closes(path):
    with quickdb.Store(path) as db:
        assert isinstance(db, quickdb.Store)
        db.define_table("books", COLUMNS)
        db.upsert_rows("books", [(1, ["أ", "ب", 1])])
        assert not db.closed
    assert db.closed
    with quickdb.Store(path, create=False) as db2:
        assert db2.get_row("books", 1) == {"id": 1, "title": "أ", "city": "ب", "year": 1}


def test_context_manager_on_exception_closes_and_releases_lock(path):
    with pytest.raises(RuntimeError):
        with quickdb.Store(path) as db:
            db.define_table("t", [("x", "int")])
            raise RuntimeError("boom")
    assert db.closed
    # The writer lock is released: a new writer opens.
    with quickdb.Store(path) as db2:
        assert db2.count_rows("t") == 0


def test_close_twice_and_use_after_close(path):
    db = quickdb.Store(path)
    db.close()
    db.close()
    with pytest.raises(quickdb.QuickDBError, match="closed"):
        db.count_edges()
    with pytest.raises(quickdb.QuickDBError, match="closed"):
        db.knn("emb", [1.0])
    with pytest.raises(quickdb.QuickDBError, match="closed"):
        db.select("t")


def test_properties_and_repr(path):
    with quickdb.Store(path, sync_each_write=False) as db:
        assert db.path == path
        assert db.read_only is False
        assert "read_only=False" in repr(db) and "closed=False" in repr(db)
    with quickdb.Store.open_read_only(path) as ro:
        assert ro.read_only is True
        assert "read_only=True" in repr(ro)


def test_pathlike_accepted(tmp_path):
    with quickdb.Store(tmp_path / "p") as db:
        db.define_table("t", [("x", int)])
    assert (tmp_path / "p" / "quickdb.sqlite").is_file()


def test_create_false_on_missing_store(tmp_path):
    with pytest.raises(quickdb.StorageError):
        quickdb.Store(str(tmp_path / "missing"), create=False)
    with pytest.raises(quickdb.StorageError):
        quickdb.Store.open_read_only(str(tmp_path / "missing"))


def test_reopen_keeps_everything(path, catalogue):
    catalogue.close()
    with quickdb.Store(path, create=False) as db:
        assert db.count_rows("books") == 6
        assert db.count_vectors("emb") == 6
        assert db.count_edges() == 5
        assert db.get_row("books", 3)["title"] == "ألف ليلة وليلة"
        assert [i for i, _ in db.knn("emb", [1.0, 0.0, 0.0, 0.0], k=3)] == [1, 2, 6]
        assert db.k_hop(1, "cites", depth=2) == [(2, 1), (3, 2), (4, 2)]


def test_read_only_reads_and_refuses_writes(path, catalogue):
    with quickdb.Store.open_read_only(path) as ro:
        assert ro.count_rows("books") == 6
        assert ro.get_row("books", 4)["city"] == "قرطبة"
        assert [i for i, _ in ro.knn("emb", [1.0, 0.0, 0.0, 0.0], k=3)] == [1, 2, 6]
        assert ro.select("books", {"eq": ["city", "فاس"]}) == [
            {"id": 2, "title": "مقدمة ابن خلدون", "city": "فاس", "year": 1377},
            {"id": 5, "title": "رحلة ابن بطوطة", "city": "فاس", "year": 1355},
        ]
        with pytest.raises(quickdb.BackendError, match="read-only"):
            ro.upsert_rows("books", [(7, ["x", "y", 1])])
        with pytest.raises(quickdb.BackendError):
            ro.define_table("other", [("x", "int")])
        with pytest.raises(quickdb.BackendError):
            ro.define_vector_index("other", 2)
        with pytest.raises(quickdb.BackendError):
            ro.upsert_vectors("emb", [9], [[0.0, 0.0, 0.0, 1.0]])
        with pytest.raises(quickdb.BackendError):
            ro.upsert_edges([(1, "cites", 9)])
        with pytest.raises(quickdb.BackendError):
            ro.delete_rows("books", [1])
        with pytest.raises(quickdb.BackendError):
            ro.flush()
    # The read-only handle wrote nothing: the writer still sees six rows.
    assert catalogue.count_rows("books") == 6


def test_read_only_sees_new_rows_and_new_tables_of_the_writer(path, catalogue):
    ro = quickdb.Store.open_read_only(path)
    try:
        catalogue.upsert_rows("books", [{"id": 7, "title": "الأغاني", "city": "حلب", "year": 967}])
        catalogue.define_table("readers", [("name", "text", False)])
        catalogue.upsert_rows("readers", [{"id": 100, "name": "ليلى"}])
        catalogue.define_vector_index("late", 2, metric="l2")
        catalogue.upsert_vectors("late", [100], [[0.5, 0.5]])
        catalogue.flush()
        with pytest.raises(quickdb.NotFoundError):
            ro.knn("late", [0.5, 0.5])
        # New rows of a table that existed when the reader opened are seen at once.
        assert ro.get_row("books", 7)["title"] == "الأغاني"
        assert ro.count_rows("books") == 7
        # A table created after the reader opened is unknown to it until it reopens (NativeStore
        # loads table definitions at open).
        with pytest.raises(quickdb.NotFoundError):
            ro.get_row("readers", 100)
    finally:
        ro.close()
    with quickdb.Store.open_read_only(path) as ro2:
        assert ro2.get_row("readers", 100) == {"id": 100, "name": "ليلى"}
        assert ro2.select("readers") == [{"id": 100, "name": "ليلى"}]
        assert ro2.knn("late", [0.5, 0.5], k=1) == [(100, 0.0)]


def test_second_writer_in_same_process_fails(path):
    with quickdb.Store(path) as first:
        with pytest.raises(quickdb.BackendError, match="locked"):
            quickdb.Store(path)
        first.define_table("t", [("x", "int")])
    # After the first writer closes, a new one opens.
    with quickdb.Store(path) as again:
        assert again.count_rows("t") == 0


SECOND_PROCESS = """
import sys, quickdb
try:
    quickdb.Store(sys.argv[1])
except quickdb.BackendError as e:
    print("LOCKED" if "locked" in str(e) else "OTHER: " + str(e))
else:
    print("OPENED")
ro = quickdb.Store.open_read_only(sys.argv[1])
print("READER", ro.count_rows("t"))
"""


def test_second_writer_in_another_process_fails(path):
    with quickdb.Store(path) as first:
        first.define_table("t", [("x", "int")])
        first.upsert_rows("t", [(1, [5])])
        first.flush()
        out = subprocess.run(
            [sys.executable, "-c", SECOND_PROCESS, path], capture_output=True, text=True, timeout=120
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["LOCKED", "READER", "1"]
