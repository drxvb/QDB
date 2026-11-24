"""Shared fixtures: a fresh store directory per test, and a small Arabic manuscript catalogue."""

import pytest

import quickdb

# (id, title, city, year, vector). Ids are shared by rows, vectors and graph nodes.
MANUSCRIPTS = [
    (1, "كليلة ودمنة", "بغداد", 750, [1.0, 0.0, 0.0, 0.0]),
    (2, "مقدمة ابن خلدون", "فاس", 1377, [0.9, 0.1, 0.0, 0.0]),
    (3, "ألف ليلة وليلة", "دمشق", 1450, [0.0, 1.0, 0.0, 0.0]),
    (4, "طوق الحمامة", "قرطبة", 1022, [0.0, 0.9, 0.1, 0.0]),
    (5, "رحلة ابن بطوطة", "فاس", 1355, [0.0, 0.0, 1.0, 0.0]),
    (6, "كتاب الحيوان", "البصرة", 847, [0.8, 0.2, 0.1, 0.0]),
]

COLUMNS = [("title", "text", False), ("city", "text", True), ("year", "int", True)]

CITES = [(1, "cites", 2, 1.0), (2, "cites", 3, 1.0), (2, "cites", 4, 1.0), (3, "cites", 5, 1.0), (6, "cites", 1, 1.0)]


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "store")


@pytest.fixture
def db(path):
    store = quickdb.Store(path)
    yield store
    store.close()


@pytest.fixture
def catalogue(db):
    """`db` with table `books`, L2 index `emb` and `cites` edges, flushed."""
    db.define_table("books", COLUMNS)
    db.upsert_rows("books", [{"id": i, "title": t, "city": c, "year": y} for i, t, c, y, _ in MANUSCRIPTS])
    db.define_vector_index("emb", 4, metric="l2", m=16, ef_construction=100)
    db.upsert_vectors("emb", [m[0] for m in MANUSCRIPTS], [m[4] for m in MANUSCRIPTS])
    db.upsert_edges(CITES)
    db.flush()
    return db
