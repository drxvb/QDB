"""Tables and rows: definitions, upserts in both row forms, reads, select filters, deletes."""

import pytest

import quickdb
from conftest import COLUMNS, MANUSCRIPTS


def ids(rows):
    return [r["id"] for r in rows]


def test_define_table_type_spellings(db):
    db.define_table("a", [("i", "int", True), ("f", "float", False), ("t", "text", False), ("b", "bytes", False)])
    db.define_table("b", [("i", int, True), ("f", float, False), ("t", str, False), ("b", bytes, False)])
    db.define_table("c", [("i", "INTEGER"), ("f", "real"), ("t", "str"), ("b", "blob")])
    for t in "abc":
        db.upsert_rows(t, [(1, [1, 2.5, "نص", b"\x00\xff"])])
        assert db.get_row(t, 1) == {"id": 1, "i": 1, "f": 2.5, "t": "نص", "b": b"\x00\xff"}


def test_define_table_identical_is_noop_and_different_is_schema_error(db):
    db.define_table("books", COLUMNS)
    db.define_table("books", COLUMNS)
    with pytest.raises(quickdb.SchemaError):
        db.define_table("books", [("title", "text", True)])


@pytest.mark.parametrize(
    "name,columns",
    [
        ("Books", [("x", "int")]),
        ("books", [("id", "int")]),
        ("books", [("x", "int"), ("x", "text")]),
        ("books", [("x", "decimal")]),
        ("books", [("x",)]),
        ("books", [(1, "int")]),
        ("books", "x int"),
    ],
)
def test_define_table_rejects_bad_definitions(db, name, columns):
    with pytest.raises(quickdb.SchemaError):
        db.define_table(name, columns)


def test_arabic_round_trip_dict_and_tuple_rows(db):
    db.define_table("books", COLUMNS)
    db.upsert_rows("books", [{"id": i, "title": t, "city": c, "year": y} for i, t, c, y, _ in MANUSCRIPTS[:3]])
    db.upsert_rows("books", [(i, [t, c, y]) for i, t, c, y, _ in MANUSCRIPTS[3:]])
    for i, t, c, y, _ in MANUSCRIPTS:
        assert db.get_row("books", i) == {"id": i, "title": t, "city": c, "year": y}
    assert db.count_rows("books") == 6
    # Diacritics, Arabic-Indic digits, mixed scripts and an emoji survive unchanged.
    text = "بِسْمِ ٱللَّهِ · QDB ١٢٣ 🌙"
    db.upsert_rows("books", [{"id": 9, "title": text}])
    assert db.get_row("books", 9)["title"] == text


def test_rows_accept_any_iterable(db):
    db.define_table("t", [("x", "int")])
    db.upsert_rows("t", ((i, [i * 10]) for i in range(5)))
    db.upsert_rows("t", [[5, (50,)]])
    assert [r["x"] for r in db.select("t")] == [0, 10, 20, 30, 40, 50]


def test_missing_dict_column_is_null_and_upsert_replaces(db):
    db.define_table("books", COLUMNS)
    db.upsert_rows("books", [{"id": 1, "title": "الأول", "year": 1}])
    assert db.get_row("books", 1) == {"id": 1, "title": "الأول", "city": None, "year": 1}
    db.upsert_rows("books", [{"id": 1, "city": "تونس"}])
    assert db.get_row("books", 1) == {"id": 1, "title": None, "city": "تونس", "year": None}


def test_value_types(db):
    db.define_table("v", [("f", "float"), ("b", "bytes"), ("i", "int")])
    db.upsert_rows("v", [(1, [3, bytearray(b"ab"), True]), (2, [None, None, None])])
    assert db.get_row("v", 1) == {"id": 1, "f": 3.0, "b": b"ab", "i": 1}
    assert isinstance(db.get_row("v", 1)["f"], float)
    assert db.get_row("v", 2) == {"id": 2, "f": None, "b": None, "i": None}
    big = 2**63 - 1
    db.upsert_rows("v", [(big, [-1.5, b"", -(2**63)])])
    assert db.get_row("v", big) == {"id": big, "f": -1.5, "b": b"", "i": -(2**63)}


@pytest.mark.parametrize(
    "row",
    [
        {"id": 1, "year": "1990"},
        {"id": 1, "year": 1.5},
        {"id": 1, "year": 2**63},
        {"id": 1, "title": 5},
        {"id": 1, "title": b"bytes"},
        {"id": 1, "nope": 1},
        {"title": "no id"},
        {"id": -1},
        {"id": 2**63},
        {"id": 1.0},
        {"id": "1"},
        (1, ["only one value"]),
        (1, ["a", "b", 1, "extra"]),
        (1,),
        "not a row",
    ],
)
def test_bad_rows_are_schema_errors(db, row):
    db.define_table("books", COLUMNS)
    with pytest.raises(quickdb.SchemaError):
        db.upsert_rows("books", [row])


def test_failed_batch_leaves_nothing_applied(db):
    db.define_table("books", COLUMNS)
    with pytest.raises(quickdb.SchemaError):
        db.upsert_rows("books", [(1, ["a", "b", 1]), (2, ["a", "b", "bad"])])
    assert db.count_rows("books") == 0


def test_get_row_missing_and_unknown_table(db):
    db.define_table("books", COLUMNS)
    assert db.get_row("books", 42) is None
    with pytest.raises(quickdb.NotFoundError):
        db.get_row("nope", 1)
    with pytest.raises(quickdb.NotFoundError):
        db.upsert_rows("nope", [(1, [])])
    with pytest.raises(quickdb.NotFoundError):
        db.count_rows("nope")
    with pytest.raises(quickdb.NotFoundError):
        db.select("nope")
    with pytest.raises(quickdb.NotFoundError):
        db.delete_rows("nope", [1])
    with pytest.raises(quickdb.SchemaError):
        db.get_row("books", -5)


def test_select_filters(catalogue):
    assert ids(catalogue.select("books")) == [1, 2, 3, 4, 5, 6]
    assert ids(catalogue.select("books", None)) == [1, 2, 3, 4, 5, 6]
    assert ids(catalogue.select("books", {"all": True})) == [1, 2, 3, 4, 5, 6]
    assert ids(catalogue.select("books", limit=2)) == [1, 2]
    assert ids(catalogue.select("books", limit=0)) == []
    assert ids(catalogue.select("books", {"eq": ["city", "فاس"]})) == [2, 5]
    assert ids(catalogue.select("books", {"eq": ["year", 1022]})) == [4]
    assert ids(catalogue.select("books", {"range": ["year", 800, 1400]})) == [2, 4, 5, 6]
    assert ids(catalogue.select("books", {"range": ["year", None, 900]})) == [1, 6]
    assert ids(catalogue.select("books", {"range": ["year", 1400, None]})) == [3]
    assert ids(catalogue.select("books", {"range": ["year", 1400, 800]})) == []
    assert ids(catalogue.select("books", {"prefix": ["title", "كت"]})) == [6]
    assert ids(catalogue.select("books", {"prefix": ["title", "رحلة"]})) == [5]
    assert ids(catalogue.select("books", {"prefix": ["title", ""]})) == [1, 2, 3, 4, 5, 6]
    assert catalogue.select("books", {"eq": ["city", "فاس"]}, limit=1) == [
        {"id": 2, "title": "مقدمة ابن خلدون", "city": "فاس", "year": 1377}
    ]


@pytest.mark.parametrize(
    "flt,err",
    [
        ({"eq": ["nope", 1]}, quickdb.NotFoundError),
        ({"eq": ["year", "x"]}, quickdb.SchemaError),
        ({"range": ["title", 1, 2]}, quickdb.SchemaError),
        ({"range": ["year", 1]}, quickdb.SchemaError),
        ({"range": ["year", 1.5, 2]}, quickdb.SchemaError),
        ({"prefix": ["year", "1"]}, quickdb.SchemaError),
        ({"prefix": ["title", 1]}, quickdb.SchemaError),
        ({"like": ["title", "x"]}, quickdb.SchemaError),
        ({"eq": ["city", "a"], "prefix": ["title", "b"]}, quickdb.SchemaError),
        ({}, quickdb.SchemaError),
        (["eq", "city", "a"], quickdb.SchemaError),
    ],
)
def test_bad_filters(catalogue, flt, err):
    with pytest.raises(err):
        catalogue.select("books", flt)


def test_delete_rows(catalogue):
    assert catalogue.delete_rows("books", [1, 1, 99]) == 1
    assert catalogue.get_row("books", 1) is None
    assert catalogue.count_rows("books") == 5
    # Vectors and edges with the same id are kept.
    assert catalogue.count_vectors("emb") == 6
    assert catalogue.count_edges() == 5
    assert catalogue.delete_rows("books", []) == 0
    assert catalogue.delete_rows("books", (i for i in [2, 3])) == 2
