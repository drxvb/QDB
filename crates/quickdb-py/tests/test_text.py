"""Full-text and hybrid search through the Python binding."""

import pytest

import quickdb


def test_text_search_arabic_and_folding(catalogue):
    catalogue.define_text_index("titles", "books", ["title"])
    hits = catalogue.text_search("titles", "ابن", k=10)
    assert {i for i, _ in hits} == {2, 5}
    # harakat and alef variants fold to the same indexed form
    assert [i for i, _ in catalogue.text_search("titles", "طَوْق الحمامة")] == [4]
    assert [i for i, _ in catalogue.text_search("titles", "الف ليلة")] == [3]
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_text_index_follows_updates_and_deletes(catalogue):
    catalogue.define_text_index("titles", "books", ["title", "city"])
    catalogue.upsert_rows("books", [{"id": 7, "title": "Muqaddimah notes", "city": "Tunis", "year": 1400}])
    assert [i for i, _ in catalogue.text_search("titles", "muqaddimah")] == [7]
    catalogue.upsert_rows("books", [{"id": 7, "title": "renamed", "city": "Tunis", "year": 1400}])
    assert catalogue.text_search("titles", "muqaddimah") == []
    catalogue.delete_rows("books", [7])
    assert catalogue.text_search("titles", "renamed") == []


def test_hostile_and_empty_queries(catalogue):
    catalogue.define_text_index("titles", "books", ["title"])
    for q in ['"', "NEAR(", "*", "-", "title:x", "", "   ", "OR NOT"]:
        assert isinstance(catalogue.text_search("titles", q), list)


def test_hybrid_search(catalogue):
    catalogue.define_text_index("titles", "books", ["title"])
    hits = catalogue.hybrid_search("titles", "emb", "ابن", [0.0, 0.0, 1.0, 0.0], k=3)
    assert hits[0][0] == 5  # first in both rankings
    assert len(hits) <= 3


def test_text_search_survives_reopen(path):
    with quickdb.Store(path) as db:
        db.define_table("notes", [("body", "text", False)])
        db.upsert_rows("notes", [{"id": 1, "body": "decision about storage"}])
        db.define_text_index("t", "notes", ["body"])
    with quickdb.Store(path) as db:
        assert [i for i, _ in db.text_search("t", "storage")] == [1]
    ro = quickdb.Store.open_read_only(path)
    assert [i for i, _ in ro.text_search("t", "decision")] == [1]
    ro.close()


def test_unknown_text_index(db):
    with pytest.raises(quickdb.QuickDBError):
        db.text_search("missing", "x")


def test_match_any_for_questions(db):
    db.define_table("notes", [("body", "text", False)])
    db.upsert_rows("notes", [{"id": 1, "body": "store notes locally"}, {"id": 2, "body": "no network on tablets"},
                             {"id": 3, "body": "weekly meeting"}])
    db.define_text_index("t", "notes", ["body"])
    q = "how do we store notes without network"
    assert db.text_search("t", q) == []
    assert [i for i, _ in db.text_search("t", q, match="any")] == [1, 2]
    with pytest.raises(ValueError):
        db.text_search("t", q, match="some")
