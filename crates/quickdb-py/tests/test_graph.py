"""Graph edges: upsert, neighbors in each direction, k_hop with relation and direction filters."""

import pytest

import quickdb


def test_neighbors_by_direction(catalogue):
    assert catalogue.neighbors(2) == [(2, "cites", 3, 1.0), (2, "cites", 4, 1.0)]
    assert catalogue.neighbors(2, direction="in") == [(1, "cites", 2, 1.0)]
    assert catalogue.neighbors(2, "cites", "both") == [(1, "cites", 2, 1.0), (2, "cites", 3, 1.0), (2, "cites", 4, 1.0)]
    assert catalogue.neighbors(2, direction="both", limit=1) == [(1, "cites", 2, 1.0)]
    assert catalogue.neighbors(2, rel="other") == []
    assert catalogue.neighbors(99) == []


def test_k_hop(catalogue):
    assert catalogue.k_hop(1) == [(2, 1)]
    assert catalogue.k_hop(1, "cites", depth=2) == [(2, 1), (3, 2), (4, 2)]
    assert catalogue.k_hop(1, depth=3) == [(2, 1), (3, 2), (4, 2), (5, 3)]
    assert catalogue.k_hop(5, direction="in", depth=5) == [(3, 1), (2, 2), (1, 3), (6, 4)]
    assert catalogue.k_hop(1, direction="both", depth=1) == [(2, 1), (6, 1)]
    assert catalogue.k_hop(1, depth=3, limit=2) == [(2, 1), (3, 2)]
    assert catalogue.k_hop(1, depth=0) == []
    assert catalogue.k_hop(1, depth=2, limit=0) == []
    assert catalogue.k_hop(1, rel="other", depth=3) == []


def test_arabic_relations_weights_and_replace(catalogue):
    # A relation is any string, Arabic included; a 3-tuple has weight 1.0.
    catalogue.upsert_edges([(4, "يشرح", 6, 0.5), (6, "يشرح", 3)])
    assert catalogue.count_edges() == 7
    assert catalogue.neighbors(6, "يشرح", "both") == [(4, "يشرح", 6, 0.5), (6, "يشرح", 3, 1.0)]
    assert catalogue.k_hop(4, "يشرح", depth=2) == [(6, 1), (3, 2)]
    # The same (src, rel, dst) replaces the weight; later duplicates in a batch win.
    catalogue.upsert_edges([(4, "يشرح", 6, 2.0), (4, "يشرح", 6, 3.0)])
    assert catalogue.count_edges() == 7
    assert catalogue.neighbors(4, "يشرح") == [(4, "يشرح", 6, 3.0)]
    # Edge endpoints need no row and no vector.
    catalogue.upsert_edges([(1000, "cites", 2000, 1.0)])
    assert catalogue.k_hop(1000) == [(2000, 1)]


def test_edges_survive_reopen(path, catalogue):
    catalogue.upsert_edges([(3, "يذكر", 1, 0.25)])
    catalogue.close()
    with quickdb.Store.open_read_only(path) as ro:
        assert ro.count_edges() == 6
        assert ro.neighbors(3, "يذكر") == [(3, "يذكر", 1, 0.25)]


@pytest.mark.parametrize(
    "edge",
    [
        (1, "r", 2, float("nan")),
        (1, "r", 2, float("inf")),
        (1, "r", 2, "heavy"),
        (-1, "r", 2, 1.0),
        (1, "r", 2**63, 1.0),
        (1, 5, 2, 1.0),
        (1, "r"),
        (1, "r", 2, 1.0, "extra"),
    ],
)
def test_bad_edges(catalogue, edge):
    with pytest.raises(quickdb.SchemaError):
        catalogue.upsert_edges([(7, "ok", 8, 1.0), edge])
    # The whole batch is rejected.
    assert catalogue.count_edges() == 5


def test_bad_direction_and_start(catalogue):
    with pytest.raises(quickdb.SchemaError, match="direction"):
        catalogue.k_hop(1, direction="up")
    with pytest.raises(quickdb.SchemaError, match="direction"):
        catalogue.neighbors(1, direction="sideways")
    with pytest.raises(quickdb.SchemaError):
        catalogue.k_hop(-1)
