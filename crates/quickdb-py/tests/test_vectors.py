"""Vector indexes: definitions, list / numpy / array.array input, knn and filtered knn."""

import array
import math
import subprocess
import sys

import pytest

import quickdb
Q = [1.0, 0.0, 0.0, 0.0]


def knn_ids(hits):
    return [i for i, _ in hits]


def test_define_vector_index_defaults_and_redefinition(db):
    db.define_vector_index("emb", 8)
    db.define_vector_index("emb", 8, metric="cosine", m=16, ef_construction=200)
    with pytest.raises(quickdb.SchemaError):
        db.define_vector_index("emb", 8, metric="l2")
    with pytest.raises(quickdb.SchemaError):
        db.define_vector_index("emb", 9)
    with pytest.raises(quickdb.SchemaError):
        db.define_vector_index("bad", 4, metric="dot")
    with pytest.raises(quickdb.SchemaError):
        db.define_vector_index("Bad", 4)
    assert db.count_vectors("emb") == 0


def test_plain_list_input_and_l2_distances(catalogue):
    hits = catalogue.knn("emb", Q, k=3)
    assert knn_ids(hits) == [1, 2, 6]
    # Squared Euclidean distance.
    assert hits[0] == (1, 0.0)
    assert hits[1][1] == pytest.approx(0.1**2 + 0.1**2, rel=1e-5)
    assert hits[2][1] == pytest.approx(0.2**2 + 0.2**2 + 0.1**2, rel=1e-5)
    assert catalogue.knn("emb", Q, k=0) == []
    assert len(catalogue.knn("emb", Q)) == 6  # k defaults to 10
    assert knn_ids(catalogue.knn("emb", (0.0, 0.0, 1.0, 0.0), k=1, ef=8)) == [5]


def test_cosine_metric(db):
    db.define_vector_index("cos", 2)  # cosine is the default
    db.upsert_vectors("cos", [1, 2, 3], [[1.0, 0.0], [10.0, 10.0], [0.0, 3.0]])
    hits = db.knn("cos", [2.0, 0.0], k=3)
    assert knn_ids(hits) == [1, 2, 3]
    assert hits[0][1] == pytest.approx(0.0, abs=1e-6)
    assert hits[1][1] == pytest.approx(1 - math.sqrt(0.5), rel=1e-4)
    assert hits[2][1] == pytest.approx(1.0, rel=1e-5)


def test_upsert_replaces_vector(catalogue):
    catalogue.upsert_vectors("emb", [5], [[1.0, 0.0, 0.0, 0.0]])
    assert catalogue.count_vectors("emb") == 6
    assert sorted(knn_ids(catalogue.knn("emb", Q, k=2))) == [1, 5]


def test_numpy_input(db):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    data = rng.standard_normal((300, 16)).astype(np.float32)
    db.define_vector_index("np", 16, metric="l2")
    db.upsert_vectors("np", np.arange(300, dtype=np.int64), data)
    assert db.count_vectors("np") == 300
    for i in (0, 123, 299):
        hits = db.knn("np", data[i], k=1, ef=128)
        assert hits[0][0] == i and hits[0][1] == pytest.approx(0.0, abs=1e-5)
    # Lists and numpy give the same answer.
    assert db.knn("np", data[5].tolist(), k=5) == db.knn("np", data[5], k=5)


def test_numpy_float64_noncontiguous_and_byte_order(db):
    np = pytest.importorskip("numpy")
    base = np.arange(40, dtype=np.float64).reshape(10, 4) / 40.0
    db.define_vector_index("v", 2, metric="l2")
    # Every other column: a non-contiguous float64 view.
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    db.upsert_vectors("v", range(10), view)
    assert db.knn("v", view[3], k=1)[0] == (3, 0.0)
    # Big-endian float32 and a Fortran-ordered array.
    db.upsert_vectors("v", [20], np.array([[5.0, 5.0]], dtype=">f4"))
    assert db.knn("v", np.array([5.0, 5.0], dtype=">f4"), k=1)[0] == (20, 0.0)
    db.upsert_vectors("v", [21, 22], np.asfortranarray(np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)))
    assert db.knn("v", [9.0, 10.0], k=1)[0] == (22, 0.0)
    # A list of 1-D numpy rows.
    db.upsert_vectors("v", [23], [np.array([11.0, 12.0], dtype=np.float32)])
    assert db.knn("v", [11.0, 12.0], k=1)[0] == (23, 0.0)


def test_numpy_bad_buffers(db):
    np = pytest.importorskip("numpy")
    db.define_vector_index("v", 2, metric="l2")
    with pytest.raises(quickdb.SchemaError, match="float32 or float64"):
        db.upsert_vectors("v", [1], np.zeros((1, 2), dtype=np.int32))
    with pytest.raises(quickdb.SchemaError, match="2-D"):
        db.upsert_vectors("v", [1], np.zeros(2, dtype=np.float32))
    with pytest.raises(quickdb.SchemaError, match="1-D"):
        db.knn("v", np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(quickdb.SchemaError):
        db.upsert_vectors("v", [1, 2], np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(quickdb.SchemaError):
        db.upsert_vectors("v", [1], np.full((1, 2), np.nan, dtype=np.float32))


def test_array_module_buffer_without_numpy(db):
    db.define_vector_index("v", 3, metric="l2")
    db.upsert_vectors("v", [1, 2], [array.array("f", [0.0, 0.0, 1.0]), array.array("d", [1.0, 0.0, 0.0])])
    assert db.knn("v", array.array("f", [1.0, 0.0, 0.0]), k=1)[0] == (2, 0.0)
    with pytest.raises(quickdb.SchemaError):
        db.knn("v", array.array("i", [1, 0, 0]))
    with pytest.raises(quickdb.SchemaError):
        db.knn("v", b"\x00" * 12)


def test_numpy_is_optional():
    code = "import sys; sys.modules['numpy'] = None; import quickdb, tempfile, os\n" \
        "d = os.path.join(tempfile.mkdtemp(), 's')\n" \
        "with quickdb.Store(d) as db:\n" \
        "    db.define_vector_index('v', 2, metric='l2')\n" \
        "    db.upsert_vectors('v', [1], [[0.5, 0.5]])\n" \
        "    print(db.knn('v', [0.5, 0.5], k=1))\n"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[(1, 0.0)]"


@pytest.mark.parametrize(
    "ids,vectors",
    [
        ([1], [[1.0, 2.0]]),  # wrong dim
        ([1, 2], [[1.0, 0.0, 0.0, 0.0]]),  # count mismatch
        ([1, 2], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0]]),  # ragged
        ([1], [[float("nan"), 0.0, 0.0, 0.0]]),
        ([1], [[float("inf"), 0.0, 0.0, 0.0]]),
        ([1], [["a", 0.0, 0.0, 0.0]]),
        ([1], ["abcd"]),
        ([-1], [[1.0, 0.0, 0.0, 0.0]]),
        ([2**63], [[1.0, 0.0, 0.0, 0.0]]),
    ],
)
def test_bad_vectors(catalogue, ids, vectors):
    with pytest.raises(quickdb.SchemaError):
        catalogue.upsert_vectors("emb", ids, vectors)
    assert catalogue.count_vectors("emb") == 6


def test_bad_queries_and_unknown_index(catalogue):
    with pytest.raises(quickdb.SchemaError):
        catalogue.knn("emb", [1.0, 0.0])
    with pytest.raises(quickdb.SchemaError):
        catalogue.knn("emb", [float("nan")] * 4, k=0)
    with pytest.raises(quickdb.NotFoundError):
        catalogue.knn("nope", Q)
    with pytest.raises(quickdb.NotFoundError):
        catalogue.upsert_vectors("nope", [1], [Q])
    with pytest.raises(quickdb.NotFoundError):
        catalogue.count_vectors("nope")


def test_knn_filtered(catalogue):
    # Nearest to Q among books from the year 1000 on: 2 (1377), then 4 (1022).
    assert knn_ids(catalogue.knn_filtered("emb", Q, 2, 64, "books", {"range": ["year", 1000, None]})) == [2, 4]
    assert knn_ids(catalogue.knn_filtered("emb", Q, 10, 64, "books", {"eq": ["city", "فاس"]})) == [2, 5]
    assert knn_ids(catalogue.knn_filtered("emb", Q, 1, 64, "books", {"prefix": ["title", "كتاب"]})) == [6]
    assert knn_ids(catalogue.knn_filtered("emb", Q, 3, 64, "books", None)) == [1, 2, 6]
    assert catalogue.knn_filtered("emb", Q, 0, 64, "books", None) == []
    assert catalogue.knn_filtered("emb", Q, 5, 64, "books", {"eq": ["city", "مكة"]}) == []
    # Deleting a row hides its vector from filtered search, without deleting the vector.
    catalogue.delete_rows("books", [2])
    assert knn_ids(catalogue.knn_filtered("emb", Q, 10, 64, "books", {"eq": ["city", "فاس"]})) == [5]
    assert 2 in knn_ids(catalogue.knn("emb", Q, k=3))


def test_knn_filtered_errors(catalogue):
    with pytest.raises(quickdb.NotFoundError):
        catalogue.knn_filtered("emb", Q, 2, 64, "nope", None)
    with pytest.raises(quickdb.NotFoundError):
        catalogue.knn_filtered("nope", Q, 2, 64, "books", None)
    with pytest.raises(quickdb.NotFoundError):
        catalogue.knn_filtered("emb", Q, 2, 64, "books", {"eq": ["nope", 1]})
    with pytest.raises(quickdb.SchemaError):
        catalogue.knn_filtered("emb", Q, 2, 64, "books", {"range": ["title", 1, 2]})
    with pytest.raises(quickdb.SchemaError):
        catalogue.knn_filtered("emb", [1.0], 2, 64, "books", None)


def test_same_results_as_the_rust_quickstart(db):
    """The data and queries of README-example.md give the results that file states."""
    docs = [
        (1, "intro", 2022, [1.0, 0.0, 0.0, 0.0]),
        (2, "basics", 2023, [0.9, 0.1, 0.0, 0.0]),
        (3, "indexes", 2024, [0.0, 1.0, 0.0, 0.0]),
        (4, "graphs", 2025, [0.0, 0.9, 0.1, 0.0]),
        (5, "tuning", 2025, [0.0, 0.0, 1.0, 0.0]),
        (6, "recap", 2024, [0.8, 0.2, 0.1, 0.0]),
    ]
    db.define_table("docs", [("title", "text", False), ("year", "int", True)])
    db.define_vector_index("emb", 4, metric="l2", m=16, ef_construction=100)
    db.upsert_rows("docs", [(i, [t, y]) for i, t, y, _ in docs])
    db.upsert_vectors("emb", [d[0] for d in docs], [d[3] for d in docs])
    db.upsert_edges([(s, "cites", d, 1.0) for s, d in [(1, 2), (2, 3), (2, 4), (3, 5), (6, 1)]])
    db.flush()
    assert knn_ids(db.knn("emb", Q, 3, 64)) == [1, 2, 6]
    assert knn_ids(db.knn_filtered("emb", Q, 2, 64, "docs", {"range": ["year", 2024, None]})) == [6, 4]
    assert [i for i, _ in db.k_hop(1, "cites", "out", 2, 100)] == [2, 3, 4]

