"""The example in the package README runs as written."""

import pathlib
import re

import quickdb

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def test_readme_example_runs(tmp_path, monkeypatch):
    code = re.search(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S).group(1)
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "README.md", "exec"), {})
    with quickdb.Store.open_read_only("catalogue.qdb") as db:
        assert db.count_rows("books") == 3
        assert [i for i, _ in db.knn("emb", [1, 0, 0, 0], k=2)] == [1, 2]
        assert [i for i, _ in db.knn_filtered("emb", [1, 0, 0, 0], 2, 64, "books", {"range": ["year", 1000, None]})] == [2, 3]
        assert db.k_hop(1, "cites", depth=2) == [(2, 1), (3, 2)]
        assert db.get_row("books", 2) == {"id": 2, "title": "مقدمة ابن خلدون", "city": "فاس", "year": 1377}
