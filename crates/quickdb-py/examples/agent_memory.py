"""Agent memory in one embedded store: remember notes, recall them by words and by meaning,
and follow how they relate.

Each memory is a row (kind + text), a vector (its embedding) and a graph node. Recall fuses
full-text BM25 and vector search (Reciprocal Rank Fusion), then walks the graph from the best hit.

The embedding is a toy (hashed bag of words) so this runs with no model. A real agent passes
its embedding model's vectors (a list or numpy array) instead; nothing else changes.

Run: python agent_memory.py
"""

import math
import re
import shutil
import tempfile

import quickdb

DIM = 64


def embed(text):
    """Toy embedding: hashed bag of words, L2-normalised."""
    v = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        h = 0xCBF29CE484222325
        for b in word.encode("utf-8"):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        v[h % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Memory:
    def __init__(self, path):
        self.db = quickdb.Store(path)
        self.db.define_table("memories", [("kind", "text", True), ("text", "text", False)])
        self.db.define_vector_index("emb", DIM, metric="cosine")
        self.db.define_text_index("words", "memories", ["text"])
        self.next_id = self.db.count_rows("memories") + 1

    def remember(self, kind, text, link=None):
        mid = self.next_id
        self.next_id += 1
        self.db.upsert_rows("memories", [{"id": mid, "kind": kind, "text": text}])
        self.db.upsert_vectors("emb", [mid], [embed(text)])
        if link:
            rel, to = link
            self.db.upsert_edges([(mid, rel, to, 1.0)])
        return mid

    def recall(self, question, k=3):
        return self.db.hybrid_search("words", "emb", question, embed(question), k=k)

    def text(self, mid):
        row = self.db.get_row("memories", mid)
        return row["text"] if row else ""

    def close(self):
        self.db.flush()
        self.db.close()


def main():
    path = tempfile.mkdtemp(prefix="qdb-agent-memory-")
    try:
        mem = Memory(path)
        goal = mem.remember("goal", "Ship the offline note-taking app for field teams by March")
        d1 = mem.remember("decision", "Store notes locally in one embedded database; sync later over HTTPS", ("serves", goal))
        mem.remember("fact", "Field tablets have 4 GB of RAM and often no network", ("constrains", d1))
        mem.remember("decision", "Use full-text search so notes can be found by exact words", ("follows", d1))
        mem.remember("fact", "ملاحظات الفريق الميداني تُكتب بالعربية والإنجليزية", ("about", goal))
        mem.close()

        mem = Memory(path)  # a new session
        for question in ["how do we store notes without network", "الفريق الميداني"]:
            print("Q:", question)
            hits = mem.recall(question)
            for mid, score in hits:
                print(f"  {score:.4f}  #{mid}  {mem.text(mid)}")
            if hits:
                for rid, depth in mem.db.k_hop(hits[0][0], depth=2, limit=10):
                    print(f"    -> #{rid} (depth {depth})  {mem.text(rid)}")
        decisions = mem.db.select("memories", {"eq": ["kind", "decision"]}, limit=10)
        print("decisions on record:", len(decisions))
        assert len(decisions) == 2
        mem.close()
    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
