import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('corpus_questions', HERE.parents[2] / 'scripts' / 'corpus_questions.py')
cq = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cq)

ROWS = [
    {'id': 'q1', 'lang': 'en', 'question': 'SECRET one', 'expected_doc_ids': [1], 'corpus': 'a'},
    {'id': 'q2', 'lang': 'ar', 'question': 'سر سري', 'expected_doc_ids': [2, 3], 'corpus': 'a'},
    {'id': 'q3', 'lang': 'en', 'question': 'SECRET three', 'expected_doc_ids': [], 'corpus': 'b'},
]


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE)
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (self.dir / 'questions.jsonl').write_text(text, encoding='utf-8', newline='\n')

    def args(self, **extra):
        values = dict(questions=str(self.dir), out=None, corpus=None, embed_url='http://unused.invalid/v1',
                      batch=2, no_embed=True, dim=8)
        values.update(extra)
        return argparse.Namespace(**values)

    def test_files_meta_and_determinism(self):
        self.write(ROWS)
        meta = cq.run(self.args())
        vectors = (self.dir / 'qvectors.f32').read_bytes()
        self.assertEqual(len(vectors), 3 * 8 * 4)
        self.assertEqual(meta['count'], 3)
        self.assertEqual(meta['dim'], 8)
        self.assertEqual(meta['model'], 'deterministic-random-v1')
        self.assertEqual(meta['qvectors_sha256'], hashlib.sha256(vectors).hexdigest())
        self.assertEqual(json.loads((self.dir / 'qmeta.json').read_text(encoding='utf-8')), meta)
        again = cq.run(self.args(out=str(self.dir / 'again')))
        self.assertEqual(again['qvectors_sha256'], meta['qvectors_sha256'])
        with self.assertRaises(ValueError):
            cq.run(self.args())

    def test_errors_never_carry_question_text(self):
        for rows in ([dict(ROWS[0], lang='fr')], [ROWS[0], ROWS[0]], [dict(ROWS[0], expected_doc_ids=['x'])],
                     [dict(ROWS[0], question='  ')]):
            self.write(rows)
            with self.assertRaises(ValueError) as caught:
                cq.run(self.args(out=str(self.dir / 'out')))
            self.assertNotIn('SECRET', str(caught.exception))
        (self.dir / 'questions.jsonl').write_text('SECRET not json\n', encoding='utf-8')
        with self.assertRaises(ValueError) as caught:
            cq.read_questions(self.dir / 'questions.jsonl')
        self.assertNotIn('SECRET', str(caught.exception))
        self.assertFalse((self.dir / 'out' / 'qvectors.f32').exists())

    def test_corpus_dim_mismatch_leaves_no_files(self):
        self.write(ROWS)
        corpus = self.dir / 'corpus'
        corpus.mkdir()
        (corpus / 'meta.json').write_text('{"dim": 9}', encoding='utf-8')
        with self.assertRaises(ValueError):
            cq.run(self.args(corpus=str(corpus)))
        self.assertFalse((self.dir / 'qvectors.f32').exists())
        self.assertFalse((self.dir / 'qmeta.json').exists())

    def test_embedding_request_goes_through_corpus_prepare(self):
        self.write(ROWS)
        sent = []

        def fake(url, payload=None):
            if payload is None:
                return {'data': [{'id': 'bge-m3'}]}
            sent.append(payload['input'])
            return {'data': [{'index': i, 'embedding': [float(i + 1)] * 4} for i in reversed(range(len(payload['input'])))]}

        with mock.patch.object(cq.cp, 'request_json', fake):
            meta = cq.run(self.args(no_embed=False))
        self.assertEqual(meta['model'], 'bge-m3')
        self.assertEqual([len(b) for b in sent], [2, 1])
        self.assertEqual(sent[0][0], 'SECRET one')
        self.assertEqual((self.dir / 'qvectors.f32').stat().st_size, 3 * 4 * 4)

    def test_main_prints_no_question_text(self):
        self.write(ROWS)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(
                sys, 'argv', ['x', '--questions', str(self.dir), '--no-embed', '--dim', '8']):
            self.assertEqual(cq.main(), 0)
        self.assertNotIn('SECRET', out.getvalue())


if __name__ == '__main__':
    unittest.main()
