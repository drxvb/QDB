"""Workstation-only standard-library tests for the corpus preparer, no servers."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('corpus_prepare', HERE.parents[2] / 'scripts' / 'corpus_prepare.py')
cp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cp)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE)
        self.base = Path(self.temp.name)
        self.root = self.base / 'research'
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def args(self, name='out', batch=2, maximum=100):
        return argparse.Namespace(roots=[str(self.root)], out=str(self.base / name),
                                  max_chunks=maximum, embed_url='http://unused.invalid/v1',
                                  batch=batch, no_embed=True, dim=8)

    def read(self, out, name):
        return [json.loads(line) for line in (out / name).read_text(encoding='utf-8').splitlines()]

    def test_graph_counts_arabic_and_determinism(self):
        (self.root / 'sub').mkdir()
        (self.root / 'a.md').write_text('العربية في ترتيبها المنطقي.\n\n' + 'paragraph words ' * 200 + '\n\n[child](sub/b.txt#part) [remote](https://example.org) ![img](sub/b.txt)', encoding='utf-8')
        (self.root / 'sub' / 'b.txt').write_text('Second document.', encoding='utf-8')
        first = cp.prepare(self.args())
        second = cp.prepare(self.args('out2', batch=7))
        out = self.base / 'out'
        chunks = self.read(out, 'chunks.jsonl')
        docs = self.read(out, 'docs.jsonl')
        edges = self.read(out, 'edges.jsonl')
        self.assertEqual(first['chunks'], len(chunks))
        self.assertEqual(first['docs'], len(docs))
        self.assertEqual(first['edges'], len(edges))
        self.assertEqual(first['vector_bytes'], len(chunks) * 8 * 4)
        self.assertEqual(first['directories'], 2)
        self.assertEqual(second['chunks'], first['chunks'])
        self.assertEqual((out / 'vectors.f32').read_bytes(), (self.base / 'out2' / 'vectors.f32').read_bytes())
        self.assertIn('العربية', chunks[0]['text'])
        self.assertTrue(all(c['chars'] == len(c['text']) <= 1200 for c in chunks))
        self.assertEqual({e['rel'] for e in edges}, {'has', 'contains', 'next', 'links'})
        self.assertEqual(len([e for e in edges if e['rel'] == 'links']), 1)
        self.assertEqual({c['doc_id'] for c in chunks}, {d['id'] for d in docs})
        self.assertEqual(len(edges), len({(e['src'], e['rel'], e['dst']) for e in edges}))
        self.assertFalse((out / '.prepare.sqlite3').exists())

    def test_filters_limits_and_no_overwrite(self):
        (self.root / 'a.txt').write_text('word ' * 1000, encoding='utf-8')
        for name in ('secret.txt', 'credential.txt', 'token.txt', 'password.txt', '.env.txt', 'key.txt'):
            (self.root / name).write_text('must not be imported', encoding='utf-8')
        for name in cp.SKIP_DIRS:
            (self.root / name).mkdir()
            (self.root / name / 'skip.txt').write_text('excluded directory', encoding='utf-8')
        (self.root / 'big.txt').write_bytes(b'x' * (cp.MAX_BYTES + 1))
        (self.root / 'bad.txt').write_bytes(b'\xff\xfe\x80')
        limited = cp.prepare(self.args(maximum=2))
        self.assertEqual(limited['chunks'], 2)
        self.assertEqual(limited['docs'], 1)
        complete = cp.prepare(self.args('complete'))
        self.assertEqual(complete['docs'], 1)
        self.assertEqual(complete['skipped']['too_large'], 1)
        self.assertEqual(complete['skipped']['invalid_utf8'], 1)
        with self.assertRaisesRegex(ValueError, 'empty'):
            cp.prepare(self.args())

    def test_paragraph_chunks_and_unbroken_text(self):
        self.assertEqual(list(cp.chunks('first\n\nsecond')), ['first\n\nsecond'])
        text = 'ع' * 2501
        pieces = list(cp.chunks(text))
        self.assertEqual(''.join(pieces), text)
        self.assertEqual(list(map(len, pieces)), [1200, 1200, 101])

    def test_embeddings_reorder_and_model(self):
        args = self.args()
        args.no_embed = False
        output = io.BytesIO()
        with mock.patch.object(cp, 'request_json', side_effect=[{'data': [{'id': 'model'}]},
                {'data': [{'index': 1, 'embedding': [3.0, 4.0]}, {'index': 0, 'embedding': [1.0, 2.0]}]}]) as request:
            embed = cp.Embeddings(args, output)
            embed.add('a')
            embed.add('b')
            self.assertEqual(embed.dim, 2)
            self.assertEqual(embed.model, 'model')
            self.assertEqual(struct.unpack('<4f', output.getvalue()), (1, 2, 3, 4))
            self.assertEqual(request.call_args.args[1]['input'], ['a', 'b'])

    def test_embeddings_reject_bad_indexes_and_nonfinite(self):
        args = self.args()
        args.no_embed = False
        for result in ({'data': [{'index': 0, 'embedding': [1.0]}, {'index': 0, 'embedding': [2.0]}]},
                       {'data': [{'index': 0, 'embedding': [float('nan')]}, {'index': 1, 'embedding': [1.0]}]}):
            with mock.patch.object(cp, 'request_json', side_effect=[{'data': [{'id': 'm'}]}, result]):
                embed = cp.Embeddings(args, io.BytesIO())
                embed.add('a')
                with self.assertRaises(RuntimeError):
                    embed.add('b')

    def test_memory_bound_for_maximum_file(self):
        (self.root / 'large.txt').write_text('bounded words ' * (cp.MAX_BYTES // 14), encoding='utf-8')
        args = self.args(maximum=2000)
        args.dim = 1024
        args.batch = 32
        meta = cp.prepare(args)
        self.assertGreater(meta['chunks'], 1000)
        self.assertIsNotNone(meta['peak_memory_bytes'])
        self.assertLess(meta['peak_memory_bytes'], 500 * 1024 * 1024)
        print(f"CORPUS_MEMORY peak_bytes={meta['peak_memory_bytes']} chunks={meta['chunks']} dim={meta['dim']}")

    def test_three_retries(self):
        with mock.patch.object(cp.urllib.request, 'urlopen', side_effect=OSError('offline')) as call, \
             mock.patch.object(cp.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'three retries'):
                cp.request_json('http://unused.invalid/models')
            self.assertEqual(call.call_count, 4)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2, 4])


if __name__ == '__main__':
    unittest.main(verbosity=2)
