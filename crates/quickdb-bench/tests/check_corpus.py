"""Verify generated corpus counts and f32 shape without printing source text."""
import json
from pathlib import Path
import struct
import sys


def check(path):
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    actual = {}
    for name in ('chunks', 'docs', 'edges'):
        count = 0
        with (path / (name + '.jsonl')).open(encoding='utf-8') as source:
            for line in source:
                item = json.loads(line)
                if name == 'chunks':
                    assert item['chars'] == len(item['text']), 'chunk chars mismatch'
                count += 1
        assert count == meta[name], f'{name}: expected {meta[name]}, got {count}'
        actual[name] = count
    size = (path / 'vectors.f32').stat().st_size
    assert size == meta['chunks'] * meta['dim'] * 4, 'vector byte length mismatch'
    vectors = 0
    with (path / 'vectors.f32').open('rb') as source:
        while row := source.read(meta['dim'] * 4):
            assert len(row) == meta['dim'] * 4
            assert all(abs(v) <= 3.4028234663852886e38 for v in struct.unpack('<' + 'f' * meta['dim'], row)), 'non-finite vector'
            vectors += 1
    assert vectors == meta['chunks']
    assert sum(c['chunks'] for c in meta['per_corpus'].values()) == meta['chunks']
    assert sum(c['docs'] for c in meta['per_corpus'].values()) == meta['docs']
    print(json.dumps({'check': 'pass', **actual, 'dim': meta['dim'], 'vector_rows': vectors, 'vector_bytes': size, 'model': meta['model']}))


if __name__ == '__main__':
    check(sys.argv[1])
