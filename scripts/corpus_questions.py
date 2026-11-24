#!/usr/bin/env python3
"""Embed written questions for `qbench corpus --questions`. Question text is never printed.

Reads <dir>/questions.jsonl, one object per line:
  {"id": "q001", "lang": "en"|"ar", "question": "...", "expected_doc_ids": [int, ...], "corpus": "<folder>"}
Writes qvectors.f32 (little-endian float32, one row per question in file order) and qmeta.json
(model, dim, count, SHA-256 of both files, no_embed flag). The embedding request shape and the
deterministic --no-embed mode are those of corpus_prepare.py, whose code is reused.
"""
import argparse
import hashlib
import importlib.util
import os
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location('corpus_prepare', HERE / 'corpus_prepare.py')
cp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp)

LANGS = {'en', 'ar'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_questions(path):
    """Return the question texts in file order. Errors name the line, never the content."""
    texts, seen = [], set()
    with Path(path).open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError:
                raise ValueError(f'questions.jsonl:{number}: not valid JSON') from None
            if not isinstance(item, dict):
                raise ValueError(f'questions.jsonl:{number}: not an object')
            ident, text, ids = item.get('id'), item.get('question'), item.get('expected_doc_ids')
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError(f'questions.jsonl:{number}: id missing or duplicate')
            if item.get('lang') not in LANGS:
                raise ValueError(f'questions.jsonl:{number}: lang must be en or ar')
            if not isinstance(item.get('corpus'), str) or not item['corpus']:
                raise ValueError(f'questions.jsonl:{number}: corpus missing')
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'questions.jsonl:{number}: question missing')
            if not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
                raise ValueError(f'questions.jsonl:{number}: expected_doc_ids must be a list of integers')
            seen.add(ident)
            texts.append(text)
    if not texts:
        raise ValueError('questions.jsonl has no questions')
    return texts


def embed(args, texts, vectors_path):
    with vectors_path.open('wb') as output:
        engine = cp.Embeddings(args, output)
        for text in texts:
            engine.add(text)
        engine.flush()
        if engine.count != len(texts):
            raise RuntimeError('internal embedding count mismatch')
    return engine.model, engine.dim


def run(args):
    source = Path(args.questions)
    out = Path(args.out) if args.out else source
    texts = read_questions(source / 'questions.jsonl')
    out.mkdir(parents=True, exist_ok=True)
    vectors_path, meta_path = out / 'qvectors.f32', out / 'qmeta.json'
    if vectors_path.exists() or meta_path.exists():
        raise ValueError('qvectors.f32 or qmeta.json already exists, refusing to overwrite')
    try:
        model, dim = embed(args, texts, vectors_path)
        if args.corpus:
            corpus_dim = json.loads((Path(args.corpus) / 'meta.json').read_text(encoding='utf-8'))['dim']
            if corpus_dim != dim:
                raise ValueError(f'question dim {dim} differs from corpus dim {corpus_dim}')
        if vectors_path.stat().st_size != len(texts) * dim * 4:
            raise RuntimeError('qvectors.f32 has the wrong length')
        meta = {'model': model, 'dim': dim, 'count': len(texts), 'no_embed': args.no_embed,
                'questions_sha256': sha256(source / 'questions.jsonl'),
                'qvectors_sha256': sha256(vectors_path)}
        meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8', newline='\n')
    except BaseException:
        vectors_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', required=True, help='directory holding questions.jsonl')
    parser.add_argument('--out', help='output directory (default: the questions directory)')
    parser.add_argument('--corpus', help='optional prepared corpus directory; its meta.json dim must match')
    parser.add_argument('--embed-url', default=os.environ.get('QUICKDB_EMBED_URL', 'http://127.0.0.1:8081/v1'))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--no-embed', action='store_true')
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()
    if min(args.batch, args.dim) <= 0 or args.batch > 256 or args.dim > 65536 or args.batch * args.dim > 1048576:
        parser.error('positive counts required, batch <=256, dim <=65536, batch*dim <=1048576')
    try:
        print(json.dumps(run(args)))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'corpus_questions: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
