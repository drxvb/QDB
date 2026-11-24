#!/usr/bin/env python3
"""Prepare QuickDB JSONL and little-endian vectors using bounded batches and disk indexes."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import struct
import sys
import time
import urllib.parse
import urllib.request

SKIP_DIRS = {'.git', 'node_modules', 'target', 'graphify-out', '.venv', 'venv',
             '__pycache__', '.worktrees', '.backups', '.claude'}
SENSITIVE = ('secret', 'credential', 'token', 'password', '.env', 'key')
EXTENSIONS = {'.md', '.txt', '.tex', '.rst'}
MAX_BYTES = 2 * 1024 * 1024
LINK = re.compile(r'(?<!!)\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def peak_memory_bytes():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_working', 'working', 'peak_paged',
                'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]

        values = Counters()
        values.cb = ctypes.sizeof(values)
        api = ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        api.restype = wintypes.BOOL
        if api(ctypes.c_void_p(-1), ctypes.byref(values), values.cb):
            return values.peak_working
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024


def sensitive(name):
    return any(part in name.lower() for part in SENSITIVE)


def chunks(text, target=1200):
    """Keep paragraph boundaries where possible, split oversized paragraphs at whitespace."""
    pending = ''
    for paragraph in re.split(r'\n\s*\n', text.replace('\r\n', '\n')):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if pending and len(pending) + len(paragraph) + 2 > target:
            yield pending
            pending = ''
        while len(paragraph) > target:
            cut = paragraph.rfind(' ', target // 2, target + 1)
            if cut < 0:
                cut = target
            yield paragraph[:cut]
            paragraph = paragraph[cut:].lstrip()
        if paragraph:
            pending = pending + '\n\n' + paragraph if pending else paragraph
    if pending:
        yield pending


def request_json(url, payload=None):
    last = None
    for attempt in range(4):
        try:
            data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=180) as response:
                body = response.read(32 * 1024 * 1024 + 1)
                if len(body) > 32 * 1024 * 1024:
                    raise ValueError('embedding response exceeds 32 MiB limit')
                return json.loads(body)
        except (OSError, ValueError) as exc:
            last = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'embedding request failed after three retries: {url}: {last}')


def write_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


class Embeddings:
    def __init__(self, args, output):
        self.args = args
        self.output = output
        self.dim = args.dim if args.no_embed else None
        self.model = 'deterministic-random-v1' if args.no_embed else None
        if not args.no_embed:
            models = request_json(args.embed_url.rstrip('/') + '/models')
            try:
                self.model = models['data'][0]['id']
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError('embedding /models returned no model id') from exc
        self.batch = []
        self.rng = random.Random(0x514442)
        self.count = 0

    def add(self, text):
        self.batch.append(text)
        if len(self.batch) >= self.args.batch:
            self.flush()

    def flush(self):
        if not self.batch:
            return
        if self.args.no_embed:
            vectors = [[self.rng.uniform(-1, 1) for _ in range(self.dim)] for _ in self.batch]
        else:
            result = request_json(self.args.embed_url.rstrip('/') + '/embeddings',
                                  {'model': self.model, 'input': list(self.batch), 'encoding_format': 'float'})
            try:
                items = sorted(result['data'], key=lambda item: item['index'])
                if [item['index'] for item in items] != list(range(len(self.batch))):
                    raise ValueError('missing or duplicate embedding indexes')
                vectors = [item['embedding'] for item in items]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f'invalid embedding response: {exc}') from exc
        for vector in vectors:
            if not isinstance(vector, list):
                raise RuntimeError('embedding must be a numeric array')
            if self.dim is None:
                self.dim = len(vector)
            if self.dim > 65536 or self.dim * self.args.batch > 1048576:
                raise RuntimeError('embedding dimensions exceed the bounded-memory batch limit')
            if not self.dim or len(vector) != self.dim or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                raise RuntimeError('embedding has inconsistent dimension or non-finite values')
            try:
                self.output.write(struct.pack('<' + 'f' * self.dim, *vector))
            except (OverflowError, struct.error) as exc:
                raise RuntimeError('embedding is outside finite f32 range') from exc
            self.count += 1
        self.batch.clear()


def walk_files(root, out, db, skipped):
    """Sort traversal on disk, so a wide directory cannot exhaust process memory."""
    db.execute('INSERT INTO walk_queue VALUES (?)', (str(root),))
    while True:
        row = db.execute('SELECT path FROM walk_queue ORDER BY path LIMIT 1').fetchone()
        if row is None:
            return
        parent = Path(row[0])
        db.execute('DELETE FROM walk_queue WHERE path=?', row)
        try:
            with os.scandir(parent) as entries:
                for entry in entries:
                    if sensitive(entry.name) or entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    if getattr(path, 'is_junction', lambda: False)():
                        continue
                    resolved = path.resolve()
                    if resolved == out or not resolved.is_relative_to(root):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.lower() not in SKIP_DIRS:
                            db.execute('INSERT OR IGNORE INTO walk_queue VALUES (?)', (str(path),))
                    elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in EXTENSIONS:
                        db.execute('INSERT INTO walk_files VALUES (?)', (str(path),))
        except OSError:
            skipped['unreadable'] += 1
        db.commit()
        for item in db.execute('SELECT path FROM walk_files ORDER BY path'):
            yield parent, (Path(item[0]).name,)
        db.execute('DELETE FROM walk_files')


def prepare(args):
    started = utc_now()
    roots = [Path(root).resolve(strict=True) for root in args.roots]
    if len(set(roots)) != len(roots):
        raise ValueError('roots must be distinct')
    for i, root in enumerate(roots):
        if not root.is_dir() or sensitive(root.name):
            raise ValueError(f'not an allowed root directory: {root}')
        if any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots[:i]):
            raise ValueError('overlapping roots are not allowed')
    out = Path(args.out).resolve()
    if any(out == root or root.is_relative_to(out) for root in roots):
        raise ValueError('output must not contain a root')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output directory must be empty, refusing to overwrite')
    db_path = out / '.prepare.sqlite3'
    db = sqlite3.connect(db_path)
    db.execute('PRAGMA cache_size=-8192')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE docs(path TEXT PRIMARY KEY, id INTEGER UNIQUE)')
    db.execute('CREATE TABLE dirs(path TEXT PRIMARY KEY, id INTEGER UNIQUE)')
    db.execute('CREATE TABLE links(src INTEGER, path TEXT, PRIMARY KEY(src,path))')
    db.execute('CREATE TABLE walk_queue(path TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE walk_files(path TEXT PRIMARY KEY)')
    counts = {'chunks': 0, 'docs': 0, 'edges': 0, 'directories': 0}
    corpora = {}
    skipped = {'too_large': 0, 'unreadable': 0, 'invalid_utf8': 0}
    try:
        with (out / 'chunks.jsonl').open('w', encoding='utf-8', newline='\n') as chunk_file, \
             (out / 'docs.jsonl').open('w', encoding='utf-8', newline='\n') as doc_file, \
             (out / 'edges.jsonl').open('w', encoding='utf-8', newline='\n') as edge_file, \
             (out / 'vectors.f32').open('wb') as vector_file:
            embed = Embeddings(args, vector_file)

            def edge(src, rel, dst):
                write_jsonl(edge_file, {'src': src, 'rel': rel, 'dst': dst, 'weight': 1.0})
                counts['edges'] += 1

            def directory(path, root):
                row = db.execute('SELECT id FROM dirs WHERE path=?', (str(path),)).fetchone()
                if row:
                    return row[0]
                parent_id = directory(path.parent, root) if path != root else None
                if counts['directories'] >= 500000:
                    raise ValueError('directory id namespace exhausted')
                ident = 500000 + counts['directories']
                counts['directories'] += 1
                db.execute('INSERT INTO dirs VALUES (?,?)', (str(path), ident))
                if parent_id is not None:
                    edge(parent_id, 'has', ident)
                return ident

            for root in roots:
                corpus = root.name
                per = corpora.setdefault(corpus, {'chunks': 0, 'docs': 0})
                for parent, files in walk_files(root, out, db, skipped):
                    for name in sorted(files):
                        if counts['chunks'] >= args.max_chunks:
                            break
                        if sensitive(name) or Path(name).suffix.lower() not in EXTENSIONS:
                            continue
                        path = parent / name
                        try:
                            if path.is_symlink() or not path.is_file():
                                continue
                            if path.stat().st_size > MAX_BYTES:
                                skipped['too_large'] += 1
                                continue
                            with path.open('rb') as source:
                                raw = source.read(MAX_BYTES + 1)
                            if len(raw) > MAX_BYTES:
                                skipped['too_large'] += 1
                                continue
                            try:
                                text = raw.decode('utf-8-sig')
                            except UnicodeDecodeError:
                                skipped['invalid_utf8'] += 1
                                continue
                        except OSError:
                            skipped['unreadable'] += 1
                            continue
                        pieces = iter(chunks(text))
                        first = next(pieces, None)
                        if first is None:
                            continue
                        if counts['docs'] >= 499999:
                            raise ValueError('document id namespace exhausted')
                        doc_id = counts['docs'] + 1
                        counts['docs'] += 1
                        per['docs'] += 1
                        db.execute('INSERT INTO docs VALUES (?,?)', (str(path.resolve()), doc_id))
                        edge(directory(parent, root), 'has', doc_id)
                        relpath = path.relative_to(root.parent).as_posix()
                        previous = None
                        seq = 0
                        piece = first
                        while piece is not None and counts['chunks'] < args.max_chunks:
                            ident = 1000000 + counts['chunks']
                            write_jsonl(chunk_file, {'id': ident, 'doc_id': doc_id, 'corpus': corpus,
                                                    'path': relpath, 'seq': seq, 'chars': len(piece), 'text': piece})
                            embed.add(piece)
                            edge(doc_id, 'contains', ident)
                            if previous is not None:
                                edge(previous, 'next', ident)
                            previous = ident
                            seq += 1
                            counts['chunks'] += 1
                            per['chunks'] += 1
                            piece = next(pieces, None)
                        write_jsonl(doc_file, {'id': doc_id, 'corpus': corpus, 'path': relpath,
                                              'chunks': seq, 'chars': len(text)})
                        if path.suffix.lower() == '.md':
                            for match in LINK.finditer(text):
                                target = match.group(1).strip('<>')
                                parsed = urllib.parse.urlsplit(target)
                                if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith(('/', '\\')):
                                    continue
                                dest = (parent / urllib.parse.unquote(parsed.path)).resolve()
                                db.execute('INSERT OR IGNORE INTO links VALUES (?,?)', (doc_id, str(dest)))
                        db.commit()
                    if counts['chunks'] >= args.max_chunks:
                        break
                if counts['chunks'] >= args.max_chunks:
                    break
            embed.flush()
            for src, dst in db.execute('SELECT links.src, docs.id FROM links JOIN docs ON docs.path=links.path ORDER BY links.src,docs.id'):
                edge(src, 'links', dst)
            if embed.count != counts['chunks']:
                raise RuntimeError('internal embedding count mismatch')
        meta = {**counts, 'dim': embed.dim or args.dim, 'model': embed.model,
                'roots': [str(root) for root in roots], 'per_corpus': corpora,
                'start_time': started, 'end_time': utc_now(), 'skipped': skipped,
                'no_embed': args.no_embed, 'vector_bytes': (out / 'vectors.f32').stat().st_size,
                'peak_memory_bytes': peak_memory_bytes()}
        with (out / 'meta.json').open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return meta
    finally:
        db.close()
        db_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-chunks', type=int, default=150000)
    parser.add_argument('--embed-url', default=os.environ.get('QUICKDB_EMBED_URL', 'http://127.0.0.1:8081/v1'))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--no-embed', action='store_true')
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()
    if min(args.max_chunks, args.batch, args.dim) <= 0 or args.batch > 256 or args.dim > 65536 or args.batch * args.dim > 1048576:
        parser.error('positive counts required, batch <=256, dim <=65536, batch*dim <=1048576')
    try:
        print(json.dumps(prepare(args), ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f'corpus_prepare: {exc}. Output is incomplete unless meta.json exists.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
