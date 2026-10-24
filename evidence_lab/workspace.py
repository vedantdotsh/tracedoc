"""Mixed-source adapter over SQLite and separately cached OpenRouter vectors."""
import argparse
import json
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
from . import baseline, evidence_eval
from .sources import sha256_file
from .retrieval import fuse


class Workspace:
    def __init__(self, index=Path('data/workspace'), *, allow_paid=False):
        index = Path(index)
        self.db = index if index.suffix == '.sqlite' else index / 'evidence.sqlite'
        with closing(baseline._connect(self.db)) as con:
            self.meta = dict(con.execute('SELECT key,value FROM meta'))
            self.rows = [dict(zip(('doc_id', 'page', 'chunk_id', 'text'), r)) for r in
                         con.execute('SELECT doc_id,page,chunk_id,text FROM chunks ORDER BY rowid')]
            self.evidence = {r[0]: dict(zip(('kind', 'source_sha256', 'system', 'revision', 'location'),
                                           (*r[1:5], json.loads(r[5])))) for r in
                            con.execute('SELECT chunk_id,kind,source_sha256,system,revision,location FROM evidence')}
        self.paths = json.loads(self.meta['source_paths'])
        self.by_id = {r['chunk_id']: r for r in self.rows}
        self.embedder = None
        self.vectors = None
        if allow_paid:
            from .openrouter_embeddings import Embedder
            self.embedder = Embedder(self.db.parent / 'openrouter-vectors')

    def documents(self):
        return sorted(self.paths)

    def located(self, hit):
        ev = self.evidence[hit['chunk_id']]
        location = {'kind': ev['kind'], **ev['location']}
        with closing(baseline._connect(self.db)) as con:
            extra = json.loads(con.execute('SELECT extra FROM evidence WHERE chunk_id=?', (hit['chunk_id'],)).fetchone()[0])
        if ev['kind'] == 'xlsx_row':
            location['cells'] = ', '.join(f['cell'] for f in extra['fields'])
            location['sheet_state'] = extra['sheet_state']
        return {**hit, **ev, 'page': int(hit['page']) if 'page' in ev['location'] else None,
                'location': location, 'extraction': {k: extra[k] for k in
                    ('engine', 'model', 'source_verified', 'confidence') if k in extra}}

    def search(self, question, doc_id=None, limit=5, method='bm25'):
        baseline._check_int('limit', limit, 1)
        if method not in ('bm25', 'dense', 'hybrid'):
            raise ValueError('Unsupported search method')
        if doc_id is not None and doc_id not in self.paths:
            raise ValueError('Unknown document')
        if method == 'bm25':
            return [self.located(h) for h in baseline.search(self.db, question, doc_id, limit)]
        if self.embedder is None:
            raise ValueError('Enable paid embeddings explicitly to use dense/hybrid search')
        if self.vectors is None:
            # Prebuild explicitly; a chat request must never embed the corpus unexpectedly.
            self.vectors = np.asarray(self.embedder.cached([r['text'] for r in self.rows]))
        q = np.asarray(self.embedder.query_embed(question))
        candidates = [(i, r) for i, r in enumerate(self.rows) if doc_id is None or r['doc_id'] == doc_id]
        ranked = sorted(({**r, 'score': float(self.vectors[i] @ q)} for i, r in candidates),
                        key=lambda h: (-h['score'], h['chunk_id']))[:max(limit, 30)]
        if method == 'hybrid':
            ranked = fuse([baseline.search(self.db, question, doc_id, max(limit, 30)), ranked], limit)
        return [self.located(h) for h in ranked[:limit]]

    def source(self, chunk_id):
        if chunk_id not in self.by_id:
            raise ValueError('Unknown evidence ID')
        hit = self.located(self.by_id[chunk_id])
        path = Path(self.paths[hit['doc_id']])
        if not path.is_file() or sha256_file(path) != hit['source_sha256']:
            raise ValueError('Source changed or is missing; reindex before inspection')
        if path.suffix.lower() in ('.pdf', '.png', '.jpg', '.jpeg'):
            hit['image_url'] = '/api/image?' + urlencode({'chunk_id': chunk_id})
        return hit

    def image(self, chunk_id):
        import pymupdf
        hit = self.source(chunk_id)
        path = Path(self.paths[hit['doc_id']])
        if path.suffix.lower() == '.pdf':
            with pymupdf.open(path) as pdf:
                page = pdf[hit['page'] - 1]
                scale = min(1.5, 2400 / max(page.rect.width, page.rect.height))
                return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes('png')
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError('Source is not a drawing image')
        return pymupdf.Pixmap(str(path)).tobytes('png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, default=Path('data/workspace'))
    p.add_argument('--embed', action='store_true')
    p.add_argument('--allow-paid', action='store_true')
    p.add_argument('--query')
    p.add_argument('--method', choices=('bm25', 'dense', 'hybrid'), default='bm25')
    a = p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(override=False)
    ws = Workspace(a.index, allow_paid=a.allow_paid)
    if a.embed:
        if not ws.embedder:
            p.error('--embed requires --allow-paid')
        list(ws.embedder.embed([r['text'] for r in ws.rows]))
        print(json.dumps({'embedded_chunks': len(ws.rows), 'model': 'openai/text-embedding-3-small'}))
    if a.query:
        print(json.dumps(ws.search(a.query, method=a.method), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
