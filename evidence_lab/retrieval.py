"""CPU semantic retrieval, reciprocal rank fusion and a small cross encoder.

Only source chunk text is embedded. Exact NumPy search is adequate at this scale.
"""
import hashlib
import json
import os
import tempfile
from contextlib import closing
from pathlib import Path
import numpy as np
from . import baseline

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
RERANK_MODEL = 'ms-marco-TinyBERT-L-2-v2'
METHODS = ('bm25', 'dense', 'hybrid', 'reranked')

def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def fuse(rankings, top_k=30, constant=60):
    baseline._check_int('top_k', top_k, 1)
    if constant <= 0:
        raise ValueError('constant must be positive')
    scores, hits = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = hit['chunk_id']
            if key in seen:
                continue
            seen.add(key)
            hits[key] = hit
            scores[key] = scores.get(key, 0) + 1 / (constant + rank)
    return [{**hits[k], 'score': scores[k]} for k in sorted(scores, key=lambda k: (-scores[k], k))[:top_k]]

class Retriever:
    def __init__(self, db='data/baseline.sqlite', cache='data/dense-v1', *, embedder=None, reranker=None):
        self.db, self.cache = Path(db), Path(cache)
        with closing(baseline._connect(db)) as con:
            self.rows = [dict(zip(('doc_id','page','chunk_id','text'), r)) for r in
                         con.execute('SELECT doc_id,page,chunk_id,text FROM chunks ORDER BY rowid')]
        for row in self.rows:
            row['page'] = int(row['page'])
        self.embedder, self.reranker, self.vectors = embedder, reranker, None
        self.digest = fingerprint(self.rows)

    def _embedder(self):
        if self.embedder is None:
            from .model_audit import verify
            verify(self.cache)
            from fastembed import TextEmbedding
            snapshots=list((self.cache/'models').glob('**/snapshots/*/model_optimized.onnx'))
            if len(snapshots)>1 or (not snapshots and (self.cache/'model-provenance.json').exists()):
                raise ValueError('Ambiguous or missing recorded embedding snapshot; use a new cache')
            options={'specific_model_path':str(snapshots[0].parent)} if len(snapshots)==1 else {}
            self.embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(self.cache/'models'), threads=4, **options)
        return self.embedder

    def build(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        meta = self.cache/'manifest.json'
        expected = {'model': EMBED_MODEL, 'source_chunks_sha256': self.digest, 'chunks': len(self.rows)}
        if meta.exists():
            if json.loads(meta.read_text('utf-8')) != expected:
                raise ValueError('Dense cache does not match source chunks/model; use a new cache directory')
            self.vectors = np.load(self.cache/'vectors.npy', allow_pickle=False)
        else:
            self.vectors = np.asarray(list(self._embedder().embed([r['text'] for r in self.rows], batch_size=64)), dtype=np.float32)
            self.vectors /= np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)
            with tempfile.NamedTemporaryFile(dir=self.cache, suffix='.npy', delete=False) as stream:
                tmp = Path(stream.name)
                np.save(stream, self.vectors, allow_pickle=False)
            os.replace(tmp,self.cache/'vectors.npy')
            from .corpus import write_json_atomic
            write_json_atomic(meta,expected)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.rows) or not np.isfinite(self.vectors).all():
            raise ValueError('Invalid dense vectors')
        return expected

    def search(self, query, doc_id=None, top_k=5, method='hybrid'):
        baseline._check_int('top_k', top_k, 1)
        if method not in METHODS:
            raise ValueError('Unknown retrieval method')
        if not query.strip():
            return []
        if method == 'bm25':
            return baseline.search(self.db, query, doc_id, top_k)
        if self.vectors is None:
            self.build()
        q = np.asarray(next(iter(self._embedder().query_embed(query))), dtype=np.float32)
        if q.shape != (self.vectors.shape[1],) or not np.isfinite(q).all() or np.linalg.norm(q)==0:
            raise ValueError('Invalid query vector')
        q /= max(float(np.linalg.norm(q)), 1e-12)
        indices = np.array([i for i,r in enumerate(self.rows) if doc_id is None or r['doc_id'] == doc_id], dtype=int)
        if not len(indices):
            return []
        scores = self.vectors[indices] @ q
        count = top_k if method == 'dense' else 30
        dense = [{**self.rows[int(indices[i])], 'score': float(scores[i])} for i in np.argsort(-scores, kind='stable')[:count]]
        if method == 'dense':
            return dense
        hits = fuse([baseline.search(self.db, query, doc_id, count), dense], count)
        if method == 'reranked':
            if self.reranker is None:
                from flashrank import Ranker
                self.reranker = Ranker(model_name=RERANK_MODEL, cache_dir=str(self.cache/'reranker'), max_length=512)
            from flashrank import RerankRequest
            passages = [{'id': i, 'text': h['text']} for i,h in enumerate(hits)]
            ranked = self.reranker.rerank(RerankRequest(query=query, passages=passages))
            hits = [{**hits[r['id']], 'score': float(r['score'])} for r in ranked]
        return hits[:top_k]
