"""Read-only chat adapter for the audited OCR PDF corpus and saved OpenAI vectors."""

import hashlib
import json
import os
import tempfile
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from . import baseline
from .openrouter_embeddings import DIMENSIONS, MODEL, Embedder
from .retrieval import Retriever, fuse
from .sources import sha256_file


class CorpusWorkspace:
    """Serve source-grounded search without ever re-embedding the document corpus."""

    def __init__(self, root, *, allow_paid=False):
        self.root = Path(root).resolve()
        self.db = self.root / 'data/ocr-index/ocr.sqlite'
        self.cache = self.root / 'data/openai-corpus-v1'
        self.retriever = Retriever(self.db, self.cache)
        self.rows = self.retriever.rows
        self.by_id = {row['chunk_id']: row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError('Duplicate OCR chunk IDs')
        corpus_manifest = json.loads((self.root / 'data/manifest.json').read_text('utf-8'))
        self.paths = {}
        self.hashes = {}
        for doc in corpus_manifest['documents']:
            doc_id = doc['doc_id']
            path = (self.root / 'data/documents' / doc_id).resolve()
            if path.parent != (self.root / 'data/documents').resolve() or path.suffix.lower() != '.pdf':
                raise ValueError(f'Invalid source PDF path: {doc_id}')
            self.paths[doc_id], self.hashes[doc_id] = path, doc['sha256']
        with closing(baseline._connect(self.db)) as con:
            indexed = {doc_id: sha for doc_id, sha in con.execute('SELECT doc_id,sha256 FROM docs')}
            self.ocr = {cid: json.loads(prov) for cid, prov in
                        con.execute('SELECT chunk_id,provenance FROM ocr_evidence')}
        if indexed != self.hashes or set(self.paths) != {p.name for p in (self.root / 'data/documents').glob('*.pdf')}:
            raise ValueError('Source PDF inventory or hashes differ from OCR index')
        for doc_id, path in self.paths.items():
            if sha256_file(path) != self.hashes[doc_id]:
                raise ValueError(f'Source PDF hash changed: {doc_id}')
        manifest_path = self.cache / 'manifest.json'
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        db_hash = sha256_file(self.db)
        if (manifest.get('model') != MODEL or manifest.get('dimensions') != DIMENSIONS
                or manifest.get('source_db_sha256') != db_hash
                or manifest.get('source_chunks_sha256') != self.retriever.digest
                or manifest.get('chunks') != len(self.rows)):
            raise ValueError('OpenAI vector manifest does not match OCR chunks, model, dimensions or database')
        self._cache_embedder = Embedder(self.cache)
        self.embedder = self._cache_embedder if allow_paid else None
        self.allow_paid = allow_paid
        self.default_method = 'hybrid' if allow_paid else 'bm25'
        self.library_label = (f'MMLongBench-Doc · {len(self.paths)} PDFs · OCR and saved OpenAI '
                              'embeddings · Hybrid search may use a paid query embedding')
        self.vectors = self._load_vectors(hashlib.sha256(manifest_bytes).hexdigest(), db_hash)

    def _load_vectors(self, manifest_hash, db_hash):
        matrix_path = self.cache / 'corpus-matrix.npy'
        meta_path = self.cache / 'corpus-matrix.json'
        identity = {'manifest_sha256': manifest_hash, 'source_db_sha256': db_hash,
                    'source_chunks_sha256': self.retriever.digest, 'model': MODEL,
                    'dimensions': DIMENSIONS, 'chunks': len(self.rows)}
        if matrix_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text('utf-8'))
            if all(meta.get(k) == v for k, v in identity.items()):
                if meta.get('matrix_sha256') != sha256_file(matrix_path):
                    raise ValueError('Consolidated vector matrix hash mismatch')
                matrix = np.load(matrix_path, allow_pickle=False)
                return self._checked_vectors(matrix)
        # The paid embedding run saved one vector per unique text. Loading these is
        # strictly cache-only; missing or invalid files fail before the server starts.
        matrix = self._checked_vectors(self._cache_embedder.cached([r['text'] for r in self.rows]))
        with tempfile.NamedTemporaryFile(dir=self.cache, suffix='.npy', delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, matrix, allow_pickle=False)
        try:
            digest = sha256_file(temporary)
            os.replace(temporary, matrix_path)
            from .corpus import write_json_atomic
            write_json_atomic(meta_path, {**identity, 'matrix_sha256': digest})
        finally:
            temporary.unlink(missing_ok=True)
        return matrix

    def _checked_vectors(self, vectors):
        if (vectors.shape != (len(self.rows), DIMENSIONS) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError('Invalid OpenAI corpus vector matrix')
        return vectors

    def documents(self):
        return sorted(self.paths)

    def located(self, hit):
        row = self.by_id[hit['chunk_id']]
        provenance = self.ocr.get(row['chunk_id'])
        is_ocr = provenance is not None
        extraction = {'engine': 'OCR' if is_ocr else 'native PDF text',
                      'model': 'local OCR' if is_ocr else 'PyMuPDF'}
        if is_ocr:
            extraction['source_verified'] = False
            extraction['ocr_provenance'] = provenance
        return {**hit, 'page': row['page'], 'kind': 'pdf_page',
                'source_sha256': self.hashes[row['doc_id']],
                'location': {'kind': 'pdf_page', 'page': row['page']},
                'extraction': extraction}

    def search(self, question, doc_id=None, limit=5, method='bm25'):
        baseline._check_int('limit', limit, 1)
        if method not in ('bm25', 'dense', 'hybrid'):
            raise ValueError('Unsupported search method')
        if doc_id is not None and doc_id not in self.paths:
            raise ValueError('Unknown document')
        if not isinstance(question, str) or not question.strip():
            return []
        count = 30
        bm = baseline.search(self.db, question, doc_id, count) if method != 'dense' else []
        if method == 'bm25':
            ranked = bm
        else:
            # Offline queries use only a previously cached vector. A new query
            # may reach OpenRouter only when the caller explicitly enabled it.
            q = (self._cache_embedder.query_embed(question) if self.allow_paid else
                 self._cache_embedder.cached([question])[0])
            q = np.asarray(q, dtype=np.float32)
            if q.shape != (DIMENSIONS,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
                raise ValueError('Invalid query vector')
            q /= np.linalg.norm(q)
            indices = np.asarray([i for i, row in enumerate(self.rows)
                                  if doc_id is None or row['doc_id'] == doc_id], dtype=int)
            scores = self.vectors[indices] @ q
            order = np.argsort(-scores, kind='stable')[:count]
            dense = [{**self.rows[int(indices[i])], 'score': float(scores[i])} for i in order]
            ranked = dense if method == 'dense' else fuse([bm, dense], count)
        result, pages = [], set()
        for hit in ranked:
            key = hit['doc_id'], hit['page']
            if key not in pages:
                pages.add(key)
                result.append(self.located(hit))
                if len(result) == limit:
                    break
        return result

    def source(self, chunk_id):
        if chunk_id not in self.by_id:
            raise ValueError('Unknown evidence ID')
        hit = self.located(self.by_id[chunk_id])
        path = self.paths[hit['doc_id']]
        if not path.is_file() or sha256_file(path) != hit['source_sha256']:
            raise ValueError('Source changed or is missing; reindex before inspection')
        hit['image_url'] = '/api/image?' + urlencode({'chunk_id': chunk_id})
        return hit

    def image(self, chunk_id):
        import pymupdf
        hit = self.source(chunk_id)
        with pymupdf.open(self.paths[hit['doc_id']]) as pdf:
            page = pdf[hit['page'] - 1]
            scale = min(1.5, 2400 / max(page.rect.width, page.rect.height))
            return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes('png')
