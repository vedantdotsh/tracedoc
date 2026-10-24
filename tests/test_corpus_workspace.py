import hashlib
import json
import sqlite3

import numpy as np
import pytest

from evidence_lab.corpus_workspace import CorpusWorkspace
from evidence_lab.openrouter_embeddings import DIMENSIONS, MODEL, _key
from evidence_lab.retrieval import Retriever
from evidence_lab.sources import sha256_file


def _vector(x=1.0, y=0.0):
    v = np.zeros(DIMENSIONS, dtype=np.float32)
    v[:2] = (x, y)
    return v / np.linalg.norm(v)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path
    pdf_dir = root / 'data/documents'
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / 'a.pdf'
    pdf.write_bytes(b'source PDF fixture')
    source_hash = sha256_file(pdf)
    (root / 'data/manifest.json').write_text(json.dumps({
        'documents': [{'doc_id': pdf.name, 'sha256': source_hash}],
    }), encoding='utf-8')
    db = root / 'data/ocr-index/ocr.sqlite'
    db.parent.mkdir(parents=True)
    rows = [
        ('a.pdf', 1, 'a.pdf#p1-c0', 'pump temperature'),
        ('a.pdf', 1, 'a.pdf#p1-c1', 'pump pressure'),
        ('a.pdf', 2, 'a.pdf#p2-ocr-c0', 'pump maximum'),
        ('a.pdf', 3, 'a.pdf#p3-c0', 'pump steel'),
    ]
    with sqlite3.connect(db) as con:
        con.execute('CREATE VIRTUAL TABLE chunks USING fts5(doc_id UNINDEXED,page UNINDEXED,chunk_id UNINDEXED,text)')
        con.execute('CREATE TABLE docs(doc_id TEXT PRIMARY KEY, sha256 TEXT, pages INTEGER)')
        con.execute('CREATE TABLE ocr_evidence(chunk_id TEXT PRIMARY KEY, provenance TEXT)')
        con.executemany('INSERT INTO chunks VALUES (?,?,?,?)', rows)
        con.execute('INSERT INTO docs VALUES (?,?,?)', ('a.pdf', source_hash, 3))
        con.execute('INSERT INTO ocr_evidence VALUES (?,?)',
                    ('a.pdf#p2-ocr-c0', json.dumps({'kind': 'ocr_text', 'source_verified': False})))
    cache = root / 'data/openai-corpus-v1'
    vector_dir = cache / 'vectors'
    vector_dir.mkdir(parents=True)
    vectors = [_vector(1, 0), _vector(.9, .1), _vector(.8, .2), _vector(.7, .3)]
    for row, vec in zip(rows, vectors):
        np.save(vector_dir / f'{_key(row[3])}.npy', vec)
    np.save(vector_dir / f'{_key("pump")}.npy', _vector())
    retriever = Retriever(db, cache)
    manifest = {'model': MODEL, 'dimensions': DIMENSIONS,
                'source_db_sha256': sha256_file(db),
                'source_chunks_sha256': retriever.digest, 'chunks': len(rows)}
    (cache / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root, pdf, cache, rows


def test_search_uses_saved_vectors_and_unique_pages_without_api(corpus, monkeypatch):
    root, pdf, cache, rows = corpus
    def no_request(*args):
        raise AssertionError('Unexpected corpus embedding API request')
    monkeypatch.setattr('evidence_lab.openrouter_embeddings.Embedder._request', no_request)
    ws = CorpusWorkspace(root)
    assert ws.documents() == ['a.pdf']
    assert ws.embedder is None
    assert (cache / 'corpus-matrix.npy').is_file()
    # The second construction must use the validated consolidated matrix.
    for row in rows:
        (cache / 'vectors' / f'{_key(row[3])}.npy').unlink()
    ws = CorpusWorkspace(root)
    for method in ('bm25', 'dense', 'hybrid'):
        hits = ws.search('pump', limit=5, method=method)
        assert [h['page'] for h in hits] == [1, 2, 3]
        assert len({(h['doc_id'], h['page']) for h in hits}) == len(hits)
        assert all(h['chunk_id'] in {r[2] for r in rows} for h in hits)
        assert all(h['extraction'].get('source_verified') is False
                   for h in hits if h['chunk_id'] == 'a.pdf#p2-ocr-c0')
    ocr = ws.source('a.pdf#p2-ocr-c0')
    assert ocr['extraction']['ocr_provenance']['kind'] == 'ocr_text'
    assert 'source_verified' not in ws.source('a.pdf#p1-c0')['extraction']
    assert ocr['image_url'].startswith('/api/image?')
    pdf.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        ws.source('a.pdf#p2-ocr-c0')


def test_rejects_vector_and_manifest_mismatch(corpus):
    root, _, cache, rows = corpus
    vector = cache / 'vectors' / f'{_key(rows[0][3])}.npy'
    np.save(vector, np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match='Invalid cached embedding'):
        CorpusWorkspace(root)
    np.save(vector, _vector())
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    manifest['source_chunks_sha256'] = hashlib.sha256(b'wrong chunks').hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest does not match'):
        CorpusWorkspace(root)


def test_rejects_changed_pdf_at_startup(corpus):
    root, pdf, _, _ = corpus
    pdf.write_bytes(b'changed after OCR indexing')
    with pytest.raises(ValueError, match='Source PDF hash changed'):
        CorpusWorkspace(root)


def test_offline_missing_query_and_explicit_paid_flag(corpus, monkeypatch):
    root, _, _, _ = corpus
    ws = CorpusWorkspace(root)
    with pytest.raises(ValueError, match='Embeddings missing'):
        ws.search('not cached', method='dense')
    seen = []
    def query(self, question):
        seen.append(question)
        return _vector()
    monkeypatch.setattr('evidence_lab.openrouter_embeddings.Embedder.query_embed', query)
    paid = CorpusWorkspace(root, allow_paid=True)
    assert paid.embedder is not None and paid.default_method == 'hybrid'
    assert paid.search('new question', method='dense')[0]['page'] == 1
    assert seen == ['new question']
