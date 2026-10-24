import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_openai_corpus as corpus


def test_preflight_preserves_chunk_text_and_counts_queries_without_writing(tmp_path, monkeypatch):
    rows = [{"text": "original OCR chunk", "doc_id": "a.pdf", "page": 1, "chunk_id": "a-1"},
            {"text": "original OCR chunk", "doc_id": "a.pdf", "page": 1, "chunk_id": "a-2"}]
    fake = SimpleNamespace(rows=rows, digest="source-fingerprint")
    monkeypatch.setattr(corpus, "Retriever", lambda *args: fake)
    monkeypatch.setattr(corpus, "checked_source", lambda *args: ("eligible-sha", [{"question": "question"}]))
    monkeypatch.setattr(corpus, "sha256", lambda *args: "db-sha")
    _, chunks, queries, texts, plan = corpus.preflight(tmp_path)
    assert chunks == ["original OCR chunk", "original OCR chunk"]
    assert queries == ["question"]
    assert texts == ["original OCR chunk", "question"]
    assert plan["source_chunks_sha256"] == "source-fingerprint"
    assert plan["pending_texts"] == 2
    assert not (tmp_path / corpus.CACHE).exists()


def test_cached_query_adapter_yields_one_vector_without_network():
    class FakeEmbedder:
        def cached(self, texts):
            assert texts == ["question"]
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    vectors = list(corpus.CachedQueries(FakeEmbedder()).query_embed("question"))
    assert len(vectors) == 1
    assert vectors[0].tolist() == [1.0, 0.0]


def test_comparison_rejects_any_bm25_drift(tmp_path):
    root = tmp_path
    before = root / corpus.OLD_REPORT
    after = root / corpus.REPORT
    before.mkdir(parents=True)
    after.mkdir(parents=True)
    metadata = {"words": "180"}
    dense = {"source_chunks_sha256": "same"}
    summary = {"eligible_sha256": "same", "source_revision": "same", "pdf_files": 135,
               "pdf_pages": 6529, "indexed_pages_with_text": 6510,
               "index_metadata": metadata, "dense_manifest": dense,
               "retrieval": {f"{s}_{m}": {"mean_page_recall_at_5": 1.0} for s in corpus.SCOPES for m in corpus.METHODS}}
    for directory in (before, after):
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    base = root / "reports/corpus-audit-v1"
    base.mkdir(parents=True)
    (base / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    retrieval = {f"{s}_{m}": {"recall": 1.0, "hit": 1.0, "mrr": 1.0, "ranked_pages": [{"doc_id": "a.pdf", "page": 1}]}
                 for s in corpus.SCOPES for m in corpus.METHODS}
    case = {"source_row": 4, "doc_id": "a.pdf", "question": "question", "gold_pages": [1],
            "evidence_sources": ["Text"], "retrieval": retrieval}
    (before / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    changed = json.loads(json.dumps(case))
    changed["retrieval"]["global_bm25"]["hit"] = 0.0
    (after / "cases.jsonl").write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BM25 changed"):
        corpus.compare(root, after)
