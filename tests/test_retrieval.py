"""Offline retrieval checks with tiny FTS rows and injected vector models."""

import sqlite3
import sys
import types

import numpy as np
import pytest

from evidence_lab.retrieval import Retriever, fuse


ROWS = [
    ("a.pdf", 1, "a#1", "apple policy"),
    ("a.pdf", 2, "a#2", "banana semantic"),
    ("b.pdf", 1, "b#1", "apple external"),
]


class FakeEmbedder:
    def __init__(self):
        self.embedded = []
        self.queries = []

    def embed(self, texts, batch_size=64):
        self.embedded.extend(texts)
        vectors = {"apple policy": [0.0, 1.0], "banana semantic": [1.0, 0.0],
                   "apple external": [1.0, 0.0]}
        return (np.asarray(vectors[text], dtype=np.float32) for text in texts)

    def query_embed(self, query):
        self.queries.append(query)
        yield np.asarray([1.0, 0.0], dtype=np.float32)


@pytest.fixture
def indexed(tmp_path):
    db = tmp_path / "index.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5("
                    "doc_id UNINDEXED, page UNINDEXED, chunk_id UNINDEXED, text)")
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", ROWS)
    return db


def test_fusion_rewards_agreement_deduplicates_and_breaks_ties():
    first = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    second = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]
    result = fuse([first, second], top_k=3)
    assert [hit["chunk_id"] for hit in result] == ["b", "a", "c"]
    assert result[0]["score"] > result[1]["score"]
    assert len({hit["chunk_id"] for hit in result}) == 3
    assert [hit["chunk_id"] for hit in fuse([first], top_k=1)] == ["a"]
    with pytest.raises(ValueError):
        fuse([first], top_k=0)
    with pytest.raises(ValueError):
        fuse([first], constant=0)


def test_dense_scope_and_hybrid_use_only_source_chunks(indexed, tmp_path):
    embedder = FakeEmbedder()
    retriever = Retriever(indexed, tmp_path / "cache", embedder=embedder)
    lexical = retriever.search("apple", doc_id="a.pdf", method="bm25")
    assert [hit["chunk_id"] for hit in lexical] == ["a#1"]
    assert embedder.embedded == []  # BM25 does not load a model
    dense = retriever.search("apple", doc_id="a.pdf", method="dense")
    assert dense[0]["chunk_id"] == "a#2"
    assert all(hit["doc_id"] == "a.pdf" for hit in dense)
    assert retriever.search("apple", doc_id="missing.pdf", method="dense") == []
    hybrid = retriever.search("apple", doc_id="a.pdf", top_k=2, method="hybrid")
    assert {hit["chunk_id"] for hit in hybrid} == {"a#1", "a#2"}
    assert embedder.embedded == [row[3] for row in ROWS]
    assert embedder.queries == ["apple", "apple", "apple"]
    assert retriever.search("  ", method="dense") == []
    with pytest.raises(ValueError, match="method"):
        retriever.search("apple", method="unknown")


def test_dense_cache_reuse_and_stale_source_rejection(indexed, tmp_path):
    cache = tmp_path / "cache"
    first = Retriever(indexed, cache, embedder=FakeEmbedder())
    expected = first.build()
    assert expected["chunks"] == len(ROWS)
    reused_embedder = FakeEmbedder()
    reused = Retriever(indexed, cache, embedder=reused_embedder)
    assert reused.build() == expected
    assert reused_embedder.embedded == []
    with sqlite3.connect(indexed) as con:
        con.execute("UPDATE chunks SET text = 'modified source' WHERE chunk_id = 'a#1'")
    stale = Retriever(indexed, cache, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="does not match source chunks"):
        stale.build()


def test_reranking_uses_bounded_candidate_ids_and_preserves_citations(
        indexed, tmp_path, monkeypatch):
    class FakeRequest:
        def __init__(self, query, passages):
            self.query, self.passages = query, passages

    class FakeReranker:
        def __init__(self):
            self.request = None

        def rerank(self, request):
            self.request = request
            return [{"id": p["id"], "score": float(p["id"] + 1)}
                    for p in reversed(request.passages)]

    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(RerankRequest=FakeRequest))
    reranker = FakeReranker()
    retriever = Retriever(indexed, tmp_path / "cache", embedder=FakeEmbedder(),
                          reranker=reranker)
    hits = retriever.search("apple", doc_id="a.pdf", method="reranked", top_k=1)
    assert reranker.request.query == "apple"
    assert all(set(p) == {"id", "text"} for p in reranker.request.passages)
    assert len(hits) == 1
    assert hits[0]["doc_id"] == "a.pdf"
    assert hits[0]["chunk_id"] in {"a#1", "a#2"}
    assert hits[0]["page"] in {1, 2}


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0]] * 3, dtype=np.float32),
                                 np.zeros((2, 2), dtype=np.float32),
                                 np.zeros((3, 2, 1), dtype=np.float32)])
def test_corrupt_dense_cache_is_rejected(indexed, tmp_path, bad):
    cache = tmp_path / "cache"
    Retriever(indexed, cache, embedder=FakeEmbedder()).build()
    np.save(cache / "vectors.npy", bad, allow_pickle=False)
    with pytest.raises(ValueError, match="Invalid dense vectors"):
        Retriever(indexed, cache, embedder=FakeEmbedder()).search("apple", method="dense")


def test_nonfinite_query_vector_is_rejected(indexed, tmp_path):
    class BadQuery(FakeEmbedder):
        def query_embed(self, query):
            yield np.asarray([np.inf, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Invalid query vector"):
        Retriever(indexed, tmp_path / "cache", embedder=BadQuery()).search("apple", method="dense")


def test_reranker_never_receives_more_than_thirty_candidates(indexed, tmp_path, monkeypatch):
    with sqlite3.connect(indexed) as con:
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                        [("a.pdf", i, f"a#{i}", f"apple item {i}") for i in range(3, 63)])

    class Embedder:
        def embed(self, texts, batch_size=64):
            return (np.asarray([1.0, 0.0], dtype=np.float32) for _ in texts)

        def query_embed(self, query):
            yield np.asarray([1.0, 0.0], dtype=np.float32)

    class FakeRequest:
        def __init__(self, query, passages):
            self.passages = passages

    class Reranker:
        def rerank(self, request):
            assert len(request.passages) <= 30
            assert all(set(p) == {"id", "text"} for p in request.passages)
            return [{"id": p["id"], "score": 1.0} for p in request.passages]

    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(RerankRequest=FakeRequest))
    retriever = Retriever(indexed, tmp_path / "cache", embedder=Embedder(), reranker=Reranker())
    hits = retriever.search("apple", doc_id="a.pdf", method="reranked", top_k=30)
    assert len(hits) == 30
    assert all(hit["doc_id"] == "a.pdf" and hit["chunk_id"].startswith("a#")
               and isinstance(hit["page"], int) for hit in hits)
