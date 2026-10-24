"""Controlled retrieval comparison with no model downloads or network calls."""

import json

import pytest

from evidence_lab import compare_retrieval
from evidence_lab.retrieval import METHODS


def samples():
    return [{"source_row": i, "doc_id": f"doc{i}.pdf", "question": f"question {i}",
             "answer": "GOLD_SECRET", "evidence_pages": [1],
             "evidence_sources": ["Text"]} for i in range(4)]


def test_evaluate_collapses_distinct_pages_and_never_passes_labels():
    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def search(self, query, doc_id, top_k, method):
            self.calls.append((query, doc_id, top_k, method))
            return [{"page": 3}, {"page": 3}, {"page": 1}, {"page": 2}]

    retriever = FakeRetriever()
    rows = [{"source_row": 7, "doc_id": "a.pdf", "question": "find value",
             "answer": "GOLD_SECRET", "evidence_pages": [1, 2]},
            {"source_row": 8, "doc_id": "a.pdf", "question": "missing",
             "answer": "Not answerable", "evidence_pages": []}]
    report = compare_retrieval.evaluate(retriever, rows, "hybrid")
    assert retriever.calls == [("find value", "a.pdf", 30, "hybrid"),
                               ("missing", "a.pdf", 30, "hybrid")]
    assert report["answerable"] == report["unanswerable"] == 1
    assert report["cases"][0]["retrieved_pages"] == [3, 1, 2]
    assert report["mean_recall"] == report["mean_hit"] == 1.0
    assert report["mrr"] == 0.5
    assert report["cases"][1]["answerable"] is False
    assert "GOLD_SECRET" not in json.dumps(report)


def test_selection_uses_recall_then_mrr_then_method_order():
    reports = {name: {"mean_recall": 0.5, "mrr": 0.5} for name in METHODS}
    assert compare_retrieval.select_method(reports) == "bm25"
    reports["hybrid"]["mrr"] = 0.6
    assert compare_retrieval.select_method(reports) == "hybrid"
    reports["reranked"]["mean_recall"] = 0.6
    assert compare_retrieval.select_method(reports) == "reranked"


def test_run_selects_on_development_then_evaluates_reserved_once(
        tmp_path, monkeypatch):
    source = tmp_path / "data/benchmark/eligible.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(samples()), encoding="utf-8")
    calls = []

    class FakeRetriever:
        def __init__(self, db, cache):
            self.db, self.cache = db, cache

        def build(self):
            return {"source_chunks_sha256": "fixture", "chunks": 4}

    def fake_evaluate(retriever, selected, method):
        rows = [s["source_row"] for s in selected]
        calls.append((method, rows))
        score = {"bm25": 0.4, "dense": 0.5, "hybrid": 0.7,
                 "reranked": 0.6}[method]
        return {"method": method, "mean_recall": score, "mrr": score,
                "cases": [{"source_row": row} for row in rows]}

    monkeypatch.setattr(compare_retrieval, "Retriever", FakeRetriever)
    monkeypatch.setattr(compare_retrieval, "evaluate", fake_evaluate)
    result = compare_retrieval.run(tmp_path, "retrieval-fixture")
    split = json.loads((tmp_path / "reports/experiment-split-v1.json").read_text())
    dev = {s["source_row"] for s in samples()
           if s["doc_id"] in split["development_docs"]}
    reserved = {s["source_row"] for s in samples()
                if s["doc_id"] in split["reserved_docs"]}
    assert result["selected"] == "hybrid"
    assert [method for method, _ in calls] == list(METHODS) + ["bm25", "hybrid"]
    assert all(set(rows) == dev for _, rows in calls[:4])
    assert all(set(rows) == reserved for _, rows in calls[4:])
    out = tmp_path / "reports/retrieval-fixture"
    selection = json.loads((out / "selection.json").read_text())
    assert selection["method"] == "hybrid"
    assert selection["reserved_consulted"] is False
    assert not (out / "reserved-dense.json").exists()
    assert not (out / "reserved-reranked.json").exists()
    assert json.loads((out / "config.json").read_text())["scope"] == "document-scoped"


def test_run_refuses_existing_report_and_path_name(tmp_path):
    with pytest.raises(ValueError, match="directory name"):
        compare_retrieval.run(tmp_path, "../escape")
    existing = tmp_path / "reports/retrieval-v1"
    existing.mkdir(parents=True)
    marker = existing / "marker.txt"
    marker.write_text("prior", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        compare_retrieval.run(tmp_path)
    assert marker.read_text(encoding="utf-8") == "prior"
