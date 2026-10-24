"""The paid evaluation contract, exercised without credentials or network calls."""

import json

import pytest

from evidence_lab import evaluate_answers
from evidence_lab.experiments import freeze_split


def prepared(tmp_path, method="hybrid"):
    samples = []
    for doc in range(4):
        for index, pages in enumerate(([2], [], [1], [])):
            row = doc * 4 + index
            samples.append({"source_row": row, "doc_id": f"doc{doc}.pdf",
                            "question": f"Question {row}", "answer": f"GOLD_SECRET_{row}",
                            "evidence_pages": pages, "evidence_sources": ["Text"]})
    source = tmp_path / "data/benchmark/eligible.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(samples), encoding="utf-8")
    split = freeze_split(samples, tmp_path / "reports/experiment-split-v1.json")
    selection = tmp_path / "reports/retrieval-v1/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"method": method, "reserved_consulted": False}), encoding="utf-8")
    return samples, split


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, doc_id, top_k, method):
        self.calls.append((query, doc_id, top_k, method))
        return [{"doc_id": doc_id, "page": 2, "chunk_id": "evidence#1",
                 "text": "Public evidence only", "score": 1.0}]


def answer_result(*, failure=False):
    return {"execution_status": "provider_error" if failure else "ok",
            "status": "provider_error" if failure else "abstained",
            "answer": None if failure else "No answer in the evidence",
            "abstained": None if failure else True, "citations": [],
            "usage": {"cost": 0.0001}}


def test_frozen_selection_and_eight_predeclared_requests(tmp_path):
    samples, split = prepared(tmp_path)
    expected = evaluate_answers.select_cases(samples, split)
    assert len(expected) == 4
    assert [bool(x["evidence_pages"]) for x in expected] == [True, True, False, False]
    assert [x["source_row"] for x in expected[:2]] == sorted(x["source_row"] for x in expected[:2])
    retriever = FakeRetriever()
    observed = []

    def answer_fn(question, hits, **kwargs):
        checkpoint = json.loads((tmp_path / "reports/answer-evaluation-v1/report.json").read_text())
        assert checkpoint["status"] == "running"
        assert checkpoint["cases"][-1]["request_state"] == "started"
        assert "reference_answer" not in checkpoint["cases"][-1]
        assert "GOLD_SECRET" not in json.dumps([question, hits, kwargs])
        observed.append((question, kwargs))
        return answer_result()

    report = evaluate_answers.run(tmp_path, True, retriever=retriever,
                                   answer_fn=answer_fn, price_loader=lambda: {"model": "fixture"})
    assert report["status"] == "completed"
    assert len(report["cases"]) == report["max_requests"] == 8
    assert [c["source_row"] for c in report["cases"]] == [x["source_row"] for x in expected] * 2
    assert [c[3] for c in retriever.calls] == ["bm25"] * 4 + ["hybrid"] * 4
    assert all(c[2] == 5 for c in retriever.calls)
    assert all(kwargs["provider_max_price"] == evaluate_answers.PRICE_CEILINGS for _, kwargs in observed)
    assert report["aggregate_cost_upper_bound_usd"] <= 0.10
    assert report["summary"]["bm25"]["negative_abstentions"] == 2
    assert report["summary"]["hybrid"]["negative_abstentions"] == 2
    with pytest.raises(ValueError, match="already exists"):
        evaluate_answers.run(tmp_path, True, retriever=retriever,
                             answer_fn=answer_fn, price_loader=lambda: {})
    assert len(observed) == 8


def test_paid_opt_in_precedes_report_or_pricing(tmp_path):
    prepared(tmp_path)
    with pytest.raises(ValueError, match="allow-paid"):
        evaluate_answers.run(tmp_path, False, price_loader=lambda: pytest.fail("pricing queried"))
    assert not (tmp_path / "reports/answer-evaluation-v1").exists()


def test_preflight_budget_blocks_every_request(tmp_path, monkeypatch):
    prepared(tmp_path)
    monkeypatch.setattr(evaluate_answers, "request_bound", lambda *args: {
        "input_token_upper_bound": 1, "output_token_limit": 384,
        "cost_upper_bound_usd": 0.02})
    with pytest.raises(ValueError, match="budget"):
        evaluate_answers.run(tmp_path, True, retriever=FakeRetriever(),
                             answer_fn=lambda *a, **kw: pytest.fail("provider called"),
                             price_loader=lambda: {"model": "fixture"})
    report = json.loads((tmp_path / "reports/answer-evaluation-v1/report.json").read_text())
    assert report["status"] == "failed" and report["cases"] == []


def test_provider_error_stops_without_retry_or_false_abstention(tmp_path):
    prepared(tmp_path)
    calls = []

    def fail_once(*args, **kwargs):
        calls.append((args, kwargs))
        return answer_result(failure=True)

    report = evaluate_answers.run(tmp_path, True, retriever=FakeRetriever(),
                                   answer_fn=fail_once, price_loader=lambda: {"model": "fixture"})
    assert report["status"] == "stopped_provider_error"
    assert len(calls) == len(report["cases"]) == 1
    scores = report["cases"][0]["scores"]
    assert scores["execution_success"] is False
    assert scores["abstained"] is None
    with pytest.raises(ValueError, match="already exists"):
        evaluate_answers.run(tmp_path, True)


@pytest.mark.parametrize("prompt,completion,request_fee", [
    ("0.0000002", "0.0000002", "0.01"),
    ("0.0000002", "nan", "0"),
    ("1000", "0.0000002", "0"),
])
def test_current_pricing_fails_closed_on_fees_or_invalid_prices(monkeypatch, prompt, completion, request_fee):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"data": [{"id": evaluate_answers.answering.DEFAULT_MODEL,
                "pricing": {"prompt": prompt, "completion": completion, "request": request_fee}}]}).encode()

    monkeypatch.setattr(evaluate_answers.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(ValueError):
        evaluate_answers.current_prices()
