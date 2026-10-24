"""Runner checks use disposable fixtures and never contact a model provider."""

import json
import sqlite3

import pytest

from evidence_lab import answering, compare_chunking, generate_pilot
from evidence_lab.chunking import STRATEGIES


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_db(path, fingerprint="same-source"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO meta VALUES ('sources_sha256', ?)", (fingerprint,))
        con.execute("CREATE TABLE chunks(text TEXT)")
        con.execute("INSERT INTO chunks VALUES ('source words')")


def samples():
    return [{"source_row": i, "doc_id": f"doc{i}.pdf", "question": f"question {i}",
             "answer": "gold", "evidence_pages": [1], "evidence_sources": ["Table"]}
            for i in range(4)]


def comparison_fixture(tmp_path):
    save(tmp_path / "data/benchmark/eligible.json", samples())
    make_db(tmp_path / "data/baseline.sqlite")
    return tmp_path


def test_comparison_uses_development_only_same_sources_and_counts_pairs(tmp_path, monkeypatch):
    root = comparison_fixture(tmp_path)
    evaluated = []

    def fake_build(pdf_dir, db, strategy, **kwargs):
        assert pdf_dir == root / "data/documents"
        assert strategy in STRATEGIES[1:]
        assert kwargs == ({"workers": 3} if strategy == "table" else {})
        make_db(db)
        return {"docs": 4, "chunks": 1}

    def fake_evaluate(db, selected):
        method = "fixed" if db.name == "baseline.sqlite" else db.stem
        evaluated.append((method, [s["source_row"] for s in selected]))
        values = {"fixed": [0.0, 1.0], "paragraph": [1.0, 0.0],
                  "heading": [1.0, 1.0], "table": [0.0, 0.0]}[method]
        cases = [{"source_row": s["source_row"], "answerable": True,
                  "recall": value} for s, value in zip(selected, values)]
        return {"cases": cases, "answerable": len(cases), "unanswerable": 0,
                "mean_recall": sum(values) / len(values), "mean_hit": 0.5,
                "mrr": 0.5, "latency_ms": {"p50": 1, "p95": 2}}

    monkeypatch.setattr(compare_chunking.baseline, "build_index", fake_build)
    monkeypatch.setattr(compare_chunking.baseline, "evaluate", fake_evaluate)
    result = compare_chunking.run(root)
    split = json.loads((root / "reports/experiment-split-v1.json").read_text())
    development = {s["source_row"] for s in samples()
                   if s["doc_id"] in split["development_docs"]}
    assert len(development) == 2
    assert [method for method, _ in evaluated] == list(STRATEGIES)
    assert all(set(rows) == development for _, rows in evaluated)
    assert all(len(rows) == 2 for _, rows in evaluated)
    assert result["partition"] == "development"
    assert result["methods"]["paragraph"]["paired_recall_vs_fixed"] == {
        "wins": 1, "losses": 1, "ties": 0}
    assert result["methods"]["heading"]["paired_recall_vs_fixed"] == {
        "wins": 1, "losses": 0, "ties": 1}
    assert result["methods"]["table"]["paired_recall_vs_fixed"] == {
        "wins": 0, "losses": 1, "ties": 1}


def test_comparison_refuses_existing_report_without_overwrite(tmp_path, monkeypatch):
    root = comparison_fixture(tmp_path)
    target = root / "reports/chunking-v1/fixed.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"prior result")
    called = []
    monkeypatch.setattr(compare_chunking.baseline, "evaluate", lambda *a: called.append(a))
    with pytest.raises(ValueError, match="already exists"):
        compare_chunking.run(root)
    assert target.read_bytes() == b"prior result"
    assert called == []


def test_comparison_rejects_mismatched_source_fingerprint(tmp_path, monkeypatch):
    root = comparison_fixture(tmp_path)

    def mismatched_build(pdf_dir, db, strategy, **kwargs):
        make_db(db, fingerprint="different-source")
        return {"docs": 4, "chunks": 1}

    monkeypatch.setattr(compare_chunking.baseline, "build_index", mismatched_build)
    monkeypatch.setattr(compare_chunking.baseline, "evaluate", lambda *a: {
        "cases": [], "answerable": 0, "unanswerable": 0, "mean_recall": 0.0,
        "mean_hit": 0.0, "mrr": 0.0, "latency_ms": {}})
    with pytest.raises(ValueError, match="Corpus differs"):
        compare_chunking.run(root)
    assert not (root / "reports/chunking-v1/paragraph.json").exists()


@pytest.mark.parametrize("run_name", ["../escape", "chunking-../escape",
                                            "Chunking-v2", "chunking_v2", "other-v2"])
def test_comparison_rejects_malformed_run_name(tmp_path, run_name):
    with pytest.raises(ValueError, match="run name"):
        compare_chunking.run(tmp_path, run_name=run_name)
    assert not (tmp_path / "reports").exists()


def resumable_report(rows, strategy="fixed"):
    return {"config": {"index": {"sources_sha256": "same-source",
                                 "strategy": strategy, "words": "180", "overlap": "30"},
                       "top_k": 5, "scope": "document-scoped"},
            "cases": [{"source_row": row, "answerable": True, "recall": 1.0}
                      for row in rows],
            "answerable": len(rows), "unanswerable": 0,
            "mean_recall": 1.0, "mean_hit": 1.0, "mrr": 1.0,
            "latency_ms": {"p50": 1, "p95": 1}, "build": {"chunks": 1}}


@pytest.mark.parametrize("change", ["source", "split", "strategy", "words",
                                             "overlap", "top_k", "scope"])
def test_resume_refuses_incompatible_completed_report(tmp_path, change):
    root = comparison_fixture(tmp_path)
    split = compare_chunking.freeze_split(samples(), root / "reports/experiment-split-v1.json")
    dev = compare_chunking.select_samples(samples(), split)
    source_rows = [s["source_row"] for s in dev]
    report = resumable_report(source_rows)
    if change == "source":
        report["config"]["index"]["sources_sha256"] = "different-source"
    elif change == "split":
        report["cases"][0]["source_row"] = 999
    elif change in ("strategy", "words", "overlap"):
        report["config"]["index"][change] = "wrong"
    elif change == "top_k":
        report["config"]["top_k"] = 3
    else:
        report["config"]["scope"] = "collection-wide"
    target = root / "reports/chunking-v1/fixed.json"
    save(target, report)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        compare_chunking.run(root, resume=True)
    assert target.read_bytes() == before
    assert not (target.parent / "paragraph.json").exists()


def test_resume_reuses_compatible_reports_without_rebuilding(tmp_path, monkeypatch):
    root = comparison_fixture(tmp_path)
    split = compare_chunking.freeze_split(samples(), root / "reports/experiment-split-v1.json")
    rows = [s["source_row"] for s in compare_chunking.select_samples(samples(), split)]
    for strategy in STRATEGIES:
        save(root / f"reports/chunking-v1/{strategy}.json",
             resumable_report(rows, strategy))
    monkeypatch.setattr(compare_chunking.baseline, "build_index",
                        lambda *a, **k: pytest.fail("compatible resume rebuilt an index"))
    monkeypatch.setattr(compare_chunking.baseline, "evaluate",
                        lambda *a, **k: pytest.fail("compatible resume reevaluated rows"))
    result = compare_chunking.run(root, resume=True)
    assert set(result["methods"]) == set(STRATEGIES)
    assert all(result["methods"][s]["paired_recall_vs_fixed"] == {
        "wins": 0, "losses": 0, "ties": len(rows)} for s in STRATEGIES[1:])


def pilot_fixture(tmp_path):
    root = tmp_path
    eligible = samples()
    save(root / "data/benchmark/eligible.json", eligible)
    save(root / "reports/chunking-v1/fixed.json", {
        "cases": [{"source_row": s["source_row"], "answerable": True} for s in eligible]})
    return root


def test_pilot_requires_opt_in_without_loading_key_or_calling_model(tmp_path, monkeypatch):
    root = pilot_fixture(tmp_path)
    called = []
    monkeypatch.setattr(generate_pilot, "load_dotenv", lambda *a, **k: called.append("env"))
    monkeypatch.setattr(answering, "answer", lambda *a, **k: called.append("model"))
    with pytest.raises(ValueError, match="allow-paid"):
        generate_pilot.run(root)
    assert called == []
    assert not (root / "reports/chunking-v1/generation-pilot.json").exists()


def test_pilot_makes_four_same_question_calls_without_labels(tmp_path, monkeypatch):
    root = pilot_fixture(tmp_path)
    monkeypatch.setattr(generate_pilot, "load_dotenv", lambda *a, **k: None)
    calls = []

    def fake_answer(*args, **kwargs):
        calls.append((args, kwargs))
        target = root / "reports/chunking-v1/generation-pilot.json"
        assert target.exists()  # written before any possible billed request
        return {"status": "answered", "usage": {"cost": 0.001}}

    monkeypatch.setattr(answering, "answer", fake_answer)
    pilot = generate_pilot.run(root, allow_paid=True)
    assert len(calls) == pilot["max_requests"] == 4
    assert [args[0].name for args, _ in calls] == [
        "baseline.sqlite", "paragraph.sqlite", "heading.sqlite", "table.sqlite"]
    assert len({args[1] for args, _ in calls}) == 1
    assert len({args[2] for args, _ in calls}) == 1
    assert all(kwargs == {"allow_paid": True} for _, kwargs in calls)
    assert pilot["model"] == answering.DEFAULT_MODEL
    assert set(pilot["results"]) == set(STRATEGIES)
    assert json.loads((root / "reports/chunking-v1/generation-pilot.json").read_text()) == pilot


def test_pilot_refuses_repeat_and_stops_on_provider_error(tmp_path, monkeypatch):
    root = pilot_fixture(tmp_path)
    monkeypatch.setattr(generate_pilot, "load_dotenv", lambda *a, **k: None)
    calls = []

    def fake_answer(*args, **kwargs):
        calls.append(args[0].name)
        return {"status": "provider_error" if len(calls) == 2 else "answered",
                "usage": {}}

    monkeypatch.setattr(answering, "answer", fake_answer)
    first = generate_pilot.run(root, allow_paid=True)
    assert calls == ["baseline.sqlite", "paragraph.sqlite"]
    assert set(first["results"]) == {"fixed", "paragraph"}
    saved = (root / "reports/chunking-v1/generation-pilot.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        generate_pilot.run(root, allow_paid=True)
    assert calls == ["baseline.sqlite", "paragraph.sqlite"]
    assert (root / "reports/chunking-v1/generation-pilot.json").read_bytes() == saved
