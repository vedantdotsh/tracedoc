"""Paired OCR comparison uses identical questions and counts gains and losses."""

import json
import sqlite3

import pytest

from scripts.compare_ocr_accuracy import compare


def _fixture(tmp_path):
    before = tmp_path / "reports/corpus-audit-v1"
    after = tmp_path / "reports/corpus-audit-ocr-v1"
    before.mkdir(parents=True)
    after.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    with sqlite3.connect(data / "baseline.sqlite") as con:
        con.execute("CREATE TABLE chunks(doc_id TEXT, page INTEGER)")
        con.executemany("INSERT INTO chunks VALUES (?, ?)",
                        [("a.pdf", 1), ("b.pdf", 1)])

    def case(source_row, doc, gold, recall, hit):
        return {"source_row": source_row, "doc_id": doc,
                "question": f"question {source_row}", "gold_pages": gold,
                "evidence_sources": ["Pure-text (Plain-text)"],
                "retrieval": {"global_hybrid": {"recall": recall, "hit": hit,
                                                 "mrr": hit, "ranked_pages": []}} if gold else {}}

    old = [case(1, "a.pdf", [1], 1.0, 1.0),
           case(2, "b.pdf", [1, 2], 0.5, 1.0),
           case(3, "c.pdf", [3], 0.0, 0.0),
           case(4, "c.pdf", [], None, None)]
    new = [case(1, "a.pdf", [1], 0.0, 0.0),
           case(2, "b.pdf", [1, 2], 1.0, 1.0),
           case(3, "c.pdf", [3], 1.0, 1.0),
           case(4, "c.pdf", [], None, None)]

    def write(folder, rows):
        (folder / "cases.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    write(before, old)
    write(after, new)
    summary = {"eligible_sha256": "eligible", "source_revision": "revision",
               "pdf_files": 3, "pdf_pages": 9,
               "index_metadata": {"words": "180", "overlap": "30", "page_base": "1",
                                  "tokenizer": "fts5 unicode61", "sources_sha256": "sources"},
               "dense_manifest": {"model": "local-model"}}
    for folder in (before, after):
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return before, after, write, new


def test_paired_comparison_counts_new_lost_and_partial_recall(tmp_path):
    _, after, _, _ = _fixture(tmp_path)
    report = compare(tmp_path)
    all_cases = report["groups"]["all_positive_questions"]["global_hybrid"]
    assert all_cases["questions"] == 3
    assert all_cases["new_hits"] == 1 and all_cases["lost_hits"] == 1
    assert all_cases["recall_improved"] == 2 and all_cases["recall_worsened"] == 1
    assert all_cases["before_hit"] == all_cases["after_hit"] == pytest.approx(2 / 3)
    groups = report["groups"]
    assert groups["all_gold_pages_previously_indexed"]["global_hybrid"]["lost_hits"] == 1
    assert groups["some_gold_pages_previously_indexed"]["global_hybrid"]["recall_improved"] == 1
    assert groups["no_gold_pages_previously_indexed"]["global_hybrid"]["new_hits"] == 1
    assert len(report["global_hybrid_changed_cases"]) == 3
    assert json.loads((after / "comparison.json").read_text("utf-8")) == report


def test_changed_question_refused_before_writing_comparison(tmp_path):
    _, after, write, new = _fixture(tmp_path)
    new[0]["question"] = "changed query"
    write(after, new)
    with pytest.raises(ValueError, match="Evaluation field changed"):
        compare(tmp_path)
    assert not (after / "comparison.json").exists()


@pytest.mark.parametrize("changed,expected", [
    ("model", "Embedding model changed"),
    ("words", "Index control changed: words"),
])
def test_changed_model_or_chunking_refused_before_writing_comparison(tmp_path, changed, expected):
    _, after, _, _ = _fixture(tmp_path)
    summary_file = after / "summary.json"
    summary = json.loads(summary_file.read_text("utf-8"))
    if changed == "model":
        summary["dense_manifest"]["model"] = "different-model"
    else:
        summary["index_metadata"]["words"] = "200"
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        compare(tmp_path)
    assert not (after / "comparison.json").exists()
