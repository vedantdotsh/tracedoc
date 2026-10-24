"""Pair the unchanged benchmark before and after OCR; never run answer generation."""
import json
from pathlib import Path
import sqlite3


def read_cases(folder):
    rows = [json.loads(line) for line in (folder / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    result = {r["source_row"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate source rows in evaluation")
    return result


def compare(root):
    root = Path(root)
    before_dir = root / "reports/corpus-audit-v1"
    after_dir = root / "reports/corpus-audit-ocr-v1"
    summaries = [json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                 for folder in (before_dir, after_dir)]
    for field in ("eligible_sha256", "source_revision", "pdf_files", "pdf_pages"):
        if summaries[0][field] != summaries[1][field]:
            raise ValueError(f"Evaluation provenance changed: {field}")
    for field in ("words", "overlap", "page_base", "tokenizer", "sources_sha256"):
        if summaries[0]["index_metadata"][field] != summaries[1]["index_metadata"][field]:
            raise ValueError(f"Index control changed: {field}")
    if summaries[0]["dense_manifest"]["model"] != summaries[1]["dense_manifest"]["model"]:
        raise ValueError("Embedding model changed")
    before, after = read_cases(before_dir), read_cases(after_dir)
    if set(before) != set(after):
        raise ValueError("Evaluation questions changed between runs")
    for key in before:
        for field in ("doc_id", "question", "gold_pages", "evidence_sources"):
            if before[key][field] != after[key][field]:
                raise ValueError(f"Evaluation field changed: {key} {field}")
    positives = [key for key, row in before.items() if row["gold_pages"]]
    with sqlite3.connect((root / "data/baseline.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as con:
        native = set(con.execute("SELECT DISTINCT doc_id,page FROM chunks"))
    groups = {"all_positive_questions": positives,
              "all_gold_pages_previously_indexed": [],
              "some_gold_pages_previously_indexed": [],
              "no_gold_pages_previously_indexed": []}
    for key in positives:
        row = before[key]
        count = sum((row["doc_id"], page) in native for page in row["gold_pages"])
        label = "all" if count == len(row["gold_pages"]) else "some" if count else "no"
        groups[f"{label}_gold_pages_previously_indexed"].append(key)
    results = {}
    for label, keys in groups.items():
        settings = {}
        for setting in before[positives[0]]["retrieval"]:
            old = [before[k]["retrieval"][setting] for k in keys]
            new = [after[k]["retrieval"][setting] for k in keys]
            settings[setting] = {
                "questions": len(keys),
                "before_recall": sum(r["recall"] for r in old) / len(keys) if keys else None,
                "after_recall": sum(r["recall"] for r in new) / len(keys) if keys else None,
                "before_hit": sum(r["hit"] for r in old) / len(keys) if keys else None,
                "after_hit": sum(r["hit"] for r in new) / len(keys) if keys else None,
                "new_hits": sum(a["hit"] > b["hit"] for b, a in zip(old, new)),
                "lost_hits": sum(a["hit"] < b["hit"] for b, a in zip(old, new)),
                "recall_improved": sum(a["recall"] > b["recall"] for b, a in zip(old, new)),
                "recall_worsened": sum(a["recall"] < b["recall"] for b, a in zip(old, new)),
            }
        results[label] = settings
    changes = []
    for key in positives:
        old, new = before[key]["retrieval"]["global_hybrid"], after[key]["retrieval"]["global_hybrid"]
        if old["recall"] != new["recall"]:
            changes.append({"source_row": key, "doc_id": before[key]["doc_id"],
                            "question": before[key]["question"], "gold_pages": before[key]["gold_pages"],
                            "before": old, "after": new})
    report = {"groups": results, "global_hybrid_changed_cases": changes,
              "control": "Same source PDFs, benchmark questions, chunk size/overlap, embedding model and retrieval budgets. Only OCR text on previously unindexed pages is added to baseline.",
              "limits": "Known benchmark, not untouched test data. These are page retrieval outcomes, not transcription accuracy, answer accuracy or verified numerical correctness."}
    (after_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["all_positive_questions"], indent=2))
    return report


if __name__ == "__main__":
    compare(Path(__file__).resolve().parents[1])
