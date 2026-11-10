"""Recompute the published retrieval figures from the per-question results.

Run from the repository root with ``python scripts/verify_reported_metrics.py``.
Uses only the files under ``reports/``; no documents, indexes or network access
are needed. Exits with status 1 if any recomputed value differs from the
published summaries.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

RUNS = (
    ("corpus-audit-v1", "Native PDF text, BGE-small"),
    ("corpus-audit-ocr-v1", "With OCR text, BGE-small"),
    ("corpus-audit-openai-v1", "With OCR text, text-embedding-3-small"),
)
KEYS = tuple(f"{scope}_{method}" for scope in ("document", "global") for method in ("bm25", "dense", "hybrid"))
FIELDS = (("recall", "mean_page_recall_at_5"), ("hit", "hit_at_5"), ("mrr", "mrr_at_5"))


def score(gold, ranked):
    rank = next((i for i, page in enumerate(ranked, 1) if page in gold), None)
    return {
        "recall": len(set(gold) & set(ranked)) / len(set(gold)),
        "hit": float(rank is not None),
        "mrr": 1 / rank if rank else 0.0,
    }


def close(a, b):
    return abs(a - b) < 1e-9


def check_run(folder):
    """Return (figures, problems) for one report folder."""
    problems = []
    summary = json.loads((folder / "summary.json").read_text("utf-8"))
    cases = [json.loads(line) for line in (folder / "cases.jsonl").read_text("utf-8").splitlines()]
    with (folder / "documents.csv").open(encoding="utf-8", newline="") as stream:
        documents = list(csv.DictReader(stream))
    positives = [c for c in cases if c["gold_pages"]]

    counts = {
        "pdf_files": len(documents),
        "pdf_pages": sum(int(d["pdf_pages"]) for d in documents),
        "indexed_pages_with_text": sum(int(d["indexed_pages_with_text"]) for d in documents),
        "structurally_eligible_questions": len(cases),
        "positive_questions": len(positives),
        "no_gold_questions": len(cases) - len(positives),
    }
    for name, value in counts.items():
        if summary[name] != value:
            problems.append(f"{name}: summary {summary[name]} != recomputed {value}")
    if {c["doc_id"] for c in cases} != {d["doc_id"] for d in documents}:
        problems.append("documents in cases.jsonl and documents.csv differ")
    if len({c["source_row"] for c in cases}) != len(cases):
        problems.append("duplicate source rows")

    metrics = {}
    for key in KEYS:
        totals = dict.fromkeys(("recall", "hit", "mrr"), 0.0)
        for case in positives:
            saved = case["retrieval"][key]
            ranked = [tuple(page) for page in saved["ranked_pages"]]
            if len(ranked) > 5 or len(set(ranked)) != len(ranked):
                problems.append(f"row {case['source_row']} {key}: not five distinct pages")
            fresh = score([(case["doc_id"], n) for n in case["gold_pages"]], ranked)
            for field in totals:
                if not close(fresh[field], saved[field]):
                    problems.append(f"row {case['source_row']} {key} {field}: saved {saved[field]} != recomputed {fresh[field]}")
                totals[field] += fresh[field]
        metrics[key] = {"hits": int(totals["hit"])}
        for field, name in FIELDS:
            metrics[key][name] = totals[field] / len(positives)
            if not close(metrics[key][name], summary["retrieval"][key][name]):
                problems.append(f"{key} {name}: summary {summary['retrieval'][key][name]} != recomputed {metrics[key][name]}")
    return {**counts, "retrieval": metrics}, problems


def paired_hits(before_dir, after_dir, key="global_hybrid"):
    def hits(folder):
        rows = (json.loads(line) for line in (folder / "cases.jsonl").read_text("utf-8").splitlines())
        return {r["source_row"]: (r["doc_id"], r["gold_pages"], r["retrieval"][key]["hit"]) for r in rows if r["gold_pages"]}
    before, after = hits(before_dir), hits(after_dir)
    if {k: v[:2] for k, v in before.items()} != {k: v[:2] for k, v in after.items()}:
        raise ValueError(f"questions differ between {before_dir.name} and {after_dir.name}")
    return {"new_hits": sum(after[k][2] > before[k][2] for k in before),
            "lost_hits": sum(after[k][2] < before[k][2] for k in before)}


def main(reports):
    results, problems = {}, []
    for name, _ in RUNS:
        results[name], found = check_run(reports / name)
        problems += [f"{name}: {p}" for p in found]
    first = results[RUNS[0][0]]
    for name, _ in RUNS[1:]:
        for field in ("pdf_files", "pdf_pages", "structurally_eligible_questions", "positive_questions"):
            if results[name][field] != first[field]:
                problems.append(f"{name}: {field} differs from {RUNS[0][0]}")

    print(f"PDFs: {first['pdf_files']}  pages: {first['pdf_pages']}  "
          f"scored questions: {first['positive_questions']} of {first['structurally_eligible_questions']} eligible")
    print()
    print(f"{'Index and embedding':<40}{'pages with text':>16}{'whole-library hybrid hit@5':>30}{'recall@5':>10}{'single-document hybrid hit@5':>31}")
    for name, label in RUNS:
        r = results[name]
        g, d = r["retrieval"]["global_hybrid"], r["retrieval"]["document_hybrid"]
        print(f"{label:<40}{r['indexed_pages_with_text']:>16}"
              f"{g['hits']:>19}/{r['positive_questions']} = {g['hit_at_5']:.1%}{g['mean_page_recall_at_5']:>10.1%}"
              f"{d['hits']:>20}/{r['positive_questions']} = {d['hit_at_5']:.1%}")
    print()
    for (before, _), (after, _) in zip(RUNS, RUNS[1:]):
        change = paired_hits(reports / before, reports / after)
        print(f"{before} -> {after} (whole-library hybrid): {change['new_hits']} new hits, {change['lost_hits']} lost hits")
    print()
    if problems:
        print(f"{len(problems)} mismatches:")
        for problem in problems[:50]:
            print("  " + problem)
        return 1
    print("All recomputed values match the published summaries.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    sys.exit(main(parser.parse_args().reports))
