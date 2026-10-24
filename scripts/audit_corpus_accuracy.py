"""Offline, positive-only page retrieval audit for the complete annotated corpus.

Run from the repository root with ``python scripts/audit_corpus_accuracy.py``.
No answer generation or network request is made.
"""
import argparse
import ast
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evidence_lab import baseline
from evidence_lab.retrieval import Retriever, fuse


METHODS = ("bm25", "dense", "hybrid")
SCOPES = ("document", "global")


def pages(hits):
    return list(dict.fromkeys((h["doc_id"], h["page"]) for h in hits))[:5]


def metrics(cases, key):
    positives = [c["retrieval"][key] for c in cases if c["gold_pages"]]
    return {
        "positive_questions": len(positives),
        "mean_page_recall_at_5": baseline._mean([x["recall"] for x in positives]),
        "hit_at_5": baseline._mean([x["hit"] for x in positives]),
        "mrr_at_5": baseline._mean([x["mrr"] for x in positives]),
    }


def score(gold, ranked):
    rank = next((i for i, page in enumerate(ranked, 1) if page in gold), None)
    return {
        "recall": len(set(gold) & set(ranked)) / len(set(gold)),
        "hit": float(rank is not None),
        "mrr": 1 / rank if rank else 0.0,
    }


def run(root: Path, output: Path, index_variant="baseline", *, retriever=None, dense_manifest=None):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    eligible_bytes = (root / "data/benchmark/eligible.json").read_bytes()
    samples = json.loads(eligible_bytes)
    raw = json.loads((root / "data/benchmark/samples.json").read_text("utf-8"))
    quarantine = json.loads((root / "reports/annotation-audit.json").read_text("utf-8"))["quarantine"]
    if retriever is not None:
        if dense_manifest is None or index_variant != "ocr":
            raise ValueError("injected retriever requires an OCR dense manifest")
        db_path = root / "data/ocr-index/ocr.sqlite"
        if retriever.db.resolve() != db_path.resolve() or retriever.vectors is None:
            raise ValueError("injected retriever does not match the OCR index or has no vectors")
    elif index_variant == "baseline":
        db_path, cache_path, embedder = root / "data/baseline.sqlite", root / "data/dense-v1", None
    elif index_variant in ("augmented", "ocr"):
        original = Retriever(root / "data/baseline.sqlite", root / "data/dense-v1")
        original.build()
        embedder = original._embedder()  # audited local snapshot; augmented cache has vectors only
        if index_variant == "augmented":
            db_path, cache_path = root / "data/extension-index/augmented.sqlite", root / "data/extension-index/dense"
        else:
            db_path, cache_path = root / "data/ocr-index/ocr.sqlite", root / "data/ocr-index/dense"
    else:
        raise ValueError(f"unknown index variant: {index_variant}")
    if retriever is None:
        retriever = Retriever(db_path, cache_path, embedder=embedder)
        dense_manifest = retriever.build()  # checks vectors match the current index
    with sqlite3.connect(db_path) as con:
        docs = dict(con.execute("SELECT doc_id,pages FROM docs"))
        chunk_pages = set(con.execute("SELECT DISTINCT doc_id,page FROM chunks"))
        index_meta = dict(con.execute("SELECT key,value FROM meta"))
    pdfs = {p.name for p in (root / "data/documents").glob("*.pdf")}
    if len(pdfs) != 135 or set(docs) != pdfs:
        raise ValueError("PDF files and indexed documents differ; rebuild or investigate before auditing")
    if len(samples) + len(quarantine) != len(raw):
        raise ValueError("eligibility and raw annotations do not reconcile")
    if index_meta.get("page_base") != "1":
        raise ValueError("unexpected index page coordinates")

    rows_by_doc = defaultdict(list)
    for i, row in enumerate(retriever.rows):
        rows_by_doc[row["doc_id"]].append(i)
    by_doc = defaultdict(list)
    cases = []
    source_counts = Counter()
    visual_questions = 0
    table_questions = 0
    no_text_gold_questions = 0
    no_text_gold_pages = 0

    for sample in samples:
        doc_id = sample["doc_id"]
        gold_nums = sorted(set(ast.literal_eval(sample["evidence_pages"])))
        gold = [(doc_id, n) for n in gold_nums]
        source = ast.literal_eval(sample["evidence_sources"])
        case = {"source_row": sample["source_row"], "doc_id": doc_id,
                "question": sample["question"], "gold_pages": gold_nums,
                "evidence_sources": source, "retrieval": {}}
        by_doc[doc_id].append(case)
        cases.append(case)
        if not gold:
            continue  # negative questions have no positive retrieval target
        source_counts.update(set(source))
        visual_questions += bool(set(source) & {"Figure", "Chart"})
        table_questions += "Table" in source
        absent = [n for n in gold_nums if (doc_id, n) not in chunk_pages]
        no_text_gold_questions += bool(absent)
        no_text_gold_pages += len(absent)

        # Match Retriever.search: 30 candidate chunks, then five unique pages.
        # Embed each query once for the two retrieval scopes.
        q = np.asarray(next(iter(retriever._embedder().query_embed(sample["question"]))), dtype=np.float32)
        if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise ValueError(f"invalid query vector for source row {sample['source_row']}")
        q /= max(float(np.linalg.norm(q)), 1e-12)
        if q.shape != (retriever.vectors.shape[1],):
            raise ValueError("query embedding dimension mismatch")
        for scope in SCOPES:
            scoped_doc = doc_id if scope == "document" else None
            bm = baseline.search(retriever.db, sample["question"], scoped_doc, 30)
            indices = np.asarray(rows_by_doc[doc_id] if scoped_doc else range(len(retriever.rows)), dtype=int)
            similarities = retriever.vectors[indices] @ q
            order = np.argsort(-similarities, kind="stable")[:30]
            dense = [{**retriever.rows[int(indices[i])], "score": float(similarities[i])} for i in order]
            rankings = {"bm25": bm, "dense": dense, "hybrid": fuse([bm, dense], 30)}
            for method, hits in rankings.items():
                ranked = pages(hits)
                case["retrieval"][f"{scope}_{method}"] = {
                    **score(gold, ranked),
                    "ranked_pages": [{"doc_id": d, "page": p} for d, p in ranked],
                }

    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    per_doc = []
    for doc_id in sorted(docs):
        doc_cases = by_doc[doc_id]
        positives = [c for c in doc_cases if c["gold_pages"]]
        result = {"doc_id": doc_id, "pdf_pages": docs[doc_id],
                  "indexed_pages_with_text": sum((doc_id, p) in chunk_pages for p in range(1, docs[doc_id] + 1)),
                  "annotated_questions": len(doc_cases), "positive_questions": len(positives),
                  "no_gold_questions": len(doc_cases) - len(positives),
                  "positive_questions_with_no_text_gold_page": sum(
                      any((doc_id, p) not in chunk_pages for p in c["gold_pages"]) for c in positives)}
        for scope in SCOPES:
            for method in METHODS:
                m = metrics(positives, f"{scope}_{method}")
                for name in ("mean_page_recall_at_5", "hit_at_5", "mrr_at_5"):
                    result[f"{scope}_{method}_{name}"] = m[name]
        per_doc.append(result)
    with (output / "documents.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_doc[0].keys())
        writer.writeheader()
        writer.writerows(per_doc)
    positives = [c for c in cases if c["gold_pages"]]
    coverage = Counter()
    for case in positives:
        available = sum((case["doc_id"], p) in chunk_pages for p in case["gold_pages"])
        coverage["all" if available == len(case["gold_pages"]) else "none" if available == 0 else "some"] += 1
    modalities = {}
    for label in sorted(source_counts):
        subset = [c for c in positives if label in c["evidence_sources"]]
        modalities[label] = {"positive_questions": len(subset),
                             "document": {method: metrics(subset, f"document_{method}") for method in METHODS},
                             "global": {method: metrics(subset, f"global_{method}") for method in METHODS}}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index_variant": index_variant,
        "evaluation": "offline retrieval of annotated evidence pages; no generated answers graded",
        "source_revision": json.loads((root / "data/manifest.json").read_text("utf-8"))["revision"],
        "eligible_sha256": hashlib.sha256(eligible_bytes).hexdigest(),
        "index_metadata": index_meta, "dense_manifest": dense_manifest,
        "pdf_files": len(pdfs), "indexed_documents": len(docs),
        "pdf_pages": sum(docs.values()), "indexed_pages_with_text": len(chunk_pages),
        "pages_without_indexed_text": sum(docs.values()) - len(chunk_pages),
        "raw_questions": len(raw), "structurally_eligible_questions": len(samples),
        "quarantined_questions": len(quarantine),
        "quarantine_reasons": dict(Counter(reason for q in quarantine for reason in q["reasons"])),
        "positive_questions": sum(bool(c["gold_pages"]) for c in cases),
        "no_gold_questions": sum(not c["gold_pages"] for c in cases),
        "positive_evidence_pages": sum(len(c["gold_pages"]) for c in cases),
        "positive_questions_with_no_text_gold_page": no_text_gold_questions,
        "positive_evidence_pages_without_indexed_text": no_text_gold_pages,
        "positive_gold_text_coverage": {k: coverage[k] for k in ("all", "some", "none")},
        "visual_figure_or_chart_positive_questions": visual_questions,
        "table_positive_questions": table_questions,
        "evidence_source_positive_question_counts": dict(sorted(source_counts.items())),
        "overlapping_modality_breakdown": modalities,
        "retrieval": {f"{scope}_{method}": metrics(cases, f"{scope}_{method}")
                      for scope in SCOPES for method in METHODS},
        "definitions": {
            "positive_denominator": "Structurally eligible questions with one or more physical gold evidence pages (837 expected).",
            "scope": "document searches only the provided doc_id; global searches all 135 PDFs without the doc_id filter. Global hits count only when both doc_id and page match gold.",
            "page_recall": "Fraction of each question's gold pages in top five distinct retrieved pages, then macro averaged across positive questions.",
            "hit": "Fraction of positive questions with at least one gold page in top five.",
            "mrr": "Reciprocal rank of first gold page in top five, averaged across positive questions.",
            "no_gold": "Unanswerable questions have no positive retrieval target and are excluded from recall, hit and MRR; abstention needs a separate evaluation.",
            "unsupported_modality": "Figure/Chart and Table counts describe annotation sources, not verified extractability or answer accuracy. The text index has no image understanding or table reasoning.",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index-variant", choices=("baseline", "augmented", "ocr"), default="baseline")
    args = parser.parse_args()
    names = {"baseline": "corpus-audit-v1", "augmented": "corpus-audit-augmented-v1", "ocr": "corpus-audit-ocr-v1"}
    target = args.output or args.root / "reports" / names[args.index_variant]
    result = run(args.root, target, args.index_variant)
    print(json.dumps({"output": str(target), "retrieval": result["retrieval"]}, indent=2))
