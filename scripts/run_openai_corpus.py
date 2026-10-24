"""Re-embed the unchanged 135-PDF OCR index and compare page retrieval with BGE.

Run --preflight before --execute. Execute uses the existing OpenRouter cache and
spend journal; interrupted runs resume from durable per-text vectors.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_lab.corpus import write_json_atomic
from evidence_lab.openrouter_embeddings import (
    DIMENSIONS, MAX_INPUT_TOKENS, MODEL, USD_PER_TOKEN, Embedder, _count, _key,
)
from evidence_lab.retrieval import Retriever
from scripts.audit_corpus_accuracy import METHODS, SCOPES, run

CACHE = Path("data/openai-corpus-v1")
REPORT = Path("reports/corpus-audit-openai-v1")
OLD_REPORT = Path("reports/corpus-audit-ocr-v1")
DB = Path("data/ocr-index/ocr.sqlite")
BUDGET_USD = 0.10


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(texts):
    return list(dict.fromkeys(texts))


def checked_source(root, retriever):
    manifest = json.loads((root / "data/manifest.json").read_text("utf-8"))
    docs = manifest["documents"]
    pdf_dir = root / "data/documents"
    if len(docs) != 135 or {p.name for p in pdf_dir.glob("*.pdf")} != {d["doc_id"] for d in docs}:
        raise ValueError("PDF inventory differs from the 135 source documents")
    for doc in docs:
        if sha256(pdf_dir / doc["doc_id"]) != doc["sha256"]:
            raise ValueError(f"PDF hash differs from source manifest: {doc['doc_id']}")
    old = json.loads((root / OLD_REPORT / "summary.json").read_text("utf-8"))
    eligible_sha = sha256(root / "data/benchmark/eligible.json")
    if old["source_revision"] != manifest["revision"] or old["eligible_sha256"] != eligible_sha:
        raise ValueError("source or eligible annotations differ from BGE OCR audit")
    if old["dense_manifest"]["source_chunks_sha256"] != retriever.digest or len(retriever.rows) != 14756:
        raise ValueError("OCR chunks differ from BGE OCR audit")
    samples = json.loads((root / "data/benchmark/eligible.json").read_text("utf-8"))
    positive = [s for s in samples if ast.literal_eval(s["evidence_pages"])]
    if len(positive) != 837:
        raise ValueError("positive question count changed")
    return eligible_sha, positive


def preflight(root, cache=CACHE):
    root = Path(root).resolve()
    cache = root / cache
    retriever = Retriever(root / DB, cache)
    eligible_sha, positive = checked_source(root, retriever)
    chunks = [row["text"] for row in retriever.rows]
    queries = [sample["question"] for sample in positive]
    texts = unique(chunks + queries)
    counts = [_count(t) for t in texts]
    if any(not t.strip() for t in texts) or max(counts) > MAX_INPUT_TOKENS:
        raise ValueError("empty or over-limit embedding input; controlled chunks cannot be changed")
    missing = [t for t in texts if not (cache / "vectors" / f"{_key(t)}.npy").is_file()]
    journal = cache / "spend.jsonl"
    spent = sum(json.loads(line)["usd"] for line in journal.read_text("utf-8").splitlines() if line.strip()) if journal.exists() else 0.0
    pending_cost = sum(_count(t) for t in missing) * USD_PER_TOKEN
    if spent + pending_cost > BUDGET_USD + 1e-12:
        raise ValueError(f"budget exceeded: reserved ${spent:.8f} plus pending ${pending_cost:.8f} > ${BUDGET_USD:.2f}")
    source_db_sha = sha256(root / DB)
    summary = {"model": MODEL, "dimensions": DIMENSIONS, "source_db_sha256": source_db_sha,
               "source_chunks_sha256": retriever.digest, "eligible_sha256": eligible_sha,
               "chunks": len(chunks), "positive_questions": len(positive),
               "unique_texts": len(texts), "total_estimated_tokens": sum(counts),
               "max_input_tokens": max(counts), "cached_texts": len(texts) - len(missing),
               "pending_texts": len(missing), "pending_estimated_tokens": sum(_count(t) for t in missing),
               "pending_estimated_usd": pending_cost, "already_reserved_usd": spent,
               "budget_usd": BUDGET_USD}
    return retriever, chunks, queries, texts, summary


class CachedQueries:
    def __init__(self, embedder):
        self.embedder = embedder

    def query_embed(self, text):
        return iter(self.embedder.cached([text]))


def compare(root, report):
    old_dir = root / OLD_REPORT
    old = json.loads((old_dir / "summary.json").read_text("utf-8"))
    baseline = json.loads((root / "reports/corpus-audit-v1/summary.json").read_text("utf-8"))
    new = json.loads((report / "summary.json").read_text("utf-8"))
    for field in ("eligible_sha256", "source_revision", "pdf_files", "pdf_pages", "indexed_pages_with_text"):
        if old[field] != new[field]:
            raise ValueError(f"comparison control changed: {field}")
    if old["index_metadata"] != new["index_metadata"] or old["dense_manifest"]["source_chunks_sha256"] != new["dense_manifest"]["source_chunks_sha256"]:
        raise ValueError("comparison index chunks or metadata changed")
    def cases(folder):
        rows = [json.loads(line) for line in (folder / "cases.jsonl").read_text("utf-8").splitlines()]
        result = {row["source_row"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate source rows")
        return result
    before, after = cases(old_dir), cases(report)
    if set(before) != set(after):
        raise ValueError("comparison questions changed")
    paired = []
    gains = {}
    for key in sorted(before):
        a, b = before[key], after[key]
        for field in ("doc_id", "question", "gold_pages", "evidence_sources"):
            if a[field] != b[field]:
                raise ValueError(f"comparison annotation changed: {key} {field}")
        if not a["gold_pages"]:
            continue
        delta = {}
        for scope in SCOPES:
            for method in METHODS:
                setting = f"{scope}_{method}"
                x, y = a["retrieval"][setting], b["retrieval"][setting]
                if method == "bm25" and x != y:
                    raise ValueError(f"BM25 changed for source row {key} in {scope} scope")
                delta[setting] = {metric: y[metric] - x[metric] for metric in ("recall", "hit", "mrr")}
        paired.append({"source_row": key, "doc_id": a["doc_id"], "delta": delta})
    for scope in SCOPES:
        for method in METHODS:
            setting = f"{scope}_{method}"
            gains[setting] = {"before": old["retrieval"][setting], "after": new["retrieval"][setting],
                              "recall_gains": sum(p["delta"][setting]["recall"] > 0 for p in paired),
                              "recall_regressions": sum(p["delta"][setting]["recall"] < 0 for p in paired),
                              "new_hits": sum(p["delta"][setting]["hit"] > 0 for p in paired),
                              "lost_hits": sum(p["delta"][setting]["hit"] < 0 for p in paired)}
    result = {"control": "Identical OCR SQLite chunks and eligible questions; only dense embedding model changed. BM25 must match exactly.",
              "positive_questions": len(paired), "metrics": gains, "paired_cases": paired,
              "pre_ocr_bge_baseline_context": baseline["retrieval"],
              "limit": "Known benchmark, not unseen test data. Page retrieval only; no answer accuracy, numerical correctness, or latency measured. 215 no-gold and 30 quarantined questions are excluded from positive retrieval scores."}
    write_json_atomic(report / "comparison.json", result)
    lines = ["# OpenAI embedding versus BGE on the 135-PDF OCR corpus", "",
             "The same 14,756 OCR chunks and 837 positive questions were used. Both audits rank 30 candidate chunks and score five distinct pages. BM25 results match exactly.", "",
             "| Scope / method | Pre-OCR BGE recall@5 (context) | OCR BGE recall@5 | OCR OpenAI recall@5 | New hits | Lost hits |", "|---|---:|---:|---:|---:|---:|"]
    for setting, item in gains.items():
        lines.append(f"| {setting} | {baseline['retrieval'][setting]['mean_page_recall_at_5']:.4f} | {item['before']['mean_page_recall_at_5']:.4f} | {item['after']['mean_page_recall_at_5']:.4f} | {item['new_hits']} | {item['lost_hits']} |")
    lines += ["", "The pre-OCR BGE column is context only: adding OCR changed the indexed text, so only the two OCR columns isolate the embedding model.",
              "This is a known benchmark, not unseen test data. Scores measure retrieval of annotated evidence pages, not generated answer accuracy or numerical correctness. The 215 no-gold questions and 30 quarantined questions are excluded from positive retrieval scores. Queries were pre-cached, so these results do not compare model latency. Paired gains and regressions are in comparison.json.", ""]
    (report / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def execute(root):
    root = Path(root).resolve()
    retriever, chunks, queries, texts, plan = preflight(root)
    report = root / REPORT
    if report.exists():
        raise ValueError(f"report already exists: {report}")
    from dotenv import load_dotenv
    load_dotenv(root / ".env", override=False)
    embedder = Embedder(root / CACHE, budget_usd=BUDGET_USD)
    for start in range(0, len(texts), 256):
        embedder.embed(texts[start:start + 256])
        print(json.dumps({"embedded_or_cached": min(start + 256, len(texts)), "total": len(texts), "reserved_usd": embedder.spent()}), flush=True)
    vectors = embedder.cached(chunks)
    if vectors.shape != (len(chunks), DIMENSIONS):
        raise ValueError("cached vector matrix shape mismatch")
    if len(embedder.cached(queries)) != len(queries):
        raise ValueError("query vectors missing")
    retriever.vectors = vectors
    retriever.embedder = CachedQueries(embedder)
    manifest = {k: v for k, v in plan.items() if k not in ("cached_texts", "pending_texts", "pending_estimated_tokens", "pending_estimated_usd", "already_reserved_usd")}
    manifest["reserved_usd"] = embedder.spent()
    write_json_atomic(root / CACHE / "manifest.json", manifest)
    run(root, report, "ocr", retriever=retriever, dense_manifest=manifest)
    return compare(root, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Read-only source, chunk, token and budget validation")
    mode.add_argument("--execute", action="store_true", help="Make paid embedding requests within the $0.10 cap, then audit")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.root)[-1], indent=2))
    else:
        result = execute(args.root)
        print(json.dumps({"report": str(args.root / REPORT), "positive_questions": result["positive_questions"], "metrics": result["metrics"]}, indent=2))
