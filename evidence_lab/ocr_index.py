"""Add OCR text from previously unindexed pages to a separate baseline copy.

The experiment changes extraction coverage only: chunking, embedding model and
retrieval budgets remain unchanged. OCR text is candidate evidence, not verified fact.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from . import baseline
from .retrieval import EMBED_MODEL, Retriever, fingerprint


def additional_records(root):
    root = Path(root).resolve()
    manifest_path = root / "data/ocr-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings_hash = hashlib.sha256(json.dumps(
        manifest["settings"], sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if settings_hash != manifest["settings_sha256"]:
        raise ValueError("OCR manifest settings hash mismatch")
    with closing(baseline._connect(root / "data/baseline.sqlite")) as con:
        docs = {d: (sha, count) for d, sha, count in con.execute("SELECT doc_id,sha256,pages FROM docs")}
        existing = set(con.execute("SELECT DISTINCT doc_id,page FROM chunks"))
        meta = dict(con.execute("SELECT key,value FROM meta"))
    expected = {(d, page) for d, (_, count) in docs.items()
                for page in range(1, count + 1) if (d, page) not in existing}
    listed = [(p["doc_id"], p["page"]) for p in manifest["pages"]]
    if len(listed) != len(set(listed)) or set(listed) != expected:
        raise ValueError("OCR selection must cover each originally unindexed page exactly once")
    # Reject stale extraction if the PDF changed after the baseline was indexed.
    for doc_id in sorted({d for d, _ in expected}):
        pdf = (root / "data/documents" / doc_id).resolve()
        if pdf.parent != (root / "data/documents").resolve():
            raise ValueError("Invalid source document path")
        with pdf.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != docs[doc_id][0]:
            raise ValueError(f"Source PDF hash changed: {doc_id}")
    records, page_status = [], {"ok": 0, "blank": 0}
    record_hashes = []
    for item in sorted(manifest["pages"], key=lambda p: (p["doc_id"], p["page"])):
        doc_id, page = item["doc_id"], item["page"]
        if isinstance(page, bool) or not isinstance(page, int):
            raise ValueError("OCR page number must be an integer")
        cache = (manifest_path.parent / item["cache_path"]).resolve()
        if not cache.is_relative_to(root / "data/ocr-v1"):
            raise ValueError("OCR cache path outside run folder")
        raw = cache.read_bytes()
        result = json.loads(raw)
        for key, expected_value in (("doc_id", doc_id), ("page", page),
                                    ("source_sha256", docs[doc_id][0]),
                                    ("settings_sha256", settings_hash)):
            if result.get(key) != expected_value:
                raise ValueError(f"OCR result {key} mismatch: {doc_id} page {page}")
        if item["source_sha256"] != docs[doc_id][0]:
            raise ValueError("OCR manifest source hash mismatch")
        status, text = result.get("status"), result.get("text")
        if status not in page_status or not isinstance(text, str):
            raise ValueError(f"Incomplete or failed OCR: {doc_id} page {page}")
        if (status == "ok") != bool(text.strip()):
            raise ValueError("OCR status contradicts recovered text")
        page_status[status] += 1
        digest = hashlib.sha256(raw).hexdigest()
        record_hashes.append((doc_id, page, digest))
        provenance = {"kind": "ocr_text", "source_sha256": docs[doc_id][0],
                      "settings_sha256": settings_hash, "cache_path": item["cache_path"],
                      "cache_sha256": digest, "source_verified": False}
        for n, part in enumerate(baseline.chunk_page(text, int(meta["words"]), int(meta["overlap"]))):
            records.append({"doc_id": doc_id, "page": page,
                            "chunk_id": f"{doc_id}#p{page}-ocr-c{n}",
                            "text": part, "provenance": provenance})
    return records, {"selected_pages": len(expected), "page_status": page_status,
                     "settings_sha256": settings_hash,
                     "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                     "page_artifacts_sha256": hashlib.sha256(json.dumps(record_hashes).encode()).hexdigest(),
                     "words": int(meta["words"]), "overlap": int(meta["overlap"])}


def build(root):
    root = Path(root).resolve()
    final = root / "data/ocr-index"
    if final.exists():
        raise FileExistsError("Completed OCR index already exists")
    additions, audit = additional_records(root)
    staging = Path(tempfile.mkdtemp(prefix=".ocr-index-", dir=root / "data"))
    db = staging / "ocr.sqlite"
    with closing(baseline._connect(root / "data/baseline.sqlite")) as source:
        with closing(sqlite3.connect(db)) as target:
            source.backup(target)
            target.execute("CREATE TABLE ocr_evidence(chunk_id TEXT PRIMARY KEY, provenance TEXT NOT NULL)")
            target.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                               [(r["doc_id"], r["page"], r["chunk_id"], r["text"]) for r in additions])
            target.executemany("INSERT INTO ocr_evidence VALUES (?, ?)",
                               [(r["chunk_id"], json.dumps(r["provenance"])) for r in additions])
            target.execute("INSERT INTO meta VALUES ('ocr_settings_sha256', ?)", (audit["settings_sha256"],))
            target.commit()
    original = Retriever(root / "data/baseline.sqlite", root / "data/dense-v1")
    original.build()
    embedder = original._embedder() if additions else None
    augmented = Retriever(db, staging / "dense", embedder=embedder)
    if augmented.rows[:len(original.rows)] != original.rows:
        raise ValueError("Baseline row ordering changed; cannot reuse vectors")
    extra = []
    if additions:
        for n, vector in enumerate(embedder.embed([r["text"] for r in additions], batch_size=32), 1):
            extra.append(vector)
            if n % 256 == 0:
                print(f"Embedded OCR chunks: {n}/{len(additions)}", flush=True)
        extra = np.asarray(extra, dtype=np.float32)
        if (extra.shape != (len(additions), original.vectors.shape[1])
                or not np.isfinite(extra).all()
                or np.any(np.linalg.norm(extra, axis=1) == 0)):
            raise ValueError("Invalid OCR embeddings")
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        vectors = np.concatenate((original.vectors, extra))
    else:
        vectors = original.vectors.copy()
    augmented.cache.mkdir()
    np.save(augmented.cache / "vectors.npy", vectors, allow_pickle=False)
    manifest = {"model": EMBED_MODEL, "source_chunks_sha256": fingerprint(augmented.rows),
                "chunks": len(augmented.rows)}
    (augmented.cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {**audit, "baseline_chunks": len(original.rows), "added_chunks": len(additions),
              "total_chunks": len(augmented.rows), "model": EMBED_MODEL,
              "baseline_documents_with_text": len({r["doc_id"] for r in original.rows}),
              "total_documents_with_text": len({r["doc_id"] for r in augmented.rows}),
              "experiment": "Original text index plus OCR on all previously unindexed pages; no table/caption additions.",
              "note": "OCR text is unverified candidate evidence. Gold annotations are not used to select or index pages."}
    (staging / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging.rename(final)
    return report


if __name__ == "__main__":
    print(json.dumps(build(Path.cwd()), indent=2))
