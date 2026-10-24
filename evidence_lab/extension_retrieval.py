"""Add structured cells and derived diagram descriptions without changing the baseline.

The augmented index retains all original chunks. Descriptions are always labeled
as model-derived, unverified evidence. Search is not numerical verification.
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
    root = Path(root)
    tables = json.loads((root / "data/tables-v1/extraction/tables.json").read_text("utf-8"))
    descriptions = json.loads((root / "reports/vision-v1/reviewed-descriptions.json").read_text("utf-8"))
    rows = []
    for table in tables:
        for fact in table.get("facts", []):
            provenance = fact["provenance"]
            content = {
                "row_id": fact["row_id"], "field": fact["column"],
                "value": fact["value"], "unit": fact.get("unit"),
                "conditions": fact.get("conditions", {}), "source_cell_text": fact["source_text"],
            }
            text = "STRUCTURED TABLE EXTRACTION; verify against source. " + json.dumps(content, ensure_ascii=False)
            rows.append({"doc_id": provenance["doc_id"], "page": provenance["page"],
                         "text": text, "kind": "extracted_table_cell", "provenance": provenance})
    for item in descriptions["rows"]:
        if item["status"] != "described" or item.get("review", {}).get("status") not in ("accepted", "corrected"):
            continue
        rows.append({"doc_id": item["doc_id"], "page": item["page"],
                     "text": "DERIVED DIAGRAM DESCRIPTION; MANUALLY REVIEWED PILOT CROP, NOT ORIGINAL TEXT. " + item["description"],
                     "kind": "derived_diagram_description", "provenance": {
                         key: item[key] for key in ("source_id", "image_path", "image_sha256",
                                                    "pdf_sha256", "crop_pdf_points", "model_requested", "review")}})
    for index, row in enumerate(rows):
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()[:12]
        row["chunk_id"] = f"extension-{index}-{digest}"
    return rows


def build(root):
    """Copy the old index and reuse its audited vectors; embed only new records."""
    root = Path(root)
    final_folder = root / "data/extension-index"
    if final_folder.exists():
        raise FileExistsError("A completed extension index already exists")
    # Preserve failed staging directories for diagnosis; a failure cannot block a fresh build.
    folder = Path(tempfile.mkdtemp(prefix=".extension-index-", dir=root / "data"))
    db = folder / "augmented.sqlite"
    additions = additional_records(root)
    with closing(baseline._connect(root / "data/baseline.sqlite")) as source:
        with closing(sqlite3.connect(db)) as destination:
            source.backup(destination)
            destination.execute("CREATE TABLE extension_evidence (chunk_id TEXT PRIMARY KEY, kind TEXT, provenance TEXT)")
            destination.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                                    [(r["doc_id"], r["page"], r["chunk_id"], r["text"]) for r in additions])
            destination.executemany("INSERT INTO extension_evidence VALUES (?, ?, ?)",
                                    [(r["chunk_id"], r["kind"], json.dumps(r["provenance"])) for r in additions])
            destination.commit()

    original = Retriever(db=root / "data/baseline.sqlite", cache=root / "data/dense-v1")
    original.build()
    embedder = original._embedder()  # Verifies and loads the existing local model snapshot.
    augmented = Retriever(db=db, cache=folder / "dense", embedder=embedder)
    if augmented.rows[:len(original.rows)] != original.rows:
        raise ValueError("Original row ordering changed; cached vectors cannot be reused")
    extra = np.asarray(list(embedder.embed([r["text"] for r in additions], batch_size=32)), dtype=np.float32)
    if extra.ndim != 2 or extra.shape[1] != original.vectors.shape[1] or not np.isfinite(extra).all():
        raise ValueError("Invalid added vectors")
    extra /= np.maximum(np.linalg.norm(extra, axis=1, keepdims=True), 1e-12)
    vectors = np.concatenate((original.vectors, extra))
    augmented.cache.mkdir()
    np.save(augmented.cache / "vectors.npy", vectors, allow_pickle=False)
    manifest = {"model": EMBED_MODEL, "source_chunks_sha256": fingerprint(augmented.rows),
                "chunks": len(augmented.rows)}
    (augmented.cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"baseline_chunks": len(original.rows), "added_chunks": len(additions),
              "total_chunks": len(augmented.rows), "model": EMBED_MODEL,
              "kinds": {kind: sum(r["kind"] == kind for r in additions)
                        for kind in sorted({r["kind"] for r in additions})},
              "note": "Baseline unchanged. Added embeddings use its audited model. No benchmark labels indexed.",
              "additional_records_sha256": hashlib.sha256(json.dumps(additions, sort_keys=True).encode()).hexdigest()}
    (folder / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    folder.rename(final_folder)  # Windows refuses an existing destination; no silent replacement.
    return report


def search(root, question, method="hybrid", top_k=5):
    """Retrieve evidence from the whole augmented corpus, with explicit source kinds."""
    root = Path(root)
    folder = root / "data/extension-index"
    original = Retriever(db=root / "data/baseline.sqlite", cache=root / "data/dense-v1")
    embedder = original._embedder() if method != "bm25" else None
    retriever = Retriever(db=folder / "augmented.sqlite", cache=folder / "dense", embedder=embedder)
    hits = retriever.search(question, method=method, top_k=top_k)
    with closing(baseline._connect(folder / "augmented.sqlite")) as con:
        for hit in hits:
            metadata = con.execute("SELECT kind,provenance FROM extension_evidence WHERE chunk_id=?",
                                   (hit["chunk_id"],)).fetchone()
            hit["evidence_kind"] = metadata[0] if metadata else "native_pdf_text"
            hit["provenance"] = json.loads(metadata[1]) if metadata else {"doc_id": hit["doc_id"], "page": hit["page"]}
            hit["support_verified"] = False
    return hits
