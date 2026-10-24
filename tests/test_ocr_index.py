"""OCR index integrity and row alignment without model inference."""

import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from evidence_lab import ocr_index
from evidence_lab.retrieval import fingerprint


def _fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    documents = data / "documents"
    documents.mkdir()
    pdf = documents / "a.pdf"
    pdf.write_bytes(b"source snapshot")
    source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    db = data / "baseline.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5("
                    "doc_id UNINDEXED, page UNINDEXED, chunk_id UNINDEXED, text)")
        con.execute("CREATE TABLE docs(doc_id TEXT PRIMARY KEY, sha256 TEXT, pages INTEGER)")
        con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("INSERT INTO docs VALUES (?, ?, ?)", (pdf.name, source_hash, 3))
        con.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    (pdf.name, 1, "native-p1", "Original evidence"))
        con.executemany("INSERT INTO meta VALUES (?, ?)",
                        [("words", "2"), ("overlap", "0"), ("page_base", "1")])
    output = data / "ocr-v1"
    output.mkdir()
    settings = {"engine": "offline-test", "dpi": 150}
    settings_hash = hashlib.sha256(json.dumps(
        settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    pages = []
    for page, status, text in ((2, "ok", "alpha beta gamma delta"), (3, "blank", "")):
        relpath = f"pages/a/p{page:05d}.json"
        pages.append({"doc_id": pdf.name, "page": page, "source_sha256": source_hash,
                      "cache_path": relpath})
        cache = output / relpath
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"doc_id": pdf.name, "page": page,
                                     "source_sha256": source_hash,
                                     "settings_sha256": settings_hash,
                                     "status": status, "text": text}), encoding="utf-8")
    manifest = {"settings": settings, "settings_sha256": settings_hash, "pages": pages}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return db, pdf, manifest


def test_records_keep_physical_pages_chunk_settings_and_audited_status(tmp_path):
    _, _, manifest = _fixture(tmp_path)
    records, audit = ocr_index.additional_records(tmp_path)
    assert [(r["page"], r["text"]) for r in records] == [
        (2, "alpha beta"), (2, "gamma delta")]
    assert [r["chunk_id"] for r in records] == [
        "a.pdf#p2-ocr-c0", "a.pdf#p2-ocr-c1"]
    assert audit["selected_pages"] == 2
    assert audit["page_status"] == {"ok": 1, "blank": 1}
    assert (audit["words"], audit["overlap"]) == (2, 0)
    assert all(r["provenance"]["settings_sha256"] == manifest["settings_sha256"]
               and r["provenance"]["source_verified"] is False for r in records)


@pytest.mark.parametrize("corrupt", ["selection", "missing", "error", "contradiction",
                                      "settings", "page", "source", "cache_escape"])
def test_rejects_incomplete_stale_or_failed_ocr(tmp_path, corrupt):
    _, pdf, manifest = _fixture(tmp_path)
    output = tmp_path / "data/ocr-v1"
    cache = output / manifest["pages"][0]["cache_path"]
    row = json.loads(cache.read_text("utf-8"))
    if corrupt == "selection":
        manifest["pages"].pop()
    elif corrupt == "missing":
        cache.unlink()
    elif corrupt == "error":
        row.update(status="error", text="", error="recognizer failed")
    elif corrupt == "contradiction":
        row.update(status="blank", text="unexpected text")
    elif corrupt == "settings":
        row["settings_sha256"] = "stale settings"
    elif corrupt == "page":
        row["page"] = 3
    elif corrupt == "source":
        pdf.write_bytes(b"new source")
    elif corrupt == "cache_escape":
        manifest["pages"][0]["cache_path"] = "../../outside.json"
    if corrupt in ("selection", "cache_escape"):
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif corrupt not in ("missing", "source"):
        cache.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        ocr_index.additional_records(tmp_path)


def test_build_preserves_baseline_and_aligns_vectors_with_rows(tmp_path, monkeypatch):
    db, _, _ = _fixture(tmp_path)
    before = db.read_bytes()

    class FakeEmbedder:
        def embed(self, texts, batch_size):
            assert texts == ["alpha beta", "gamma delta"] and batch_size == 32
            return iter([[0, 2], [0, 3]])

    class FakeRetriever:
        def __init__(self, db, cache, embedder=None):
            with closing(sqlite3.connect(db)) as con:
                self.rows = [dict(zip(("doc_id", "page", "chunk_id", "text"), row))
                             for row in con.execute(
                                 "SELECT doc_id,page,chunk_id,text FROM chunks ORDER BY rowid")]
            self.vectors = np.asarray([[1, 0]], dtype=np.float32)
            self.cache = cache

        def build(self):
            return None

        def _embedder(self):
            return FakeEmbedder()

    monkeypatch.setattr(ocr_index, "Retriever", FakeRetriever)
    report = ocr_index.build(tmp_path)
    assert report["baseline_chunks"] == 1 and report["added_chunks"] == 2
    assert report["page_status"] == {"ok": 1, "blank": 1}
    assert db.read_bytes() == before
    folder = tmp_path / "data/ocr-index"
    with sqlite3.connect(folder / "ocr.sqlite") as con:
        rows = [dict(zip(("doc_id", "page", "chunk_id", "text"), row))
                for row in con.execute("SELECT doc_id,page,chunk_id,text FROM chunks ORDER BY rowid")]
        evidence = dict(con.execute("SELECT chunk_id,provenance FROM ocr_evidence"))
    assert [r["chunk_id"] for r in rows] == [
        "native-p1", "a.pdf#p2-ocr-c0", "a.pdf#p2-ocr-c1"]
    assert set(evidence) == {r["chunk_id"] for r in rows[1:]}
    assert all(json.loads(value)["kind"] == "ocr_text" for value in evidence.values())
    vectors = np.load(folder / "dense/vectors.npy", allow_pickle=False)
    np.testing.assert_array_equal(vectors, [[1, 0], [0, 1], [0, 1]])
    dense_manifest = json.loads((folder / "dense/manifest.json").read_text("utf-8"))
    assert dense_manifest["source_chunks_sha256"] == fingerprint(rows)
    with pytest.raises(FileExistsError):
        ocr_index.build(tmp_path)
