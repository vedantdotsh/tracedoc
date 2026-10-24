"""Augmented index records and cache ordering, with no real model inference."""

import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from evidence_lab import extension_retrieval
from evidence_lab.retrieval import fingerprint


def inputs(root):
    table_path = root / "data/tables-v1/extraction/tables.json"
    table_path.parent.mkdir(parents=True)
    table_path.write_text(json.dumps([{"facts": [{
        "row_id": "AB-12", "column": "Capacity", "value": 42, "unit": "MW",
        "conditions": {"temperature": "25 C"}, "source_text": "42 MW",
        "provenance": {"doc_id": "a.pdf", "page": 1,
                       "cell_bbox": [1, 2, 3, 4]},
    }]}]), encoding="utf-8")
    descriptions = root / "reports/vision-v1/reviewed-descriptions.json"
    descriptions.parent.mkdir(parents=True)
    descriptions.write_text(json.dumps({"rows": [
        {"status": "described", "review": {"status": "corrected"},
         "doc_id": "a.pdf", "page": 2, "description": "Four visible panels.",
         "source_id": "crop-1", "image_path": "crop.png", "image_sha256": "f" * 64,
         "pdf_sha256": "a" * 64, "crop_pdf_points": [0, 0, 20, 20],
         "model_requested": "test-model", "expected_answer": "SHOULD NOT LEAK"},
        {"status": "described", "review": {"status": "pending"},
         "doc_id": "a.pdf", "page": 3, "description": "UNREVIEWED CAPTION"},
        {"status": "abstained", "review": {"status": "accepted"},
         "doc_id": "a.pdf", "page": 4, "description": "ABSTENTION"},
    ]}), encoding="utf-8")


def test_additional_records_include_cells_and_reviewed_descriptions_only(tmp_path):
    inputs(tmp_path)
    rows = extension_retrieval.additional_records(tmp_path)
    assert [row["kind"] for row in rows] == [
        "extracted_table_cell", "derived_diagram_description"]
    assert "AB-12" in rows[0]["text"] and "42" in rows[0]["text"]
    assert "MANUALLY REVIEWED" in rows[1]["text"]
    assert all("SHOULD NOT LEAK" not in json.dumps(row) for row in rows)
    assert all("UNREVIEWED CAPTION" not in json.dumps(row) for row in rows)
    assert all("ABSTENTION" not in json.dumps(row) for row in rows)
    assert len({row["chunk_id"] for row in rows}) == 2


def test_build_reuses_original_vectors_and_fingerprints_ordered_rows(tmp_path, monkeypatch):
    inputs(tmp_path)
    db = tmp_path / "data/baseline.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5("
                    "doc_id UNINDEXED, page UNINDEXED, chunk_id UNINDEXED, text)")
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", [
            ("a.pdf", 1, "original-1", "first text"),
            ("a.pdf", 2, "original-2", "second text"),
        ])

    class FakeEmbedder:
        def embed(self, texts, batch_size):
            assert batch_size == 32
            return iter([[0.0, 3.0] for _ in texts])

    class FakeRetriever:
        def __init__(self, db, cache, embedder=None):
            self.cache = cache
            with closing(sqlite3.connect(db)) as con:
                self.rows = [dict(zip(("doc_id", "page", "chunk_id", "text"), row))
                             for row in con.execute(
                                 "SELECT doc_id,page,chunk_id,text FROM chunks ORDER BY rowid")]
            self.vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def build(self):
            return None

        def _embedder(self):
            return FakeEmbedder()

    monkeypatch.setattr(extension_retrieval, "Retriever", FakeRetriever)
    report = extension_retrieval.build(tmp_path)
    assert report["baseline_chunks"] == 2
    assert report["added_chunks"] == 2
    cache = tmp_path / "data/extension-index/dense"
    vectors = np.load(cache / "vectors.npy", allow_pickle=False)
    np.testing.assert_array_equal(vectors[:2], [[1, 0], [0, 1]])
    np.testing.assert_allclose(vectors[2:], [[0, 1], [0, 1]])
    with sqlite3.connect(tmp_path / "data/extension-index/augmented.sqlite") as con:
        rows = [dict(zip(("doc_id", "page", "chunk_id", "text"), row))
                for row in con.execute("SELECT doc_id,page,chunk_id,text FROM chunks ORDER BY rowid")]
    manifest = json.loads((cache / "manifest.json").read_text("utf-8"))
    assert [row["chunk_id"] for row in rows[:2]] == ["original-1", "original-2"]
    assert manifest["source_chunks_sha256"] == fingerprint(rows)
    with pytest.raises(FileExistsError):
        extension_retrieval.build(tmp_path)


def test_failed_build_keeps_final_name_free_for_retry(tmp_path, monkeypatch):
    inputs(tmp_path)
    db = tmp_path / "data/baseline.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5("
                    "doc_id UNINDEXED, page UNINDEXED, chunk_id UNINDEXED, text)")
        con.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    ("a.pdf", 1, "original", "source text"))

    fail = [True]

    class FakeEmbedder:
        def embed(self, texts, batch_size):
            if fail[0]:
                raise RuntimeError("synthetic embedding failure")
            return iter([[1.0, 0.0] for _ in texts])

    class FakeRetriever:
        def __init__(self, db, cache, embedder=None):
            self.cache = cache
            with closing(sqlite3.connect(db)) as con:
                self.rows = [dict(zip(("doc_id", "page", "chunk_id", "text"), row))
                             for row in con.execute(
                                 "SELECT doc_id,page,chunk_id,text FROM chunks ORDER BY rowid")]
            self.vectors = np.array([[1.0, 0.0]], dtype=np.float32)

        def build(self):
            return None

        def _embedder(self):
            return FakeEmbedder()

    monkeypatch.setattr(extension_retrieval, "Retriever", FakeRetriever)
    with pytest.raises(RuntimeError, match="synthetic embedding failure"):
        extension_retrieval.build(tmp_path)
    assert not (tmp_path / "data/extension-index").exists()
    fail[0] = False
    assert extension_retrieval.build(tmp_path)["added_chunks"] == 2
    assert (tmp_path / "data/extension-index/dense/manifest.json").is_file()
