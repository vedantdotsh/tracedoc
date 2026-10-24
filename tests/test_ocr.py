"""Local OCR selection and cache contracts; no engine or network download."""

import hashlib
import json
import sqlite3
import sys
import types

import numpy as np
import pymupdf
import pytest

from evidence_lab import ocr


def _selection_fixture(tmp_path, monkeypatch):
    package = tmp_path / "rapidocr"
    models = package / "models"
    models.mkdir(parents=True)
    for name in ("PP-OCRv6_det_small.onnx", "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                 "PP-OCRv6_rec_small.onnx"):
        (models / name).write_bytes(name.encode())
    (package / "config.yaml").write_text("offline: true", encoding="utf-8")
    (package / "__init__.py").touch()
    monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setattr(ocr.importlib.metadata, "version", lambda name: "test-version")
    pdf_dir = tmp_path / "documents"
    pdf_dir.mkdir()
    source = pdf_dir / "a.pdf"
    source.write_bytes(b"test source bytes")
    db = tmp_path / "baseline.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE docs(doc_id TEXT PRIMARY KEY, sha256 TEXT, pages INTEGER)")
        con.execute("CREATE TABLE chunks(doc_id TEXT, page INTEGER)")
        con.execute("INSERT INTO docs VALUES (?, ?, ?)",
                    (source.name, hashlib.sha256(source.read_bytes()).hexdigest(), 3))
        con.execute("INSERT INTO chunks VALUES (?, ?)", (source.name, 2))
    return db, pdf_dir, source


def test_manifest_selects_only_missing_physical_pages_and_fingerprints_settings(tmp_path, monkeypatch):
    db, pdf_dir, source = _selection_fixture(tmp_path, monkeypatch)
    output = tmp_path / "ocr"
    manifest = ocr.prepare_manifest(db, pdf_dir, output)
    assert [(p["doc_id"], p["page"]) for p in manifest["pages"]] == [
        ("a.pdf", 1), ("a.pdf", 3)]
    assert all(p["source_sha256"] == ocr.sha256(source) for p in manifest["pages"])
    assert len({p["cache_path"] for p in manifest["pages"]}) == 2
    assert all(not p["cache_path"].startswith("/") for p in manifest["pages"])
    assert manifest["settings_sha256"] == hashlib.sha256(
        json.dumps(manifest["settings"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert json.loads((output / "manifest.json").read_text("utf-8")) == manifest
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Source differs"):
        ocr.prepare_manifest(db, pdf_dir, tmp_path / "second")


def test_cache_is_bound_to_page_source_and_settings(tmp_path):
    entry = {"doc_id": "a.pdf", "page": 3, "source_sha256": "source"}
    cache = tmp_path / "page.json"
    row = {**entry, "settings_sha256": "settings", "status": "ok", "text": "readable"}
    cache.write_text(json.dumps(row), encoding="utf-8")
    assert ocr.valid_cache(cache, entry, "settings")
    assert not ocr.valid_cache(cache, {**entry, "page": 2}, "settings")
    assert not ocr.valid_cache(cache, {**entry, "source_sha256": "new source"}, "settings")
    assert not ocr.valid_cache(cache, entry, "new settings")
    cache.write_text(json.dumps({**row, "status": "blank"}), encoding="utf-8")
    assert not ocr.valid_cache(cache, entry, "settings")
    cache.write_text(json.dumps({**row, "text": ""}), encoding="utf-8")
    assert not ocr.valid_cache(cache, entry, "settings")
    cache.write_text(json.dumps({**row, "status": "error", "text": "",
                                 "error": "temporary failure"}), encoding="utf-8")
    assert ocr.valid_cache(cache, entry, "settings")  # retained for audit; --retry-errors reruns it
    cache.write_text("{bad json", encoding="utf-8")
    assert not ocr.valid_cache(cache, entry, "settings")


def test_recover_page_uses_one_based_coordinates_and_distinguishes_blank_from_error(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "documents"
    pdf_dir.mkdir()
    pdf = pdf_dir / "a.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(pdf)
    entry = {"doc_id": pdf.name, "page": 2, "source_sha256": ocr.sha256(pdf),
             "cache_path": "pages/a/p00002.json"}
    seen = []

    class Engine:
        def __init__(self, params):
            assert params == {"fixture": True}

        def __call__(self, pixels):
            seen.append(pixels.shape[:2])
            return types.SimpleNamespace(txts=["Second page text"], scores=[0.9],
                                         boxes=np.asarray([[[1, 2], [3, 4]]]))

    monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=Engine))
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        setNumThreads=lambda count: None, COLOR_RGB2BGR=1, cvtColor=lambda pixels, _: pixels))
    monkeypatch.setattr(ocr, "_ENGINE", None)
    settings = {"engine": "RapidOCR", "params": {"fixture": True}, "dpi": 72,
                "max_render_side": 2400}
    recovered = ocr.recover_page(entry, settings, "settings", pdf_dir, tmp_path / "ocr")
    assert recovered["status"] == "ok"
    assert recovered["page"] == 2 and recovered["text"] == "Second page text"
    assert recovered["lines"][0]["confidence"] == 0.9
    assert recovered["render"]["full_page"] is True and seen
    assert json.loads((tmp_path / "ocr" / entry["cache_path"]).read_text("utf-8")) == recovered

    monkeypatch.setattr(ocr, "_ENGINE", lambda pixels: types.SimpleNamespace(
        txts=[], scores=[], boxes=None))
    blank = ocr.recover_page(entry, settings, "settings", pdf_dir, tmp_path / "ocr")
    assert blank["status"] == "blank" and blank["text"] == ""
    failed = ocr.recover_page({**entry, "page": 4}, settings, "settings", pdf_dir,
                              tmp_path / "ocr")
    assert failed["status"] == "error" and failed["text"] == "" and failed["error"]
