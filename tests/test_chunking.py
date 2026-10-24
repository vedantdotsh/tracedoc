"""Compare page-local chunk strategies on synthetic PDFs."""

from collections import Counter
import sqlite3

import pymupdf
import pytest

from evidence_lab.baseline import build_index, chunk_page, evaluate, search
from evidence_lab import chunking
from evidence_lab.chunking import STRATEGIES, pack_units, page_chunks, table_rows


def make_pdf(path):
    with pymupdf.open() as doc:
        first = doc.new_page()
        first.insert_text((72, 72), "Overview", fontsize=18)
        first.insert_text((72, 100), "First section has unique ALPHATOKEN information.",
                          fontsize=10)
        first.insert_text((72, 135), "Details", fontsize=18)
        first.insert_text((72, 163), "Second section has unique BETATOKEN information.",
                          fontsize=10)
        second = doc.new_page()
        second.insert_text((72, 72), "MARSBOUND cargo allocation.", fontsize=10)
        xs, ys = (72, 190, 310), (110, 140, 170)
        for x in xs:
            second.draw_line((x, ys[0]), (x, ys[-1]))
        for y in ys:
            second.draw_line((xs[0], y), (xs[-1], y))
        for x, y, value in [(80, 130, "Product"), (198, 130, "Units"),
                            (80, 160, "Widget"), (198, 160, "42")]:
            second.insert_text((x, y), value, fontsize=10)
        doc.save(path)


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    make_pdf(path)
    return path


def test_fixed_strategy_matches_existing_word_windows(pdf):
    with pymupdf.open(pdf) as doc:
        for page in doc:
            assert page_chunks(page, "fixed", words=6, overlap=2) == chunk_page(
                page.get_text("text", sort=True), words=6, overlap=2)


def test_heading_repeats_heading_with_body_without_losing_text(pdf):
    with pymupdf.open(pdf) as doc:
        page = doc[0]
        chunks = page_chunks(page, "heading", words=8, overlap=2)
        original = Counter(page.get_text("text", sort=True).split())
    retained = Counter(" ".join(chunks).split())
    assert all(retained[token] >= count for token, count in original.items())
    assert any("Overview" in chunk and "ALPHATOKEN" in chunk for chunk in chunks)
    assert any("Details" in chunk and "BETATOKEN" in chunk for chunk in chunks)


def test_detected_table_pairs_headers_with_values(pdf):
    with pymupdf.open(pdf) as doc:
        page = doc[1]
        assert page.find_tables().tables
        chunks = page_chunks(page, "table")
    row = next(chunk for chunk in chunks if "Widget" in chunk and "42" in chunk)
    assert '"column": "Product", "value": "Widget"' in row
    assert '"column": "Units", "value": "42"' in row
    assert any("MARSBOUND" in chunk for chunk in chunks)


def test_complex_graphics_fall_back_to_full_fixed_text_without_table_scan():
    class ComplexPage:
        number = 3

        def get_text(self, kind, sort=False):
            if kind == "text":
                return "alpha beta gamma delta epsilon"
            if kind == "dict":
                return {"blocks": []}
            raise AssertionError(kind)

        def get_drawings(self):
            return [{"items": [None] * (chunking.MAX_DRAWING_ITEMS + 1)}]

        def find_tables(self, **kwargs):
            raise AssertionError("complex page must not invoke table detection")

    diagnostics = []
    chunks = page_chunks(ComplexPage(), "table", words=3, overlap=1,
                         diagnostics=diagnostics)
    assert chunks == chunk_page("alpha beta gamma delta epsilon", words=3, overlap=1)
    assert diagnostics == [{"page": 4, "reason": "table_graphics_complexity_limit",
                            "drawing_items": chunking.MAX_DRAWING_ITEMS + 1,
                            "limit": chunking.MAX_DRAWING_ITEMS,
                            "fallback": "fixed_text_chunks"}]


def test_index_reports_fallback_document_and_page(pdf, tmp_path, monkeypatch):
    monkeypatch.setattr(chunking, "MAX_DRAWING_ITEMS", 1)
    db = tmp_path / "guarded.sqlite"
    result = build_index(pdf.parent, db, strategy="table")
    assert result["diagnostics"]
    assert all(d["doc_id"] == pdf.name and d["page"] == 2
               and d["drawing_items"] > 1 and d["fallback"] == "fixed_text_chunks"
               for d in result["diagnostics"])
    assert search(db, "MARSBOUND", doc_id=pdf.name)[0]["page"] == 2


def test_pack_units_preserves_whole_short_units_and_validates_settings():
    assert pack_units(["alpha beta", "gamma delta"], words=3, overlap=1) == [
        "alpha beta", "gamma delta"]
    rows = table_rows(["Amount"], [["42"]])
    assert '"column": "Amount", "value": "42"' in rows[0]
    assert table_rows(["Amount"], [[None]]) == []
    with pytest.raises(ValueError):
        pack_units(["alpha"], words=0)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_mini_index_and_evaluation_for_each_strategy(pdf, tmp_path, strategy):
    db = tmp_path / f"{strategy}.sqlite"
    result = build_index(pdf.parent, db, strategy=strategy, words=16, overlap=3)
    assert result["strategy"] == strategy
    assert result["docs"] == 1 and result["pages"] == 2
    assert result["chunks"] >= 2
    hits = search(db, "MARSBOUND", doc_id=pdf.name)
    assert hits and hits[0]["page"] == 2
    report = evaluate(db, [{"doc_id": pdf.name, "question": "MARSBOUND",
                            "answer": "cargo allocation", "evidence_pages": [2],
                            "evidence_sources": ["Text"]}], top_k=1)
    assert report["config"]["index"]["strategy"] == strategy
    assert report["mean_recall"] == report["mean_hit"] == report["mrr"] == 1.0


def test_invalid_strategy_rejected_before_index_creation(pdf, tmp_path):
    db = tmp_path / "bad.sqlite"
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        build_index(pdf.parent, db, strategy="unknown")
    assert not db.exists()
    with pymupdf.open(pdf) as doc:
        with pytest.raises(ValueError, match="overlap"):
            page_chunks(doc[0], "paragraph", words=2, overlap=2)


def test_two_worker_index_matches_one_worker_index(pdf, tmp_path):
    make_pdf(pdf.parent / "other.pdf")
    one, two = tmp_path / "one.sqlite", tmp_path / "two.sqlite"
    first = build_index(pdf.parent, one, strategy="table", workers=1)
    second = build_index(pdf.parent, two, strategy="table", workers=2)
    assert first["extraction_workers"] == 1
    assert second["extraction_workers"] == 2
    assert {k: v for k, v in first.items() if k not in ("db", "extraction_workers")} == {
        k: v for k, v in second.items() if k not in ("db", "extraction_workers")}
    with sqlite3.connect(one) as a, sqlite3.connect(two) as b:
        for sql in ("SELECT doc_id, sha256, pages FROM docs ORDER BY doc_id",
                    "SELECT doc_id, page, chunk_id, text FROM chunks ORDER BY doc_id, page, chunk_id",
                    "SELECT key, value FROM meta ORDER BY key"):
            assert a.execute(sql).fetchall() == b.execute(sql).fetchall()
    assert search(one, "MARSBOUND", top_k=10) == search(two, "MARSBOUND", top_k=10)


def test_two_worker_failed_rebuild_preserves_prior_index(pdf, tmp_path):
    db = tmp_path / "index.sqlite"
    build_index(pdf.parent, db, workers=1)
    before = db.read_bytes()
    (pdf.parent / "broken.pdf").write_bytes(b"not a PDF")
    with pytest.raises(RuntimeError, match="unreadable PDF"):
        build_index(pdf.parent, db, workers=2)
    assert db.read_bytes() == before


@pytest.mark.parametrize("workers", [0, -1, 5, True, 1.5])
def test_invalid_worker_counts_are_rejected_before_index_creation(pdf, tmp_path, workers):
    db = tmp_path / "invalid.sqlite"
    with pytest.raises(ValueError, match="workers"):
        build_index(pdf.parent, db, workers=workers)
    assert not db.exists()
