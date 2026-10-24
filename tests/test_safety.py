"""CLI data-preservation checks on small, isolated fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest

from evidence_lab.baseline import build_index
from evidence_lab.corpus import write_atomic


PROJECT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "evidence_lab", *map(str, args)],
        cwd=PROJECT, capture_output=True, text=True,
    )


@pytest.fixture
def fixture(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf = pdf_dir / "a.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "alpha evidence value")
        doc.save(pdf)
    db = tmp_path / "index.sqlite"
    build_index(pdf_dir, db)
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{
        "doc_id": "a.pdf", "question": "alpha", "answer": "value",
        "evidence_pages": [1], "evidence_sources": ["Text"],
    }]), encoding="utf-8")
    return pdf_dir, pdf, db, samples


@pytest.mark.parametrize("command,output_flag,target", [
    ("audit", "--output", "samples"),
    ("audit", "--eligible", "samples"),
    ("audit", "--eligible", "db"),
    ("evaluate", "--output", "samples"),
    ("evaluate", "--output", "db"),
    ("ask", "--output", "db"),
])
def test_cli_rejects_input_output_aliases(fixture, tmp_path, command, output_flag, target):
    _, _, db, samples = fixture
    protected = samples if target == "samples" else db
    before = protected.read_bytes()
    args = [command]
    if command == "ask":
        args += ["alpha"]
    else:
        args += ["--samples", samples]
    args += ["--db", db, output_flag, protected]
    if command == "audit":
        args += ["--eligible" if output_flag == "--output" else "--output",
                 tmp_path / "other.json", "--overwrite"]
    elif command == "evaluate":
        args += ["--overwrite"]
    result = cli(*args)
    assert result.returncode == 1, result.stderr
    assert "distinct" in result.stderr
    assert protected.read_bytes() == before


def test_index_rejects_source_pdf_as_database(fixture):
    pdf_dir, pdf, _, _ = fixture
    before = pdf.read_bytes()
    result = cli("index", "--pdf-dir", pdf_dir, "--db", pdf)
    assert result.returncode == 1, result.stderr
    assert "distinct" in result.stderr
    assert pdf.read_bytes() == before


def test_audit_rejects_malformed_row_without_writing_outputs(fixture, tmp_path):
    _, _, db, _ = fixture
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps([{"doc_id": "a.pdf"}]), encoding="utf-8")
    report, eligible = tmp_path / "report.json", tmp_path / "eligible.json"
    result = cli("audit", "--samples", malformed, "--db", db,
                 "--output", report, "--eligible", eligible)
    assert result.returncode == 1
    assert "sample 0: missing keys" in result.stderr
    assert "Traceback" not in result.stderr
    assert not report.exists() and not eligible.exists()


def test_atomic_write_preserves_unrelated_tmp_file(tmp_path):
    target = tmp_path / "output.json"
    existing = tmp_path / "output.json.tmp"
    existing.write_bytes(b"unrelated")
    write_atomic(target, b"new result")
    assert target.read_bytes() == b"new result"
    assert existing.read_bytes() == b"unrelated"


def test_index_preserves_unrelated_tmp_file(fixture):
    pdf_dir, _, db, _ = fixture
    existing = db.with_name(db.name + ".tmp")
    existing.write_bytes(b"unrelated")
    build_index(pdf_dir, db)
    assert existing.read_bytes() == b"unrelated"
    assert db.read_bytes().startswith(b"SQLite format 3")


def test_audit_requires_overwrite_for_existing_output(fixture, tmp_path):
    _, _, db, samples = fixture
    report = tmp_path / "report.json"
    report.write_bytes(b"prior report")
    eligible = tmp_path / "eligible.json"
    result = cli("audit", "--samples", samples, "--db", db,
                 "--output", report, "--eligible", eligible)
    assert result.returncode == 1
    assert "--overwrite" in result.stderr
    assert report.read_bytes() == b"prior report"
    assert not eligible.exists()
