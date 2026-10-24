"""Downloader integrity checks; all HTTP responses are local test doubles."""

import io

import pytest

from evidence_lab import corpus


def test_bad_download_checksum_preserves_prior_file(tmp_path, monkeypatch):
    target = tmp_path / "document.pdf"
    target.write_bytes(b"previous verified file")
    received = b"%PDF-invalid replacement"
    monkeypatch.setattr(corpus.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(received))
    with pytest.raises(ConnectionError, match="sha1 mismatch"):
        corpus._fetch_file("https://example.test/file", target, len(received), "0" * 40, True)
    assert target.read_bytes() == b"previous verified file"
    assert not target.with_name(target.name + ".part").exists()


def test_unsafe_names_are_reported_and_never_downloaded(tmp_path, monkeypatch):
    names = ["../escape.pdf", "subdir/file.pdf", "C:escape.pdf", "..\\escape.pdf"]
    tree = {"tree": [{"path": corpus.SAMPLES_PATH, "type": "blob"}] + [
        {"path": corpus.DOC_PREFIX + name, "type": "blob"} for name in names]}
    monkeypatch.setattr(corpus, "_load_tree", lambda root: tree)
    calls = []

    def sync(entry, dest, pdf):
        calls.append((dest, pdf))
        return {"status": "existing"}

    monkeypatch.setattr(corpus, "_sync", sync)
    result = corpus.download(tmp_path)
    assert calls == [(tmp_path / "benchmark" / "samples.json", False)]
    assert {f["doc_id"] for f in result["failures"] if f["error"] == "unsafe filename"} == set(names)
    assert result["verified_pdfs"] == 0
    assert result["ok"] is False


def test_workers_are_bounded_and_invalid_counts_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "_load_tree", lambda root: {
        "tree": [{"path": corpus.SAMPLES_PATH, "type": "blob"}]})
    monkeypatch.setattr(corpus, "_sync", lambda *a: {"status": "existing"})
    actual_executor = corpus.cf.ThreadPoolExecutor
    requested = []

    def executor(workers):
        requested.append(workers)
        return actual_executor(workers)

    monkeypatch.setattr(corpus.cf, "ThreadPoolExecutor", executor)
    corpus.download(tmp_path, workers=1000)
    assert requested == [corpus.MAX_WORKERS]
    for workers in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            corpus.download(tmp_path, workers=workers)
