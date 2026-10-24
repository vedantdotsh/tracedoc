"""Baseline behavior checks using synthetic PDFs, never benchmark answers."""

from pathlib import Path

import pytest

from evidence_lab.baseline import build_index, chunk_page, evaluate, score_pages, search


def test_chunks_preserve_words_and_bounded_overlap():
    tokens = [f"word{i}" for i in range(23)]
    chunks = [part.split() for part in chunk_page(" ".join(tokens), words=10, overlap=3)]
    assert all(0 < len(part) <= 10 for part in chunks)
    assert chunks[0] == tokens[:10]
    reconstructed = chunks[0].copy()
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-3:] == current[:3]
        reconstructed.extend(current[3:])
    assert reconstructed == tokens


def test_empty_and_short_pages():
    assert chunk_page(" \n\t ") == []
    assert chunk_page("Only three words", words=10, overlap=0) == ["Only three words"]


def test_invalid_chunk_settings():
    for words, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]:
        with pytest.raises(ValueError):
            chunk_page("Some source text", words=words, overlap=overlap)


def make_index(tmp_path: Path) -> Path:
    import pymupdf

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    documents = {
        "alpha": ["Overview of the apparatus.", "ZX102 pressure is 12 bar at room temperature."],
        "beta": ["ZX102 pressure is 18 bar in the alternative model."],
    }
    for name, pages in documents.items():
        with pymupdf.open() as document:
            for text in pages:
                document.new_page().insert_text((72, 72), text)
            document.save(pdf_dir / f"{name}.pdf")
    db_path = tmp_path / "index.sqlite"
    assert isinstance(build_index(pdf_dir, db_path), dict)
    return db_path


def test_search_retains_source_page_and_filters_document(tmp_path):
    db_path = make_index(tmp_path)
    hits = search(db_path, "ZX102")
    assert {hit["doc_id"] for hit in hits} == {"alpha.pdf", "beta.pdf"}
    filtered = search(db_path, "ZX102", doc_id="alpha.pdf")
    assert len(filtered) == 1
    hit = filtered[0]
    assert hit["doc_id"] == "alpha.pdf"
    assert hit["page"] == 2
    assert "12 bar" in hit["text"]
    assert hit["chunk_id"] is not None
    assert isinstance(hit["score"], (int, float))
    assert search(db_path, "ZX102", doc_id="missing") == []
    assert len(search(db_path, "ZX102", top_k=1)) == 1


def test_query_punctuation_is_safe_and_does_not_mutate_index(tmp_path):
    db_path = make_index(tmp_path)
    expected = search(db_path, "ZX102")
    assert search(db_path, "") == []
    assert search(db_path, ' \" (); * ') == []
    assert isinstance(search(db_path, "ZX102'); DROP TABLE chunks; --"), list)
    assert search(db_path, "ZX102", doc_id="alpha' OR 1=1 --") == []
    assert search(db_path, "ZX102") == expected


def test_invalid_result_limit(tmp_path):
    db_path = make_index(tmp_path)
    for limit in (0, -1):
        with pytest.raises(ValueError):
            search(db_path, "ZX102", top_k=limit)


def test_page_recall_is_not_hit_rate():
    assert score_pages([1, 2], [3, 2, 2]) == {"recall": 0.5, "hit": 1.0, "mrr": 0.5}
    assert score_pages([1, 2], [3]) == {"recall": 0.0, "hit": 0.0, "mrr": 0.0}
    with pytest.raises(ValueError):
        score_pages([], [1])


def test_evaluation_excludes_unanswerable_questions_from_retrieval_metrics(tmp_path):
    db_path = make_index(tmp_path)
    samples = [
        {"doc_id": "alpha.pdf", "question": "ZX102", "evidence_pages": "[1, 2]",
         "evidence_sources": "['Text']"},
        {"doc_id": "alpha.pdf", "question": "unavailable", "evidence_pages": [],
         "evidence_sources": [], "answer": "Not answerable"},
    ]
    result = evaluate(db_path, samples, top_k=1)
    assert result["answerable"] == result["unanswerable"] == 1
    assert result["mean_recall"] == 0.5
    assert result["mean_hit"] == result["mrr"] == 1.0
    assert result["cases"][0]["retrieved_pages"] == [2]
    assert result["cases"][1]["recall"] is None
    only_unanswerable = evaluate(db_path, samples[1:])
    assert only_unanswerable["mean_recall"] is None


def test_evaluation_rejects_unknown_documents_and_invalid_evidence(tmp_path):
    db_path = make_index(tmp_path)
    sample = {"doc_id": "alpha.pdf", "question": "ZX102",
              "evidence_pages": [2], "evidence_sources": ["Text"]}
    for change in ({"doc_id": "missing.pdf"}, {"evidence_pages": [0]},
                   {"evidence_pages": [3]}, {"evidence_pages": [True]}):
        with pytest.raises(ValueError, match="sample 0:"):
            evaluate(db_path, [{**sample, **change}])


def test_reference_answers_are_neither_indexed_nor_used_as_queries(tmp_path):
    db_path = make_index(tmp_path)
    (tmp_path / "pdfs" / "samples.json").write_text(
        '{"answer": "LEAKEDSENTINEL"}', encoding="utf-8")
    build_index(tmp_path / "pdfs", db_path)
    sample = {"doc_id": "alpha.pdf", "question": "UNSEENQUESTION",
              "answer": "ZX102 LEAKEDSENTINEL", "evidence_pages": [2],
              "evidence_sources": ["Text"]}
    result = evaluate(db_path, [sample])
    assert result["cases"][0]["retrieved_pages"] == []
    assert search(db_path, "LEAKEDSENTINEL") == []


def test_evaluation_does_not_classify_missing_evidence_as_unanswerable(tmp_path):
    db_path = make_index(tmp_path)
    sample = {"doc_id": "alpha.pdf", "question": "ZX102", "answer": "12 bar",
              "evidence_pages": [], "evidence_sources": []}
    with pytest.raises(ValueError, match="without page evidence"):
        evaluate(db_path, [sample])


def test_failed_rebuild_preserves_working_index(tmp_path):
    db_path = make_index(tmp_path)
    before = db_path.read_bytes()
    expected = search(db_path, "ZX102")
    (tmp_path / "pdfs" / "broken.pdf").write_bytes(b"This is not a PDF")
    with pytest.raises(RuntimeError, match="unreadable PDF"):
        build_index(tmp_path / "pdfs", db_path)
    assert db_path.read_bytes() == before
    assert search(db_path, "ZX102") == expected
