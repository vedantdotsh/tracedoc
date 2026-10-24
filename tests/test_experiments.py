"""Notebook helpers keep partitions stable and display untrusted text safely."""

import pytest

from evidence_lab import answering, experiments


SAMPLES = [
    {"doc_id": doc, "question": f"question {i}", "answer": "gold",
     "evidence_pages": [1], "evidence_sources": ["Text"]}
    for i, doc in enumerate(["a.pdf", "b.pdf", "a.pdf", "c.pdf", "d.pdf"])
]


def test_frozen_split_is_deterministic_disjoint_and_complete(tmp_path):
    path = tmp_path / "split.json"
    first = experiments.freeze_split(SAMPLES, path)
    assert experiments.freeze_split(SAMPLES, path) == first
    dev, reserved = set(first["development_docs"]), set(first["reserved_docs"])
    assert dev and reserved
    assert dev.isdisjoint(reserved)
    assert dev | reserved == {s["doc_id"] for s in SAMPLES}
    selected = (experiments.select_samples(SAMPLES, first, "development")
                + experiments.select_samples(SAMPLES, first, "reserved"))
    assert len(selected) == len(SAMPLES)
    assert sorted(s["question"] for s in selected) == sorted(s["question"] for s in SAMPLES)
    assert len({part for part in ("development", "reserved")
                if "a.pdf" in first[f"{part}_docs"]}) == 1


def test_changed_split_is_refused_without_overwriting(tmp_path):
    path = tmp_path / "split.json"
    experiments.freeze_split(SAMPLES, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Frozen split differs"):
        experiments.freeze_split(SAMPLES, path, seed="changed")
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="Frozen split differs"):
        experiments.freeze_split(SAMPLES[:-1], path)
    assert path.read_bytes() == before


def test_inspect_case_escapes_untrusted_text_and_does_not_generate(monkeypatch):
    sample = {"doc_id": "<doc>.pdf", "question": "<script>question</script>",
              "answer": "<img src=x>", "evidence_pages": [1]}
    seen = []

    def fake_search(db, query, doc_id, top_k):
        seen.append((query, doc_id, top_k))
        return [{"doc_id": doc_id, "page": 1, "chunk_id": "c1",
                 "text": "<script>source</script>", "score": -1.0}]

    def forbidden(*args, **kwargs):
        raise AssertionError("generation should not run")

    monkeypatch.setattr(experiments.baseline, "search", fake_search)
    monkeypatch.setattr(answering, "answer", forbidden)
    html = experiments.inspect_case("unused.sqlite", sample)
    assert seen == [(sample["question"], sample["doc_id"], experiments.baseline.ALL_CHUNKS)]
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;source&lt;/script&gt;" in html
    assert "&lt;script&gt;question&lt;/script&gt;" in html
    assert "&lt;img src=x&gt;" in html
    assert "Generated answer: not run" in html
