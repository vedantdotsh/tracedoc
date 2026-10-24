"""Offline checks for the optional evidence gate; no provider or gold labels."""

import json

import pytest

from evidence_lab import abstention, answering


def source(text, *, doc_id="a.pdf", source_id="S1"):
    return {"source_id": source_id, "doc_id": doc_id, "page": 1,
            "chunk_id": f"{doc_id}#p1-c0", "text": text}


def hit(text, *, doc_id="a.pdf"):
    return {key: value for key, value in source(text, doc_id=doc_id).items()
            if key != "source_id"}


@pytest.mark.parametrize("question,evidence,reason", [
    ("What was revenue in 2021?", "Revenue was $10 million in 2022.",
     "missing_requested_year"),
    ("What percentage of participants completed the survey in 2024?",
     "The survey had a 60 percent completion rate in 2023.", "missing_requested_year"),
    ("What does section 7 require?", "Section 8 requires an audit.",
     "missing_typed_identifier"),
    ("Calculate the percentage required by section 7.",
     "Section 8 requires a 60 percent threshold.", "missing_typed_identifier"),
    ("How many participants completed the survey?", "The survey was published in 2021.",
     "missing_numeric_detail"),
])
def test_missing_requested_detail_abstains(question, evidence, reason):
    decision = abstention.check(question, [source(evidence)])
    assert decision["abstain"] is True
    assert reason in decision["reasons"]
    assert isinstance(decision["details"], list)


@pytest.mark.parametrize("question,evidence", [
    ("Is the warranty 3 years?", "The warranty lasts 2 years."),
    ("Was the 2021 revenue $10 million?", "The 2022 revenue was $12 million."),
    ("What is 20 percent of 80?", "The formula for a percentage is given here."),
    ("Compare revenue in 2021 and 2022.", "Revenue was $10 million in 2021."),
    ("What does section 7 require?", "Section 7 requires an audit."),
    ("How many participants completed the survey?", "In total, twelve participants completed it."),
    ("How many participants completed the survey?", "2021 participants completed the survey."),
])
def test_gate_does_not_infer_missing_support_from_question_digits_alone(question, evidence):
    """A verification claim, computation, or relevant number needs model judgment."""
    decision = abstention.check(question, [source(evidence)])
    assert decision["abstain"] is False, decision


def test_passing_gate_never_certifies_support():
    decision = abstention.check(
        "What does section 7 require?", [source("Section 7 requires an audit.")])
    assert decision["abstain"] is False
    assert decision["support_verified"] is False


def test_conflicting_numeric_facts_with_same_scope_abstain():
    question = "What was the annual revenue?"
    sources = [source("Annual revenue was 10 million dollars.", doc_id="a.pdf"),
               source("Annual revenue was 12 million dollars.",
                      doc_id="b.pdf", source_id="S2")]
    decision = abstention.check(question, sources)
    assert decision["abstain"] is True
    assert "conflicting_numeric_facts" in decision["reasons"]


@pytest.mark.parametrize("question,second", [
    ("Compare annual revenue for the two companies.", "Annual revenue was 12 million dollars."),
    ("What was the annual revenue?", "Quarterly revenue was 12 million dollars."),
])
def test_distinct_numeric_scopes_do_not_trigger_conflict(question, second):
    sources = [source("Annual revenue was 10 million dollars.", doc_id="a.pdf"),
               source(second, doc_id="b.pdf", source_id="S2")]
    assert "conflicting_numeric_facts" not in abstention.check(question, sources)["reasons"]


def test_same_document_numeric_variants_do_not_trigger_cross_document_conflict():
    sources = [source("Annual revenue was 10 million dollars."),
               source("Annual revenue was 12 million dollars.", source_id="S2")]
    assert "conflicting_numeric_facts" not in abstention.check(
        "What was the annual revenue?", sources)["reasons"]


def provider_reply(answer="Twelve months.", abstained=False):
    content = {"answer": answer, "citations": [] if abstained else ["S1"],
               "abstained": abstained}
    return json.dumps({"model": "fixture", "usage": {}, "choices": [{
        "finish_reason": "stop", "message": {"content": json.dumps(content)}}]}).encode()


def test_gate_blocks_generation_without_loading_credentials(monkeypatch):
    monkeypatch.setattr(answering, "_post", lambda *a, **kw: pytest.fail("provider called"))
    result = answering.answer_from_hits(
        "What was revenue in 2021?", [hit("Revenue was $10 million in 2022.")],
        allow_paid=True, evidence_gate=True)
    assert result["status"] == "gate_abstained"
    assert result["execution_status"] == "not_run"
    assert result["abstained"] is True
    assert result["abstention_origin"] == "evidence_gate"
    assert result["evidence_gate"]["abstain"] is True
    assert result["raw_response"] is None
    assert result["usage"] == {}


def test_gate_off_uses_same_evidence_and_preserves_old_result_shape(monkeypatch):
    calls = []

    def fake_post(messages, model, *args):
        calls.append(messages)
        return provider_reply()

    monkeypatch.setattr(answering, "_post", fake_post)
    args = ("What was revenue in 2021?", [hit("Revenue was $10 million in 2022.")])
    baseline = answering.answer_from_hits(*args, allow_paid=True)
    explicit_off = answering.answer_from_hits(*args, allow_paid=True, evidence_gate=False)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert baseline["status"] == explicit_off["status"] == "answered"
    assert baseline["sources"] == explicit_off["sources"]
    assert baseline["messages"] == explicit_off["messages"]
    assert "evidence_gate" not in baseline and "evidence_gate" not in explicit_off
    assert "abstention_origin" not in baseline and "abstention_origin" not in explicit_off


def test_model_refusal_and_provider_failure_remain_distinct(monkeypatch):
    args = ("What is the warranty duration?", [hit("The warranty lasts twelve months.")])
    monkeypatch.setattr(answering, "_post", lambda *a, **kw: provider_reply(
        "The evidence does not say.", abstained=True))
    refusal = answering.answer_from_hits(*args, allow_paid=True, evidence_gate=True)
    assert refusal["status"] == "abstained"
    assert refusal["execution_status"] == "ok"
    assert refusal["abstention_origin"] == "model"
    monkeypatch.setattr(answering, "_post", lambda *a, **kw: (_ for _ in ()).throw(
        TimeoutError("fixture timeout")))
    failure = answering.answer_from_hits(*args, allow_paid=True, evidence_gate=True)
    assert failure["status"] == failure["execution_status"] == "provider_error"
    assert failure["abstained"] is None
    assert failure.get("abstention_origin") not in ("model", "evidence_gate")
