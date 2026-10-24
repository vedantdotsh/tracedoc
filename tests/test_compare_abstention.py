"""Counterfactual replay must reuse saved generation and evidence, entirely offline."""

from copy import deepcopy

import pytest

from evidence_lab import answering, compare_abstention


def case(question, evidence, *, answer="Saved answer", abstained=False,
         execution_status="ok", answerable=True):
    result = {
        "question": question, "sources": [{"source_id": "S1", "doc_id": "a.pdf",
            "page": 1, "chunk_id": "a.pdf#p1-c0", "text": evidence}],
        "status": "abstained" if abstained else "answered",
        "execution_status": execution_status,
        "answer": answer, "abstained": abstained,
        "citations": [] if abstained else [{"source_id": "S1", "doc_id": "a.pdf",
            "page": 1, "chunk_id": "a.pdf#p1-c0"}],
        "raw_response": "SAVED_PROVIDER_RESPONSE", "usage": {"total_tokens": 17},
        "messages": [{"role": "user", "content": "SAVED_REQUEST"}],
    }
    return {"method": "bm25", "source_row": 3, "doc_id": "a.pdf",
            "reference_answer": "GOLD_SECRET", "expected_pages": [1] if answerable else [],
            "scores": {"answerable": answerable}, "result": result}


def test_replay_reuses_exact_saved_generation_and_sources_without_network(monkeypatch):
    monkeypatch.setattr(answering, "_post", lambda *a, **kw: pytest.fail("provider called"))
    original = case("What does section 7 require?", "Section 7 requires an audit.")
    before = deepcopy(original)
    replay = compare_abstention.replay_case(original)
    assert original == before
    assert replay["baseline"] == original["result"]
    assert replay["gated"]["answer"] == original["result"]["answer"]
    assert replay["gated"]["sources"] == original["result"]["sources"]
    assert replay["gated"]["raw_response"] == "SAVED_PROVIDER_RESPONSE"
    assert replay["gated"]["usage"] == {"total_tokens": 17}
    assert replay["decision"]["abstain"] is False
    assert replay["gated"]["replay_original_generation"] is True


def test_replay_block_is_gate_only_and_preserves_original_response(monkeypatch):
    monkeypatch.setattr(answering, "_post", lambda *a, **kw: pytest.fail("provider called"))
    original = case("What was revenue in 2021?", "Revenue was $10 million in 2022.")
    replay = compare_abstention.replay_case(original)
    assert replay["baseline"]["status"] == "answered"
    assert replay["gated"]["status"] == "gate_abstained"
    assert replay["gated"]["execution_status"] == "not_run"
    assert replay["gated"]["abstained"] is True
    assert replay["gated"]["abstention_origin"] == "evidence_gate"
    assert replay["gated"]["raw_response"] == original["result"]["raw_response"]
    assert replay["decision"]["abstain"] is True
    assert original["result"]["status"] == "answered"


def test_replay_decision_does_not_use_gold_or_model_answer():
    original = case("What was revenue in 2021?", "Revenue was $10 million in 2022.")
    altered = deepcopy(original)
    altered["reference_answer"] = "THE_ANSWER_IS_HIDDEN"
    altered["expected_pages"] = []
    altered["scores"]["answerable"] = False
    altered["result"]["answer"] = "Completely different saved answer"
    assert compare_abstention.replay_case(original)["decision"] == (
        compare_abstention.replay_case(altered)["decision"])


def test_provider_failure_stays_failure_in_replay():
    failed = case("What was revenue in 2021?", "Revenue was $10 million in 2022.",
                  answer=None, execution_status="provider_error")
    failed["result"].update(status="provider_error", abstained=None)
    replay = compare_abstention.replay_case(failed)
    assert replay["baseline"]["execution_status"] == "provider_error"
    assert replay["gated"]["execution_status"] == "provider_error"
    assert replay["gated"]["abstained"] is None
    assert replay["gated"].get("abstention_origin") not in ("evidence_gate", "model")


def test_summary_separates_gate_tradeoff_from_model_and_provider():
    positive_blocked = compare_abstention.replay_case(case(
        "What was revenue in 2021?", "Revenue was $10 million in 2022."))
    negative_answer = compare_abstention.replay_case(case(
        "What does section 7 require?", "Section 8 requires an audit.",
        answerable=False))
    negative_refusal = compare_abstention.replay_case(case(
        "Is the warranty 3 years?", "The warranty lasts 2 years.",
        abstained=True, answerable=False))
    failure = case("What was revenue in 2021?", "Revenue was $10 million in 2022.",
                   answer=None, execution_status="provider_error", answerable=False)
    failure["result"].update(status="provider_error", abstained=None)
    failed_replay = compare_abstention.replay_case(failure)
    summary = compare_abstention.summarize([
        positive_blocked, negative_answer, negative_refusal, failed_replay])["bm25"]
    assert summary["baseline"]["gate_abstentions"] == 0
    assert summary["gated"]["gate_abstentions"] == 2
    assert summary["gated"]["false_abstentions"] == 1
    assert summary["baseline"]["negative_false_answers"] == 1
    assert summary["gated"]["negative_false_answers"] == 0
    assert summary["gated"]["model_abstentions"] == 1
    assert summary["gated"]["original_execution_failures"] == 1
