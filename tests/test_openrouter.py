"""OpenRouter behavior uses a mocked transport and a disposable test key."""

import json
import urllib.error
from unittest.mock import MagicMock

import pytest

from evidence_lab import answering


HITS = [{"doc_id": "a.pdf", "page": 2, "chunk_id": "a.pdf#p2-c0",
         "text": "The warranty lasts twelve months.", "score": -1.0}]
ANSWER = {"answer": "Twelve months.", "citations": ["S1"], "abstained": False}


def response(body):
    obj = MagicMock()
    obj.__enter__.return_value.read.return_value = json.dumps(body).encode()
    return obj


def openrouter_body(content=ANSWER, *, finish_reason="stop"):
    return {"model": answering.DEFAULT_MODEL, "usage": {"total_tokens": 22},
            "choices": [{"finish_reason": finish_reason,
                         "message": {"content": json.dumps(content)}}]}


def test_openrouter_requires_explicit_opt_in(monkeypatch):
    called = MagicMock()
    monkeypatch.setattr(answering.urllib.request, "urlopen", called)
    with pytest.raises(ValueError, match="allow_paid"):
        answering.answer_from_hits("warranty?", HITS)
    called.assert_not_called()


def test_openrouter_requires_local_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    called = MagicMock()
    monkeypatch.setattr(answering.urllib.request, "urlopen", called)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        answering.answer_from_hits("warranty?", HITS, allow_paid=True)
    called.assert_not_called()


def test_request_uses_supplied_evidence_and_hides_key(monkeypatch):
    key = "disposable-test-key-never-sent"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    called = MagicMock(return_value=response(openrouter_body()))
    monkeypatch.setattr(answering.urllib.request, "urlopen", called)
    result = answering.answer_from_hits("warranty?", HITS, allow_paid=True)
    req = called.call_args.args[0]
    assert result["status"] == "answered"
    assert result["model_requested"] == answering.DEFAULT_MODEL
    assert result["citations"][0]["page"] == HITS[0]["page"]
    assert req.full_url == answering.OPENROUTER_URL
    assert req.get_header("Authorization") == f"Bearer {key}"
    assert json.loads(req.data)["messages"] == result["messages"]
    assert HITS[0]["text"] in result["messages"][1]["content"]
    assert key not in json.dumps(result)
    called.assert_called_once()


@pytest.mark.parametrize("body", [
    {"choices": [{"finish_reason": "length", "message": {"content": json.dumps(ANSWER)}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
    openrouter_body({"answer": "wrong", "citations": ["S9"], "abstained": False}),
    {"choices": []},
])
def test_openrouter_rejects_incomplete_or_invalid_responses(monkeypatch, body):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    called = MagicMock(return_value=response(body))
    monkeypatch.setattr(answering.urllib.request, "urlopen", called)
    result = answering.answer_from_hits("warranty?", HITS, allow_paid=True)
    assert result["execution_status"] == "invalid_response"
    assert result["answer"] is None
    assert result["abstained"] is None
    assert result["raw_response"]
    called.assert_called_once()


def test_openrouter_transport_failure_has_no_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    called = MagicMock(side_effect=urllib.error.URLError("offline"))
    monkeypatch.setattr(answering.urllib.request, "urlopen", called)
    result = answering.answer_from_hits("warranty?", HITS, allow_paid=True)
    assert result["execution_status"] == "provider_error"
    assert result["answer"] is None
    assert result["abstained"] is None
    called.assert_called_once()
