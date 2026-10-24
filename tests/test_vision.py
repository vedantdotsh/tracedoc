"""Vision pilot guards tested offline with synthetic inputs and no paid calls."""

import json

import pytest

from evidence_lab import vision


def test_image_bound_rejects_nonpositive_oversized_and_overbudget_dimensions():
    assert 0 < vision.image_token_bound(256, 256) <= 16384
    assert vision.image_token_bound(1024, 1024) == 16384
    for width, height in ((0, 256), (-1, 256), (256, 0),
                          (2049, 128), (2048, 2048)):
        with pytest.raises(ValueError):
            vision.image_token_bound(width, height)


def test_request_bound_counts_image_pixels_and_caps_output():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What does this show?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 1000}},
    ]}]
    bound = vision.request_bound(messages, 256, 256)
    assert bound["image_token_bound"] == 16384
    assert bound["output_token_bound"] == vision.MAX_TOKENS
    assert bound["input_token_bound"] >= 16384 + len("What does this show?") + 512
    assert bound["usd_bound"] < vision.BUDGET_USD


def test_answer_validator_rejects_unknown_citation_and_invalid_schema():
    accepted = vision.validate_answer(json.dumps({
        "answer": "No supported value appears in the source.",
        "citations": [], "abstained": True,
    }))
    assert accepted["abstained"] is True
    for candidate in (
        {"answer": "99", "citations": ["S2"], "abstained": False},
        {"answer": "99", "citations": [], "abstained": False},
        {"answer": "", "citations": ["S1"], "abstained": False},
        {"answer": "99", "citations": ["S1"], "abstained": False,
         "verified": True},
    ):
        with pytest.raises(ValueError):
            vision.validate_answer(json.dumps(candidate))


def test_ledger_reservation_is_irreversible_even_without_completed_call(tmp_path):
    plan = {"case_ids": ["synthetic-1"], "model": vision.MODEL,
            "reserved_usd": 0.001}
    reserved = vision.reserve_run(tmp_path, plan)
    assert reserved.is_file()
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        vision.reserve_run(tmp_path, plan)
    assert reserved.is_file()


def test_default_pilot_replay_does_not_make_network_call(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("replay attempted a network call")

    monkeypatch.setattr(vision.urllib.request, "urlopen", forbidden)
    result = vision.run_pilot(tmp_path)
    assert result["status"] == "not_run"


def test_preflight_enforces_price_cap_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "load_fixture", lambda root: {
        "questions": [], "sources": [],
    })
    fixture = tmp_path / "data/vision-v1/fixture.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("{}", encoding="utf-8")
    base = {"id": vision.MODEL,
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_parameters": ["response_format"],
            "pricing": {"prompt": str(vision.PROMPT_PRICE),
                        "completion": str(vision.COMPLETION_PRICE)}}
    assert vision.preflight(tmp_path, pricing=base)["reserved_usd"] == 0
    overpriced = json.loads(json.dumps(base))
    overpriced["pricing"]["prompt"] = str(vision.PROMPT_PRICE * 1.01)
    with pytest.raises(ValueError, match="price"):
        vision.preflight(tmp_path, pricing=overpriced)
    surcharge = json.loads(json.dumps(base))
    surcharge["pricing"]["image"] = "0.001"
    with pytest.raises(ValueError, match="extra pricing"):
        vision.preflight(tmp_path, pricing=surcharge)


def pilot_fixture():
    return {"questions": [{"id": "T1", "source_id": "source",
                            "question": "What is shown?", "expected_answer": "PRIVATE-GOLD"}],
            "sources": [{"id": "source", "doc_id": "source.pdf", "page": 1,
                         "crop_pdf_points": [0, 0, 10, 10], "image_sha256": "f" * 64,
                         "text": "Visible source text", "width": 10, "height": 10}]}


def pilot_plan():
    return {"reserved_usd": 0.001, "requests": [{"question_id": "T1",
            "mode": "text_only", "input_token_bound": 1000,
            "output_token_bound": vision.MAX_TOKENS, "usd_bound": 0.001}]}


def test_provider_failure_stops_once_and_is_not_abstention(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "preflight", lambda root: pilot_plan())
    monkeypatch.setattr(vision, "load_fixture", lambda root: pilot_fixture())
    monkeypatch.setenv("OPENROUTER_API_KEY", "disposable-test-key")
    calls = []

    def fail(payload, key):
        calls.append((payload, key))
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr(vision, "_post", fail)
    result = vision.run_pilot(tmp_path, execute=True)
    assert len(calls) == 1
    assert result["status"] == "stopped"
    assert result["rows"][0]["status"] == "request_or_validation_failed"
    assert result["rows"][0].get("answer") is None
    assert "disposable-test-key" not in json.dumps(result)
    assert (tmp_path / "data/vision-v1/charged-run-ledger.json").is_file()
    with pytest.raises(FileExistsError):
        vision.run_pilot(tmp_path, execute=True)
    assert len(calls) == 1


def test_echoed_credential_never_appears_in_saved_raw_response(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "preflight", lambda root: pilot_plan())
    monkeypatch.setattr(vision, "load_fixture", lambda root: pilot_fixture())
    monkeypatch.setenv("OPENROUTER_API_KEY", "disposable-test-key")
    monkeypatch.setattr(vision, "_post", lambda payload, key: {
        "model": vision.MODEL, "usage": {"prompt_tokens": 10,
                                         "completion_tokens": 10},
        "diagnostic": "disposable-test-key",
        "choices": [{"message": {"content": json.dumps({
            "answer": "999", "citations": ["S1"], "abstained": False,
        })}}],
    })
    result = vision.run_pilot(tmp_path, execute=True)
    raw = tmp_path / "data/vision-v1/raw/T1-text_only.json"
    assert result["status"] == "completed"
    assert raw.is_file()
    assert "disposable-test-key" not in raw.read_text(encoding="utf-8")
    assert "PRIVATE-GOLD" not in json.dumps(result)
    assert "unverified" in result["rows"][0]["verification"]


def test_generic_description_prompt_excludes_frozen_question_and_answer(tmp_path):
    image = tmp_path / "crop.png"
    image.write_bytes(b"fixture image bytes")
    source = {**pilot_fixture()["sources"][0], "image_path": "crop.png"}
    messages = vision.build_messages(
        {"question": vision.DESCRIPTION_PROMPT}, source, "image_text", tmp_path)
    sent = json.dumps(messages)
    assert "What is shown?" not in sent
    assert "PRIVATE-GOLD" not in sent
    assert "Describe only visible contents" in sent


def test_description_preflight_counts_existing_reservation(tmp_path, monkeypatch):
    image = tmp_path / "crop.png"
    image.write_bytes(b"fixture image bytes")
    source = {**pilot_fixture()["sources"][0], "image_path": "crop.png"}
    monkeypatch.setattr(vision, "load_fixture", lambda root: {
        "sources": [source], "questions": pilot_fixture()["questions"],
    })
    monkeypatch.setattr(vision, "preflight", lambda root: {
        "pricing": {}, "checked_at": "test", "fixture_sha256": "f" * 64,
    })
    ledger = tmp_path / "data/vision-v1/charged-run-ledger.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"plan": {"reserved_usd": 0.01}}), encoding="utf-8")
    plan = vision.description_preflight(tmp_path)
    assert plan["aggregate_reserved_usd"] == pytest.approx(
        plan["prior_reserved_usd"] + plan["reserved_usd"])
    ledger.write_text(json.dumps({"plan": {"reserved_usd": vision.BUDGET_USD}}),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="Combined"):
        vision.description_preflight(tmp_path)
