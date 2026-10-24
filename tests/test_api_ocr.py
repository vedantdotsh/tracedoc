import json

import pymupdf
import pytest

from evidence_lab.api_ocr import OpenRouterOCR
from evidence_lab import api_ocr


def image(width=80, height=40):
    return pymupdf.Pixmap(pymupdf.csRGB, (0, 0, width, height), False).tobytes('png')


def pricing():
    return {'id': api_ocr.vision.MODEL,
            'architecture': {'input_modalities': ['text', 'image']},
            'supported_parameters': ['response_format'],
            'pricing': {'prompt': '0.00000025', 'completion': '0.00000075'}}


def response(lines=['A2 [unclear] mm'], finish='stop'):
    return {'model': api_ocr.vision.MODEL, 'choices': [
        {'finish_reason': finish, 'message': {'content': json.dumps({'lines': lines})}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'cost': 0.00005}}


def test_transcribes_once_and_reuses_valid_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret')
    monkeypatch.setattr(api_ocr.vision, 'get_pricing', pricing)
    calls = []

    def post(payload, key):
        calls.append(payload)
        assert key == 'secret'
        assert payload['max_tokens'] == 2048
        assert payload['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
        return response()

    monkeypatch.setattr(api_ocr.vision, '_post', post)
    ocr = OpenRouterOCR(tmp_path)
    data = image()
    expected = [{'text': 'A2 [unclear] mm', 'confidence': None, 'box': None,
                 'engine': 'openrouter_vision', 'model': api_ocr.vision.MODEL,
                 'source_verified': False}]
    assert ocr(data) == expected
    assert ocr(data) == expected
    assert len(calls) == 1
    journal = [json.loads(s) for s in (tmp_path / 'spend.jsonl').read_text().splitlines()]
    assert len([s for s in journal if 'reserved_usd' in s]) == 1
    assert journal[1]['actual_usage']['cost'] == 0.00005
    cached = json.loads(next((tmp_path / 'responses').glob('*.json')).read_text())
    assert cached['source_size_px'] == [80, 40]
    assert cached['sent_size_px'] == [80, 40]


def test_truncated_response_keeps_reservation_without_cache_or_retry(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret')
    monkeypatch.setattr(api_ocr.vision, 'get_pricing', pricing)
    calls = []

    def post(payload, key):
        calls.append(1)
        return response(finish='length')

    monkeypatch.setattr(api_ocr.vision, '_post', post)
    ocr = OpenRouterOCR(tmp_path)
    with pytest.raises(ValueError, match='Incomplete'):
        ocr(image())
    assert len(calls) == 1
    assert not list((tmp_path / 'responses').glob('*.json'))
    assert 'reserved_usd' in (tmp_path / 'spend.jsonl').read_text()


def test_budget_and_price_caps_prevent_post(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret')
    monkeypatch.setattr(api_ocr.vision, 'get_pricing', pricing)
    monkeypatch.setattr(api_ocr.vision, '_post', lambda *_: pytest.fail('must not post'))
    with pytest.raises(RuntimeError, match='budget'):
        OpenRouterOCR(tmp_path, budget=.00001)(image())
    assert not (tmp_path / 'spend.jsonl').exists()

    bad = pricing()
    bad['pricing']['completion'] = '0.000001'
    monkeypatch.setattr(api_ocr.vision, 'get_pricing', lambda: bad)
    with pytest.raises(ValueError, match='price'):
        OpenRouterOCR(tmp_path)(image())
    assert not (tmp_path / 'spend.jsonl').exists()


def test_large_image_is_resized_for_request(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret')
    monkeypatch.setattr(api_ocr.vision, 'get_pricing', pricing)
    monkeypatch.setattr(api_ocr.vision, '_post', lambda *_: response())
    OpenRouterOCR(tmp_path)(image(2200, 1200))
    cached = json.loads(next((tmp_path / 'responses').glob('*.json')).read_text())
    assert cached['source_size_px'] == [2200, 1200]
    assert max(cached['sent_size_px']) <= 2048
    api_ocr.vision.image_token_bound(*cached['sent_size_px'])
