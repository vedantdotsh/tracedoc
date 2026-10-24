import json
import urllib.error

import numpy as np
import pytest

from evidence_lab import openrouter_embeddings as oe


class _Resp:
    def __init__(self, body):
        self._raw = json.dumps(body).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Fake:
    """Returns a unit-ish vector with its peak at len(text), in reversed order."""

    def __init__(self):
        self.calls = []
        self.mutate = None

    def __call__(self, req, timeout=None):
        body = json.loads(req.data)
        self.calls.append((req, body))
        data = []
        for i, text in enumerate(body["input"]):
            vec = [0.0] * oe.DIMENSIONS
            vec[len(text)] = 5.0
            data.append({"index": i, "embedding": vec})
        data.reverse()
        if self.mutate:
            self.mutate(data)
        return _Resp({"data": data})


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake = Fake()
    monkeypatch.setattr(oe.urllib.request, "urlopen", fake)
    return fake


def test_index_order_normalization_and_request_shape(tmp_path, http):
    vecs = oe.Embedder(tmp_path).embed(["a", "bbb", "cc"])
    assert vecs.shape == (3, oe.DIMENSIONS)
    assert [int(v.argmax()) for v in vecs] == [1, 3, 2]
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
    req, body = http.calls[0]
    assert req.full_url == oe.URL
    assert req.get_header("Authorization") == "Bearer test-key"
    assert body == {
        "model": "openai/text-embedding-3-small",
        "input": ["a", "bbb", "cc"],
        "dimensions": 1536,
        "encoding_format": "float",
        "provider": {"allow_fallbacks": False, "max_price": {"prompt": 0.02}},
    }


def test_cache_hits_need_no_key_and_no_http(tmp_path, http, monkeypatch):
    first = oe.Embedder(tmp_path).embed(["a", "bb"])
    monkeypatch.delenv("OPENROUTER_API_KEY")
    again = oe.Embedder(tmp_path)
    assert np.array_equal(again.embed(["bb", "a"]), first[::-1])
    assert int(again.query_embed("a").argmax()) == 1
    assert len(http.calls) == 1
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        again.embed(["a", "uncached"])
    assert len(http.calls) == 1


def test_only_unique_missing_texts_are_requested(tmp_path, http):
    emb = oe.Embedder(tmp_path)
    emb.embed(["a"])
    out = emb.embed(["a", "bb", "bb"])
    assert http.calls[1][1]["input"] == ["bb"]
    assert out.shape[0] == 3 and np.array_equal(out[1], out[2])


def test_token_cap_rejects_before_any_request(tmp_path, http):
    with pytest.raises(ValueError, match="cap"):
        oe.Embedder(tmp_path).embed(["ok", "word " * 9000])
    assert http.calls == []
    assert not (tmp_path / "spend.jsonl").exists()


def test_budget_preflight_covers_whole_batch(tmp_path, http):
    emb = oe.Embedder(tmp_path, budget_usd=1e-6)  # ~50 tokens
    with pytest.raises(RuntimeError, match="batch costs"):
        emb.embed(["short", "word " * 100])  # "short" alone would fit
    assert http.calls == []
    assert emb.spent() == 0.0


def test_reserve_happens_before_call_and_is_never_refunded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    journal = tmp_path / "spend.jsonl"
    seen = []

    def boom(req, timeout=None):
        seen.append(journal.read_text())
        raise urllib.error.URLError("down")

    monkeypatch.setattr(oe.urllib.request, "urlopen", boom)
    text = "word " * 100
    tokens = oe._count(text)
    emb = oe.Embedder(tmp_path, budget_usd=tokens * oe.USD_PER_TOKEN * 1.5)
    with pytest.raises(urllib.error.URLError):
        emb.embed([text])
    assert len(seen) == 1 and json.loads(seen[0])["tokens"] == tokens  # reserved first, no retry
    assert emb.spent() == pytest.approx(tokens * oe.USD_PER_TOKEN)  # not refunded
    with pytest.raises(RuntimeError, match="batch costs"):
        emb.embed([text])  # remaining budget is too small for a second attempt
    assert len(seen) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d[0].update(index=d[1]["index"]),
        lambda d: d[0].update(embedding=[1.0] * 10),
        lambda d: d[0]["embedding"].__setitem__(0, float("nan")),
        lambda d: d[0].update(embedding=[0.0] * oe.DIMENSIONS),
        lambda d: d.pop(),
    ],
    ids=["dup-index", "dimension", "nan", "zero", "missing"],
)
def test_invalid_responses_are_rejected_and_not_cached(tmp_path, http, mutate):
    http.mutate = mutate
    emb = oe.Embedder(tmp_path)
    with pytest.raises(ValueError):
        emb.embed(["a", "bb"])
    assert list(emb.vectors.glob("*.npy")) == []


def test_chunk_text_preserves_everything_and_respects_limit():
    text = ("héllo wörld 日本語 😀 text.\n\n" * 60) + "tail  \n"
    chunks = oe.chunk_text(text, max_tokens=50)
    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(oe._count(c) <= 50 for c in chunks)


def test_chunk_text_falls_back_to_characters_for_unbroken_text():
    text = "😀日本語" * 200
    chunks = oe.chunk_text(text, max_tokens=20)
    assert "".join(chunks) == text
    assert all(oe._count(c) <= 20 for c in chunks)


def test_chunk_text_small_and_empty():
    assert oe.chunk_text("short text") == ["short text"]
    assert oe.chunk_text("") == []
