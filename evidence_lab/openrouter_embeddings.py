"""OpenRouter text-embedding-3-small client with a per-vector cache and a durable spend journal."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken

MODEL = "openai/text-embedding-3-small"
URL = "https://openrouter.ai/api/v1/embeddings"
DIMENSIONS = 1536
MAX_INPUT_TOKENS = 8192
USD_PER_TOKEN = 0.02 / 1_000_000
BATCH = 64

_ENC = tiktoken.get_encoding("cl100k_base")
_SEPARATORS = ("\n\n", "\n", " ")


def _count(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def chunk_text(text: str, max_tokens: int = 800) -> list[str]:
    """Split text so each chunk is <= max_tokens; "".join(chunks) == text."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    return _split(text, max_tokens, _SEPARATORS)


def _split(text: str, max_tokens: int, seps: tuple[str, ...]) -> list[str]:
    if not text:
        return []
    if _count(text) <= max_tokens:
        return [text]
    if len(text) == 1:
        raise ValueError('A Unicode character exceeds the requested token limit')
    if seps:
        sep = seps[0]
        parts = [p + sep for p in text.split(sep)]
        parts[-1] = parts[-1][: -len(sep)]
    else:
        parts = list(text)
    out: list[str] = []
    cur = ""
    for part in parts:
        pieces = _split(part, max_tokens, seps[1:]) if _count(part) > max_tokens else [part]
        for piece in pieces:
            if cur and _count(cur + piece) > max_tokens:
                out.append(cur)
                cur = ""
            cur += piece
    if cur:
        out.append(cur)
    return out


def _key(text: str) -> str:
    return hashlib.sha256((MODEL + text).encode("utf-8")).hexdigest()


class Embedder:
    def __init__(self, cache: Path, budget_usd: float = 0.1):
        if not np.isfinite(budget_usd) or budget_usd <= 0:
            raise ValueError('Budget must be positive and finite')
        self.cache = Path(cache)
        self.budget_usd = budget_usd
        self.vectors = self.cache / "vectors"
        self.vectors.mkdir(parents=True, exist_ok=True)
        self.journal = self.cache / "spend.jsonl"

    def spent(self) -> float:
        if not self.journal.exists():
            return 0.0
        lines = self.journal.read_text(encoding="utf-8").splitlines()
        return sum(json.loads(line)["usd"] for line in lines if line.strip())

    def embed(self, texts: list[str]) -> np.ndarray:
        lock = self.cache / 'request.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError('Embedding cache is in use; inspect request.lock after a crashed process') from exc
        os.close(fd)
        try:
            return self._embed(texts)
        finally:
            lock.unlink()

    def _embed(self, texts):
        texts = list(texts)
        counts: dict[str, int] = {}
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                raise ValueError('Embedding text must be nonempty')
            n = _count(t)
            if n > MAX_INPUT_TOKENS:
                raise ValueError(f"input has {n} tokens; cap is {MAX_INPUT_TOKENS}")
            counts[t] = n

        found: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for t in counts:
            path = self._path(_key(t))
            if path.exists():
                found[t] = self._load(path)
            else:
                missing.append(t)

        if missing:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for uncached texts")
            cost = sum(counts[t] for t in missing) * USD_PER_TOKEN
            if self.spent() + cost > self.budget_usd:
                raise RuntimeError(
                    f"batch costs ${cost:.8f}; ${self.spent():.8f} of ${self.budget_usd} already reserved"
                )
            # 32 * 8192 stays below the provider's aggregate request ceiling.
            for i in range(0, len(missing), 32):
                batch = missing[i : i + 32]
                self._reserve(sum(counts[t] for t in batch))
                for t, vec in zip(batch, self._request(api_key, batch)):
                    self._save(_key(t), vec)
                    found[t] = vec

        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        return np.stack([found[t] for t in texts])

    def cached(self, texts):
        return np.asarray([self._load(self._path(_key(t))) for t in texts])

    def _load(self, path):
        if not path.is_file():
            raise ValueError('Embeddings missing: run workspace --embed --allow-paid first')
        vec = np.load(path, allow_pickle=False)
        if vec.shape != (DIMENSIONS,) or not np.isfinite(vec).all() or not np.isclose(np.linalg.norm(vec), 1, atol=1e-5):
            raise ValueError('Invalid cached embedding')
        return vec

    def query_embed(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _path(self, key: str) -> Path:
        return self.vectors / f"{key}.npy"

    def _save(self, key: str, vec: np.ndarray) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.save(f, vec)
        os.replace(tmp, path)

    def _reserve(self, tokens: int) -> None:
        """Durably journal the spend before the call; never refunded, even on failure."""
        line = json.dumps({"model": MODEL, "tokens": tokens, "usd": tokens * USD_PER_TOKEN}) + "\n"
        with open(self.journal, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _request(self, api_key: str, batch: list[str]) -> list[np.ndarray]:
        payload = {
            "model": MODEL,
            "input": batch,
            "dimensions": DIMENSIONS,
            "encoding_format": "float",
            "provider": {"allow_fallbacks": False, "max_price": {"prompt": 0.02}},
        }
        req = urllib.request.Request(
            URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            response = json.loads(resp.read())
            if response.get('model', MODEL) not in (MODEL, 'text-embedding-3-small'):
                raise ValueError('Unexpected embedding model')
            items = response['data']

        if any(type(item.get('index')) is not int for item in items) or sorted(item["index"] for item in items) != list(range(len(batch))):
            raise ValueError("response indexes do not match inputs")
        out: list[np.ndarray | None] = [None] * len(batch)
        for item in items:
            vec = np.asarray(item["embedding"], dtype=np.float64)
            if vec.shape != (DIMENSIONS,):
                raise ValueError(f"expected {DIMENSIONS} dimensions, got shape {vec.shape}")
            if not np.isfinite(vec).all():
                raise ValueError("embedding contains non-finite values")
            norm = np.linalg.norm(vec)
            if norm == 0:
                raise ValueError("embedding is all zeros")
            out[item["index"]] = (vec / norm).astype(np.float32)
        return out  # type: ignore[return-value]
