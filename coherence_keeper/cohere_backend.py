"""Cohere-backed retrieval: dense Embed → cross-encoder Rerank.

This is the real retrieval pipeline, built on Cohere's production stack:
  1. Embed every passage (search_document) and the claim (search_query) with
     embed-english-v3.0; cosine similarity gives a dense first-stage ranking.
  2. Rerank the dense top-N against the claim with rerank-english-v3.0 (a real
     cross-encoder) to get the final top-k.

Doc embeddings are cached on disk so repeated `keeper eval` runs don't re-embed the
corpus (and stay well under the trial key's rate limits). Calls retry with backoff
on rate-limit errors. Needs COHERE_API_KEY (loaded from .env) and `pip install cohere`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from . import config
from .corpus import Passage

_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "data" / "cache"
_EMBED_MODEL = "embed-english-v3.0"
_RERANK_MODEL = "rerank-english-v3.0"
_DENSE_N = 12  # dense candidates handed to the reranker


def _client():
    config.load_env()
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise RuntimeError("COHERE_API_KEY not set (put it in .env). See .env.example.")
    import cohere
    return cohere.Client(key)


def _with_backoff(fn, *, tries=5, base=2.0):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — SDK raises several rate-limit types
            msg = str(e).lower()
            transient = "429" in msg or "rate" in msg or "timeout" in msg
            if not transient or attempt == tries - 1:
                raise
            time.sleep(base * (attempt + 1))
    raise RuntimeError("unreachable")


def _embeddings(resp):
    """embed-v3 response: .embeddings is a list of float vectors (no embedding_types)."""
    emb = resp.embeddings
    return getattr(emb, "float", emb)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class CohereBackend:
    def __init__(self):
        self.co = _client()
        self._doc_cache: dict[str, list[float]] = {}

    def _corpus_key(self, passages: list[Passage]) -> str:
        h = hashlib.sha256()
        for p in sorted(passages, key=lambda x: x.id):
            h.update(p.id.encode())
            h.update(p.text.encode())
        return h.hexdigest()[:16]

    def _doc_embeddings(self, passages: list[Passage]) -> dict[str, list[float]]:
        if self._doc_cache:
            return self._doc_cache
        _CACHE.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE / f"doc-embeds-{self._corpus_key(passages)}.json"
        if cache_file.exists():
            self._doc_cache = json.loads(cache_file.read_text())
            return self._doc_cache
        resp = _with_backoff(lambda: self.co.embed(
            texts=[p.text for p in passages], model=_EMBED_MODEL,
            input_type="search_document"))
        vecs = _embeddings(resp)
        self._doc_cache = {p.id: list(v) for p, v in zip(passages, vecs)}
        cache_file.write_text(json.dumps(self._doc_cache))
        return self._doc_cache

    def retrieve(self, claim: str, passages: list[Passage], k: int = 5) -> list[str]:
        docs = self._doc_embeddings(passages)
        qresp = _with_backoff(lambda: self.co.embed(
            texts=[claim], model=_EMBED_MODEL, input_type="search_query"))
        qvec = list(_embeddings(qresp)[0])

        dense = sorted(docs.items(), key=lambda kv: -_cosine(qvec, kv[1]))[:_DENSE_N]
        dense_ids = [pid for pid, _ in dense]
        by_id = {p.id: p for p in passages}

        rr = _with_backoff(lambda: self.co.rerank(
            model=_RERANK_MODEL, query=claim,
            documents=[by_id[pid].text for pid in dense_ids], top_n=k))
        return [dense_ids[r.index] for r in rr.results]
