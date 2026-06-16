"""Day-1 BASELINE retriever + contradiction judge — deliberately weak, on purpose.

The point of building the eval first is to have an honest yardstick BEFORE the real
system exists. This baseline is the floor: lexical token-overlap retrieval and a
crude negation-cue judge. It scores poorly, and that's the feature — when the real
dense-retrieval + cross-encoder rerank + LLM-judge backend lands (see retrieve.py /
judge.py, Day 2), the same `keeper eval` shows the lift, with numbers.

Interfaces the real backend will implement:
    retrieve(claim, passages, k) -> list[passage_id]      # ranked, best first
    judge(claim, passages_by_id)  -> dict[passage_id, float]  # P(contradicts)
"""
from __future__ import annotations

import re

from .corpus import Passage

_WORD = re.compile(r"[a-z]+")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by", "for", "with",
    "was", "were", "is", "are", "be", "been", "his", "her", "its", "their", "he", "she",
    "it", "they", "that", "this", "who", "whom", "from", "as", "but", "not", "no", "all",
    "any", "each", "into", "out", "up", "down", "him", "them", "had", "has", "have",
}
# Cues that often mark a flat denial / reversal. Weak by design.
_NEGATION = {"no", "never", "nothing", "not", "none", "without", "freely", "neither", "nor"}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


def retrieve(claim: str, passages: list[Passage], k: int = 5) -> list[str]:
    """Rank passages by token-overlap (Jaccard) with the claim. Ties broken by id
    for determinism."""
    cset = set(_tokens(claim))
    scored = []
    for p in passages:
        pset = set(_tokens(p.text))
        union = cset | pset
        j = len(cset & pset) / len(union) if union else 0.0
        scored.append((j, p.id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [pid for _, pid in scored[:k]]


def judge(claim: str, passages_by_id: dict[str, Passage]) -> dict[str, float]:
    """Crude contradiction confidence: a passage that shares topic words with the
    claim AND carries a negation cue gets a higher score. No real entailment — this
    is the floor the real LLM judge has to beat."""
    cset = set(_tokens(claim))
    out: dict[str, float] = {}
    for pid, p in passages_by_id.items():
        ptoks = _tokens(p.text)
        pset = set(ptoks)
        overlap = len(cset & pset) / len(cset) if cset else 0.0
        has_negation = any(w in _NEGATION for w in p.text.lower().split())
        # confidence: topical overlap gated by a negation cue, capped honest-low.
        conf = round(min(0.6, overlap) * (1.0 if has_negation else 0.25), 3)
        out[pid] = conf
    return out
