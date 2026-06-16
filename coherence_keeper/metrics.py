"""Retrieval + contradiction-detection metrics. Pure stdlib, deterministic, unit-tested.

Two families, because the system has two jobs and they fail differently:

  RETRIEVAL  — did the passages that actually bear on the claim get pulled and
               ranked near the top?  MRR, NDCG@k, precision@k, recall@k over the
               set of ground-truth "relevant" passage ids.

  CONTRADICTION — of the passages the judge FLAGGED as contradicting the claim,
               how many really do (precision), how many of the real ones did it
               catch (recall), and how often did it cry wolf (false-positive
               rate)?  Measured at a confidence threshold so the README can show
               exactly where the detector breaks.

Ground-truth labels live in the golden set and are NEVER shown to the system —
the harness reads them, the retriever/judge do not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# --- Retrieval metrics ------------------------------------------------------

def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant id (1-indexed). 0 if none retrieved."""
    for i, pid in enumerate(ranked_ids, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for pid in topk if pid in relevant) / len(topk)


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & relevant) / len(relevant)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance NDCG@k. 1.0 = all relevant ids ranked first, in any order."""
    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 if pid in relevant else 0.0) / math.log2(pos + 1)
            for pos, pid in enumerate(ids[:k], start=1)
        )
    actual = dcg(ranked_ids)
    # Ideal: every relevant id packed into the top positions.
    ideal_ids = list(relevant) + [pid for pid in ranked_ids if pid not in relevant]
    ideal = dcg(ideal_ids)
    return actual / ideal if ideal > 0 else 0.0


# --- Contradiction-detection metrics ----------------------------------------

@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    tp: int
    fp: int
    fn: int
    threshold: float


def contradiction_prf(
    flagged: dict[str, float],   # passage_id -> judge confidence it contradicts
    truth: set[str],             # passage_ids that genuinely contradict the claim
    considered: set[str],        # all passage_ids the judge actually looked at
    threshold: float,
) -> PRF:
    """Precision / recall / F1 / false-positive-rate at a confidence threshold.

    A passage counts as "called a contradiction" if its confidence >= threshold.
    FPR is over the considered non-contradictions, so it answers "of the passages
    that do NOT contradict the claim, how many did we wrongly flag?" — the metric
    that exposes crying-wolf on arbitrary input.
    """
    called = {pid for pid, c in flagged.items() if c >= threshold and pid in considered}
    truth = truth & considered
    negatives = considered - truth

    tp = len(called & truth)
    fp = len(called - truth)
    fn = len(truth - called)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / len(negatives) if negatives else 0.0
    return PRF(round(precision, 3), round(recall, 3), round(f1, 3), round(fpr, 3),
               tp, fp, fn, threshold)


def calibration_bins(
    points: list[tuple[float, bool]],  # (confidence, was_actually_a_contradiction)
    n_bins: int = 5,
) -> list[dict]:
    """Reliability curve: bucket predictions by confidence, compare mean confidence
    to the actual hit rate in each bucket. A well-calibrated judge has the two close.
    """
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        # last bin is closed on the right so confidence == 1.0 lands somewhere
        in_bin = [p for p in points if (lo <= p[0] < hi) or (b == n_bins - 1 and p[0] == 1.0)]
        if not in_bin:
            bins.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0, "mean_confidence": None, "actual_rate": None})
            continue
        mean_conf = sum(c for c, _ in in_bin) / len(in_bin)
        actual = sum(1 for _, hit in in_bin if hit) / len(in_bin)
        bins.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": len(in_bin),
            "mean_confidence": round(mean_conf, 3),
            "actual_rate": round(actual, 3),
        })
    return bins


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
