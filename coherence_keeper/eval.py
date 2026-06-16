"""The eval harness. Runs a (retrieve, judge) system over the golden set and
reports retrieval + contradiction metrics. The labels are read HERE and never
handed to the system.

`run_eval` takes the system as two callables so any backend (baseline today,
dense+rerank+LLM-judge next) plugs in unchanged — the eval is the fixed yardstick.
"""
from __future__ import annotations

from typing import Callable

from . import metrics
from .corpus import Claim, Passage, load_corpus, load_golden

RetrieveFn = Callable[[str, list[Passage], int], list[str]]
JudgeFn = Callable[[str, dict[str, Passage]], dict[str, float]]


def run_eval(
    corpus_path,
    golden_path,
    retrieve: RetrieveFn,
    judge: JudgeFn,
    *,
    k: int = 5,
    threshold: float = 0.5,
) -> dict:
    passages = load_corpus(corpus_path)
    by_id = {p.id: p for p in passages}
    claims = load_golden(golden_path)

    rr, ndcg, p_at_k, r_at_k = [], [], [], []
    prfs: list[metrics.PRF] = []
    calib_points: list[tuple[float, bool]] = []
    per_claim = []

    for c in claims:
        ranked = retrieve(c.claim, passages, k)          # system sees claim + passages only
        considered = set(ranked)
        flagged = judge(c.claim, {pid: by_id[pid] for pid in ranked})

        # Retrieval: did we surface the genuine contradiction(s)?
        rr.append(metrics.reciprocal_rank(ranked, c.contradicts))
        ndcg.append(metrics.ndcg_at_k(ranked, c.contradicts, k))
        p_at_k.append(metrics.precision_at_k(ranked, c.contradicts, k))
        r_at_k.append(metrics.recall_at_k(ranked, c.contradicts, k))

        # Contradiction: of what we flagged, how much was real?
        prf = metrics.contradiction_prf(flagged, c.contradicts, considered, threshold)
        prfs.append(prf)

        for pid, conf in flagged.items():
            calib_points.append((conf, pid in c.contradicts))

        per_claim.append({
            "id": c.id, "claim": c.claim,
            "retrieved": ranked,
            "true_contradictions": sorted(c.contradicts),
            "flagged": {pid: conf for pid, conf in flagged.items() if conf >= threshold},
            "precision": prf.precision, "recall": prf.recall, "fpr": prf.false_positive_rate,
        })

    return {
        "n_claims": len(claims),
        "k": k,
        "threshold": threshold,
        "retrieval": {
            "mrr": round(metrics.mean(rr), 3),
            f"ndcg@{k}": round(metrics.mean(ndcg), 3),
            f"precision@{k}": round(metrics.mean(p_at_k), 3),
            f"recall@{k}": round(metrics.mean(r_at_k), 3),
        },
        "contradiction": {
            "precision": round(metrics.mean([p.precision for p in prfs]), 3),
            "recall": round(metrics.mean([p.recall for p in prfs]), 3),
            "f1": round(metrics.mean([p.f1 for p in prfs]), 3),
            "false_positive_rate": round(metrics.mean([p.false_positive_rate for p in prfs]), 3),
        },
        "calibration": metrics.calibration_bins(calib_points),
        "per_claim": per_claim,
    }


def format_report(result: dict) -> str:
    k = result["k"]
    r, c = result["retrieval"], result["contradiction"]
    lines = [
        "Coherence Keeper — eval report",
        "=" * 40,
        f"claims: {result['n_claims']}   k: {k}   threshold: {result['threshold']}",
        "",
        "RETRIEVAL  (did the contradicting passage surface near the top?)",
        f"  MRR             {r['mrr']}",
        f"  NDCG@{k}          {r[f'ndcg@{k}']}",
        f"  precision@{k}     {r[f'precision@{k}']}",
        f"  recall@{k}        {r[f'recall@{k}']}",
        "",
        "CONTRADICTION  (of what we flagged, how much was real?)",
        f"  precision       {c['precision']}",
        f"  recall          {c['recall']}",
        f"  F1              {c['f1']}",
        f"  false-pos rate  {c['false_positive_rate']}   (crying wolf — lower is better)",
        "",
        "CALIBRATION  (mean confidence vs actual hit-rate per bucket)",
    ]
    for b in result["calibration"]:
        if b["n"] == 0:
            continue
        lines.append(f"  {b['range']}  n={b['n']:<3} conf={b['mean_confidence']}  actual={b['actual_rate']}")
    return "\n".join(lines)
