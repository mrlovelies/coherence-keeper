"""Record real-backend predictions to a committed artifact, and replay them
offline so a stranger reproduces the headline numbers with NO API key.

A real run (Cohere Embed/Rerank + the LLM judge) is non-deterministic and needs
secrets. The eval *metrics*, however, are a pure function of (retrieved ids,
judge confidences, golden labels). So we record the system's raw outputs once,
commit them, and `eval --from-cache` recomputes the identical metrics through the
exact same harness with zero deps. The golden labels still live only in
data/golden and are never written into the prediction file — so the eval stays
honest on replay: it scores the recorded predictions against held-out labels,
it does not replay the labels back to itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import eval as eval_mod


def run_and_save(corpus_path, golden_path, retrieve, judge, save_path, *,
                 k: int = 5, threshold: float = 0.5, meta: dict | None = None) -> dict:
    """Run the real (retrieve, judge) system, capturing each claim's raw outputs,
    and write them + the resulting metrics to ``save_path``. Returns the eval result."""
    records: dict[str, dict] = {}

    def cap_retrieve(claim, passages, kk):
        ranked = retrieve(claim, passages, kk)
        records.setdefault(claim, {})["retrieved"] = list(ranked)
        return ranked

    def cap_judge(claim, cand):
        flagged = judge(claim, cand)
        records.setdefault(claim, {})["flagged"] = {pid: float(c) for pid, c in flagged.items()}
        return flagged

    result = eval_mod.run_eval(corpus_path, golden_path, cap_retrieve, cap_judge,
                               k=k, threshold=threshold)
    payload = {
        "meta": meta or {},
        "k": k,
        "threshold": threshold,
        "metrics": {
            "retrieval": result["retrieval"],
            "contradiction": result["contradiction"],
        },
        "predictions": records,
    }
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return result


def replay(corpus_path, golden_path, predictions_path, *,
           k: int | None = None, threshold: float | None = None) -> dict:
    """Recompute the eval metrics offline from committed predictions. No key, no
    model, no deps — the recorded (retrieved, flagged) outputs flow through the
    same harness and the labels are read fresh from data/golden."""
    data = json.loads(Path(predictions_path).read_text())
    preds = data["predictions"]
    kk = data["k"] if k is None else k
    th = data["threshold"] if threshold is None else threshold

    def cached_retrieve(claim, passages, k_):
        return list(preds[claim]["retrieved"])

    def cached_judge(claim, cand):
        return {pid: float(c) for pid, c in preds[claim]["flagged"].items()}

    return eval_mod.run_eval(corpus_path, golden_path, cached_retrieve, cached_judge,
                             k=kk, threshold=th)
