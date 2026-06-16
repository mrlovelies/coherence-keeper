"""Replay tests for the committed prediction artifacts.

WHAT THESE GUARD: metric-code regressions and corruption/tampering of the
committed prediction files. If the metric core changes or a results file is
edited, the recomputed numbers stop matching the recorded ones and CI fails —
so the README's numbers, the committed artifact, and the scoring code can't
silently drift apart.

WHAT THESE DO **NOT** PROVE: that the live model still hits these numbers. The
predictions are recorded model outputs, not a fresh call — replaying good
outputs scores well by construction. Re-proving the model means regenerating
with a key (`eval --retriever cohere --judge llm`), which CI cannot do offline.
The artifacts exist so a stranger can verify the *scoring is honest* (labels held
out in data/golden, F1 correctly derived from predictions) without a key — not to
stand in for the model.
"""
from __future__ import annotations

import json
from pathlib import Path

from coherence_keeper import replay

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "greek-myth.jsonl"
GOLDEN = ROOT / "data" / "golden" / "claims.jsonl"
RESULTS = ROOT / "data" / "results"


def _recompute(name: str) -> tuple[dict, dict]:
    path = RESULTS / name
    recorded = json.loads(path.read_text())["metrics"]
    got = replay.replay(CORPUS, GOLDEN, path)
    return got, recorded


def test_baseline_artifact_reproduces():
    got, recorded = _recompute("baseline.json")
    assert got["contradiction"] == recorded["contradiction"]
    assert got["retrieval"] == recorded["retrieval"]
    assert got["contradiction"]["f1"] == 0.5


def test_cohere_artifact_reproduces():
    got, recorded = _recompute("cohere-llm.json")
    assert got["contradiction"] == recorded["contradiction"]
    assert got["retrieval"] == recorded["retrieval"]
    assert got["contradiction"]["f1"] == 1.0
    assert got["contradiction"]["false_positive_rate"] == 0.0


def test_recorded_lift_is_real_in_the_artifacts():
    # The committed predictions encode a genuine baseline -> Cohere lift, scored
    # by the same harness against the same held-out labels.
    base = replay.replay(CORPUS, GOLDEN, RESULTS / "baseline.json")
    coh = replay.replay(CORPUS, GOLDEN, RESULTS / "cohere-llm.json")
    assert coh["contradiction"]["f1"] > base["contradiction"]["f1"]
