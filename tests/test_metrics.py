"""Deterministic tests for the metric math — no model, no network. These are the
load-bearing correctness guarantees: if MRR/NDCG/PRF are wrong, every reported
number is wrong, so they're pinned to hand-worked examples."""
import math

from coherence_keeper import metrics


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert metrics.reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_precision_at_k():
    assert metrics.precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5
    assert metrics.precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    assert metrics.precision_at_k([], {"a"}, 5) == 0.0


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5
    assert metrics.recall_at_k(["a"], {"a"}, 1) == 1.0
    assert metrics.recall_at_k(["a"], set(), 1) == 0.0


def test_ndcg_perfect_and_worst():
    # relevant item first -> perfect
    assert metrics.ndcg_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    # single relevant item at position 2: DCG = 1/log2(3), IDCG = 1/log2(2)=1
    got = metrics.ndcg_at_k(["x", "a", "y"], {"a"}, 3)
    assert math.isclose(got, (1 / math.log2(3)) / 1.0, rel_tol=1e-9)
    # no relevant retrieved
    assert metrics.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_contradiction_prf_basic():
    flagged = {"a": 0.9, "b": 0.8, "c": 0.2}
    truth = {"a"}
    considered = {"a", "b", "c"}
    prf = metrics.contradiction_prf(flagged, truth, considered, threshold=0.5)
    # called = {a, b}; tp={a}=1, fp={b}=1, fn=0
    assert prf.tp == 1 and prf.fp == 1 and prf.fn == 0
    assert prf.precision == 0.5
    assert prf.recall == 1.0
    # negatives = considered - truth = {b, c}; fp among them = {b} -> 1/2
    assert prf.false_positive_rate == 0.5


def test_contradiction_prf_threshold_excludes():
    flagged = {"a": 0.4}
    prf = metrics.contradiction_prf(flagged, {"a"}, {"a"}, threshold=0.5)
    # nothing called -> recall 0, precision 0
    assert prf.tp == 0 and prf.fn == 1
    assert prf.recall == 0.0


def test_calibration_bins_counts():
    points = [(0.9, True), (0.85, True), (0.1, False), (0.2, False)]
    bins = metrics.calibration_bins(points, n_bins=5)
    top = bins[-1]      # 0.8-1.0
    assert top["n"] == 2 and top["actual_rate"] == 1.0
    bottom = bins[0]    # 0.0-0.2
    assert bottom["n"] == 1 and bottom["actual_rate"] == 0.0
