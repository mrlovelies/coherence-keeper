"""LLM contradiction judge via the `claude` CLI (headless, free on a subscription).

Given a claim and the retrieved passages, it returns, per passage, a calibrated
confidence that the passage CONTRADICTS the claim — states something that cannot be
true at the same time as the claim. It is told NOT to flag passages that merely
discuss the same topic, or that support the claim. The harness, not the judge, owns
the threshold and the scoring; the judge only surfaces candidates with confidence.

One CLI call per claim (all retrieved passages in a single prompt). Falls back to a
clear error if `claude` isn't on PATH — use the baseline judge offline.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

from .corpus import Passage

_SYSTEM = (
    "You are a careful fact-checker. You are given a CLAIM and several PASSAGES. "
    "For each passage decide whether it CONTRADICTS the claim — asserts something "
    "that cannot be true at the same time as the claim. A passage that is merely on "
    "the same topic, or that SUPPORTS/agrees with the claim, does NOT contradict it. "
    "Be conservative: only call a contradiction when the two statements genuinely "
    "cannot both hold. Give a calibrated confidence in [0,1] that it contradicts."
)


def _prompt(claim: str, passages_by_id: dict[str, Passage]) -> str:
    items = [{"id": pid, "text": p.text} for pid, p in passages_by_id.items()]
    return (
        f'CLAIM: "{claim}"\n\n'
        f"PASSAGES (JSON):\n{json.dumps(items, ensure_ascii=False)}\n\n"
        "Reply with ONLY a JSON object mapping each passage id to "
        '{"contradicts": true|false, "confidence": 0.0-1.0}. '
        "confidence is your probability that the passage contradicts the claim "
        "(near 0 if it clearly does not, near 1 if it clearly does). No prose."
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    return json.loads(text[start:])


def judge(claim: str, passages_by_id: dict[str, Passage], *, model: str | None = None,
          timeout: int = 180) -> dict[str, float]:
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("`claude` CLI not found — use --judge baseline for offline.")
    full = f"{_SYSTEM}\n\n---\n\n{_prompt(claim, passages_by_id)}"
    cmd = [claude, "-p", full]
    if model:
        cmd += ["--model", model]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI exited {r.returncode}: {(r.stderr or '')[-300:]}")
    parsed = _extract_json(r.stdout or "")
    out: dict[str, float] = {}
    for pid in passages_by_id:
        entry = parsed.get(pid, {})
        conf = float(entry.get("confidence", 0.0)) if isinstance(entry, dict) else 0.0
        out[pid] = round(max(0.0, min(1.0, conf)), 3)
    return out
