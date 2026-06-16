# Coherence Keeper — working notes

Retrieval + grounded contradiction surfacing, built eval-first. Give it a claim; it
finds passages that contradict it, with citations + confidence, for a human to judge.

## Invariants (do not break)
- **Eval before logic.** The golden set + harness are the fixed yardstick. Labels in
  `data/golden/` are NEVER passed to a retriever or judge — only `eval.py` reads them.
- **Surfaces, never pronounces.** The tool returns candidate contradictions for a human
  to judge. Don't add a feature that auto-decides truth.
- **Numbers, not adjectives.** Every capability claim in the README must trace to a
  `keeper eval` number. Report the false-positive rate honestly.
- **The eval is CI-gated** (.github/workflows/eval.yml): pytest + `keeper eval
  --min-precision`. Raise the gate as the backend improves; never lower it to pass.

## Layout
- `coherence_keeper/metrics.py` — retrieval + contradiction metric math (unit-tested)
- `coherence_keeper/eval.py` — the harness (reads labels; scores a system)
- `coherence_keeper/baseline.py` — Day-1 weak baseline (the floor to beat)
- `coherence_keeper/__main__.py` — CLI: `eval`, `check`
- `data/corpus/`, `data/golden/` — synthetic corpus + held-out planted labels
- `tests/` — deterministic metric tests, no model/network

## Next
Dense + hybrid retrieval + cross-encoder rerank + LLM-judge; Cohere Embed/Rerank adapter
alongside a local default. The interface is in `baseline.py`: retrieve(claim, passages, k)
and judge(claim, passages_by_id). Swap the backend; keep the eval fixed.
