# Coherence Keeper

[![eval](https://github.com/mrlovelies/coherence-keeper/actions/workflows/eval.yml/badge.svg)](https://github.com/mrlovelies/coherence-keeper/actions/workflows/eval.yml)
&nbsp;[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A second pair of eyes for a large body of writing. Give it a claim, and it finds the
passages that **contradict** it — with a citation back to the source and a confidence
score — so a human can catch where a corpus disagrees with itself. Lore bibles, a year
of notes, product docs, a knowledge base: anywhere there's too much text to hold in one
head.

It **surfaces candidates for a human to judge** — it never pronounces a verdict. That
line is the whole design stance: the tool brings you the evidence, you stay the author.

```
$ keeper check "Theseus killed the Minotaur inside the Labyrinth."

Possible contradictions (for a human to judge):
  [0.71] p21 · myth-corpus
        The Minotaur was never slain at all; it died at last of old age,
        and Theseus never once set foot in the Labyrinth.
```

## Eval first — this is the point

This repo was built **eval before logic**. Before any retrieval code existed, there was
a golden set with *planted* contradictions and a harness that scores how well the system
finds them. The reason: for a tool whose job is "tell me where my corpus disagrees with
itself," the only question that matters is *does it actually work, and where does it
break* — and the honest answer is a number, not an adjective.

The eval (`keeper eval`) plants, into a synthetic corpus, passages that deliberately
contradict seeded claims — alongside on-topic-but-consistent passages that are precision
traps. The ground-truth labels live in `data/golden/` and are **never shown to the
retriever or the judge** (the harness strips them). It reports:

- **Retrieval** — MRR, NDCG@k, precision@k, recall@k: did the contradicting passage even
  surface near the top?
- **Contradiction** — precision, recall, F1, and **false-positive rate**: of what the
  judge flagged, how much really contradicts the claim — and how often it cried wolf.
- **Calibration** — mean confidence vs. actual hit-rate per bucket.

The build is **CI-gated**: GitHub Actions runs the unit tests, the offline baseline eval
(failing if contradiction precision drops below the threshold — the DeepEval-in-CI
pattern), and a replay of the committed real-run predictions to confirm the scoring still
reproduces the reported numbers. A regression in the code or the metric blocks the merge.
(CI can't call the model with no key, so it verifies the *scoring*, not a fresh model run
— regenerating with a key is how you re-prove performance.)

### Results — the lift, measured by the same eval

The eval is the fixed yardstick; the backend swaps. On the 10-claim Greek-myth set
(7 flat contradictions + 3 subtler ones — a number swap, a partial-detail change, an
attribute swap), with on-topic *supporting* passages planted as precision traps:

| backend | contradiction P / R / F1 | false-pos rate | retrieval MRR / NDCG@5 / recall@5 |
|---|---|---|---|
| baseline (offline: lexical + negation-cue) | 0.50 / 0.50 / 0.50 | 0.00 | 0.57 / 0.68 / 1.00 |
| **Cohere Embed + Rerank · LLM judge** | **1.00 / 1.00 / 1.00** | 0.00 | 0.48 / 0.62 / 1.00 |

Read honestly, because the interesting parts aren't the headline number:

- **The lift is the point.** Same eval, same corpus: swapping the weak baseline for real
  retrieval + an LLM judge takes contradiction F1 from 0.50 → 1.00. The subtler cases
  (twelve-vs-ten labours, gold-vs-silver) are exactly where the negation-cue baseline
  fails and the real judge holds.
- **Retrieval is honestly imperfect — and that's the architecture.** `recall@5 = 1.0`
  means the contradicting passage is always retrieved, but `MRR ≈ 0.48` means it's often
  *not* ranked first: the reranker puts the *supporting* passage above the contradicting
  one (it matches the claim's wording more closely). The system works by retrieving
  broadly and letting the judge separate support from contradiction — which is why the
  false-positive rate is 0.0 with supporting passages sitting right there in the results.
- **Calibrated, not lucky.** Non-contradictions get mean confidence 0.015; real
  contradictions 0.95 — a cleanly bimodal judge, threshold shown and tunable.
- **Small and clear by design.** This is 10 clear-to-moderate cases; a 1.0 means "clears
  the current bar," not "solved." The harness exists precisely to hold the line as harder
  cases (partial truths, temporal qualifiers) and a bigger corpus get added — raising the
  bar is the ongoing work, and the number will (correctly) drop when it gets hard enough.

**Reproduce both rows yourself — no key required.** The real run's per-claim predictions
are committed, so the harness recomputes the metrics offline (the labels are still read
fresh from `data/golden/`, never from the prediction file — it scores recorded predictions
against held-out labels, it doesn't replay the answers back to itself):

```bash
python -m coherence_keeper eval --from-cache data/results/baseline.json    # → F1 0.50
python -m coherence_keeper eval --from-cache data/results/cohere-llm.json   # → F1 1.00
```

Only the one run's predictions are committed here, but in local testing the Cohere+judge F1
held at 1.00 across repeated runs. To re-prove that yourself, regenerate from scratch with
`keeper eval --retriever cohere --judge llm` (needs `COHERE_API_KEY` + the `claude` CLI).

What the cache replay does and doesn't prove, stated honestly: it lets anyone verify the
**scoring is honest** (labels held out, F1 correctly derived from predictions) without a
key — it is *not* a re-run of the model, so it can't prove the model still performs (that
needs a key). CI replays the committed predictions only to catch metric-code drift or
artifact tampering, not as a stand-in for the model.

## How it works

```
claim ─▶ dense retrieve (Cohere Embed, cosine) ─▶ rerank (Cohere Rerank cross-encoder)
      ─▶ judge: does this passage contradict the claim? (LLM, calibrated confidence)
      ─▶ ranked contradictions + citations + confidence
                          │                                          ▲
                  the human reads the harness's            held-out planted labels
                  honest numbers, then decides             (never shown to the system)
```

Retrieval runs on **Cohere's production stack** — `embed-english-v3.0` for dense first-stage
ranking, `rerank-english-v3.0` (a real cross-encoder) to reorder the top candidates. Doc
embeddings are cached on disk so repeated eval runs stay well under the trial key's limits.
The contradiction judge returns a **calibrated confidence**, not a binary oracle —
paraphrase, hedging, and temporal qualifiers make flat yes/no answers wrong, so the
threshold is shown and tunable. The backend is swappable behind one interface
(`retrieve` / `judge`); an offline `baseline` (no key, no deps) is the floor and what CI runs.

## Status & honest scope

- **Eval + harness (done):** planted-contradiction golden set, deterministic metric core
  with unit tests, CLI, CI gate. Built first, before any retrieval logic.
- **Real backend (done):** Cohere Embed + Rerank retrieval and an LLM contradiction judge
  (via the `claude` CLI) — `--retriever cohere --judge llm`. Numbers above.
- **Next:** harder/subtler planted cases and a larger corpus (to push the contradiction
  number off 1.0 where it should be), a local-embeddings backend so the real path runs
  with no API key, and per-call observability surfaced in the README.

Built as a focused project with AI assistance. The design calls are the point: plant the
failures, strip the answer key, gate the build, and keep a human as the judge.

## Usage

```bash
# Offline baseline — no key, no deps, standard library only (this is what CI runs):
python -m coherence_keeper eval
python -m coherence_keeper check "<your claim>"

# Real backend — Cohere Embed + Rerank retrieval + LLM judge:
#   pip install cohere   and put COHERE_API_KEY in .env (see .env.example),
#   plus the `claude` CLI on PATH for the judge.
python -m coherence_keeper eval  --retriever cohere --judge llm
python -m coherence_keeper check "Everything King Midas touched turned to gold." --retriever cohere --judge llm
```

## License

MIT — see [LICENSE](LICENSE).
