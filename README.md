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

The build is **CI-gated**: GitHub Actions runs the unit tests and the eval, and fails if
contradiction precision drops below the threshold (the DeepEval-in-CI pattern). A
regression blocks the merge.

### Where it stands today (honest numbers, weak baseline)

Day 1 ships the eval harness plus a **deliberately weak baseline** — lexical
token-overlap retrieval and a crude negation-cue judge — so there's a real floor to beat.
On the 7-claim Greek-myth set (`keeper eval`):

```
RETRIEVAL       MRR 0.60 · NDCG@5 0.70 · recall@5 1.00   (it finds the contradictions)
CONTRADICTION   precision 0.57 · recall 0.57 · FPR 0.00  (the weak judge misses ~half, but never cries wolf)
```

`recall@5 = 1.0` says the contradicting passage is always retrieved; `contradiction
recall 0.57` says the baseline *judge* only confirms about half of them. That gap is
exactly what the real backend closes next — and the eval will show the lift in the same
numbers.

## How it works

```
claim ─▶ retrieve (lexical today; dense + hybrid next) ─▶ rerank (cross-encoder, next)
      ─▶ judge: does this passage contradict the claim?  ─▶ ranked contradictions + citations + confidence
                          │                                          ▲
                  the human reads the harness's            held-out planted labels
                  honest numbers, then decides
```

The judge returns a **calibrated confidence**, not a binary oracle — paraphrase, hedging,
and temporal qualifiers make flat yes/no answers wrong, so the threshold is shown and
tunable.

## Status & honest scope

- **Day 1 (done):** eval harness, planted-contradiction golden set, deterministic metric
  core with unit tests, weak baseline backend, CLI, CI gate.
- **Next:** dense retrieval + hybrid (BM25 + embeddings) + cross-encoder rerank, and an
  LLM-judge contradiction step — with a **Cohere Embed + Rerank** adapter alongside the
  local default, so retrieval can run on a production reranking stack. The eval stays the
  fixed yardstick; only the backend swaps.

Built as a focused project with AI assistance. The design calls are the point: plant the
failures, strip the answer key, gate the build, and keep a human as the judge.

## Usage

```bash
python -m coherence_keeper eval                 # run the planted-contradiction eval
python -m coherence_keeper eval --min-precision 0.5   # CI gate
python -m coherence_keeper check "<your claim>"  # surface contradicting passages
```

No dependencies to run the eval or the baseline (standard library only).

## License

MIT — see [LICENSE](LICENSE).
