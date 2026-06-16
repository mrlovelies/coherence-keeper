"""CLI: `python -m coherence_keeper eval|check`.

Day 1 ships the eval harness and a deliberately-weak baseline backend, so the
numbers are real (and honestly low) before the real retrieval lands.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import baseline, eval as eval_mod
from .corpus import load_corpus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "corpus" / "greek-myth.jsonl"
DEFAULT_GOLDEN = ROOT / "data" / "golden" / "claims.jsonl"


def _build_system(retriever: str, judge_name: str):
    """Return (retrieve_fn, judge_fn) for the chosen backends.
    baseline = offline, no deps (CI default). cohere/llm = the real stack."""
    if retriever == "cohere":
        from .cohere_backend import CohereBackend
        retrieve_fn = CohereBackend().retrieve
    else:
        retrieve_fn = baseline.retrieve
    if judge_name == "llm":
        from . import judge_llm
        judge_fn = judge_llm.judge
    else:
        judge_fn = baseline.judge
    return retrieve_fn, judge_fn


def main(argv=None):
    ap = argparse.ArgumentParser(prog="coherence_keeper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="run the planted-contradiction eval")
    pe.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    pe.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    pe.add_argument("--retriever", choices=["baseline", "cohere"], default="baseline")
    pe.add_argument("--judge", choices=["baseline", "llm"], default="baseline")
    pe.add_argument("--k", type=int, default=5)
    pe.add_argument("--threshold", type=float, default=0.5)
    pe.add_argument("--min-precision", type=float, default=None,
                    help="CI gate: exit non-zero if contradiction precision is below this")
    pe.add_argument("--save", type=Path, default=None,
                    help="record real-backend predictions to a committed JSON artifact")
    pe.add_argument("--from-cache", type=Path, default=None,
                    help="replay saved predictions offline (no key) — reproduces the metrics")

    pc = sub.add_parser("check", help="surface passages that contradict a claim")
    pc.add_argument("claim")
    pc.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    pc.add_argument("--retriever", choices=["baseline", "cohere"], default="baseline")
    pc.add_argument("--judge", choices=["baseline", "llm"], default="baseline")
    pc.add_argument("--k", type=int, default=5)
    pc.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args(argv)

    if args.cmd == "eval":
        if args.from_cache is not None:
            from . import replay
            result = replay.replay(args.corpus, args.golden, args.from_cache,
                                   k=args.k, threshold=args.threshold)
        else:
            retrieve_fn, judge_fn = _build_system(args.retriever, args.judge)
            if args.save is not None:
                from . import replay
                meta = {"retriever": args.retriever, "judge": args.judge,
                        "k": args.k, "threshold": args.threshold}
                result = replay.run_and_save(args.corpus, args.golden, retrieve_fn,
                                             judge_fn, args.save, k=args.k,
                                             threshold=args.threshold, meta=meta)
            else:
                result = eval_mod.run_eval(args.corpus, args.golden, retrieve_fn,
                                           judge_fn, k=args.k, threshold=args.threshold)
        print(eval_mod.format_report(result))
        if args.min_precision is not None:
            prec = result["contradiction"]["precision"]
            if prec < args.min_precision:
                print(f"\nFAIL: contradiction precision {prec} < gate {args.min_precision}")
                return 1
        return 0

    if args.cmd == "check":
        retrieve_fn, judge_fn = _build_system(args.retriever, args.judge)
        passages = load_corpus(args.corpus)
        by_id = {p.id: p for p in passages}
        ranked = retrieve_fn(args.claim, passages, args.k)
        flagged = judge_fn(args.claim, {pid: by_id[pid] for pid in ranked})
        hits = sorted(((c, pid) for pid, c in flagged.items() if c >= args.threshold),
                      reverse=True)
        print(f'claim: "{args.claim}"\n')
        if not hits:
            print("No contradicting passages surfaced above the confidence threshold.")
            print("(Day-1 baseline backend — weak by design; real retrieval lands next.)")
            return 0
        print("Possible contradictions (for a human to judge):")
        for conf, pid in hits:
            print(f"  [{conf:.2f}] {pid} · {by_id[pid].source}")
            print(f"        {by_id[pid].text}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
