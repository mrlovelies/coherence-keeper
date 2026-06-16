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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="coherence_keeper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="run the planted-contradiction eval")
    pe.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    pe.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    pe.add_argument("--k", type=int, default=5)
    pe.add_argument("--threshold", type=float, default=0.5)
    pe.add_argument("--min-precision", type=float, default=None,
                    help="CI gate: exit non-zero if contradiction precision is below this")

    pc = sub.add_parser("check", help="surface passages that contradict a claim")
    pc.add_argument("claim")
    pc.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    pc.add_argument("--k", type=int, default=5)
    pc.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args(argv)

    if args.cmd == "eval":
        result = eval_mod.run_eval(args.corpus, args.golden, baseline.retrieve,
                                   baseline.judge, k=args.k, threshold=args.threshold)
        print(eval_mod.format_report(result))
        if args.min_precision is not None:
            prec = result["contradiction"]["precision"]
            if prec < args.min_precision:
                print(f"\nFAIL: contradiction precision {prec} < gate {args.min_precision}")
                return 1
        return 0

    if args.cmd == "check":
        passages = load_corpus(args.corpus)
        by_id = {p.id: p for p in passages}
        ranked = baseline.retrieve(args.claim, passages, args.k)
        flagged = baseline.judge(args.claim, {pid: by_id[pid] for pid in ranked})
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
