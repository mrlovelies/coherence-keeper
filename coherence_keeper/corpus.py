"""Load the corpus and the golden set. JSONL with `#` comment lines."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Passage:
    id: str
    text: str
    source: str = "myth-corpus"


@dataclass
class Claim:
    id: str
    claim: str
    contradicts: set[str] = field(default_factory=set)
    supports: set[str] = field(default_factory=set)
    related: set[str] = field(default_factory=set)


def _rows(path: str | Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        yield json.loads(line)


def load_corpus(path: str | Path) -> list[Passage]:
    return [Passage(id=r["id"], text=r["text"], source=r.get("source", "myth-corpus"))
            for r in _rows(path)]


def load_golden(path: str | Path) -> list[Claim]:
    """Loads ground truth. Callers MUST NOT pass these label fields to a retriever
    or judge — only the harness is allowed to see them."""
    out = []
    for r in _rows(path):
        out.append(Claim(
            id=r["id"], claim=r["claim"],
            contradicts=set(r.get("contradicts", [])),
            supports=set(r.get("supports", [])),
            related=set(r.get("related", [])),
        ))
    return out
