"""Minimal .env loader — no dependency. Reads KEY=VALUE lines from a .env at the
repo root into os.environ (without overwriting anything already set). Secrets stay
in the gitignored .env; nothing here ever prints a value."""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | Path | None = None) -> None:
    p = Path(path) if path else _ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def has_cohere_key() -> bool:
    load_env()
    return bool(os.environ.get("COHERE_API_KEY"))
