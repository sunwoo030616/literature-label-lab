"""Paths and JSON helpers; no model loading at import time."""

import hashlib
import json
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
DB = V4 / "data/corpus.sqlite3"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
