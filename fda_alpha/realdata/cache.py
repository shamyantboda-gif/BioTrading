"""
Dead-simple on-disk cache for API responses.

The free endpoints are rate-limited (Yahoo 429s aggressively, openFDA throttles
per-IP). Caching turns a reproducible research run from "hammer three APIs for
two minutes and maybe get blocked" into "hit the network once, then read local
files forever". The cache is content-addressed by a caller-supplied key and
lives under ``data/cache/`` (git-ignored).

Two payload kinds:

* ``json``  — arbitrary JSON from openFDA / CT.gov / Yahoo chart endpoints.
* ``frame`` — a parsed price DataFrame, stored as Parquet when available and
  CSV otherwise, so a rerun does not re-parse Yahoo's chart JSON.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Repo-root/data/cache. __file__ is fda_alpha/realdata/cache.py -> parents[2].
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _key_to_path(key: str, suffix: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}{suffix}"


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_json(key: str) -> Any | None:
    p = _key_to_path(key, ".json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def put_json(key: str, value: Any) -> None:
    _ensure_dir()
    _key_to_path(key, ".json").write_text(
        json.dumps(value, default=str), encoding="utf-8"
    )


def get_frame(key: str) -> pd.DataFrame | None:
    parq = _key_to_path(key, ".parquet")
    if parq.exists():
        try:
            return pd.read_parquet(parq)
        except Exception:  # pragma: no cover - missing pyarrow, fall through
            pass
    csv = _key_to_path(key, ".csv")
    if csv.exists():
        df = pd.read_csv(csv, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    return None


def put_frame(key: str, df: pd.DataFrame) -> None:
    _ensure_dir()
    try:
        df.to_parquet(_key_to_path(key, ".parquet"))
    except Exception:  # pragma: no cover - no parquet engine installed
        df.to_csv(_key_to_path(key, ".csv"))


def clear() -> int:
    """Delete every cached file. Returns the number removed."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.iterdir():
        if p.is_file():
            p.unlink()
            n += 1
    return n
