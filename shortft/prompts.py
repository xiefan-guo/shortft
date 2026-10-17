"""Prompt dataset loading for the .jsonl files produced by scripts/prepare_data.py.

Records are dicts with at least a ``prompt`` key; ``category`` / ``source`` /
``uid`` are preserved when present.

The HPDv2 eval file is written grouped by category (all anime, then all
concept-art, ...), so taking the first N prompts would silently evaluate on a
single category.  Subsampling is therefore stratified by default.
"""
from __future__ import annotations

import json
import os
import random


def load_prompt_records(path: str, limit: int | None = None, seed: int = 0,
                        stratify: bool = True) -> list[dict]:
    """Read a .jsonl prompt file into a list of record dicts.

    Args:
        path: one JSON object per line; a bare string per line also works.
        limit: keep at most this many records.
        seed: subsample RNG seed, fixed by default so the subset is reproducible.
        stratify: subsample proportionally across ``category`` when every record
            has one.  Set False for a plain random subsample.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt file not found: {path}")

    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = line
            if isinstance(obj, str):
                obj = {"prompt": obj}
            elif not isinstance(obj, dict):
                obj = {"prompt": str(obj)}
            elif "prompt" not in obj and obj:
                obj = {"prompt": str(next(iter(obj.values())))}
            if obj.get("prompt"):
                records.append(obj)

    if limit is not None and limit < len(records):
        records = _subsample(records, limit, seed, stratify)
    return records


def _subsample(records: list[dict], limit: int, seed: int,
               stratify: bool) -> list[dict]:
    rng = random.Random(seed)
    if not (stratify and all("category" in r for r in records)):
        return rng.sample(records, limit)

    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    cats = sorted(by_cat)

    # Largest-remainder allocation so the subset mirrors the category mix.
    exact = [limit * len(by_cat[c]) / len(records) for c in cats]
    counts = [int(e) for e in exact]
    for i in sorted(range(len(cats)), key=lambda i: exact[i] - counts[i],
                    reverse=True)[: limit - sum(counts)]:
        counts[i] += 1

    out = []
    for c, n in zip(cats, counts):
        out += rng.sample(by_cat[c], min(n, len(by_cat[c])))
    rng.shuffle(out)
    return out


def load_prompts(path: str, limit: int | None = None, seed: int = 0) -> list[str]:
    """Convenience wrapper returning just the prompt strings."""
    return [r["prompt"] for r in load_prompt_records(path, limit=limit, seed=seed)]


def categories_of(records: list[dict]) -> list[str]:
    """Per-record category, defaulting to ``"all"`` when the file has none."""
    return [r.get("category", "all") for r in records]
