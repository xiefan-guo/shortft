"""Build the ShortFT train / eval prompt sets from the raw HPDv2 download.

Inputs (fetched by scripts/download_assets.py into $SHORTFT_DATA_ROOT/raw/hpdv2):
    benchmark/{anime,concept-art,paintings,photo}.json   800 prompts each -> eval
    train.json                                           preference pairs -> train

Train and eval are kept disjoint: any prompt appearing in the benchmark split
is removed from the train split.

Outputs:
    train/hpdv2_train.jsonl    de-duplicated unique train prompts
    eval/hpdv2_test.jsonl      full benchmark split, labelled by category
    eval/hpdv2_test_400.jsonl  stratified 100-per-category subset, fixed seed
"""
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import (  # noqa: E402
    DATA_RAW_DIR,
    HPDV2_EVAL_PROMPTS,
    HPDV2_EVAL_PROMPTS_400,
    HPDV2_TRAIN_PROMPTS,
)

# Fixed order, so a prompt present in two categories gets a deterministic label.
CATEGORIES = ["anime", "concept-art", "paintings", "photo"]
SUBSET_PER_CATEGORY = 100
SUBSET_SEED = 20261004


def norm(prompt: str) -> str:
    """Canonical de-duplication key: collapse whitespace, strip."""
    return " ".join(prompt.split())


def load_raw(filename: str):
    with open(os.path.join(DATA_RAW_DIR, filename)) as f:
        return json.load(f)


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  -> {path}  {len(rows)} rows")


def main():
    eval_rows, seen_eval = [], {}
    for cat in CATEGORIES:
        prompts = load_raw(f"benchmark/{cat}.json")
        kept = 0
        for prompt in prompts:
            key = norm(prompt)
            if key in seen_eval:
                continue
            seen_eval[key] = cat
            eval_rows.append({
                "prompt": prompt,
                "category": cat,
                "source": "hpdv2_benchmark",
                "uid": hashlib.md5(key.encode()).hexdigest()[:12],
            })
            kept += 1
        print(f"[eval] {cat:12s} raw={len(prompts):4d}  kept={kept:4d}")

    rng = random.Random(SUBSET_SEED)
    by_cat: dict[str, list[dict]] = {}
    for row in eval_rows:
        by_cat.setdefault(row["category"], []).append(row)
    subset = []
    for cat in CATEGORIES:
        subset += rng.sample(by_cat[cat],
                             min(SUBSET_PER_CATEGORY, len(by_cat[cat])))
    rng.shuffle(subset)

    train_raw = load_raw("train.json")
    print(f"[train] {len(train_raw)} preference pairs")

    train_rows, seen_train = [], set()
    skipped_overlap = skipped_dup = 0
    for record in train_raw:
        prompt = record["prompt"]
        key = norm(prompt)
        if not key:
            continue
        if key in seen_eval:
            skipped_overlap += 1
            continue
        if key in seen_train:
            skipped_dup += 1
            continue
        seen_train.add(key)
        train_rows.append({
            "prompt": prompt,
            "source": "hpdv2_train",
            "uid": hashlib.md5(key.encode()).hexdigest()[:12],
        })
    print(f"[train] kept={len(train_rows)}  dup={skipped_dup}  "
          f"overlap-with-eval={skipped_overlap}")

    write_jsonl(HPDV2_TRAIN_PROMPTS, train_rows)
    write_jsonl(HPDV2_EVAL_PROMPTS, eval_rows)
    write_jsonl(HPDV2_EVAL_PROMPTS_400, subset)

    assert not (seen_train & set(seen_eval)), "train/eval overlap"
    print("[done] train and eval splits are disjoint")


if __name__ == "__main__":
    main()
