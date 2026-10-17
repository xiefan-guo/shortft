"""Score a trained ShortFT LoRA against baseline FLUX.1-dev.

Generates images for both models from the same prompts and seeds, scores them
with one or more reward models, and writes a per-prompt breakdown.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import EXAMPLE_PROMPTS, setup_hf_env  # noqa: E402
from shortft.prompts import categories_of, load_prompt_records  # noqa: E402
from shortft.rewards import build_rewards, parse_reward_spec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_shortft_flux import load_pipeline  # noqa: E402


def load_cached_images(directory: str | None, indices: list[int]):
    """Cached PNGs as [N, 3, H, W] float tensors, or None if incomplete.

    Uses the same 8-bit PNG round-trip as `generate_images`, so reusing a
    cached baseline gives identical scores to regenerating it.
    """
    if not directory or not os.path.isdir(directory):
        return None
    paths = [os.path.join(directory, f"{i:04d}.png") for i in indices]
    if not all(os.path.exists(p) for p in paths):
        return None
    imgs = [torch.from_numpy(np.array(Image.open(p).convert("RGB")))
            .float().div(255.0).permute(2, 0, 1) for p in paths]
    return torch.stack(imgs)


def generate_images(pipe, prompts, seeds, num_steps=28, guidance=3.5,
                    image_size=512, out_dir=None, label=""):
    imgs = []
    for i, (prompt, seed) in enumerate(tqdm(list(zip(prompts, seeds)),
                                            desc=f"gen [{label}]")):
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=image_size,
            width=image_size,
            generator=generator,
        ).images[0]
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            image.save(os.path.join(out_dir, f"{i:04d}.png"))
        imgs.append(torch.from_numpy(np.array(image)).float().div(255.0)
                    .permute(2, 0, 1))
    return torch.stack(imgs)


def compute_scores(reward_fn, images, prompts, batch=4):
    scores = []
    for i in range(0, len(images), batch):
        imgs = images[i:i + batch].to(reward_fn.device)
        imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear",
                             align_corners=False)
        scores.extend(reward_fn.score(imgs, prompts[i:i + batch]).tolist())
    return scores


def main():
    p = argparse.ArgumentParser(description="ShortFT evaluation")
    p.add_argument("--lora_path", required=True)
    p.add_argument("--prompt_file", default=EXAMPLE_PROMPTS)
    p.add_argument("--n_prompts", type=int, default=None,
                   help="Subsample to N prompts, stratified across categories.")
    p.add_argument("--baseline_dir", default=None,
                   help="Directory with cached baseline PNGs (0000.png ...). "
                        "Baseline generation is skipped when they all exist, so "
                        "several runs can share one baseline.")
    p.add_argument("--output_dir", default="outputs/eval")
    p.add_argument("--reward_spec", default="hpsv2",
                   help="Comma-separated reward names; all are scored over the "
                        "same images and the first is the headline number.")
    p.add_argument("--num_segments", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_scale", type=float, default=0.75)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    setup_hf_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    records = load_prompt_records(args.prompt_file, limit=args.n_prompts,
                                  seed=args.seed)
    prompts = [r["prompt"] for r in records]
    cats = categories_of(records)
    seeds = [args.seed] * len(prompts)
    print(f"[eval] {len(prompts)} prompts from {args.prompt_file}")
    if any(c != "all" for c in cats):
        print(f"[eval] categories: {dict(collections.Counter(cats))}")

    # The baseline is LoRA-free and seed-fixed, hence deterministic and
    # shareable.  Cached PNGs are named by position in the *full* prompt file,
    # so map the subsample back to those indices.
    full_records = load_prompt_records(args.prompt_file)
    position = {r["prompt"]: i for i, r in enumerate(full_records)}
    base_idx = [position[p] for p in prompts]

    imgs_base = load_cached_images(args.baseline_dir, base_idx)
    if imgs_base is not None:
        print(f"[eval] reusing cached baseline: {args.baseline_dir}")
    else:
        pipe_base = load_pipeline(None, device=device, dtype=dtype)
        imgs_base = generate_images(
            pipe_base, prompts, seeds, num_steps=args.num_steps,
            image_size=args.image_size,
            out_dir=os.path.join(args.output_dir, "baseline"), label="base")
        del pipe_base
        torch.cuda.empty_cache()

    pipe_ft = load_pipeline(
        args.lora_path, num_segments=args.num_segments,
        lora_rank=args.lora_rank, lora_scale=args.lora_scale,
        device=device, dtype=dtype)
    imgs_ft = generate_images(
        pipe_ft, prompts, seeds, num_steps=args.num_steps,
        image_size=args.image_size,
        out_dir=os.path.join(args.output_dir, "shortft"), label="shortft")
    del pipe_ft
    torch.cuda.empty_cache()

    names = [n for n, _ in parse_reward_spec(args.reward_spec)]
    all_base, all_ft = {}, {}
    for name, _weight, fn in build_rewards(args.reward_spec, device=device,
                                           dtype=dtype):
        all_base[name] = compute_scores(fn, imgs_base, prompts)
        all_ft[name] = compute_scores(fn, imgs_ft, prompts)
        del fn
        torch.cuda.empty_cache()

    primary = names[0]
    base_scores, ft_scores = all_base[primary], all_ft[primary]

    print(f"\n{'=' * 62}")
    print(f"  N={len(prompts)}   primary reward: {primary}")
    for n in names:
        b = sum(all_base[n]) / len(all_base[n])
        f = sum(all_ft[n]) / len(all_ft[n])
        print(f"  {n:<10} baseline {b:8.4f} -> shortft {f:8.4f}   "
              f"delta {f - b:+.4f}")
    print(f"{'=' * 62}\n")

    # A gain concentrated in one style is a different finding from a uniform one.
    per_cat = {}
    for c in sorted(set(cats)):
        idx = [i for i, x in enumerate(cats) if x == c]
        per_cat[c] = {
            "n": len(idx),
            "baseline": round(sum(base_scores[i] for i in idx) / len(idx), 4),
            "shortft": round(sum(ft_scores[i] for i in idx) / len(idx), 4),
        }
        per_cat[c]["delta"] = round(
            per_cat[c]["shortft"] - per_cat[c]["baseline"], 4)
    if len(per_cat) > 1:
        print(f"  {'category':<14}{'n':>5}{'baseline':>11}{'shortft':>10}{'delta':>9}")
        for c, v in sorted(per_cat.items()):
            print(f"  {c:<14}{v['n']:>5}{v['baseline']:>11.4f}"
                  f"{v['shortft']:>10.4f}{v['delta']:>+9.4f}")
        print()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_json = os.path.join(args.output_dir, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "summary": {
                "n": len(prompts),
                "prompt_file": args.prompt_file,
                "lora_path": args.lora_path,
                "lora_rank": args.lora_rank,
                "lora_scale": args.lora_scale,
                "num_segments": args.num_segments,
                "num_steps": args.num_steps,
                "image_size": args.image_size,
                "seed": args.seed,
                "reward_spec": args.reward_spec,
                "per_reward": {
                    n: {
                        "baseline": round(sum(all_base[n]) / len(all_base[n]), 4),
                        "shortft": round(sum(all_ft[n]) / len(all_ft[n]), 4),
                        "delta": round(sum(all_ft[n]) / len(all_ft[n])
                                       - sum(all_base[n]) / len(all_base[n]), 4),
                    } for n in names
                },
                "per_category": per_cat,
            },
            "per_prompt": [{
                "prompt": prompt,
                "category": cats[i],
                "baseline": round(base_scores[i], 4),
                "shortft": round(ft_scores[i], 4),
                "delta": round(ft_scores[i] - base_scores[i], 4),
            } for i, prompt in enumerate(prompts)],
        }, f, indent=2)
    print(f"[eval] results written to {out_json}")


if __name__ == "__main__":
    main()
