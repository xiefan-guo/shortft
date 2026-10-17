"""Generate images with a trained ShortFT LoRA checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from diffusers import FluxPipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import EXAMPLE_PROMPTS, resolve_flux_path, setup_hf_env  # noqa: E402
from shortft.models.lora import add_timestep_aware_lora  # noqa: E402
from shortft.prompts import load_prompts  # noqa: E402


def load_pipeline(lora_path: str | None, num_segments: int = 4,
                  lora_rank: int = 64, lora_scale: float = 0.75,
                  device="cuda", dtype=torch.bfloat16) -> FluxPipeline:
    """Load FLUX.1-dev, optionally with ShortFT adapters merged in.

    The shortcut is not used at inference: the full chain runs with the stacked
    adapters active.
    """
    model_path = resolve_flux_path()
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    if not lora_path:
        return pipe

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"--lora_path does not exist: {lora_path}")

    meta_path = os.path.join(os.path.dirname(lora_path), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        num_segments = meta.get("num_segments", num_segments)
        lora_rank = meta.get("lora_rank", lora_rank)
        print(f"[inference] meta.json: k={num_segments} r={lora_rank}")

    state = torch.load(lora_path, map_location="cpu")
    pipe.transformer = add_timestep_aware_lora(
        pipe.transformer, num_segments=num_segments, rank=lora_rank)

    # `missing_keys` lists model parameters the checkpoint does not provide,
    # which for LoRA-only loading is every base weight.  What must be empty is
    # the set of checkpoint keys that matched nothing.
    model_keys = set(dict(pipe.transformer.named_parameters())) \
        | set(dict(pipe.transformer.named_buffers()))
    unloaded = [k for k in state if k not in model_keys]
    if unloaded:
        raise RuntimeError(
            f"{len(unloaded)}/{len(state)} LoRA keys did not load "
            f"(e.g. {unloaded[:3]}). Check --num_segments / --lora_rank.")
    pipe.transformer.load_state_dict(state, strict=False)
    print(f"[inference] LoRA loaded: {lora_path} ({len(state)} keys)")

    # Scaling the adapters down at inference trades a little reward for
    # noticeably less reward over-optimisation.
    if lora_scale != 1.0:
        for name, param in pipe.transformer.named_parameters():
            if "lora_" in name:
                param.data.mul_(lora_scale)
        print(f"[inference] LoRA scaled by {lora_scale}")

    return pipe


def main():
    p = argparse.ArgumentParser(description="ShortFT FLUX.1-dev inference")
    p.add_argument("--lora_path", default=None,
                   help="Trained lora_weights.pt; omit for the baseline model.")
    p.add_argument("--prompt_file", default=EXAMPLE_PROMPTS)
    p.add_argument("--n_prompts", type=int, default=None,
                   help="Subsample to N prompts (stratified by category).")
    p.add_argument("--output_dir", default="./outputs/inference")
    p.add_argument("--num_segments", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_scale", type=float, default=0.75)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    setup_hf_env()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = load_prompts(args.prompt_file, limit=args.n_prompts, seed=args.seed)
    print(f"[inference] {len(prompts)} prompts from {args.prompt_file}")

    pipe = load_pipeline(
        args.lora_path,
        num_segments=args.num_segments,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
    )

    for i, prompt in enumerate(prompts):
        generator = torch.Generator("cuda").manual_seed(args.seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            height=args.image_size,
            width=args.image_size,
            generator=generator,
        ).images[0]
        out = os.path.join(args.output_dir, f"{i:04d}.png")
        image.save(out)
        print(f"  [{i:04d}] {prompt[:60]} -> {out}")

    print(f"[done] {len(prompts)} images -> {args.output_dir}")


if __name__ == "__main__":
    main()
