"""ShortFT training loop for FLUX.1-dev.

Each training step runs three phases:

    Phase 1  steps 0 .. seg_start-1     vanilla denoising, no grad
    Phase 2  steps seg_start .. seg_end ShortFT LoRA, with grad (truncated)
    Phase 3  remaining steps            Hyper-FLUX shortcut to x_0, differentiable
                 |
                 v
             VAE decode (differentiable) -> reward -> backward

Phase 3 and the VAE decode must stay differentiable: the reward gradient
travels back through them to reach the ShortFT adapters.  That short path is
what replaces backpropagation through the full denoising chain.

Progressive training visits k stages; stage i optimises adapters i..k-1 and
places the differentiable window on segment i, so reward supervision
eventually reaches the whole chain.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import FluxTransformer2DModel
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import (  # noqa: E402
    EXAMPLE_PROMPTS,
    HYPER_FLUX_8STEP_LORA,
    resolve_flux_path,
    setup_hf_env,
)
from shortft.models.flux_wrapper import FluxWrapper  # noqa: E402
from shortft.models.lora import (  # noqa: E402
    add_timestep_aware_lora,
    flag_aware_checkpoint_func,
    get_stage_params,
    get_trainable_params,
    load_hyper_shortcut,
    lora_state_dict,
    set_active_lora,
    shortcut_mode,
    zero_lora_grads,
)
from shortft.prompts import load_prompts  # noqa: E402
from shortft.rewards import build_rewards, parse_reward_spec  # noqa: E402


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


def resize_for_reward(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    if images.shape[-1] == size and images.shape[-2] == size:
        return images
    return F.interpolate(images, size=(size, size), mode="bilinear",
                         align_corners=False)


def build_segment_boundaries(num_steps: int, num_segments: int) -> list[int]:
    """Step index of the last step of each segment, for a k-segment split."""
    seg_size = num_steps // num_segments
    boundaries = [(j + 1) * seg_size - 1 for j in range(num_segments)]
    boundaries[-1] = num_steps - 1
    return boundaries


def save_checkpoint(transformer, output_dir: str, step: int, stage: int,
                    num_segments: int, lora_rank: int) -> None:
    ckpt_dir = Path(output_dir) / f"checkpoint-{step:05d}-stage{stage}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(transformer), ckpt_dir / "lora_weights.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps(
        {"step": step, "stage": stage, "num_segments": num_segments,
         "lora_rank": lora_rank}))
    print(f"[ckpt] saved {ckpt_dir}")


def train(args):
    setup_hf_env()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    device = accelerator.device
    weight_dtype = torch.bfloat16

    model_path = resolve_flux_path()
    print(f"[init] FLUX path: {model_path}")

    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=weight_dtype).to(device)

    transformer = add_timestep_aware_lora(
        transformer,
        num_segments=args.num_segments,
        rank=args.lora_rank,
        lora_alpha=args.lora_rank,
    )

    flux = FluxWrapper(
        model_path=model_path,
        transformer=transformer,
        device=device,
        weight_dtype=weight_dtype,
        image_size=args.image_size,
    )

    if args.text_encoders_on_cpu:
        # T5-XXL is ~9.6 GB in bf16.  Encoding is no_grad, so parking it on
        # CPU costs wall-clock and nothing else.
        print("[init] parking text encoders on CPU")
        flux.text_encoder.to("cpu")
        flux.text_encoder_2.to("cpu")
        flux.text_encoder_device = torch.device("cpu")
        torch.cuda.empty_cache()

    # Hyper-FLUX is itself a LoRA, so it occupies a frozen adapter slot on the
    # transformer we already hold rather than needing a second model copy.
    load_hyper_shortcut(transformer, HYPER_FLUX_8STEP_LORA, scale=1.0)

    rewards = build_rewards(args.reward_spec, device=device, dtype=weight_dtype)
    print(f"[init] rewards: {args.reward_spec}")
    for name, weight, fn in rewards:
        print(f"  {name:10s} weight={weight:<8g} score_scale={fn.score_scale:g}")

    prompts = load_prompts(args.train_data_path)
    print(f"[init] {len(prompts)} prompts from {args.train_data_path}")
    dataloader = torch.utils.data.DataLoader(
        PromptDataset(prompts),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    lora_params = get_trainable_params(transformer)
    print(f"[init] trainable params: {sum(p.numel() for p in lora_params):,}")
    optimizer = AdamW(lora_params, lr=args.learning_rate,
                      betas=(0.9, 0.999), weight_decay=0.1)

    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader)
    # The shortcut flag lives on the raw module, so keep a direct handle.
    unwrapped = accelerator.unwrap_model(transformer)
    flux.transformer = transformer

    if args.gradient_checkpointing:
        # The differentiable chain spans several transformer steps plus the VAE
        # decode and does not fit otherwise.  The flag-aware wrapper is required
        # so that a block recomputed during backward() runs on the same branch
        # it was originally run on.
        unwrapped.enable_gradient_checkpointing(
            gradient_checkpointing_func=flag_aware_checkpoint_func(unwrapped))
        print("[init] gradient checkpointing enabled (flag-aware)")

    sigmas, _timesteps = flux.get_schedule(args.num_inference_steps)
    seg_boundaries = build_segment_boundaries(
        args.num_inference_steps, args.num_segments)
    print(f"[init] segment boundaries (step indices): {seg_boundaries}")

    global_step = 0
    start_time = time.time()
    reward_terms: dict[str, float] = {}

    if args.stage_schedule == "cyclic":
        # Rotating through the stages keeps every adapter under reward
        # supervision throughout training, instead of only during its block.
        visits = [s for _ in range(args.num_cycles)
                  for s in range(args.num_segments)]
    else:
        visits = list(range(args.num_segments))
    print(f"[init] schedule: {args.stage_schedule}, "
          f"{len(visits)} stage visits x {args.steps_per_stage} steps")

    for stage in visits:
        # Under DDP the set of parameters with requires_grad=True must not
        # change after the model is wrapped, so there every adapter stays
        # trainable and the optimizer is restricted instead.
        set_active_lora(
            unwrapped,
            active_from=stage,
            num_segments=args.num_segments,
            freeze_inactive=(accelerator.num_processes == 1),
        )
        zero_lora_grads(unwrapped)
        active_params = get_stage_params(unwrapped, stage)
        for group in optimizer.param_groups:
            group["params"] = active_params

        seg_start = 0 if stage == 0 else seg_boundaries[stage - 1] + 1
        seg_end = seg_boundaries[stage]
        print(f"\n[stage {stage}] optimising lora_{stage}..."
              f"lora_{args.num_segments - 1} | denoising steps "
              f"{seg_start}-{seg_end} with grad | "
              f"{sum(p.numel() for p in active_params):,} params")

        pbar = tqdm(total=args.steps_per_stage, desc=f"stage {stage}")
        stage_step = 0

        while stage_step < args.steps_per_stage:
            for batch in dataloader:
                if stage_step >= args.steps_per_stage:
                    break
                prompts_batch = batch["prompt"]
                B = len(prompts_batch)

                with accelerator.accumulate(transformer):
                    generator = torch.Generator(device=device).manual_seed(
                        random.randint(0, 2 ** 31))

                    with torch.no_grad():
                        t5_embeds, pooled = flux.encode_prompt(prompts_batch)

                    latents = flux.sample_noise(B, generator=generator)
                    img_ids, txt_ids = flux.make_ids(t5_embeds.shape[1])

                    # Phase 1: frozen base model walks to the segment start.
                    with torch.no_grad():
                        for i in range(seg_start):
                            latents = flux.denoising_step(
                                latents, sigmas[i], sigmas[i + 1],
                                t5_embeds, pooled, img_ids, txt_ids,
                                guidance_scale=args.guidance_scale,
                                enable_grad=False)

                    # Phase 2: the trainable LoRA stack runs the segment.
                    # Only the last `truncate_backprop_to` steps keep
                    # gradients, which caps the depth of the chain.
                    grad_from = max(seg_start,
                                    seg_end + 1 - args.truncate_backprop_to)
                    for i in range(seg_start, seg_end + 1):
                        with_grad = i >= grad_from
                        if with_grad and not latents.requires_grad:
                            latents = latents.detach()
                        latents = flux.denoising_step(
                            latents, sigmas[i], sigmas[i + 1],
                            t5_embeds, pooled, img_ids, txt_ids,
                            guidance_scale=args.guidance_scale,
                            enable_grad=with_grad)

                    # Phase 3: the frozen shortcut jumps to x_0.  Its weights
                    # do not train, but the mapping stays differentiable.
                    sigma_shortcut_start = sigmas[seg_end + 1]
                    latents_clean = latents
                    if sigma_shortcut_start.item() > 1e-4:
                        sc = torch.linspace(
                            sigma_shortcut_start.item(), 0.0,
                            args.shortcut_steps + 1,
                            device=device, dtype=weight_dtype)
                        with shortcut_mode(unwrapped):
                            for si in range(args.shortcut_steps):
                                latents_clean = flux.denoising_step(
                                    latents_clean, sc[si], sc[si + 1],
                                    t5_embeds, pooled, img_ids, txt_ids,
                                    guidance_scale=args.guidance_scale,
                                    enable_grad=True)

                    images = flux.decode_latents(latents_clean)
                    images_rw = resize_for_reward(images, 224)

                    # Weighted sum on the score scale: score_scale maps each
                    # reward's loss back to -mean_score, so a spec like
                    # "pickscore:10,hpsv2:2,aesthetic:0.05" means what it says.
                    loss = None
                    for _name, weight, fn in rewards:
                        term = weight * fn.score_scale * fn(images_rw, prompts_batch)
                        loss = term if loss is None else loss + term

                    with torch.no_grad():
                        for name, _weight, fn in rewards:
                            reward_terms[name] = fn.score(
                                images_rw.detach(), prompts_batch).mean().item()

                    # Safe outside shortcut_mode: the flag-aware checkpoint
                    # wrapper restores each block's saved branch.
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(active_params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    stage_step += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        **{f"s_{k}": f"{v:.2f}" for k, v in reward_terms.items()},
                        elapsed=f"{(time.time() - start_time) / 60:.1f}m",
                    )
                    if global_step % args.save_every == 0 and accelerator.is_main_process:
                        save_checkpoint(unwrapped, args.output_dir, global_step,
                                        stage, args.num_segments, args.lora_rank)

        pbar.close()

    if accelerator.is_main_process:
        save_checkpoint(unwrapped, args.output_dir, global_step,
                        args.num_segments - 1, args.num_segments, args.lora_rank)
        elapsed_min = (time.time() - start_time) / 60
        print(f"[done] {global_step} steps in {elapsed_min:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(description="ShortFT training on FLUX.1-dev")
    p.add_argument("--train_data_path", default=EXAMPLE_PROMPTS,
                   help="JSONL file with one prompt record per line.")
    p.add_argument("--output_dir", default="./ckpts/shortft_flux")
    p.add_argument("--reward_spec", default="hpsv2",
                   help="Comma-separated 'name:weight' list, e.g. "
                        "'pickscore:10,hpsv2:2,aesthetic:0.05'. Weights apply "
                        "on the score scale. A bare name means weight 1.")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_segments", type=int, default=4,
                   help="k: number of denoising segments / LoRA adapters.")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--steps_per_stage", type=int, default=200)
    p.add_argument("--stage_schedule", default="blocked",
                   choices=["blocked", "cyclic"],
                   help="blocked = run each stage to completion in turn; "
                        "cyclic = rotate through the stages.")
    p.add_argument("--num_cycles", type=int, default=1,
                   help="Number of stage rotations (cyclic schedule only).")
    p.add_argument("--num_inference_steps", type=int, default=28,
                   help="Total denoising steps (FLUX default is 28).")
    p.add_argument("--shortcut_steps", type=int, default=4,
                   help="Euler steps the Hyper-FLUX shortcut takes to reach x_0.")
    p.add_argument("--truncate_backprop_to", type=int, default=2,
                   help="Keep gradients only for the last N steps of a segment.")
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--text_encoders_on_cpu", action="store_true",
                   help="Keep T5/CLIP on CPU to free ~9.6 GB of VRAM.")
    args = p.parse_args()
    parse_reward_spec(args.reward_spec)  # fail fast on a typo, before loading
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
