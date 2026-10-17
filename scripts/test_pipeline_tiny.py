"""Structural check of the ShortFT training step.

Builds a FLUX transformer with the real config but only 1+1 layers, so the whole
three-phase step plus the reward backward runs on CPU in well under a minute.

The last few steps are the point of the method: a frozen Hyper-FLUX block must
still carry the reward gradient back to the trainable adapters.
"""
from __future__ import annotations

import os
import sys

import torch
from diffusers import FluxTransformer2DModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shortft.models.flux_wrapper import _pack_latents, _unpack_latents  # noqa: E402
from shortft.models.lora import (  # noqa: E402
    ShortFTLoRALinear,
    add_timestep_aware_lora,
    get_trainable_params,
    lora_state_dict,
    set_active_lora,
    shortcut_mode,
)

NUM_SEGMENTS = 4
RANK = 4
LATENT_SIZE = 64
SEQ_LEN = (LATENT_SIZE // 2) ** 2


def build_tiny_transformer(dtype=torch.float32) -> FluxTransformer2DModel:
    """Real FLUX architecture and tensor conventions, 1+1 layers."""
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=True,
        axes_dims_rope=(4, 6, 6),
    ).to(dtype)


def make_inputs(batch_size=1, dtype=torch.float32):
    latents = _pack_latents(
        torch.randn(batch_size, 16, LATENT_SIZE, LATENT_SIZE, dtype=dtype))
    t5_embeds = torch.randn(batch_size, 16, 4096, dtype=dtype)
    pooled = torch.randn(batch_size, 768, dtype=dtype)

    h = w = LATENT_SIZE // 2
    img_ids = torch.zeros(h, w, 3, dtype=dtype)
    img_ids[..., 1] += torch.arange(h, dtype=dtype)[:, None]
    img_ids[..., 2] += torch.arange(w, dtype=dtype)[None, :]
    img_ids = img_ids.reshape(h * w, 3)
    txt_ids = torch.zeros(16, 3, dtype=dtype)

    return latents, t5_embeds, pooled, img_ids, txt_ids


def euler_step(transformer, x, sigma, sigma_next, t5_embeds, pooled,
               img_ids, txt_ids, guidance=3.5):
    B = x.shape[0]
    v = transformer(
        hidden_states=x,
        timestep=torch.full((B,), sigma, dtype=x.dtype),
        encoder_hidden_states=t5_embeds,
        pooled_projections=pooled,
        txt_ids=txt_ids,
        img_ids=img_ids,
        guidance=torch.full((B,), guidance, dtype=x.dtype),
        return_dict=False,
    )[0]
    return x + (sigma_next - sigma) * v


def main():
    torch.manual_seed(0)

    print("[1] building tiny FLUX (real config, 1+1 layers)")
    transformer = build_tiny_transformer()

    print(f"[2] injecting timestep-aware LoRA  k={NUM_SEGMENTS} r={RANK}")
    add_timestep_aware_lora(transformer, num_segments=NUM_SEGMENTS, rank=RANK)

    print("[3] populating the shortcut slots with random stand-ins")
    for m in transformer.modules():
        if isinstance(m, ShortFTLoRALinear):
            m.hyper_A = torch.nn.Parameter(
                torch.randn(RANK, m.in_features) * 0.01, requires_grad=False)
            m.hyper_B = torch.nn.Parameter(
                torch.randn(m.out_features, RANK) * 0.01, requires_grad=False)
    trainable = get_trainable_params(transformer)
    print(f"    trainable params: {sum(p.numel() for p in trainable):,}")

    latents, t5_embeds, pooled, img_ids, txt_ids = make_inputs()
    args = (t5_embeds, pooled, img_ids, txt_ids)
    print(f"    packed latents {tuple(latents.shape)}  "
          f"img_ids {tuple(img_ids.shape)}")

    print("[4] pack/unpack round trip")
    original = torch.randn(1, 16, LATENT_SIZE, LATENT_SIZE)
    assert torch.allclose(
        _unpack_latents(_pack_latents(original), LATENT_SIZE, LATENT_SIZE),
        original), "pack/unpack is not invertible"

    print("[5] phase 1 - frozen base model, no grad")
    with torch.no_grad():
        latents = euler_step(transformer, latents, 1.0, 0.8, *args)
    assert not latents.requires_grad

    print("[6] phase 2 - ShortFT LoRA, with grad")
    latents = euler_step(transformer, latents, 0.8, 0.6, *args)
    assert latents.requires_grad, "phase 2 produced no gradient path"

    print("[7] phase 3 - frozen shortcut, differentiable")
    with shortcut_mode(transformer):
        latents = euler_step(transformer, latents, 0.6, 0.0, *args)
    assert transformer._shortft_shortcut_flag[0] == 0, "flag was not reset"
    assert latents.requires_grad, "gradient does not survive the shortcut"

    print("[8] reward + backward")
    loss = _unpack_latents(latents, LATENT_SIZE, LATENT_SIZE).pow(2).mean()
    loss.backward()
    with_grad = [p for p in trainable if p.grad is not None]
    grad_norm = sum(p.grad.norm().item() ** 2 for p in with_grad) ** 0.5
    print(f"    params with grad: {len(with_grad)}/{len(trainable)}  "
          f"grad norm {grad_norm:.4f}")
    assert len(with_grad) == len(trainable), "some adapters received no gradient"
    assert grad_norm > 0, "gradient is identically zero"

    print("[9] progressive staging freezes earlier adapters")
    transformer.zero_grad(set_to_none=True)
    set_active_lora(transformer, active_from=2, num_segments=NUM_SEGMENTS)
    staged = get_trainable_params(transformer)
    assert len(staged) == len(trainable) // 2, \
        f"stage 2 should train half the adapters, got {len(staged)}"

    print("[10] LoRA state dict round trip")
    sd = lora_state_dict(transformer)
    missing = [k for k in sd if k not in dict(transformer.named_parameters())]
    assert not missing, f"state dict keys do not map back: {missing[:3]}"
    print(f"    {len(sd)} tensors")

    print("\n=== PIPELINE TEST PASSED ===")


if __name__ == "__main__":
    main()
