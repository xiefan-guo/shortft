"""FLUX.1-dev wrapper: text encoding, latent packing, Euler steps, VAE decoding.

Keeps the FLUX-specific bookkeeping (RoPE ids, 2x2 latent packing, dynamic
shift) in one place.  The transformer is injected from outside so that LoRA
can be attached to it before the wrapper is built.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """FLUX dynamic-shift ``mu``: linear in the packed sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, H*W/4, C*4] (2x2 spatial patch folding)."""
    B, C, H, W = latents.shape
    x = latents.reshape(B, C, H // 2, 2, W // 2, 2)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)


def _unpack_latents(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[B, H*W/4, C*4] -> [B, C, H, W]."""
    B, _seq, c4 = packed.shape
    C = c4 // 4
    h, w = height // 2, width // 2
    x = packed.reshape(B, h, w, C, 2, 2)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, height, width)


class FluxWrapper(nn.Module):
    """Holds the frozen FLUX components and the utilities the trainer needs."""

    def __init__(
        self,
        model_path: str,
        transformer: FluxTransformer2DModel,
        device: torch.device,
        weight_dtype: torch.dtype = torch.bfloat16,
        image_size: int = 1024,
    ):
        super().__init__()
        self.device = device
        self.dtype = weight_dtype
        self.image_size = image_size
        self.latent_size = image_size // 8

        self.transformer = transformer

        # Text encoders may be parked on CPU to free VRAM; encoding is no_grad,
        # so only wall-clock is affected.
        self.text_encoder_device = device
        # The VAE decoder has to stay differentiable, and its fp32 activations
        # dominate the memory peak, so it is checkpointed.
        self.checkpoint_vae = True

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer")
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_path, subfolder="tokenizer_2")

        clip = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        self.text_encoder = clip.to(device=device, dtype=weight_dtype)
        self.text_encoder.requires_grad_(False)

        t5 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2")
        self.text_encoder_2 = t5.to(device=device, dtype=weight_dtype)
        self.text_encoder_2.requires_grad_(False)

        # fp32 VAE: bf16 decoding is numerically too coarse for reward gradients.
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        self.vae = vae.to(device=device, dtype=torch.float32)
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode_prompt(
        self, prompts: list[str], max_sequence_length: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (T5 embeddings [B, seq, 4096], CLIP pooled embeddings [B, 768])."""
        ted = self.text_encoder_device

        clip_tokens = self.tokenizer(
            prompts, padding="max_length", max_length=77, truncation=True,
            return_tensors="pt",
        ).input_ids.to(ted)
        pooled = self.text_encoder(clip_tokens, output_hidden_states=False)
        pooled = pooled.pooler_output.to(self.device, self.dtype)

        t5_tokens = self.tokenizer_2(
            prompts, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(ted)
        t5_embeds = self.text_encoder_2(t5_tokens).last_hidden_state
        t5_embeds = t5_embeds.to(self.device, self.dtype)

        return t5_embeds, pooled

    def denoising_step(
        self,
        latents_packed: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        t5_embeds: torch.Tensor,
        pooled: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance_scale: float = 3.5,
        enable_grad: bool = False,
    ) -> torch.Tensor:
        """One Euler step on the packed latents."""
        B = latents_packed.shape[0]

        def _step(x):
            v = self.transformer(
                hidden_states=x,
                timestep=sigma.expand(B).to(self.dtype),
                encoder_hidden_states=t5_embeds,
                pooled_projections=pooled,
                txt_ids=txt_ids,
                img_ids=img_ids,
                guidance=torch.full((B,), guidance_scale,
                                    device=self.device, dtype=self.dtype),
                return_dict=False,
            )[0]
            return x + (sigma_next - sigma) * v

        if enable_grad:
            return _step(latents_packed)
        with torch.no_grad():
            return _step(latents_packed)

    def decode_latents(self, latents_packed: torch.Tensor) -> torch.Tensor:
        """Decode packed latents to [B, 3, H, W] pixels in [0, 1], differentiably.

        The reward gradient travels back through the decoder to reach the
        denoising chain; the VAE weights themselves stay frozen.
        """
        H = W = self.latent_size

        def _decode(packed):
            latents = _unpack_latents(packed, H, W)
            latents = (latents / self.vae.config.scaling_factor
                       + self.vae.config.shift_factor)
            images = self.vae.decode(latents.float()).sample
            return (images / 2.0 + 0.5).clamp(0, 1)

        if self.checkpoint_vae and torch.is_grad_enabled() \
                and latents_packed.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                _decode, latents_packed, use_reentrant=False)
        return _decode(latents_packed)

    def make_ids(self, txt_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE position ids as 2-D ``[seq, 3]`` tensors (img_ids, txt_ids)."""
        h = w = self.latent_size // 2
        img_ids = torch.zeros(h, w, 3, device=self.device, dtype=self.dtype)
        img_ids[..., 1] += torch.arange(h, device=self.device, dtype=self.dtype)[:, None]
        img_ids[..., 2] += torch.arange(w, device=self.device, dtype=self.dtype)[None, :]
        img_ids = img_ids.reshape(h * w, 3)

        txt_ids = torch.zeros(txt_seq_len, 3, device=self.device, dtype=self.dtype)
        return img_ids, txt_ids

    def sample_noise(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Packed Gaussian noise latents [B, H*W/4, C*4]."""
        H = W = self.latent_size
        noise = torch.randn(batch_size, 16, H, W, generator=generator,
                            device=self.device, dtype=self.dtype)
        return _pack_latents(noise)

    def get_schedule(self, num_steps: int = 28) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sigmas [num_steps+1], timesteps [num_steps]).

        FLUX.1-dev uses dynamic shifting, which needs ``mu``, derived from the
        packed image sequence length.
        """
        seq_len = (self.latent_size // 2) ** 2
        kwargs = {}
        if getattr(self.scheduler.config, "use_dynamic_shifting", False):
            kwargs["mu"] = calculate_shift(
                seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        self.scheduler.set_timesteps(num_steps, device=self.device, **kwargs)
        return (self.scheduler.sigmas.to(self.dtype),
                self.scheduler.timesteps.to(self.dtype))
