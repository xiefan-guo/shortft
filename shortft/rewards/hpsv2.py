"""Human Preference Score v2.1, differentiable w.r.t. pixel images.

Built on ``open_clip`` rather than the ``hpsv2`` pip package, whose bundled
open_clip copy ships without its BPE vocabulary.  HPS v2 is an OpenCLIP
ViT-H/14 fine-tuned on preference data, so loading its state dict into a stock
OpenCLIP-H gives identical scores.
"""
import os

import open_clip
import torch
import torch.nn as nn

from configs.paths import HPSV2_CKPT


class HPSv2Reward(nn.Module):
    name = "HPSv2"

    # forward() returns (1 - cos).mean() and score() returns cos * 100, so
    # score_scale * forward() == -mean_score + 100.  This puts a weighted
    # multi-reward combination on a common score scale.
    score_scale = 100.0

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device, dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # The HPS checkpoint supplies every weight, so skip the LAION download.
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="fp32", device="cpu")
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

        if not os.path.exists(HPSV2_CKPT):
            raise FileNotFoundError(
                f"HPS v2.1 checkpoint not found at {HPSV2_CKPT}. "
                "Run scripts/download_assets.py first.")
        ckpt = torch.load(HPSV2_CKPT, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 20:
            raise RuntimeError(
                f"HPS v2 checkpoint does not match OpenCLIP-H: {len(missing)} "
                f"missing keys (e.g. {missing[:3]})")
        print(f"[HPSv2] loaded {HPSV2_CKPT}  "
              f"missing={len(missing)} unexpected={len(unexpected)}")

        self.model = model.to(device=device, dtype=dtype).eval()
        self.model.requires_grad_(False)

        self.register_buffer("_mean", torch.tensor(
            self.MEAN, dtype=dtype, device=device).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(
            self.STD, dtype=dtype, device=device).view(1, 3, 1, 1))

    def _embed(self, images: torch.Tensor, prompts: list):
        pixel_values = (images.to(self.dtype) - self._mean) / self._std
        tokens = self.tokenizer(prompts).to(self.device)

        img_f = self.model.encode_image(pixel_values)
        txt_f = self.model.encode_text(tokens)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        return img_f, txt_f

    def forward(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        """Mean ``1 - HPS`` loss.  ``images`` is [B, 3, 224, 224] in [0, 1]."""
        img_f, txt_f = self._embed(images, prompts)
        return (1.0 - (img_f * txt_f).sum(dim=-1)).mean()

    @torch.no_grad()
    def score(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        """Per-sample HPS v2 score on the paper's x100 scale (higher is better)."""
        img_f, txt_f = self._embed(images, prompts)
        return (img_f * txt_f).sum(dim=-1) * 100.0
