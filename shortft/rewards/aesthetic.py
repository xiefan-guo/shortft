"""LAION aesthetic predictor reward, differentiable w.r.t. pixel images."""
import os

import open_clip
import torch
import torch.nn as nn

from configs.paths import AESTHETIC_MLP_CKPT, CLIP_L14_PRETRAINED


class _AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticReward(nn.Module):
    name = "Aesthetic"

    # forward() already returns -mean_score.
    score_scale = 1.0

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device, dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # weights_only=False is required when CLIP_L14_PRETRAINED points at the
        # original OpenAI TorchScript archive rather than a plain state dict.
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=CLIP_L14_PRETRAINED, device=device,
            weights_only=False)
        clip_model.eval()
        clip_model.requires_grad_(False)
        self.clip = clip_model.to(dtype=dtype)

        if not os.path.exists(AESTHETIC_MLP_CKPT):
            raise FileNotFoundError(
                f"Aesthetic MLP head not found at {AESTHETIC_MLP_CKPT}. "
                "Run scripts/download_assets.py first.")
        mlp = _AestheticMLP(768)
        mlp.load_state_dict(torch.load(AESTHETIC_MLP_CKPT, map_location="cpu"))
        mlp.eval()
        mlp.requires_grad_(False)
        self.mlp = mlp.to(device=device, dtype=dtype)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.MEAN, device=images.device,
                            dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.STD, device=images.device,
                           dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._normalize(images.to(self.dtype))
        img_f = self.clip.encode_image(pixel_values)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        return self.mlp(img_f).squeeze(-1)

    def forward(self, images: torch.Tensor, prompts: list = None) -> torch.Tensor:
        return -self._predict(images).mean()

    @torch.no_grad()
    def score(self, images: torch.Tensor, prompts: list = None) -> torch.Tensor:
        return self._predict(images)
