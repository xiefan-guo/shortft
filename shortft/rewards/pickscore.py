"""PickScore reward, differentiable w.r.t. pixel images."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from configs.paths import PICKSCORE_REPO


class PickScoreReward(nn.Module):
    """CLIP-H fine-tuned on Pick-a-Pic preference data."""

    name = "PickScore"

    # forward() returns (1 - s/100).mean(), so score_scale * forward() ==
    # -mean_score + 100, matching the HPSv2 convention.
    score_scale = 100.0

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device, dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(PICKSCORE_REPO)
        model = AutoModel.from_pretrained(PICKSCORE_REPO).eval()
        self.model = model.to(device=device, dtype=dtype)
        self.model.requires_grad_(False)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [0, 1], normalised on-GPU so gradients survive.

        The HuggingFace processor expects PIL images, which would break the
        gradient chain, so the same CLIP mean/std is applied manually.
        """
        mean = torch.tensor(self.MEAN, device=images.device,
                            dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.STD, device=images.device,
                           dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    @staticmethod
    def _as_embedding(out, projection_dim: int) -> torch.Tensor:
        """Projected [B, D] embedding, for both transformers 4.x and 5.x.

        4.x returns a plain tensor; 5.x returns a ``BaseModelOutputWithPooling``
        whose ``pooler_output`` is already the projected embedding.
        """
        if torch.is_tensor(out):
            return out
        emb = getattr(out, "pooler_output", None)
        if emb is not None and emb.shape[-1] == projection_dim:
            return emb
        raise TypeError(f"cannot extract embedding from {type(out).__name__}")

    def _embed(self, images: torch.Tensor, prompts: list):
        text_inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True,
            max_length=77, truncation=True,
        ).to(self.device)

        pixel_values = self._normalize(images.to(self.dtype))
        d = self.model.projection_dim
        img_f = self._as_embedding(
            self.model.get_image_features(pixel_values=pixel_values), d)
        txt_f = self._as_embedding(self.model.get_text_features(**text_inputs), d)

        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        return img_f, txt_f

    def forward(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        img_f, txt_f = self._embed(images, prompts)
        scores = self.model.logit_scale.exp() * (img_f * txt_f).sum(dim=-1)
        # logit_scale.exp() is ~100, so PickScore lands near 20 and s/100 keeps
        # the loss O(1) like the other rewards.
        return (1.0 - scores / 100.0).mean()

    @torch.no_grad()
    def score(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        img_f, txt_f = self._embed(images, prompts)
        return self.model.logit_scale.exp() * (img_f * txt_f).sum(dim=-1)
