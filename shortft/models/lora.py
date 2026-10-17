"""Timestep-aware LoRA and the Hyper-FLUX denoising shortcut.

The denoising chain is split into k segments and each segment gets its own LoRA
adapter.  Adapters are stacked: running "up to adapter i" means summing the
weight deltas of adapters 0...i.

Every target ``nn.Linear`` is replaced by :class:`ShortFTLoRALinear`, which
holds k trainable adapters plus one frozen slot for the Hyper-FLUX
distillation LoRA.  A shared flag on the transformer selects which branch is
active, so a single transformer copy serves both the ShortFT path and the
shortcut path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


FLUX_LORA_TARGET_MODULES: list[str] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "proj_mlp",
    "proj_out",
]


class ShortFTLoRALinear(nn.Module):
    """``nn.Linear`` with k stacked ShortFT adapters and a frozen shortcut slot.

    On the ShortFT path the output is ``Wx + sum_j scale * B_j A_j x`` over the
    adapters in the transformer's active set.  On the shortcut path only the
    frozen Hyper-FLUX delta applies, so the module reproduces the distilled
    few-step trajectory.
    """

    def __init__(self, original: nn.Linear, num_adapters: int, rank: int,
                 lora_alpha: float):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.num_adapters = num_adapters
        self.rank = rank
        self.scale = lora_alpha / rank

        self.weight = original.weight
        self.bias = original.bias

        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for _ in range(num_adapters):
            A = nn.Parameter(torch.empty(rank, self.in_features))
            B = nn.Parameter(torch.zeros(self.out_features, rank))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lora_A.append(A)
            self.lora_B.append(B)

        self._active_set: set | None = None

        self.hyper_A: nn.Parameter | None = None
        self.hyper_B: nn.Parameter | None = None
        self.hyper_scale: float = 1.0
        self._shortcut_flag: list | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)

        if self._shortcut_flag is not None and self._shortcut_flag[0]:
            if self.hyper_A is not None:
                h = (x.to(self.hyper_A.dtype) @ self.hyper_A.T) @ self.hyper_B.T
                out = out + h.to(out.dtype) * self.hyper_scale
            return out

        active = self._active_set if self._active_set is not None \
            else set(range(self.num_adapters))
        for j in active:
            delta = (x.to(self.lora_A[j].dtype) @ self.lora_A[j].T) @ self.lora_B[j].T
            out = out + delta.to(out.dtype) * self.scale
        return out


def _matches_target(full_name: str) -> bool:
    parts = full_name.split(".")
    for target in FLUX_LORA_TARGET_MODULES:
        t_parts = target.split(".")
        if parts[-len(t_parts):] == t_parts:
            return True
    return False


def _iter_target_linears(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and _matches_target(full):
            yield full, module, name, child
        elif not isinstance(child, nn.Linear):
            yield from _iter_target_linears(child, full)


def add_timestep_aware_lora(
    transformer: nn.Module,
    num_segments: int = 4,
    rank: int = 64,
    lora_alpha: float | None = None,
) -> nn.Module:
    """Replace every target linear with a :class:`ShortFTLoRALinear`, in place.

    LoRA parameters are kept in fp32 while the base weights stay in their
    original dtype.  Everything that is not a ShortFT adapter is frozen.
    """
    if lora_alpha is None:
        lora_alpha = float(rank)

    replaced = 0
    for _full_name, parent, attr, linear in list(_iter_target_linears(transformer)):
        lora_layer = ShortFTLoRALinear(linear, num_segments, rank, lora_alpha)
        device = linear.weight.device
        for j in range(num_segments):
            lora_layer.lora_A[j].data = lora_layer.lora_A[j].data.to(
                device=device, dtype=torch.float32)
            lora_layer.lora_B[j].data = lora_layer.lora_B[j].data.to(
                device=device, dtype=torch.float32)
        setattr(parent, attr, lora_layer)
        replaced += 1

    transformer._shortft_active_adapters = set(range(num_segments))
    transformer._shortft_num_segments = num_segments
    transformer._shortft_shortcut_flag = [0]
    _wire_shared_state(transformer)

    print(f"[ShortFT LoRA] replaced {replaced} linear layers  "
          f"k={num_segments}  r={rank}")

    # from_pretrained leaves the 12B base parameters trainable.  Registering
    # gradients for them costs a full bf16 set (~24 GB) that no optimizer ever
    # consumes, so freeze everything outside the adapters.
    for name, param in transformer.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    return transformer


def _wire_shared_state(transformer: nn.Module) -> None:
    active = transformer._shortft_active_adapters
    flag = transformer._shortft_shortcut_flag
    for m in transformer.modules():
        if isinstance(m, ShortFTLoRALinear):
            m._active_set = active
            m._shortcut_flag = flag


def load_hyper_shortcut(transformer: nn.Module, lora_path: str,
                        scale: float = 1.0) -> int:
    """Load the Hyper-FLUX distillation LoRA into the frozen shortcut slots.

    The checkpoint follows diffusers naming
    (``transformer.<path>.lora_A.weight`` / ``.lora_B.weight``).  The weights
    only take effect inside :class:`shortcut_mode`.  Returns the number of
    layers that received weights.
    """
    from safetensors.torch import load_file

    sd = load_file(lora_path, device="cpu")
    layers = {name: m for name, m in transformer.named_modules()
              if isinstance(m, ShortFTLoRALinear)}

    loaded = 0
    for key in sd:
        if not key.endswith(".lora_A.weight"):
            continue
        mod_path = key[len("transformer."):-len(".lora_A.weight")] \
            if key.startswith("transformer.") else key[:-len(".lora_A.weight")]
        b_key = key.replace(".lora_A.", ".lora_B.")
        layer = layers.get(mod_path)
        if b_key not in sd or layer is None:
            continue

        A, B = sd[key], sd[b_key]
        if A.shape[1] != layer.in_features or B.shape[0] != layer.out_features:
            continue

        device = layer.weight.device
        layer.hyper_A = nn.Parameter(
            A.to(device=device, dtype=torch.float32), requires_grad=False)
        layer.hyper_B = nn.Parameter(
            B.to(device=device, dtype=torch.float32), requires_grad=False)
        layer.hyper_scale = scale
        loaded += 1

    print(f"[Hyper-FLUX shortcut] loaded into {loaded}/{len(layers)} layers")
    return loaded


class shortcut_mode:
    """Context manager that switches the transformer to the shortcut branch."""

    def __init__(self, transformer: nn.Module):
        self.flag = transformer._shortft_shortcut_flag

    def __enter__(self):
        self.flag[0] = 1
        return self

    def __exit__(self, *exc):
        self.flag[0] = 0
        return False


def flag_aware_checkpoint_func(transformer: nn.Module):
    """Gradient-checkpointing wrapper that remembers the shortcut flag.

    Pass to ``enable_gradient_checkpointing(gradient_checkpointing_func=...)``.
    A training step runs some blocks outside ``shortcut_mode`` and some inside
    it, while ``backward()`` happens after both.  Making the flag an input of
    the checkpointed call saves it alongside the other inputs, so every block
    is recomputed on the branch it originally ran on.
    """
    flag = transformer._shortft_shortcut_flag

    def gc_func(module, *args):
        saved = flag[0]

        def run(flag_tensor):
            prev = flag[0]
            flag[0] = int(flag_tensor.item())
            try:
                return module(*args)
            finally:
                flag[0] = prev

        return torch.utils.checkpoint.checkpoint(
            run,
            torch.tensor([saved], device=args[0].device, dtype=torch.long),
            use_reentrant=False,
        )

    return gc_func


def set_active_lora(transformer: nn.Module, active_from: int,
                    num_segments: int | None = None,
                    freeze_inactive: bool = True) -> None:
    """Select the adapters trained at this progressive stage.

    All adapters stay active in the forward pass; ``active_from`` only decides
    which ones are optimised.  ``freeze_inactive=False`` is the DDP-safe mode:
    DistributedDataParallel requires the set of parameters with
    ``requires_grad=True`` to be fixed once the model is wrapped, so there the
    caller restricts the optimizer with :func:`get_stage_params` instead.
    """
    k = num_segments or transformer._shortft_num_segments
    transformer._shortft_active_adapters.clear()
    transformer._shortft_active_adapters.update(range(k))

    if not freeze_inactive:
        return

    for m in transformer.modules():
        if isinstance(m, ShortFTLoRALinear):
            for j in range(m.num_adapters):
                trainable = j >= active_from
                m.lora_A[j].requires_grad_(trainable)
                m.lora_B[j].requires_grad_(trainable)


def get_trainable_params(transformer: nn.Module) -> list[nn.Parameter]:
    """All LoRA parameters with ``requires_grad=True``."""
    return [p for m in transformer.modules()
            if isinstance(m, ShortFTLoRALinear)
            for p in list(m.lora_A) + list(m.lora_B)
            if p.requires_grad]


def get_stage_params(transformer: nn.Module, active_from: int) -> list[nn.Parameter]:
    """LoRA parameters of adapters ``>= active_from``, ignoring ``requires_grad``."""
    return [p for m in transformer.modules()
            if isinstance(m, ShortFTLoRALinear)
            for j, p in enumerate(list(m.lora_A) + list(m.lora_B))
            if (j % m.num_adapters) >= active_from]


def zero_lora_grads(transformer: nn.Module) -> None:
    """Drop every LoRA gradient, so a stage boundary cannot apply a stale one."""
    for m in transformer.modules():
        if isinstance(m, ShortFTLoRALinear):
            for p in list(m.lora_A) + list(m.lora_B):
                p.grad = None


def lora_state_dict(transformer: nn.Module) -> dict[str, torch.Tensor]:
    """Extract the ShortFT adapter weights for checkpointing."""
    sd = {}
    for name, m in transformer.named_modules():
        if isinstance(m, ShortFTLoRALinear):
            for j, (A, B) in enumerate(zip(m.lora_A, m.lora_B)):
                sd[f"{name}.lora_A.{j}"] = A.data.clone()
                sd[f"{name}.lora_B.{j}"] = B.data.clone()
    return sd
