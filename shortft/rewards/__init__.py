"""Reward model registry.

Every reward is an ``nn.Module`` whose ``forward(images, prompts)`` returns a
scalar loss to minimise and whose ``score(images, prompts)`` returns the
per-sample reward (higher is better, no grad).

Each class also exposes ``score_scale`` such that

    score_scale * forward()  ==  -mean_score + const

so a weighted multi-reward combination applies its weights on the *score*
scale.  Without it the weights would act on differently-scaled losses (HPSv2
and PickScore are /100-scaled, Aesthetic is not) and silently over-weight
Aesthetic by 100x.
"""
from .aesthetic import AestheticReward
from .hpsv2 import HPSv2Reward
from .pickscore import PickScoreReward

REWARD_NAMES = ("hpsv2", "pickscore", "aesthetic")

_REGISTRY = {
    "hpsv2": HPSv2Reward,
    "pickscore": PickScoreReward,
    "aesthetic": AestheticReward,
}


def build_reward(name: str, device, dtype):
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown reward: {name}. Known: {', '.join(REWARD_NAMES)}")
    return cls(device=device, dtype=dtype)


def parse_reward_spec(spec: str) -> list[tuple[str, float]]:
    """Parse ``"pickscore:10,hpsv2:2,aesthetic:0.05"`` into ``[(name, weight)]``.

    A bare name means weight 1.0.
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, weight = part.partition(":")
        name = name.strip().lower()
        if name not in REWARD_NAMES:
            raise ValueError(
                f"Unknown reward {name!r} in spec {spec!r}. "
                f"Known: {', '.join(REWARD_NAMES)}")
        out.append((name, float(weight) if sep else 1.0))
    if not out:
        raise ValueError(f"Empty reward spec: {spec!r}")
    return out


def build_rewards(spec: str, device, dtype) -> list[tuple[str, float, object]]:
    """Build every reward in a spec.  Returns ``[(name, weight, module)]``."""
    return [(n, w, build_reward(n, device=device, dtype=dtype))
            for n, w in parse_reward_spec(spec)]
