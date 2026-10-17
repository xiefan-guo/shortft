"""File-system layout. Every path can be overridden by an environment variable."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"

# Local FLUX.1-dev directory; when unset the snapshot is looked up in the HF cache.
FLUX_MODEL_PATH = os.environ.get("FLUX_MODEL_PATH")

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Auxiliary weights fetched by scripts/download_assets.py.
ASSETS_DIR = os.environ.get("SHORTFT_ASSETS", os.path.join(PROJECT_ROOT, "assets"))

HYPER_FLUX_8STEP_LORA = os.path.join(
    ASSETS_DIR, "Hyper-FLUX.1-dev-8steps-lora.safetensors")

HPSV2_CKPT = os.environ.get(
    "HPSV2_CKPT", os.path.join(ASSETS_DIR, "HPS_v2.1_compressed.pt"))
AESTHETIC_MLP_CKPT = os.environ.get(
    "AESTHETIC_MLP_CKPT", os.path.join(ASSETS_DIR, "sac+logos+ava1-l14-linearMSE.pth"))
# open_clip tag or a path to a local CLIP ViT-L/14 checkpoint.
CLIP_L14_PRETRAINED = os.environ.get("CLIP_L14_PRETRAINED", "openai")
PICKSCORE_REPO = os.environ.get("PICKSCORE_REPO", "yuvalkirstain/PickScore_v1")

DATA_ROOT = os.environ.get("SHORTFT_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
DATA_RAW_DIR = os.path.join(DATA_ROOT, "raw", "hpdv2")

# Written by scripts/prepare_data.py; train and eval are disjoint by construction.
HPDV2_TRAIN_PROMPTS = os.path.join(DATA_ROOT, "train", "hpdv2_train.jsonl")
HPDV2_EVAL_PROMPTS = os.path.join(DATA_ROOT, "eval", "hpdv2_test.jsonl")
HPDV2_EVAL_PROMPTS_400 = os.path.join(DATA_ROOT, "eval", "hpdv2_test_400.jsonl")

# Shipped with the repo so every script runs out of the box.
EXAMPLE_PROMPTS = os.path.join(DATA_ROOT, "example_prompts.jsonl")


def resolve_flux_path() -> str:
    """Return a local FLUX.1-dev directory, falling back to the hub id."""
    if FLUX_MODEL_PATH:
        return FLUX_MODEL_PATH
    hub = os.path.join(HF_HOME, "hub", "models--" + FLUX_MODEL_ID.replace("/", "--"))
    snapshots = os.path.join(hub, "snapshots")
    if os.path.isdir(snapshots):
        dirs = [os.path.join(snapshots, d) for d in sorted(os.listdir(snapshots))]
        dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "model_index.json"))]
        if dirs:
            return dirs[-1]
    return FLUX_MODEL_ID


def setup_hf_env() -> None:
    """Point huggingface_hub at the configured cache."""
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
