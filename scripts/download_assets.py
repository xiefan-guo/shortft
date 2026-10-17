"""Download the auxiliary weights ShortFT needs.

FLUX.1-dev itself is loaded through the normal HuggingFace cache; set
FLUX_MODEL_PATH if you keep it somewhere else (see configs/paths.py).

Downloads into $SHORTFT_ASSETS (default: ./assets):
  Hyper-FLUX.1-dev-8steps-lora.safetensors   the denoising shortcut
  HPS_v2.1_compressed.pt                     HPS v2 reward head
  sac+logos+ava1-l14-linearMSE.pth           LAION aesthetic MLP head

Re-running is cheap: files already in the cache are skipped.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import ASSETS_DIR, DATA_RAW_DIR, setup_hf_env  # noqa: E402

setup_hf_env()

from huggingface_hub import hf_hub_download  # noqa: E402

TOKEN = os.environ.get("HF_TOKEN")


def fetch(repo_id, filename, local_dir=None, repo_type=None):
    print(f"[download] {repo_id}/{filename}", flush=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename,
                           local_dir=local_dir, repo_type=repo_type, token=TOKEN)
    print(f"  -> {path}", flush=True)
    return path


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # Denoising shortcut: few-step LoRA distilled from FLUX.1-dev.
    fetch("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors",
          local_dir=ASSETS_DIR)

    # Reward heads.
    fetch("xswu/HPSv2", "HPS_v2.1_compressed.pt", local_dir=ASSETS_DIR)
    fetch("trl-lib/ddpo-aesthetic-predictor",
          "aesthetic-model.pth", local_dir=ASSETS_DIR)
    os.rename(os.path.join(ASSETS_DIR, "aesthetic-model.pth"),
              os.path.join(ASSETS_DIR, "sac+logos+ava1-l14-linearMSE.pth"))

    # HPDv2 prompts.  Both are *dataset* repos; repo_type="dataset" is required.
    # zhwang/HPDv2 carries the benchmark split, ymhao/HPDv2 the train split.
    for filename in ["benchmark/anime.json", "benchmark/concept-art.json",
                     "benchmark/paintings.json", "benchmark/photo.json"]:
        fetch("zhwang/HPDv2", filename, local_dir=DATA_RAW_DIR,
              repo_type="dataset")
    fetch("ymhao/HPDv2", "train.json", local_dir=DATA_RAW_DIR,
          repo_type="dataset")

    # The image archives are skipped: ShortFT generates its own images and only
    # needs the prompt text.
    print("[done]", flush=True)
    print("[next] python scripts/prepare_data.py", flush=True)


if __name__ == "__main__":
    main()
