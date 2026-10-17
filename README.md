## ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning<br><sub>Official PyTorch code release for the ICCV 2025 paper: https://arxiv.org/abs/2507.22604</sub>

**ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning**<br>
Xiefan Guo, Miaomiao Cui, Liefeng Bo, Di Huang<br>
https://xiefan-guo.github.io/shortft<br>

Abstract: *Backpropagation-based approaches aim to align diffusion models with reward functions through end-to-end backpropagation of the reward gradient within the denoising chain, offering a promising perspective. However, due to the computational costs and the risk of gradient explosion associated with the lengthy denoising chain, existing approaches struggle to achieve complete gradient backpropagation, leading to suboptimal results. In this paper, we introduce Shortcut-based Fine-Tuning (ShortFT), an efficient fine-tuning strategy that utilizes the shorter denoising chain. More specifically, we employ the recently researched trajectory-preserving few-step diffusion model, which enables a shortcut over the original denoising chain, and construct a shortcut-based denoising chain of shorter length. The optimization on this chain notably enhances the efficiency and effectiveness of fine-tuning the foundational model. Our method has been rigorously tested and can be effectively applied to various reward functions, significantly improving alignment performance and surpassing state-of-the-art alternatives.*

---

### Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

Download the shortcut LoRA, the reward heads, and the HPDv2 prompts:

```bash
python scripts/download_assets.py
python scripts/prepare_data.py
```

FLUX.1-dev is read from the HuggingFace cache. Point `FLUX_MODEL_PATH` at a local directory if you keep it elsewhere. All paths live in `configs/paths.py` and can be overridden by environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `FLUX_MODEL_PATH` | HF cache lookup | FLUX.1-dev snapshot directory |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache |
| `SHORTFT_ASSETS` | `./assets` | Shortcut LoRA and reward checkpoints |
| `SHORTFT_DATA_ROOT` | `./data` | Prompt datasets |

Run the structural check — it builds a 1+1-layer FLUX with the real config and exercises the whole training step, including the gradient path through the frozen shortcut, on CPU in under a minute:

```bash
python scripts/test_pipeline_tiny.py
```

---

### Training

```bash
python scripts/train_shortft_flux.py \
    --train_data_path data/train/hpdv2_train.jsonl \
    --output_dir ckpts/shortft_hpsv2 \
    --reward_spec hpsv2 \
    --image_size 512 \
    --num_segments 4 \
    --lora_rank 64 \
    --num_inference_steps 28 \
    --shortcut_steps 4 \
    --truncate_backprop_to 2 \
    --steps_per_stage 250 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5
```

Multi-GPU:

```bash
accelerate launch --num_processes 8 scripts/train_shortft_flux.py [same args]
```

`--reward_spec` takes a weighted list. Weights apply on the *score* scale, so the paper's combined reward reads directly:

```bash
--reward_spec "pickscore:10,hpsv2:2,aesthetic:0.05"
```

Useful knobs:

| Flag | Meaning |
|---|---|
| `--num_segments` | `k`, the number of segments and adapters |
| `--shortcut_steps` | Euler steps the shortcut takes to reach `x_0` |
| `--truncate_backprop_to` | keep gradients only for the last N steps of a segment |
| `--stage_schedule cyclic` | rotate through stages instead of running each to completion |
| `--text_encoders_on_cpu` | park T5/CLIP on CPU, freeing ~9.6 GB of VRAM |

Checkpoints hold only the adapter tensors plus a `meta.json` recording `k` and the rank, which inference and evaluation read back automatically.

---

### Inference

```bash
python scripts/inference_shortft_flux.py \
    --lora_path ckpts/shortft_hpsv2/checkpoint-01000-stage3/lora_weights.pt \
    --prompt_file data/eval/hpdv2_test_400.jsonl \
    --output_dir outputs/shortft
```

Omit `--lora_path` for baseline images. Adapters are scaled by `--lora_scale` (default 0.75), which trades a little reward for noticeably less reward over-optimisation.

---

### Evaluation

```bash
python scripts/eval_shortft_flux.py \
    --lora_path ckpts/shortft_hpsv2/checkpoint-01000-stage3/lora_weights.pt \
    --prompt_file data/eval/hpdv2_test_400.jsonl \
    --reward_spec "hpsv2,pickscore,aesthetic" \
    --output_dir outputs/eval_hpsv2
```

Generates baseline and fine-tuned images from the same prompts and seeds, scores every reward in the spec over both, and writes `eval_results.json` with per-prompt and per-category breakdowns. `--baseline_dir` reuses a cached baseline across runs; because the baseline is LoRA-free and seed-fixed, the cached scores are identical to regenerating.

---

### Citation

```bibtex
@inproceedings{guo2025shortft,
    title     = {ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning},
    author    = {Guo, Xiefan and Cui, Miaomiao and Bo, Liefeng and Huang, Di},
    booktitle = {IEEE International Conference on Computer Vision (ICCV)},
    year      = {2025}
}
```
