# VLM UI Generator

> **Sketch → Polished Mobile UI** using a Visual Language Model + Stable Diffusion ControlNet

---

## Overview

This project implements a **VLM-powered UI generation system** that:

1. **Trains** on real mobile UI design images (HuggingFace) and hand-drawn UI sketches (Kaggle)
2. **Learns** multimodal representations that understand both UI images and natural language prompts
3. **Generates** polished mobile UI designs from a hand-drawn sketch + text prompt

### Architecture

```
Hand-drawn Sketch ──► CLIP Image Encoder ──► Patch Embeddings (B, P, d)
                                                       │
Text Prompt       ──► CLIP Text Encoder  ──► Token Embeddings (B, L, d)
                                                       │
                              Cross-Attention Fusion (Text attends to Image)
                                                       │
                                              h_ui ∈ R^d
                                           /              \
                           User Preference Module    FusedConditioningInjector
                                  p_u                         │
                                   │              Stable Diffusion UNet
                           Rating Prediction         + ControlNet (scribble)
                            r̂_ui = σ(p_u^T W_r h_ui)         │
                                                       Generated UI Image
```

### Datasets

| Dataset | Source | Role |
|---|---|---|
| `mrtoy/mobile-ui-design` | HuggingFace | Target UI images |
| `vinothpandian/uisketch` | Kaggle | Real sketch↔UI pairs |
| `antrixsh/prompt-engineering-and-responses-dataset` | Kaggle | Natural language prompts |

### Loss Functions (Multi-task)

```
L = λ1·L_rating + λ2·L_BPR + λ3·L_align + λ4·L_gen

L_rating = (1/|D|) Σ (r_ui − r̂_ui)²          # Rating regression (MSE)
L_BPR    = −Σ log σ(r̂_ui − r̂_uj)             # Pairwise ranking (BPR)
L_align  = −log[exp(sim(v,t)/τ) / Σ exp(...)]  # Cross-modal alignment (InfoNCE)
L_gen    = MSE(ε, ε̂)                           # Diffusion denoising
```

---

## Project Structure

```
vlm_ui_generator/
├── config/
│   ├── model_config.yaml       # CLIP, cross-attention, generator hyperparams
│   └── training_config.yaml    # epochs, batch size, optimizer, loss weights
│
├── data/
│   ├── dataset_loader.py       # Load & merge all 3 datasets → DataLoaders
│   ├── preprocessor.py         # Image transforms (CLIP / diffusion / sketch)
│   └── augmentation.py         # Paired augmentation (sketch ↔ target)
│
├── models/
│   ├── vlm_encoder.py          # CLIP encoder + LoRA fine-tuning
│   ├── cross_attention.py      # Cross-attention fusion module (paper §3)
│   ├── user_preference.py      # User preference learning (paper §4)
│   ├── ui_generator.py         # Stable Diffusion + ControlNet wrapper
│   └── vlm_pipeline.py         # Full end-to-end pipeline
│
├── training/
│   ├── trainer.py              # Two-phase training loop
│   ├── losses.py               # All loss functions (paper §6)
│   └── scheduler.py            # AdamW + cosine LR scheduler
│
├── inference/
│   └── inference_engine.py     # High-level generate() API
│
├── utils/
│   ├── checkpoint.py           # Save / load checkpoints
│   ├── metrics.py              # MAE, RMSE, SSIM, PSNR, FID
│   ├── visualizer.py           # Image grids, loss curves
│   └── logger.py               # Logging setup
│
├── train.py                    # ← Training entry point
├── generate.py                 # ← Inference entry point
├── evaluate.py                 # ← Evaluation entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

You also need Kaggle credentials for the Kaggle datasets:
```bash
# Place your kaggle.json in ~/.kaggle/kaggle.json
# OR set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 2. Train

```bash
python train.py
```

Override any config value inline:
```bash
python train.py training.batch_size=4 training.num_epochs=20 training.mixed_precision=fp16
```

Resume from a checkpoint:
```bash
python train.py checkpointing.resume_from=checkpoints/checkpoint_epoch0010.pth
```

### 3. Generate a UI from your sketch

```bash
python generate.py \
    --sketch my_sketch.jpg \
    --prompt "Design a login screen with email, password fields and a sign-in button" \
    --checkpoint checkpoints/best_model.pth \
    --output outputs/generated/login_ui.png \
    --compare
```

With sketch pre-processing (recommended for photos of paper sketches):
```bash
python generate.py \
    --sketch photo_of_sketch.jpg \
    --preprocess \
    --prompt "Mobile e-commerce home screen with search and product grid" \
    --seed 42
```

### 4. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test --n-samples 100
```

---

## Training Phases

| Phase | Epochs | What trains |
|---|---|---|
| **Phase 1** | 1 – 10 | Cross-attention fusion, user preference, rating head, conditioning injector |
| **Phase 2** | 11 – 30 | End-to-end (CLIP LoRA + UNet + ControlNet + all above) |

---

## Configuration

Key settings in `config/training_config.yaml`:

```yaml
training:
  num_epochs: 30
  batch_size: 8
  mixed_precision: "fp16"    # Use fp16 for faster training on GPU

loss:
  lambda1: 1.0   # Rating loss weight
  lambda2: 0.5   # BPR ranking loss weight
  lambda3: 0.5   # Cross-modal alignment weight

generator:
  num_inference_steps: 50
  guidance_scale: 7.5
  controlnet_conditioning_scale: 1.0
```

---

## Hardware Requirements

| Config | VRAM | Speed |
|---|---|---|
| fp16 + batch 8 + gradient checkpointing | ~16 GB | Recommended |
| fp16 + batch 4 | ~10 GB | Minimum |
| CPU (testing only) | – | Very slow |

---

## Example Prompts

```
"Design a mobile login screen with email and password fields and a Google sign-in button"
"Create a clean e-commerce product listing page with search bar and filter chips"
"Generate a fitness tracking app home screen showing steps, calories and heart rate"
"Design a chat messaging interface with message bubbles and an input bar"
"Create an onboarding screen for a travel app with a beautiful hero image"
```
