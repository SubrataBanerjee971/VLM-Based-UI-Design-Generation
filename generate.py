"""
generate.py
───────────
CLI script to generate a UI design from a sketch + prompt.

Usage
─────
    # Basic
    python generate.py --sketch my_sketch.jpg --prompt "Login screen with email and password"

    # With options
    python generate.py \
        --sketch my_sketch.jpg \
        --prompt "E-commerce product page with search bar and carousel" \
        --checkpoint checkpoints/best_model.pth \
        --output outputs/generated/my_ui.png \
        --steps 50 \
        --guidance 7.5 \
        --seed 42

    # Pre-process sketch first (enhances edges)
    python generate.py --sketch raw_photo.jpg --preprocess --prompt "..."
"""

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image

from inference.inference_engine import InferenceEngine
from utils.visualizer           import save_sketch_ui_pair
from utils.logger               import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="VLM UI Generator – Sketch → UI Image")

    p.add_argument("--sketch",     required=True, help="Path to hand-drawn sketch image")
    p.add_argument("--prompt",     default=None,  help="Text description of desired UI")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                   help="Path to model checkpoint")
    p.add_argument("--output",     default="outputs/generated/result.png",
                   help="Output image path")
    p.add_argument("--steps",      type=int,   default=50,  help="Denoising steps")
    p.add_argument("--guidance",   type=float, default=7.5, help="CFG guidance scale")
    p.add_argument("--controlnet-scale", type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--user-id",    type=int,   default=None,
                   help="User ID for personalised generation")
    p.add_argument("--preprocess", action="store_true",
                   help="Pre-process sketch (edge enhancement)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compare",    action="store_true",
                   help="Save side-by-side sketch vs generated comparison")

    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logger("vlm_ui.generate")

    # ── Load sketch ──────────────────────────────────────────────────────────
    sketch_path = Path(args.sketch)
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch not found: {sketch_path}")

    if args.preprocess:
        logger.info("Pre-processing sketch …")
        sketch = InferenceEngine.preprocess_sketch(sketch_path, output_size=512)
    else:
        sketch = Image.open(sketch_path).convert("RGB")

    logger.info(f"Sketch loaded: {sketch.size}")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.checkpoint} …")
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path    = args.checkpoint,
        model_config_path  = "config/model_config.yaml",
        device             = args.device,
        default_steps      = args.steps,
        default_guidance   = args.guidance,
        default_controlnet_scale = args.controlnet_scale,
    )

    # ── Generate ─────────────────────────────────────────────────────────────
    logger.info("Generating UI …")
    result = engine.generate(
        sketch_image   = sketch,
        prompt         = args.prompt,
        user_id        = args.user_id,
        num_steps      = args.steps,
        guidance_scale = args.guidance,
        seed           = args.seed,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    logger.info(f"Generated UI saved → {out_path}")

    if args.compare:
        comp_path = out_path.with_stem(out_path.stem + "_compare")
        save_sketch_ui_pair(
            sketch    = sketch,
            generated = result,
            save_path = comp_path,
            prompt    = args.prompt or "",
        )
        logger.info(f"Comparison saved → {comp_path}")


if __name__ == "__main__":
    main()
