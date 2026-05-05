"""
evaluate.py
───────────
Evaluation script.

Usage
─────
    python evaluate.py --checkpoint checkpoints/best_model.pth --split test
"""

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

from data.dataset_loader      import build_dataloaders
from models.vlm_pipeline      import VLMUIPipeline
from utils.checkpoint         import load_checkpoint_for_inference
from utils.metrics            import compute_rating_metrics, compute_ssim, compute_psnr
from utils.visualizer         import save_image_grid
from utils.logger             import setup_logger
from inference.inference_engine import InferenceEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--split",      default="test", choices=["val", "test"])
    p.add_argument("--n-samples",  type=int, default=64,
                   help="Number of samples to evaluate (0 = all)")
    p.add_argument("--output-dir", default="outputs/eval")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logger("vlm_ui.eval")

    cfg       = OmegaConf.merge(
        OmegaConf.load("config/model_config.yaml"),
        OmegaConf.load("config/training_config.yaml"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = VLMUIPipeline(cfg, num_users=50_000)
    load_checkpoint_for_inference(model, args.checkpoint, device=args.device)
    engine = InferenceEngine(model, device=args.device)

    # ── Data ───────────────────────────────────────────────────────────────
    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if args.split == "test" else val_loader

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate ───────────────────────────────────────────────────────────
    ssim_scores, psnr_scores = [], []
    generated_imgs, gt_imgs  = [], []

    n_done = 0
    max_n  = args.n_samples if args.n_samples > 0 else float("inf")

    for batch in tqdm(loader, desc="Evaluating"):
        for i in range(len(batch["prompt"])):
            if n_done >= max_n:
                break

            sketch_pil = Image.fromarray(
                ((batch["sketch"][i].permute(1,2,0).numpy() * 0.5 + 0.5) * 255).astype("uint8")
            )
            gt_pil = Image.fromarray(
                ((batch["target_ui"][i].permute(1,2,0).numpy() * 0.5 + 0.5) * 255).astype("uint8")
            )
            prompt = batch["prompt"][i]

            gen_pil = engine.generate(sketch_image=sketch_pil, prompt=prompt, seed=42)

            ssim_scores.append(compute_ssim(gen_pil, gt_pil))
            psnr_scores.append(compute_psnr(gen_pil, gt_pil))
            generated_imgs.append(gen_pil)
            gt_imgs.append(gt_pil)

            n_done += 1

        if n_done >= max_n:
            break

    # ── Report ─────────────────────────────────────────────────────────────
    import numpy as np
    results = {
        "n_samples"  : n_done,
        "mean_ssim"  : float(np.mean(ssim_scores)),
        "mean_psnr"  : float(np.mean(psnr_scores)),
    }
    logger.info("Evaluation Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    # Save results
    import json
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Visual grid ────────────────────────────────────────────────────────
    save_image_grid(
        generated_imgs[:16],
        titles=[f"Gen {i+1}" for i in range(min(16, len(generated_imgs)))],
        save_path=out_dir / "generated_grid.png",
    )
    save_image_grid(
        gt_imgs[:16],
        titles=[f"GT {i+1}" for i in range(min(16, len(gt_imgs)))],
        save_path=out_dir / "ground_truth_grid.png",
    )
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
