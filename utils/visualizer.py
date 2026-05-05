"""
visualizer.py
─────────────
Utilities for visualising training progress and generated UI samples.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_image_grid(
    images: Sequence[Image.Image],
    titles: Sequence[str] | None = None,
    save_path: str | Path = "grid.png",
    ncols: int = 4,
    figsize_per: tuple[float, float] = (3.0, 3.5),
):
    """Save a grid of PIL images with optional titles."""
    n     = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per[0], nrows * figsize_per[1]),
    )
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            if titles:
                ax.set_title(titles[i], fontsize=7, wrap=True)
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(log_csv_path: str, save_path: str = "outputs/logs/loss_curves.png"):
    """Read train_log.csv and plot all loss curves."""
    import pandas as pd
    df = pd.read_csv(log_csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    loss_keys = ["loss_total", "loss_gen", "loss_rating", "loss_align"]
    titles    = ["Total Loss", "Generation Loss", "Rating Loss", "Alignment Loss"]

    for ax, key, title in zip(axes.flatten(), loss_keys, titles):
        if key in df.columns:
            ax.plot(df["step"], df[key], linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)

    plt.suptitle("Training Loss Curves", fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved → {save_path}")


def save_sketch_ui_pair(
    sketch: Image.Image,
    generated: Image.Image,
    ground_truth: Image.Image | None = None,
    save_path: str = "pair.png",
    prompt: str = "",
):
    """Save a side-by-side comparison: sketch | generated | (optional) ground truth."""
    imgs   = [sketch, generated]
    labels = ["Input Sketch", "Generated UI"]
    if ground_truth is not None:
        imgs.append(ground_truth)
        labels.append("Ground Truth")

    ncols = len(imgs)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    if prompt:
        fig.suptitle(f"Prompt: {prompt[:100]}", fontsize=8, y=0.02)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
