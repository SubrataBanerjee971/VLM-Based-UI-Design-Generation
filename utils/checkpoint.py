"""
checkpoint.py
─────────────
Save and load model checkpoints with full training state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, path):
    """Save full training state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"        : epoch,
        "global_step"  : global_step,
        "model_state"  : model.state_dict(),
        "optimizer"    : optimizer.state_dict(),
        "scheduler"    : scheduler.state_dict(),
    }, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, scheduler, path) -> tuple[int, int]:
    """Load training state. Returns (epoch, global_step)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch       = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    logger.info(f"Checkpoint loaded ← {path} (epoch={epoch})")
    return epoch, global_step


def load_checkpoint_for_inference(model, path, device="cuda"):
    """Load only model weights (no optimizer state)."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    logger.info(f"Model weights loaded from {path}")
