"""
train.py
────────
Entry point for training the VLM UI Generator.

Usage
─────
    # Standard training
    python train.py

    # Override config values inline
    python train.py training.batch_size=4 training.num_epochs=20

    # Resume from checkpoint
    python train.py checkpointing.resume_from=checkpoints/checkpoint_epoch0010.pth
"""

import os
import random
import logging

import numpy as np
import torch
from omegaconf import OmegaConf

from data.dataset_loader  import build_dataloaders
from models.vlm_pipeline  import VLMUIPipeline
from training.trainer     import Trainer
from utils.logger         import setup_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # ── Load configs ─────────────────────────────────────────────────────────
    model_cfg    = OmegaConf.load("config/model_config.yaml")
    training_cfg = OmegaConf.load("config/training_config.yaml")
    cfg          = OmegaConf.merge(model_cfg, training_cfg)

    # CLI overrides: python train.py training.batch_size=4
    cli_cfg = OmegaConf.from_cli()
    cfg     = OmegaConf.merge(cfg, cli_cfg)

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = setup_logger("vlm_ui.train", log_file=cfg.logging.log_dir + "/train.log")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(cfg.training.seed)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info("Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Initialising model …")
    model = VLMUIPipeline(model_cfg=cfg, num_users=50_000)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
    )

    logger.info("Starting training …")
    trainer.train()


if __name__ == "__main__":
    main()
