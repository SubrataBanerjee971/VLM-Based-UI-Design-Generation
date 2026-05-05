"""
scheduler.py
────────────
Builds the AdamW optimizer and a cosine-with-warmup LR scheduler.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from omegaconf import DictConfig


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: DictConfig,
    num_training_steps: int,
):
    ocfg = cfg.optimizer
    scfg = cfg.scheduler

    # ── Parameter groups: no weight decay on biases / norms ──────────────────
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": ocfg.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        grouped_params,
        lr    = ocfg.lr,
        betas = tuple(ocfg.betas),
        eps   = ocfg.eps,
    )

    if scfg.type == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = scfg.warmup_steps,
            num_training_steps = num_training_steps,
            num_cycles         = scfg.num_cycles,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = scfg.warmup_steps,
            num_training_steps = num_training_steps,
        )

    return optimizer, scheduler
