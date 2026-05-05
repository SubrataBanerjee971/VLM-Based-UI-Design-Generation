"""
trainer.py
──────────
Training engine for the VLM UI Generator.

Implements two-phase training:
    Phase 1 (epochs 1–N1): Train cross-attention + user embeddings only
    Phase 2 (epochs N1–N):  End-to-end fine-tuning with diffusion loss

Supports:
    • Mixed precision  (fp16 / bf16)
    • Gradient accumulation
    • Gradient checkpointing
    • Checkpoint save & resume
    • WandB + CSV logging
"""

from __future__ import annotations

import os
import csv
import time
import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from models.vlm_pipeline import VLMUIPipeline
from training.losses     import MultiTaskLoss
from training.scheduler  import build_optimizer_and_scheduler
from utils.checkpoint    import save_checkpoint, load_checkpoint
from utils.metrics       import compute_rating_metrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: VLMUIPipeline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        tcfg = cfg.training
        lcfg = cfg.loss

        # ── Loss ─────────────────────────────────────────────────────────────
        self.criterion = MultiTaskLoss(
            lambda1     = lcfg.lambda1,
            lambda2     = lcfg.lambda2,
            lambda3     = lcfg.lambda3,
            lambda4     = 1.0,
            temperature = lcfg.temperature,
        ).to(device)

        # ── Optimiser & Scheduler ─────────────────────────────────────────────
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model=self.model,
            cfg=cfg,
            num_training_steps=len(train_loader) * tcfg.num_epochs // tcfg.gradient_accumulation_steps,
        )

        # ── Mixed precision ───────────────────────────────────────────────────
        self.use_amp = tcfg.mixed_precision in ("fp16", "bf16")
        self.scaler  = GradScaler(enabled=self.use_amp and tcfg.mixed_precision == "fp16")
        self.amp_dtype = torch.float16 if tcfg.mixed_precision == "fp16" else torch.bfloat16

        # ── Gradient checkpointing ────────────────────────────────────────────
        if tcfg.gradient_checkpointing:
            try:
                self.model.generator.unet.enable_gradient_checkpointing()
                self.model.generator.controlnet.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled.")
            except AttributeError:
                pass

        # ── State ────────────────────────────────────────────────────────────
        self.epoch          = 0
        self.global_step    = 0
        self.best_val_loss  = float("inf")

        # ── Logging ──────────────────────────────────────────────────────────
        log_dir = Path(cfg.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "train_log.csv"
        self._init_csv()

        if cfg.logging.use_wandb:
            import wandb
            wandb.init(project=cfg.logging.project_name, config=dict(cfg))

    # ── CSV logging ──────────────────────────────────────────────────────────

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "step", "phase",
                "loss_total", "loss_gen", "loss_rating", "loss_bpr", "loss_align",
                "lr",
            ])

    def _log_csv(self, row: dict):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "step", "phase",
                "loss_total", "loss_gen", "loss_rating", "loss_bpr", "loss_align",
                "lr",
            ])
            writer.writerow(row)

    # ── Phase management ─────────────────────────────────────────────────────

    def _set_phase(self, phase: int):
        """Freeze / unfreeze parameters depending on training phase."""
        if phase == 1:
            # Freeze encoders + generator; train only fusion + user preference
            for name, p in self.model.named_parameters():
                if any(m in name for m in ["fusion", "user_pref", "W_r", "rating_bias", "injector"]):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
            logger.info("[Phase 1] Training: fusion, user_pref, rating head, injector")
        else:
            # Unfreeze everything except frozen VAE / text encoder
            for name, p in self.model.named_parameters():
                if any(m in name for m in ["vae.", "text_encoder."]):
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
            logger.info("[Phase 2] End-to-end training (VAE + text_encoder frozen)")

    # ── Single batch step ────────────────────────────────────────────────────

    def _step(self, batch: dict, phase: int) -> dict[str, float]:
        sketch_pixels = batch["sketch"].to(self.device)          # CLIP-norm (B,3,224,224)
        target_images = batch["target_ui"].to(self.device)       # diffusion target
        sketch_cond   = batch["sketch"].to(self.device)          # ControlNet input
        prompts       = batch["prompt"]                          # list[str]
        user_ids      = batch.get(
            "user_id",
            torch.zeros(sketch_pixels.size(0), dtype=torch.long)
        ).to(self.device)

        # Dummy ratings in [1,5] for BPR if not provided
        ratings = batch.get(
            "rating",
            torch.rand(sketch_pixels.size(0)) * 4 + 1,
        ).to(self.device)

        # Tokenise prompts
        tok = self.model.encoder.tokenize(prompts, device=self.device)
        prompt_ids  = tok["input_ids"]
        prompt_mask = tok["attention_mask"]

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            out = self.model(
                sketch_pixels = sketch_pixels,
                target_images = target_images,
                sketch_cond   = sketch_cond,
                prompt_ids    = prompt_ids,
                prompt_mask   = prompt_mask,
                prompts       = prompts,
                user_ids      = user_ids,
                ratings       = ratings,
            )

            # Build negative samples for BPR (random shuffle)
            B = sketch_pixels.size(0)
            neg_idx     = torch.randperm(B, device=self.device)
            r_hat_neg   = out["r_hat"][neg_idx]

            # Image & text embeddings for alignment loss
            v_embed = self.model.encoder.get_image_embedding(sketch_pixels)
            t_embed = self.model.encoder.get_text_embedding(prompt_ids, prompt_mask)

            losses = self.criterion(
                r_hat     = out["r_hat"],
                r_true    = ratings,
                r_hat_pos = out["r_hat"],
                r_hat_neg = r_hat_neg,
                v_embed   = v_embed,
                t_embed   = t_embed,
                gen_loss  = out["gen_loss"] if phase == 2 else out["gen_loss"].detach(),
            )

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    # ── Train one epoch ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int, phase: int):
        self.model.train()
        self.optimizer.zero_grad()
        accum_steps = self.cfg.training.gradient_accumulation_steps
        log_every   = self.cfg.logging.log_every_n_steps

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Phase {phase}]", leave=False)

        for step_in_epoch, batch in enumerate(pbar):
            loss_dict = self._step(batch, phase)
            loss      = torch.tensor(loss_dict["total"], device=self.device,
                                     requires_grad=False)

            # Re-compute backward with autocast
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                tok = self.model.encoder.tokenize(batch["prompt"], device=self.device)
                out = self.model(
                    sketch_pixels = batch["sketch"].to(self.device),
                    target_images = batch["target_ui"].to(self.device),
                    sketch_cond   = batch["sketch"].to(self.device),
                    prompt_ids    = tok["input_ids"],
                    prompt_mask   = tok["attention_mask"],
                    prompts       = batch["prompt"],
                    user_ids      = batch.get(
                        "user_id", torch.zeros(batch["sketch"].size(0), dtype=torch.long)
                    ).to(self.device),
                    ratings       = batch.get(
                        "rating", torch.rand(batch["sketch"].size(0)) * 4 + 1
                    ).to(self.device),
                )
                B       = batch["sketch"].size(0)
                neg_idx = torch.randperm(B, device=self.device)
                v_emb   = self.model.encoder.get_image_embedding(batch["sketch"].to(self.device))
                t_emb   = self.model.encoder.get_text_embedding(tok["input_ids"], tok["attention_mask"])
                losses  = self.criterion(
                    r_hat     = out["r_hat"],
                    r_true    = batch.get("rating", torch.rand(B)*4+1).to(self.device),
                    r_hat_pos = out["r_hat"],
                    r_hat_neg = out["r_hat"][neg_idx],
                    v_embed   = v_emb,
                    t_embed   = t_emb,
                    gen_loss  = out["gen_loss"] if phase == 2 else out["gen_loss"].detach(),
                )
                total_step_loss = losses["total"] / accum_steps

            self.scaler.scale(total_step_loss).backward()

            if (step_in_epoch + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += losses["total"].item()
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

            if self.global_step % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                row = {
                    "epoch"       : epoch,
                    "step"        : self.global_step,
                    "phase"       : phase,
                    "loss_total"  : losses["total"].item(),
                    "loss_gen"    : losses["gen"].item(),
                    "loss_rating" : losses["rating"].item(),
                    "loss_bpr"    : losses["bpr"].item(),
                    "loss_align"  : losses["align"].item(),
                    "lr"          : lr,
                }
                self._log_csv(row)

                if self.cfg.logging.use_wandb:
                    import wandb
                    wandb.log(row, step=self.global_step)

        return total_loss / max(len(self.train_loader), 1)

    # ── Validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        all_r_hat, all_r_true = [], []

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            tok = self.model.encoder.tokenize(batch["prompt"], device=self.device)
            out = self.model(
                sketch_pixels = batch["sketch"].to(self.device),
                target_images = batch["target_ui"].to(self.device),
                sketch_cond   = batch["sketch"].to(self.device),
                prompt_ids    = tok["input_ids"],
                prompt_mask   = tok["attention_mask"],
                prompts       = batch["prompt"],
                user_ids      = batch.get(
                    "user_id", torch.zeros(batch["sketch"].size(0), dtype=torch.long)
                ).to(self.device),
                ratings       = batch.get(
                    "rating", torch.rand(batch["sketch"].size(0)) * 4 + 1
                ).to(self.device),
            )
            total_loss += out["gen_loss"].item() + out["rating_loss"].item()
            all_r_hat.append(out["r_hat"].cpu())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        metrics  = compute_rating_metrics(
            torch.cat(all_r_hat).numpy(),
            torch.ones(len(torch.cat(all_r_hat))).numpy() * 3.0,  # placeholder
        )
        logger.info(f"[Val] Epoch {epoch} | loss={avg_loss:.4f} | MAE={metrics['mae']:.4f}")
        return avg_loss

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self):
        tcfg     = self.cfg.training
        ckpt_dir = Path(self.cfg.checkpointing.save_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if self.cfg.checkpointing.resume_from:
            self.epoch, self.global_step = load_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.cfg.checkpointing.resume_from,
            )
            logger.info(f"Resumed from epoch {self.epoch}")

        total_epochs = tcfg.num_epochs
        phase1_end   = tcfg.phase1_epochs

        for epoch in range(self.epoch + 1, total_epochs + 1):
            self.epoch = epoch
            phase = 1 if epoch <= phase1_end else 2
            self._set_phase(phase)

            t0         = time.time()
            train_loss = self._train_epoch(epoch, phase)
            elapsed    = time.time() - t0

            logger.info(
                f"Epoch {epoch}/{total_epochs} | phase={phase} | "
                f"train_loss={train_loss:.4f} | time={elapsed:.1f}s"
            )

            # Validation
            if epoch % self.cfg.evaluation.eval_every_n_epochs == 0:
                val_loss = self._validate(epoch)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.global_step,
                        ckpt_dir / "best_model.pth",
                    )
                    logger.info(f"  ↳ New best model saved (val_loss={val_loss:.4f})")

            # Periodic checkpoint
            if epoch % self.cfg.checkpointing.save_every_n_epochs == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step,
                    ckpt_dir / f"checkpoint_epoch{epoch:04d}.pth",
                )

        logger.info("Training complete.")
