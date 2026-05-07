"""
run_training.py
───────────────
Training launcher.

Sets Kaggle credentials from environment variables BEFORE importing
any project modules (kagglehub reads env vars at import time).

Usage
─────
    python run_training.py

    # Override config values inline
    python run_training.py training.batch_size=4 training.num_epochs=20

    # Resume from checkpoint
    python run_training.py checkpointing.resume_from=checkpoints/checkpoint_epoch0010.pth
"""

import os

# ── Kaggle credentials ──────────────────────────────────────────────────────
# These must be set BEFORE any kagglehub import
os.environ.setdefault("KAGGLE_USERNAME", "subratabanerjeerony")
os.environ.setdefault("KAGGLE_KEY",      "KGAT_e248bc7a00abd94d8535b09a822d36a3")

# ── Run the main training entry-point ───────────────────────────────────────
from train import main

if __name__ == "__main__":
    main()
