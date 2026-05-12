"""
dataset_loader.py
─────────────────
Loads and merges three datasets:
  1. HuggingFace  : mrtoy/mobile-ui-design         (polished UI images + metadata)
  2. Kaggle       : vinothpandian/uisketch          (hand-drawn UI component sketches)
  3. Kaggle       : antrixsh/prompt-engineering-…   (natural-language prompts)

Returns a unified torch Dataset of (sketch, target_ui, prompt) triplets.

Key design decision: ALL image loading is LAZY (deferred to __getitem__).
The __init__ only builds a lightweight sample manifest — no pixels are loaded
or resized during startup, so dataset construction is fast.
"""

from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import kagglehub

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Raw loaders  (return lightweight metadata only)
# ──────────────────────────────────────────────────────────────────────────────

def load_mobile_ui_dataset(split: str = "train"):
    """Load the HuggingFace mobile-ui-design dataset (streaming-safe)."""
    logger.info("Loading mrtoy/mobile-ui-design from HuggingFace ...")
    ds = load_dataset("mrtoy/mobile-ui-design", split=split)
    logger.info(f"  -> {len(ds)} samples loaded (split={split})")
    return ds


def load_ui_sketch_dataset() -> pd.DataFrame:
    """
    Download the Kaggle UI-sketch dataset and return a manifest DataFrame.
    Columns: [name, label, medium, device, full_path]
    No images are loaded here.
    """
    logger.info("Downloading vinothpandian/uisketch from Kaggle ...")
    dataset_path = Path(kagglehub.dataset_download("vinothpandian/uisketch"))

    # Search for labels.csv anywhere in the downloaded folder
    csv_path = None
    for p in dataset_path.rglob("labels.csv"):
        csv_path = p
        break
    
    if not csv_path or not csv_path.exists():
        # Try any CSV if labels.csv isn't found
        csvs = list(dataset_path.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {dataset_path}")
        csv_path = csvs[0]

    logger.info(f"  -> Found sketch manifest at: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Correct paths if the CSV is in a subdirectory
    base_dir = csv_path.parent
    df["full_path"] = df["name"].apply(lambda n: str(base_dir / n))
    
    df = df[df["full_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    logger.info(f"  -> {len(df)} sketch records in manifest")
    return df


def load_prompt_dataset() -> pd.DataFrame:
    """
    Download the prompt dataset from Kaggle and return it as a DataFrame.
    Columns include: Prompt, Prompt_Type, Prompt_Length, Response
    """
    logger.info("Downloading antrixsh/prompt-engineering-and-responses-dataset ...")
    dataset_path = Path(kagglehub.dataset_download(
        "antrixsh/prompt-engineering-and-responses-dataset"
    ))
    
    csv_path = None
    # Search recursively for the CSV
    for p in dataset_path.rglob("*.csv"):
        csv_path = p
        break

    if not csv_path or not csv_path.exists():
        raise FileNotFoundError(f"No CSV found in {dataset_path}")

    logger.info(f"  -> Found prompt dataset at: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  -> {len(df)} prompts loaded from {csv_path.name}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  UI-centric prompt builder
# ──────────────────────────────────────────────────────────────────────────────

UI_PROMPT_TEMPLATES = [
    "Design a mobile UI based on this sketch.",
    "Convert this hand-drawn wireframe into a polished mobile UI design.",
    "Generate a professional mobile app interface from this sketch.",
    "Create a high-fidelity UI design based on this wireframe sketch.",
    "Turn this rough sketch into a clean mobile application interface.",
    "Design a modern mobile app UI following this hand-drawn layout.",
    "Based on this sketch, create a polished mobile user interface.",
    "Transform this wireframe into a complete mobile UI with proper styling.",
]


def build_ui_prompts(prompt_df: Optional[pd.DataFrame], n: int) -> list[str]:
    """
    Extract UI-relevant prompts from the prompt dataset, padded/trimmed to n.
    Falls back to built-in templates if the dataset is unavailable or too small.
    """
    prompts: list[str] = []

    if prompt_df is not None:
        text_col = None
        for col in ["Prompt", "prompt", "instruction", "text", "question", "input"]:
            if col in prompt_df.columns:
                text_col = col
                break

        if text_col:
            raw = prompt_df[text_col].dropna().tolist()
            ui_keywords = {
                "ui", "interface", "app", "design", "mobile", "screen",
                "layout", "wireframe", "sketch",
            }
            prompts = [p for p in raw
                       if any(kw in str(p).lower() for kw in ui_keywords)]
            logger.info(f"  -> {len(prompts)} UI-relevant prompts extracted")

    while len(prompts) < n:
        prompts.extend(UI_PROMPT_TEMPLATES)

    random.shuffle(prompts)
    return prompts[:n]


# ──────────────────────────────────────────────────────────────────────────────
#  Sketch simulation helper
# ──────────────────────────────────────────────────────────────────────────────

def _image_to_sketch(img: Image.Image) -> Image.Image:
    """Canny-edge sketch simulation — CPU-only, no GPU needed."""
    import cv2
    arr = np.array(img.convert("L"))
    edges = cv2.Canny(arr, threshold1=50, threshold2=150)
    return Image.fromarray(255 - edges).convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
#  Unified Dataset  (fully lazy — no pixels loaded at __init__ time)
# ──────────────────────────────────────────────────────────────────────────────

class UIGenerationDataset(Dataset):
    """
    Each item returns:
        {
            "sketch"    : torch.Tensor  – sketch image (CLIP/ControlNet input)
            "target_ui" : torch.Tensor  – polished UI design (generation target)
            "prompt"    : str           – natural language instruction
        }

    Sample manifest is built eagerly (just indices / paths) but all pixel I/O
    is deferred to __getitem__ to keep startup fast.

    Sample types in the manifest:
        {"type": "hf_synthetic",  "idx": int}              → HF image + edge sketch
        {"type": "sketch_component", "sketch_path": str, "hf_idx": int}
                                                            → uisketch image + HF target
    """

    def __init__(
        self,
        hf_dataset,
        sketch_df: Optional[pd.DataFrame],
        prompt_df: Optional[pd.DataFrame],
        transform=None,
        sketch_transform=None,
        image_size: int = 512,
    ):
        self.hf_dataset      = hf_dataset
        self.image_size      = image_size
        self.transform       = transform
        self.sketch_transform = sketch_transform

        # Auto-detect image column once
        self._image_col = self._detect_image_column(hf_dataset)

        # ── Build lightweight sample manifest (NO pixel loading) ───────────
        self.samples: list[dict] = []

        n_hf = len(hf_dataset)

        # 1. Synthetic sketch pairs from HF dataset
        for idx in range(n_hf):
            self.samples.append({"type": "hf_synthetic", "idx": idx})

        # 2. uisketch component images paired with HF targets (paper-medium only)
        if sketch_df is not None and len(sketch_df) > 0:
            hand_drawn = (
                sketch_df[sketch_df["medium"] == "paper"]
                if "medium" in sketch_df.columns else sketch_df
            )
            if len(hand_drawn) == 0:
                hand_drawn = sketch_df
            for i, (_, row) in enumerate(hand_drawn.iterrows()):
                self.samples.append({
                    "type": "sketch_component",
                    "sketch_path": str(row["full_path"]),
                    "hf_idx": i % n_hf,
                })
            logger.info(f"  -> {len(hand_drawn)} uisketch component entries added to manifest")

        # ── Prompts (just strings — always fast) ─────────────────────────
        n = len(self.samples)
        self.prompts = build_ui_prompts(prompt_df, n)
        logger.info(f"UIGenerationDataset: {n} total samples in manifest (lazy).")

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_image_column(dataset) -> str:
        for col in ["image", "img", "screenshot", "photo", "pixel_values"]:
            if col in dataset.features:
                return col
        return dataset.column_names[0]

    def _load_hf_image(self, idx: int) -> Image.Image:
        """Load and resize one HF dataset image."""
        raw = self.hf_dataset[idx][self._image_col]
        if not isinstance(raw, Image.Image):
            raw = Image.fromarray(raw)
        return raw.convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )

    def _apply_transforms(self, sketch: Image.Image, target: Image.Image):
        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)
        if self.transform:
            target = self.transform(target)
        return sketch, target

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt = self.prompts[idx % len(self.prompts)]

        if sample["type"] == "hf_synthetic":
            target = self._load_hf_image(sample["idx"])
            sketch = _image_to_sketch(target)

        else:  # sketch_component
            # Load real hand-drawn sketch from uisketch
            try:
                sketch = (
                    Image.open(sample["sketch_path"])
                    .convert("RGB")
                    .resize((self.image_size, self.image_size), Image.BICUBIC)
                )
            except Exception:
                sketch = Image.new("RGB", (self.image_size, self.image_size), 128)

            # Pair with corresponding HF target
            target = self._load_hf_image(sample["hf_idx"])

        sketch, target = self._apply_transforms(sketch, target)

        return {
            "sketch":    sketch,
            "target_ui": target,
            "prompt":    prompt,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg,
    transform=None,
    sketch_transform=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Splits the full dataset according to config ratios.
    """
    from data.preprocessor import get_transforms

    image_size = cfg.data.image_size
    t_transform, s_transform = (
        get_transforms(image_size)
        if transform is None
        else (transform, sketch_transform)
    )

    # Load raw dataset manifests (fast — no pixel I/O)
    hf_ds = load_mobile_ui_dataset("train")

    try:
        sketch_df = load_ui_sketch_dataset()
    except Exception as e:
        logger.warning(f"Could not load uisketch: {e}. Proceeding without component sketches.")
        sketch_df = None

    try:
        prompt_df = load_prompt_dataset()
    except Exception as e:
        logger.warning(f"Could not load prompt dataset: {e}. Using built-in templates.")
        prompt_df = None

    full_dataset = UIGenerationDataset(
        hf_dataset       = hf_ds,
        sketch_df        = sketch_df,
        prompt_df        = prompt_df,
        transform        = t_transform,
        sketch_transform = s_transform,
        image_size       = image_size,
    )

    n       = len(full_dataset)
    n_train = int(n * cfg.data.train_split)
    n_val   = int(n * cfg.data.val_split)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    # num_workers=0 on Windows/CPU to avoid multiprocessing spawn issues
    num_workers = getattr(cfg.data, "num_workers", 0)
    pin_memory  = getattr(cfg.data, "pin_memory", False)

    loader_kwargs = dict(
        batch_size  = cfg.training.batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    logger.info(f"DataLoaders built - train:{n_train}  val:{n_val}  test:{n_test}")
    return train_loader, val_loader, test_loader
