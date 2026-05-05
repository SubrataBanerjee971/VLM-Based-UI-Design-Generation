"""
dataset_loader.py
─────────────────
Loads and merges three datasets:
  1. HuggingFace  : mrtoy/mobile-ui-design         (polished UI images + metadata)
  2. Kaggle       : vinothpandian/uisketch          (hand-drawn UI sketches)
  3. Kaggle       : antrixsh/prompt-engineering-…   (natural-language prompts)

Returns a unified torch Dataset of (sketch, target_ui, prompt) triplets.
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
from kagglehub import KaggleDatasetAdapter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Raw loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_mobile_ui_dataset(split: str = "train"):
    """Load the HuggingFace mobile-ui-design dataset."""
    logger.info("Loading mrtoy/mobile-ui-design from HuggingFace …")
    ds = load_dataset("mrtoy/mobile-ui-design", split=split)
    logger.info(f"  ↳ {len(ds)} samples loaded (split={split})")
    return ds


def load_ui_sketch_dataset() -> pd.DataFrame:
    """Load the Kaggle UI-sketch dataset."""
    logger.info("Loading vinothpandian/uisketch from Kaggle …")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "vinothpandian/uisketch",
        "",
    )
    logger.info(f"  ↳ {len(df)} records loaded")
    return df


def load_prompt_dataset() -> pd.DataFrame:
    """Load the prompt-engineering dataset from Kaggle."""
    logger.info("Loading antrixsh/prompt-engineering-and-responses-dataset …")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "antrixsh/prompt-engineering-and-responses-dataset",
        "",
    )
    logger.info(f"  ↳ {len(df)} records loaded")
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
    Pull UI-relevant prompts from the prompt dataset, padded/trimmed to n.
    Falls back to built-in templates when the dataset is unavailable.
    """
    prompts: list[str] = []

    if prompt_df is not None:
        # Try to find a text column
        text_col = None
        for col in ["prompt", "instruction", "text", "question", "input"]:
            if col in prompt_df.columns:
                text_col = col
                break

        if text_col:
            raw = prompt_df[text_col].dropna().tolist()
            # Keep only prompts that mention UI / design / interface / app
            ui_keywords = {"ui", "interface", "app", "design", "mobile", "screen",
                           "layout", "wireframe", "sketch"}
            ui_prompts = [p for p in raw
                          if any(kw in str(p).lower() for kw in ui_keywords)]
            prompts = ui_prompts

    # Pad with templates if needed
    while len(prompts) < n:
        prompts.extend(UI_PROMPT_TEMPLATES)

    random.shuffle(prompts)
    return prompts[:n]


# ──────────────────────────────────────────────────────────────────────────────
#  Unified Dataset
# ──────────────────────────────────────────────────────────────────────────────

class UIGenerationDataset(Dataset):
    """
    Each item: {
        "sketch"    : PIL.Image  – hand-drawn sketch (input conditioning)
        "target_ui" : PIL.Image  – polished UI design (generation target)
        "prompt"    : str        – natural language instruction
    }

    Strategy
    --------
    1. Use mobile-ui-design images as *target_ui* references.
    2. Apply edge-detection / sketch simulation on target_ui to synthesise
       *sketch* images when real sketch pairs are unavailable.
    3. When real sketch pairs ARE available (uisketch), use them directly.
    4. Pair each image with a UI-relevant prompt from the prompt dataset.
    """

    def __init__(
        self,
        hf_dataset,
        sketch_df: Optional[pd.DataFrame],
        prompt_df: Optional[pd.DataFrame],
        transform=None,
        sketch_transform=None,
        image_size: int = 512,
        use_synthetic_sketches: bool = True,
    ):
        self.image_size = image_size
        self.transform = transform
        self.sketch_transform = sketch_transform
        self.use_synthetic_sketches = use_synthetic_sketches

        # ── target UI images from HuggingFace dataset ──────────────────────
        self.target_images: list[Image.Image] = []
        self.synthetic_sketches: list[Image.Image] = []  # edge-map versions

        image_col = self._detect_image_column(hf_dataset)
        for sample in hf_dataset:
            img = sample[image_col]
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img).convert("RGB")
                except Exception:
                    continue
            img = img.convert("RGB").resize((image_size, image_size))
            self.target_images.append(img)
            if use_synthetic_sketches:
                self.synthetic_sketches.append(self._to_sketch(img))

        # ── real sketch pairs from Kaggle uisketch ─────────────────────────
        self.real_sketch_pairs: list[tuple[Image.Image, Image.Image]] = []
        if sketch_df is not None:
            self._load_sketch_pairs(sketch_df, image_size)

        # ── build unified sample list ──────────────────────────────────────
        self.samples: list[dict] = []

        # Real pairs first
        for sketch, target in self.real_sketch_pairs:
            self.samples.append({"sketch": sketch, "target_ui": target})

        # Synthetic pairs
        for idx, (sketch, target) in enumerate(
            zip(self.synthetic_sketches, self.target_images)
        ):
            self.samples.append({"sketch": sketch, "target_ui": target})

        # ── prompts ────────────────────────────────────────────────────────
        n = len(self.samples)
        self.prompts = build_ui_prompts(prompt_df, n)
        logger.info(f"UIGenerationDataset: {n} total samples built.")

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_image_column(dataset) -> str:
        """Auto-detect the image column in an HF dataset."""
        for col in ["image", "img", "screenshot", "photo", "pixel_values"]:
            if col in dataset.features:
                return col
        # Fallback: first column
        return dataset.column_names[0]

    @staticmethod
    def _to_sketch(img: Image.Image) -> Image.Image:
        """
        Simulate a hand-drawn sketch via Canny edge detection.
        Works without a GPU – pure NumPy / OpenCV.
        """
        import cv2
        arr = np.array(img.convert("L"))
        edges = cv2.Canny(arr, threshold1=50, threshold2=150)
        # Invert: white background, black strokes (like real sketches)
        edges = 255 - edges
        return Image.fromarray(edges).convert("RGB")

    def _load_sketch_pairs(self, df: pd.DataFrame, image_size: int):
        """
        Load sketch↔design pairs from the Kaggle uisketch dataframe.
        Expected columns: sketch_path / design_path  or  similar.
        """
        sketch_col = None
        design_col = None
        for col in df.columns:
            lc = col.lower()
            if "sketch" in lc and sketch_col is None:
                sketch_col = col
            if any(k in lc for k in ["design", "target", "ui", "screen"]) and design_col is None:
                design_col = col

        if sketch_col is None or design_col is None:
            logger.warning("uisketch: could not identify sketch/design columns – skipping real pairs.")
            return

        for _, row in df.iterrows():
            try:
                s_img = Image.open(str(row[sketch_col])).convert("RGB").resize((image_size, image_size))
                d_img = Image.open(str(row[design_col])).convert("RGB").resize((image_size, image_size))
                self.real_sketch_pairs.append((s_img, d_img))
            except Exception:
                continue

        logger.info(f"  ↳ {len(self.real_sketch_pairs)} real sketch pairs loaded.")

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt = self.prompts[idx % len(self.prompts)]

        sketch = sample["sketch"]
        target = sample["target_ui"]

        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)
        if self.transform:
            target = self.transform(target)

        return {
            "sketch": sketch,
            "target_ui": target,
            "prompt": prompt,
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
    t_transform, s_transform = get_transforms(image_size) if transform is None else (transform, sketch_transform)

    # Load raw datasets
    hf_ds = load_mobile_ui_dataset("train")
    try:
        sketch_df = load_ui_sketch_dataset()
    except Exception as e:
        logger.warning(f"Could not load uisketch: {e}. Proceeding without real pairs.")
        sketch_df = None
    try:
        prompt_df = load_prompt_dataset()
    except Exception as e:
        logger.warning(f"Could not load prompt dataset: {e}. Using built-in templates.")
        prompt_df = None

    full_dataset = UIGenerationDataset(
        hf_dataset=hf_ds,
        sketch_df=sketch_df,
        prompt_df=prompt_df,
        transform=t_transform,
        sketch_transform=s_transform,
        image_size=image_size,
    )

    n = len(full_dataset)
    n_train = int(n * cfg.data.train_split)
    n_val   = int(n * cfg.data.val_split)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    logger.info(f"DataLoaders built — train:{n_train}  val:{n_val}  test:{n_test}")
    return train_loader, val_loader, test_loader
