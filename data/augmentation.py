"""
augmentation.py
───────────────
Augmentation strategies that preserve the sketch ↔ target-UI alignment.
All transforms are applied *identically* to both sketch and target.
"""

from __future__ import annotations

import random
import torchvision.transforms.functional as F
from PIL import Image
import torch


class PairedAugmentation:
    """Apply the same spatial transforms to (sketch, target) simultaneously."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        rotate_range: tuple[float, float] = (-5.0, 5.0),
        scale_range: tuple[float, float] = (0.9, 1.1),
        color_jitter_target: bool = True,
    ):
        self.hflip_prob      = hflip_prob
        self.rotate_range    = rotate_range
        self.scale_range     = scale_range
        self.color_jitter_target = color_jitter_target

        self._jitter = torch.nn.Sequential(
            torch.nn.Identity()   # placeholder; applied only to target
        )

    def __call__(
        self,
        sketch: Image.Image,
        target: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:

        # ── Horizontal flip (same for both) ─────────────────────────────────
        if random.random() < self.hflip_prob:
            sketch = F.hflip(sketch)
            target = F.hflip(target)

        # ── Random rotation ─────────────────────────────────────────────────
        angle = random.uniform(*self.rotate_range)
        sketch = F.rotate(sketch, angle, fill=255)   # white fill for sketches
        target = F.rotate(target, angle, fill=0)

        # ── Random scale / crop ─────────────────────────────────────────────
        scale  = random.uniform(*self.scale_range)
        w, h   = sketch.size
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        sketch = F.resize(sketch, (new_h, new_w))
        target = F.resize(target, (new_h, new_w))
        # Centre-crop back to original size
        sketch = F.center_crop(sketch, (h, w))
        target = F.center_crop(target, (h, w))

        # ── Colour jitter on target only ────────────────────────────────────
        if self.color_jitter_target:
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            target = F.adjust_brightness(target, brightness)
            target = F.adjust_contrast(target, contrast)
            target = F.adjust_saturation(target, saturation)

        return sketch, target


class SketchStyleAugmentation:
    """
    Apply random sketch-style perturbations to synthesised edge maps so
    the model becomes robust to different drawing styles.
    """

    def __init__(self, noise_std: float = 5.0, line_thickness_range=(1, 3)):
        self.noise_std   = noise_std
        self.lt_range    = line_thickness_range

    def __call__(self, sketch: Image.Image) -> Image.Image:
        import numpy as np
        import cv2

        arr = np.array(sketch.convert("L")).astype(np.float32)

        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, arr.shape)
            arr   = np.clip(arr + noise, 0, 255)

        # Random dilation to vary stroke width
        thickness = random.randint(*self.lt_range)
        kernel    = np.ones((thickness, thickness), np.uint8)
        arr_u8    = arr.astype(np.uint8)
        # Invert → dilate → invert  (dilates dark strokes)
        inv = 255 - arr_u8
        inv = cv2.dilate(inv, kernel, iterations=1)
        arr_u8 = 255 - inv

        return Image.fromarray(arr_u8).convert("RGB")
