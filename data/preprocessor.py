"""
preprocessor.py
───────────────
Image transforms for both target UI images and sketch inputs.
"""

from __future__ import annotations

import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import torch


def get_transforms(image_size: int = 512):
    """
    Returns:
        target_transform  – for polished UI images (normalize to [-1,1] for diffusion)
        sketch_transform  – for sketch inputs      (normalize to [0,1] for ControlNet)
    """

    target_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    sketch_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.Grayscale(num_output_channels=3),   # keep 3-ch for ControlNet
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return target_transform, sketch_transform


def get_clip_transform(image_size: int = 224):
    """Preprocessing for CLIP encoder (224×224, ImageNet stats)."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL → tensor in [-1, 1]."""
    t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return t(img.convert("RGB"))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor in [-1, 1] → PIL."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().float()
    arr = (arr * 0.5 + 0.5).clamp(0, 1).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))
