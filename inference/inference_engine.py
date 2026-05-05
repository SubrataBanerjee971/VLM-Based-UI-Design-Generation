"""
inference_engine.py
───────────────────
High-level inference API.

Usage
─────
    engine = InferenceEngine.from_checkpoint("checkpoints/best_model.pth", device="cuda")
    result = engine.generate(
        sketch_path = "my_sketch.jpg",
        prompt      = "Design a clean mobile login screen with email and password fields",
        user_id     = 42,
    )
    result.save("output_ui.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf

from models.vlm_pipeline import VLMUIPipeline
from data.preprocessor   import get_clip_transform, pil_to_tensor
from utils.checkpoint    import load_checkpoint_for_inference

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps VLMUIPipeline for easy single-image inference."""

    DEFAULT_PROMPTS = [
        "Design a clean modern mobile UI based on this sketch.",
        "Convert this wireframe into a polished mobile app interface.",
        "Generate a high-fidelity mobile UI design from this hand-drawn sketch.",
        "Create a professional mobile application screen based on this layout.",
    ]

    def __init__(
        self,
        model: VLMUIPipeline,
        device: str = "cuda",
        default_steps: int = 50,
        default_guidance: float = 7.5,
        default_controlnet_scale: float = 1.0,
        output_size: int = 512,
    ):
        self.model          = model.eval().to(device)
        self.device         = device
        self.default_steps  = default_steps
        self.default_guidance      = default_guidance
        self.default_controlnet_scale = default_controlnet_scale
        self.output_size    = output_size
        self.clip_transform = get_clip_transform(224)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config_path: str = "config/model_config.yaml",
        device: str = "cuda",
        num_users: int = 10_000,
        **kwargs,
    ) -> "InferenceEngine":
        model_cfg = OmegaConf.load(model_config_path)
        model     = VLMUIPipeline(model_cfg, num_users=num_users)
        load_checkpoint_for_inference(model, checkpoint_path, device=device)
        logger.info(f"Model loaded from {checkpoint_path}")
        return cls(model=model, device=device, **kwargs)

    # ── Core generate ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        sketch_path: str | Path | None = None,
        sketch_image: Image.Image | None = None,
        prompt: str | None = None,
        user_id: int | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        controlnet_scale: float | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """
        Generate a UI design image from a sketch.

        Args:
            sketch_path    : path to hand-drawn sketch image
            sketch_image   : PIL Image (alternative to sketch_path)
            prompt         : text description; uses a default if None
            user_id        : optional user ID for personalised generation
            num_steps      : denoising steps (default from config)
            guidance_scale : CFG scale (default from config)
            controlnet_scale: ControlNet conditioning strength
            seed           : random seed for reproducibility

        Returns:
            PIL.Image of the generated UI
        """
        # ── Reproducibility ─────────────────────────────────────────────────
        if seed is not None:
            torch.manual_seed(seed)
            import numpy as np, random as rnd
            np.random.seed(seed)
            rnd.seed(seed)

        # ── Load sketch ──────────────────────────────────────────────────────
        if sketch_image is None:
            if sketch_path is None:
                raise ValueError("Provide either sketch_path or sketch_image.")
            sketch_image = Image.open(sketch_path).convert("RGB")

        # ── Default prompt ───────────────────────────────────────────────────
        if prompt is None:
            import random
            prompt = random.choice(self.DEFAULT_PROMPTS)
        logger.info(f"Generating UI | prompt='{prompt[:80]}…' | user_id={user_id}")

        # ── Inference ────────────────────────────────────────────────────────
        result = self.model.generate_ui(
            sketch_image  = sketch_image,
            prompt        = prompt,
            user_id       = user_id,
            num_steps     = num_steps  or self.default_steps,
            guidance_scale= guidance_scale or self.default_guidance,
            device        = self.device,
        )
        return result

    # ── Batch generate ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_batch(
        self,
        sketches: list[Image.Image],
        prompts: list[str],
        user_ids: list[int] | None = None,
        **kwargs,
    ) -> list[Image.Image]:
        """Generate multiple UI images in sequence."""
        if user_ids is None:
            user_ids = [None] * len(sketches)
        results = []
        for sketch, prompt, uid in zip(sketches, prompts, user_ids):
            img = self.generate(
                sketch_image=sketch, prompt=prompt, user_id=uid, **kwargs
            )
            results.append(img)
        return results

    # ── Sketch pre-processing helper ─────────────────────────────────────────

    @staticmethod
    def preprocess_sketch(
        image_path: str | Path,
        output_size: int = 512,
        enhance_edges: bool = True,
    ) -> Image.Image:
        """
        Load and optionally enhance a hand-drawn sketch for better ControlNet conditioning.
        - Converts to grayscale
        - Applies adaptive thresholding to clean up strokes
        - Resizes to output_size
        """
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        if enhance_edges:
            # Adaptive threshold: robust to uneven lighting
            img = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            # Dilate to thicken strokes slightly
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img    = cv2.dilate(255 - img, kernel, iterations=1)
            img    = 255 - img

        img = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img).convert("RGB")
