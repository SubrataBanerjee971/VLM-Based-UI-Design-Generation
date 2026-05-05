"""
vlm_encoder.py
──────────────
CLIP-based visual and text encoder with optional LoRA fine-tuning.

Key outputs
───────────
encode_image_patches()  →  (B, P, d)   patch-level embeddings
encode_text_tokens()    →  (B, L, d)   token-level embeddings
get_image_embedding()   →  (B, d)      CLS / pooled image vector
get_text_embedding()    →  (B, d)      pooled text vector
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig, TaskType


class VLMEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        d_model: int = 512,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        freeze_clip: bool = False,
        use_lora: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.model_name = model_name

        # ── Load CLIP ────────────────────────────────────────────────────────
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # ── Optionally freeze all CLIP weights ──────────────────────────────
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad_(False)

        # ── LoRA fine-tuning on the vision transformer ───────────────────────
        if use_lora and not freeze_clip:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
            )
            self.clip = get_peft_model(self.clip, lora_cfg)

        # ── Projection head to common d_model ───────────────────────────────
        clip_hidden = self._get_clip_hidden_size()
        self.img_proj  = nn.Linear(clip_hidden, d_model) if clip_hidden != d_model else nn.Identity()
        self.txt_proj  = nn.Linear(clip_hidden, d_model) if clip_hidden != d_model else nn.Identity()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_clip_hidden_size(self) -> int:
        """Read the vision hidden size from CLIP config."""
        try:
            return self.clip.config.vision_config.hidden_size
        except AttributeError:
            return 768   # CLIP ViT-B/32 default

    def _unwrapped_clip(self):
        """Return the underlying CLIPModel even if wrapped with PEFT."""
        m = self.clip
        return m.base_model.model if hasattr(m, "base_model") else m

    # ── Public API ───────────────────────────────────────────────────────────

    def encode_image_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) preprocessed with CLIP norms

        Returns:
            patch_embeddings: (B, P, d_model)   P = num_patches + 1 (CLS)
        """
        clip = self._unwrapped_clip()
        vision_out = clip.vision_model(pixel_values=pixel_values)
        hidden = vision_out.last_hidden_state          # (B, P, hidden_size)
        return self.img_proj(hidden)                   # (B, P, d_model)

    def encode_text_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids      : (B, L)
            attention_mask : (B, L)

        Returns:
            token_embeddings: (B, L, d_model)
        """
        clip = self._unwrapped_clip()
        text_out = clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = text_out.last_hidden_state            # (B, L, hidden_size)
        return self.txt_proj(hidden)                   # (B, L, d_model)

    def get_image_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Global (pooled) image embedding.  Returns (B, d_model)."""
        clip = self._unwrapped_clip()
        feat = clip.get_image_features(pixel_values=pixel_values)  # (B, projection_dim)
        # project to d_model if sizes differ
        if feat.shape[-1] != self.d_model:
            feat = self.img_proj(feat)
        return feat

    def get_text_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Global (pooled) text embedding.  Returns (B, d_model)."""
        clip = self._unwrapped_clip()
        feat = clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if feat.shape[-1] != self.d_model:
            feat = self.txt_proj(feat)
        return feat

    def tokenize(
        self,
        texts: list[str],
        max_length: int = 77,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Tokenise a list of strings using the CLIP tokeniser."""
        enc = self.processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}
