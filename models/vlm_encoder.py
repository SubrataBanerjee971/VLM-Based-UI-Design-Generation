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

        # ── Projection heads to common d_model ───────────────────────────
        clip_vis_hidden = self._get_clip_hidden_size()        # patch-level (768 for ViT-B/32)
        clip_txt_hidden = self._get_clip_text_hidden_size()   # token-level (512 for ViT-B/32)
        clip_proj_dim   = self._get_clip_projection_dim()     # global feature dim (512 for ViT-B/32)

        # patch / token level projections
        self.img_proj = nn.Linear(clip_vis_hidden, d_model) if clip_vis_hidden != d_model else nn.Identity()
        self.txt_proj = nn.Linear(clip_txt_hidden, d_model) if clip_txt_hidden != d_model else nn.Identity()

        # global pooled-feature projections (CLIP already projects to projection_dim)
        self.img_proj_global = nn.Linear(clip_proj_dim, d_model) if clip_proj_dim != d_model else nn.Identity()
        self.txt_proj_global = nn.Linear(clip_proj_dim, d_model) if clip_proj_dim != d_model else nn.Identity()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_clip_hidden_size(self) -> int:
        """Read the VISION hidden size from CLIP config."""
        try:
            return self.clip.config.vision_config.hidden_size
        except AttributeError:
            return 768   # CLIP ViT-B/32 vision default

    def _get_clip_text_hidden_size(self) -> int:
        """Read the TEXT hidden size from CLIP config."""
        try:
            return self.clip.config.text_config.hidden_size
        except AttributeError:
            return 512   # CLIP ViT-B/32 text default

    def _get_clip_projection_dim(self) -> int:
        """Read CLIP's output projection dim (used by get_image/text_features)."""
        try:
            return self.clip.config.projection_dim
        except AttributeError:
            return 512   # CLIP ViT-B/32 default

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
        
        # Ensure we have a Tensor (some transformers versions return objects)
        if not isinstance(feat, torch.Tensor):
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            elif isinstance(feat, (list, tuple)):
                feat = feat[0]

        return self.img_proj_global(feat)  # identity if projection_dim == d_model

    def get_text_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Global (pooled) text embedding.  Returns (B, d_model)."""
        clip = self._unwrapped_clip()
        feat = clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (B, projection_dim) = (B, 512) for ViT-B/32

        # Ensure we have a Tensor
        if not isinstance(feat, torch.Tensor):
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            elif isinstance(feat, (list, tuple)):
                feat = feat[0]

        return self.txt_proj_global(feat)  # identity if projection_dim == d_model

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
