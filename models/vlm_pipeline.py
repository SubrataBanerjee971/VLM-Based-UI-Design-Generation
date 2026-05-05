"""
vlm_pipeline.py
───────────────
Top-level model that wires together:

    VLMEncoder          → image / text patch embeddings
    CrossAttentionFusion → fused multimodal embedding h_ui
    UserPreferenceModule → user preference vector p_u
    UIGeneratorModel     → diffusion-based image generation

Also exposes the rating-prediction head used during training
(section 5 of the paper):  r̂_ui = σ(p_u^T W_r h_ui + b)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.vlm_encoder     import VLMEncoder
from models.cross_attention  import CrossAttentionFusion
from models.user_preference  import UserPreferenceModule
from models.ui_generator     import UIGeneratorModel


class VLMUIPipeline(nn.Module):
    """
    Full VLM-based UI Generation Pipeline.

    Training mode:  returns (generation_loss, rating_loss, align_loss)
    Inference mode: returns PIL.Image of the generated UI
    """

    def __init__(self, model_cfg: DictConfig, num_users: int = 10_000):
        super().__init__()

        d = model_cfg.vlm.d_model

        # ── 1. Multimodal Encoder (CLIP + LoRA) ─────────────────────────────
        self.encoder = VLMEncoder(
            model_name   = model_cfg.vlm.clip_model_name,
            d_model      = d,
            lora_rank    = model_cfg.vlm.lora_rank,
            lora_alpha   = model_cfg.vlm.lora_alpha,
            lora_dropout = model_cfg.vlm.lora_dropout,
            freeze_clip  = model_cfg.vlm.freeze_clip,
        )

        # ── 2. Cross-Attention Fusion ────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            d_model    = d,
            num_heads  = model_cfg.cross_attention.num_heads,
            num_layers = model_cfg.cross_attention.num_layers,
            d_ff       = model_cfg.cross_attention.d_ff,
            dropout    = model_cfg.cross_attention.dropout,
            pooling    = model_cfg.cross_attention.pooling,
        )

        # ── 3. User Preference Module ────────────────────────────────────────
        self.user_pref = UserPreferenceModule(
            num_users    = num_users,
            d_model      = d,
            history_len  = model_cfg.user_preference.history_len,
            num_heads    = model_cfg.user_preference.attn_heads,
        )

        # ── 4. Rating Prediction Head ────────────────────────────────────────
        #   r̂_ui = σ(p_u^T W_r h_ui + b)
        self.W_r = nn.Linear(d, d, bias=False)
        self.rating_bias = nn.Parameter(torch.zeros(1))

        # ── 5. UI Generator (Stable Diffusion + ControlNet) ──────────────────
        self.generator = UIGeneratorModel(
            base_model      = model_cfg.generator.base_model,
            controlnet_model= model_cfg.generator.controlnet_model,
            d_vlm           = d,
        )

    # ── Shared encoder pass (used in both train and inference) ────────────────

    def _encode(
        self,
        sketch_pixels: torch.Tensor,    # (B, 3, 224, 224) CLIP-normalised
        prompt_ids: torch.Tensor,       # (B, L)
        prompt_mask: torch.Tensor,      # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_ui : (B, d)  – fused multimodal interaction embedding
            v_ui : (B, d)  – global image embedding (for alignment loss)
        """
        # Patch-level embeddings
        img_patches = self.encoder.encode_image_patches(sketch_pixels)   # (B, P, d)
        txt_tokens  = self.encoder.encode_text_tokens(prompt_ids, prompt_mask)  # (B, L, d)

        # Global embeddings (for cross-modal alignment loss)
        v_ui = self.encoder.get_image_embedding(sketch_pixels)           # (B, d)

        # Cross-attention fusion
        _, h_ui = self.fusion(txt_tokens, img_patches, prompt_mask.bool())  # (B, d)

        return h_ui, v_ui

    # ── Rating prediction ─────────────────────────────────────────────────────

    def predict_rating(
        self,
        p_u: torch.Tensor,   # (B, d)
        h_ui: torch.Tensor,  # (B, d)
    ) -> torch.Tensor:
        """Returns r̂_ui ∈ [0,1] of shape (B,)."""
        score = (p_u * self.W_r(h_ui)).sum(dim=-1) + self.rating_bias  # (B,)
        return torch.sigmoid(score)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        sketch_pixels: torch.Tensor,    # (B, 3, 224, 224)  CLIP-normalised
        target_images: torch.Tensor,    # (B, 3, H,   W  )  diffusion target
        sketch_cond: torch.Tensor,      # (B, 3, H,   W  )  ControlNet input
        prompt_ids: torch.Tensor,       # (B, L)
        prompt_mask: torch.Tensor,      # (B, L)
        prompts: list[str],
        user_ids: torch.Tensor,         # (B,) long
        ratings: torch.Tensor | None = None,  # (B,) ground-truth ratings
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
            gen_loss   – diffusion denoising loss
            rating_loss– MSE on predicted ratings (if ratings given)
            align_loss – cross-modal contrastive alignment loss
            h_ui       – (B, d) for external use / logging
            r_hat      – (B,)  predicted ratings
        """
        device = sketch_pixels.device

        # ── Encode ───────────────────────────────────────────────────────────
        h_ui, v_ui = self._encode(sketch_pixels, prompt_ids, prompt_mask)

        # ── User preference ───────────────────────────────────────────────────
        p_u = self.user_pref(user_ids, h_ui)

        # ── Rating prediction ─────────────────────────────────────────────────
        r_hat = self.predict_rating(p_u, h_ui)   # (B,)

        # ── Generation loss ───────────────────────────────────────────────────
        gen_loss = self.generator(
            target_images=target_images,
            sketch_images=sketch_cond,
            prompts=prompts,
            h_ui=h_ui,
        )

        out = {"gen_loss": gen_loss, "r_hat": r_hat, "h_ui": h_ui}

        # ── Rating regression loss ─────────────────────────────────────────────
        if ratings is not None:
            rating_loss = nn.functional.mse_loss(r_hat, ratings.float())
            out["rating_loss"] = rating_loss
        else:
            out["rating_loss"] = torch.tensor(0.0, device=device)

        # ── Cross-modal alignment loss ─────────────────────────────────────────
        t_ui = self.encoder.get_text_embedding(prompt_ids, prompt_mask)  # (B, d)
        out["align_loss"] = self._cross_modal_alignment_loss(v_ui, t_ui)

        return out

    @staticmethod
    def _cross_modal_alignment_loss(
        v: torch.Tensor,   # (B, d) image embeddings
        t: torch.Tensor,   # (B, d) text  embeddings
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        InfoNCE / CLIP-style contrastive loss.
            L_align = -log [ exp(sim(v_ui, t_ui)/τ) / Σ_j exp(sim(v_ui, t_uj)/τ) ]
        """
        v = nn.functional.normalize(v, dim=-1)
        t = nn.functional.normalize(t, dim=-1)
        sim = torch.matmul(v, t.T) / temperature   # (B, B)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = (
            nn.functional.cross_entropy(sim, labels)
            + nn.functional.cross_entropy(sim.T, labels)
        ) / 2
        return loss

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_ui(
        self,
        sketch_image,           # PIL.Image or (1,3,H,W) tensor
        prompt: str,
        user_id: int | None = None,
        clip_image_size: int = 224,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        device: str = "cuda",
    ):
        """Generate a UI image from a sketch + prompt."""
        from data.preprocessor import get_clip_transform, pil_to_tensor
        from PIL import Image

        # Prepare CLIP-normalised sketch for encoder
        if isinstance(sketch_image, Image.Image):
            clip_t = get_clip_transform(clip_image_size)
            clip_pixels = clip_t(sketch_image).unsqueeze(0).to(device)
        else:
            clip_pixels = sketch_image.to(device)

        # Tokenise prompt
        tok = self.encoder.tokenize([prompt], device=device)
        prompt_ids  = tok["input_ids"]
        prompt_mask = tok["attention_mask"]

        h_ui, _ = self._encode(clip_pixels, prompt_ids, prompt_mask)

        if user_id is not None:
            uid_t = torch.tensor([user_id], device=device)
            p_u   = self.user_pref(uid_t, h_ui)
            # Blend h_ui with user preference for personalised generation
            h_ui  = (h_ui + p_u) / 2.0

        return self.generator.generate(
            sketch_image=sketch_image,
            prompt=prompt,
            h_ui=h_ui,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
