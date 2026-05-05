"""
cross_attention.py
──────────────────
Cross-Attention Fusion Module as described in the paper:

    Q = T_ui @ W_Q         (text token queries)
    K = V_ui @ W_K         (image patch keys)
    V = V_ui @ W_V         (image patch values)

    Z_ui = CrossAttn(T_ui, V_ui)
    h_ui = Pooling(Z_ui)

This allows review/prompt text to attend over spatial image regions,
producing a fused multimodal interaction embedding h_ui ∈ R^d.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
#  Single Cross-Attention Layer
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    One cross-attention sub-layer where text queries attend to image patches.

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens: torch.Tensor,      # (B, L, d)  – queries
        image_patches: torch.Tensor,    # (B, P, d)  – keys / values
        text_mask: torch.Tensor | None = None,   # (B, L) bool mask
    ) -> torch.Tensor:
        """Returns (B, L, d) – text conditioned on image context."""
        B, L, d = text_tokens.shape
        _, P, _  = image_patches.shape
        H        = self.num_heads
        dk       = self.d_k

        Q = self.W_Q(text_tokens)                  # (B, L, d)
        K = self.W_K(image_patches)                # (B, P, d)
        V = self.W_V(image_patches)                # (B, P, d)

        # Split into heads
        Q = rearrange(Q, "b l (h dk) -> b h l dk", h=H)
        K = rearrange(K, "b p (h dk) -> b h p dk", h=H)
        V = rearrange(V, "b p (h dk) -> b h p dk", h=H)

        # Scaled dot-product attention
        scale  = math.sqrt(dk)
        scores = torch.einsum("b h l d, b h p d -> b h l p", Q, K) / scale  # (B,H,L,P)

        if text_mask is not None:
            # Expand mask for heads and patches: (B,1,L,1)
            scores = scores.masked_fill(
                (~text_mask).unsqueeze(1).unsqueeze(-1), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)           # (B, H, L, P)
        attn = self.dropout(attn)

        out = torch.einsum("b h l p, b h p d -> b h l d", attn, V)  # (B,H,L,dk)
        out = rearrange(out, "b h l dk -> b l (h dk)")               # (B,L,d)
        out = self.W_O(out)

        # Residual + LayerNorm
        out = self.norm(text_tokens + out)
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  Feed-Forward Block
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# ──────────────────────────────────────────────────────────────────────────────
#  Full Cross-Attention Fusion Module
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Stack of (CrossAttentionLayer + FeedForward) blocks.
    Text tokens attend to image patches at every layer.

    Final output h_ui ∈ R^d via configurable pooling.
    """

    def __init__(
        self,
        d_model: int  = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int      = 2048,
        dropout: float = 0.1,
        pooling: str   = "mean",   # "mean" | "cls" | "attention"
    ):
        super().__init__()
        self.pooling = pooling

        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttentionLayer(d_model, num_heads, dropout),
                FeedForward(d_model, d_ff, dropout),
            ])
            for _ in range(num_layers)
        ])

        # Attention pooling weights (used when pooling == "attention")
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.pool_attn  = CrossAttentionLayer(d_model, num_heads, dropout)

    def forward(
        self,
        text_tokens: torch.Tensor,      # (B, L, d)
        image_patches: torch.Tensor,    # (B, P, d)
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Z_ui : (B, L, d)  – full fused sequence
            h_ui : (B, d)     – pooled multimodal interaction embedding
        """
        x = text_tokens

        for cross_attn, ff in self.layers:
            x = cross_attn(x, image_patches, text_mask)
            x = ff(x)

        Z_ui = x   # (B, L, d)

        # ── Pooling ──────────────────────────────────────────────────────────
        if self.pooling == "cls":
            h_ui = Z_ui[:, 0, :]                                    # CLS token

        elif self.pooling == "attention":
            q = self.pool_query.expand(Z_ui.size(0), -1, -1)       # (B, 1, d)
            pooled = self.pool_attn(q, Z_ui)                        # (B, 1, d)
            h_ui   = pooled.squeeze(1)                               # (B, d)

        else:  # "mean"
            if text_mask is not None:
                mask_expanded = text_mask.unsqueeze(-1).float()     # (B, L, 1)
                h_ui = (Z_ui * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                h_ui = Z_ui.mean(dim=1)                             # (B, d)

        return Z_ui, h_ui
