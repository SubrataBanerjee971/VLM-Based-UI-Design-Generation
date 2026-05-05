"""
user_preference.py
──────────────────
User Preference Learning Module.

Aggregates a user's historical interaction embeddings {h_ui | i ∈ I_u}
via attention pooling to produce a stable user preference vector p_u ∈ R^d.

    H_u = {h_ui | i ∈ I_u}
    p_u = AttentionPooling(H_u)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Soft attention over a variable-length sequence of interaction embeddings.

        e_i = v^T tanh(W h_i + b)
        α_i = softmax(e_i)
        p_u = Σ_i α_i h_i
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm  = nn.LayerNorm(d_model)

    def forward(
        self,
        history: torch.Tensor,          # (B, T, d)
        key_padding_mask: torch.Tensor | None = None,  # (B, T) True = ignore
    ) -> torch.Tensor:
        """Returns p_u : (B, d)."""
        B = history.size(0)
        q = self.query.expand(B, -1, -1)             # (B, 1, d)
        out, _ = self.attn(q, history, history,
                           key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))              # (B, d)


class UserPreferenceModule(nn.Module):
    """
    Maintains and updates per-user preference embeddings.

    In training:
        Given a batch of (user_id, h_ui) pairs, updates the embedding bank
        and returns the current preference vector p_u.

    In inference:
        Looks up the stored p_u for known users;
        falls back to a zero vector for cold-start users.
    """

    def __init__(
        self,
        num_users: int,
        d_model: int,
        history_len: int = 20,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model     = d_model
        self.history_len = history_len

        # Learnable user embedding table (initialised randomly)
        self.user_embeddings = nn.Embedding(num_users, d_model)
        nn.init.normal_(self.user_embeddings.weight, std=0.02)

        self.attn_pool = AttentionPooling(d_model, num_heads)

        # Rolling history buffer: stores the last `history_len`
        # interaction embeddings per user (detached, for aggregation)
        self.register_buffer(
            "history_bank",
            torch.zeros(num_users, history_len, d_model),
        )
        self.register_buffer(
            "history_ptr",
            torch.zeros(num_users, dtype=torch.long),
        )

    # ── History management ───────────────────────────────────────────────────

    @torch.no_grad()
    def push_interaction(
        self,
        user_ids: torch.Tensor,   # (B,) long
        h_ui: torch.Tensor,       # (B, d)
    ):
        """Append new interaction embeddings to the rolling history bank."""
        for uid, emb in zip(user_ids.tolist(), h_ui.detach()):
            ptr = int(self.history_ptr[uid].item())
            self.history_bank[uid, ptr] = emb
            self.history_ptr[uid] = (ptr + 1) % self.history_len

    def get_history(
        self,
        user_ids: torch.Tensor,   # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            history      : (B, T, d)
            padding_mask : (B, T)  True where the slot is empty / padding
        """
        history = self.history_bank[user_ids]           # (B, T, d)
        # Detect zero-padded (empty) slots
        padding_mask = (history.abs().sum(-1) == 0)     # (B, T)
        return history, padding_mask

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        user_ids: torch.Tensor,   # (B,) long
        h_ui: torch.Tensor | None = None,  # (B, d) current interaction
    ) -> torch.Tensor:
        """
        Returns p_u : (B, d)

        If h_ui is provided the current interaction is injected into the
        attention pool alongside historical embeddings (training mode).
        Otherwise only stored history is used (inference mode).
        """
        history, pad_mask = self.get_history(user_ids)  # (B, T, d), (B, T)

        if h_ui is not None:
            # Prepend the current interaction as the most salient context
            cur = h_ui.unsqueeze(1)                     # (B, 1, d)
            history  = torch.cat([cur, history], dim=1) # (B, T+1, d)
            cur_mask = torch.zeros(
                history.size(0), 1,
                dtype=torch.bool, device=pad_mask.device,
            )
            pad_mask = torch.cat([cur_mask, pad_mask], dim=1)

        # If the entire history is padding (cold-start), fall back to
        # the learnable user embedding
        all_padding = pad_mask.all(dim=1)               # (B,)
        p_u_attn    = self.attn_pool(history, pad_mask) # (B, d)
        p_u_lookup  = self.user_embeddings(user_ids)    # (B, d)

        # Blend: cold-start → lookup; warm → attention pool
        p_u = torch.where(
            all_padding.unsqueeze(-1).expand_as(p_u_attn),
            p_u_lookup,
            p_u_attn,
        )
        return p_u
