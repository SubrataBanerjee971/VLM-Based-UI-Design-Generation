"""
losses.py
─────────
Multi-task loss objective from Section 6 of the paper.

    L = λ1·L_rating + λ2·L_BPR + λ3·L_align

Plus the diffusion MSE loss from the generator (added externally).

6.1  Rating Regression Loss     L_rating = (1/|D|) Σ (r_ui − r̂_ui)²
6.2  Pairwise BPR Ranking Loss  L_BPR    = −Σ log σ(r̂_ui − r̂_uj)
6.3  Cross-Modal Alignment Loss L_align  = −log [ exp(sim(v,t)/τ) / Σ exp(sim(v,tj)/τ) ]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  Individual losses
# ──────────────────────────────────────────────────────────────────────────────

def rating_regression_loss(r_hat: torch.Tensor, r_true: torch.Tensor) -> torch.Tensor:
    """
    L_rating = (1/|D|) Σ (r_ui − r̂_ui)²
    Both tensors: (B,)
    """
    return F.mse_loss(r_hat, r_true.float())


def bpr_ranking_loss(
    r_hat_pos: torch.Tensor,   # (B,) predicted rating for positive items
    r_hat_neg: torch.Tensor,   # (B,) predicted rating for negative items
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bayesian Personalised Ranking loss.
    L_BPR = −Σ log σ(r̂_ui − r̂_uj)
    """
    diff = r_hat_pos - r_hat_neg              # (B,)
    return -torch.log(torch.sigmoid(diff) + eps).mean()


def cross_modal_alignment_loss(
    v: torch.Tensor,        # (B, d) L2-normalised image embeddings
    t: torch.Tensor,        # (B, d) L2-normalised text  embeddings
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss encouraging image–text alignment.
    L_align = −log [ exp(sim(v_ui, t_ui)/τ) / Σ_j exp(sim(v_ui, t_uj)/τ) ]
    """
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = torch.matmul(v, t.T) / temperature   # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_v2t = F.cross_entropy(logits,   labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
#  Multi-task loss combiner
# ──────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Combines all losses with learnable or fixed weights.

        L = λ1·L_rating + λ2·L_BPR + λ3·L_align + λ4·L_gen
    """

    def __init__(
        self,
        lambda1: float = 1.0,   # rating
        lambda2: float = 0.5,   # BPR
        lambda3: float = 0.5,   # alignment
        lambda4: float = 1.0,   # generation (diffusion MSE)
        temperature: float = 0.07,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.temperature = temperature

        if learnable_weights:
            # log-variance trick for uncertainty weighting
            self.log_sigma = nn.Parameter(torch.zeros(4))
        else:
            self.register_buffer("_lambdas",
                torch.tensor([lambda1, lambda2, lambda3, lambda4]))
            self.log_sigma = None

    def _get_weights(self) -> torch.Tensor:
        if self.log_sigma is not None:
            # Uncertainty weighting: w_i = exp(-log_sigma_i)
            return torch.exp(-self.log_sigma)
        return self._lambdas

    def forward(
        self,
        r_hat: torch.Tensor,          # (B,)
        r_true: torch.Tensor,         # (B,)
        r_hat_pos: torch.Tensor,      # (B,) for BPR
        r_hat_neg: torch.Tensor,      # (B,) for BPR
        v_embed: torch.Tensor,        # (B, d)
        t_embed: torch.Tensor,        # (B, d)
        gen_loss: torch.Tensor,       # scalar
    ) -> dict[str, torch.Tensor]:

        w = self._get_weights()

        l_rating = rating_regression_loss(r_hat, r_true)
        l_bpr    = bpr_ranking_loss(r_hat_pos, r_hat_neg)
        l_align  = cross_modal_alignment_loss(v_embed, t_embed, self.temperature)
        l_gen    = gen_loss

        # --- Sanitize individual losses ---
        l_rating = torch.where(torch.isnan(l_rating), torch.zeros_like(l_rating), l_rating)
        l_bpr    = torch.where(torch.isnan(l_bpr), torch.zeros_like(l_bpr), l_bpr)
        l_align  = torch.where(torch.isnan(l_align), torch.zeros_like(l_align), l_align)
        l_gen    = torch.where(torch.isnan(l_gen), torch.zeros_like(l_gen), l_gen)

        total = (
            w[0] * l_rating
            + w[1] * l_bpr
            + w[2] * l_align
            + w[3] * l_gen
        )

        return {
            "total"  : total,
            "rating" : l_rating,
            "bpr"    : l_bpr,
            "align"  : l_align,
            "gen"    : l_gen,
            "w_rating": w[0].detach(),
            "w_bpr"  : w[1].detach(),
            "w_align": w[2].detach(),
            "w_gen"  : w[3].detach(),
        }
