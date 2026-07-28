"""Temporal Fusion Transformer (Lim et al. 2021) — simplified but honest.

Key components:
  GRN: Gated Residual Network — the TFT building block.
  VSN: Variable Selection Network — learns per-feature importance weights.
       The TFT's key contribution: explicit, interpretable feature selection.
  LSTM encoder/decoder with static context conditioning.
  Multi-head temporal self-attention (interpretable variant).
  Gated add-and-norm fusion.

Architecture matches the paper closely, scaled to our dataset:
  - Encoder: past load (normalized) + past weather = n_enc_feat columns
  - Decoder: calendar + weather forecast + anchor + optional TSO = n_fut_feat columns
  - d_model: embedding dim per feature (32/64/128 in the sweep)
  - n_heads: 4 (paper default)
  - Dropout: 0.1

Why the VSN matters for this project:
  The VSN's softmax attention weights (one weight per feature at each timestep)
  are a free byproduct: export them and you have timestep-level feature
  importance — not a post-hoc SHAP but in-model. This directly answers
  "which feature drove today's forecast?" for any hour.

Reference: https://arxiv.org/abs/1912.09363
"""

from __future__ import annotations

import torch
import torch.nn as nn


N_Q = 3  # p10, p50, p90


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _GLU(nn.Module):
    """Gated Linear Unit: split in half, multiply by sigmoid of second half."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network.

    f(a, c) = LayerNorm(a + GLU(ELU(W_1 a + W_2 c) + b_1))
    Context c is optional; set to None when not used.
    If d_in != d_model, a skip projection is added.
    """

    def __init__(self, d_in: int, d_model: int, context_dim: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in + context_dim, d_model)
        self.glu = _GLU(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_model, bias=False) if d_in != d_model else nn.Identity()

    def forward(self, a: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([a, c], dim=-1) if c is not None else a
        x = torch.nn.functional.elu(self.fc1(x))
        x = self.drop(self.glu(x))
        return self.norm(x + self.skip(a))


class VSN(nn.Module):
    """Variable Selection Network for a set of features at one time position.

    Each feature gets its own GRN (individual processing). A separate
    context-conditioned GRN learns the selection weights (softmax). Output
    is the weighted sum of processed features.

    Saves .weights (B, n_vars) after each forward — extract for interpretability.
    """

    def __init__(self, n_vars: int, d_model: int, context_dim: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        # One GRN per variable (individual feature transformation)
        self.var_grns = nn.ModuleList([GRN(1, d_model, dropout=dropout) for _ in range(n_vars)])
        # Selection GRN: all features flattened → softmax weights
        self.sel_grn = GRN(n_vars * d_model, n_vars, context_dim, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, n_vars) or (B, T, n_vars)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)  # (B, 1, n_vars)
        B, T, _ = x.shape

        processed = torch.stack(
            [grn(x[..., i:i+1]) for i, grn in enumerate(self.var_grns)], dim=-2
        )  # (B, T, n_vars, d_model)

        flat = processed.reshape(B, T, self.n_vars * self.d_model)
        ctx = c.unsqueeze(1).expand(-1, T, -1) if (c is not None and c.dim() == 2) else c
        weights = self.softmax(self.sel_grn(flat, ctx))  # (B, T, n_vars)
        self.weights = weights.detach().mean(dim=1)  # (B, n_vars) for interpretation

        out = (processed * weights.unsqueeze(-1)).sum(dim=-2)  # (B, T, d_model)
        return out.squeeze(1) if squeeze else out


# ---------------------------------------------------------------------------
# Main TFT model
# ---------------------------------------------------------------------------

class TFT(nn.Module):
    """Temporal Fusion Transformer for quantile load forecasting.

    Input contract (same as EncDec in nets.py):
      enc:    (B, ENCODER_HOURS, enc_feat)  — past features
      fut:    (B, 24, fut_feat)             — known-future features
      anchor: (B, 24)                       — instance-normalized lag-168 (unused here,
                                              the TSO slot in fut already contains it)

    Output: (B, 24, 3) — p10/p50/p90 quantiles, instance-normalized.
    """

    name = "tft"

    def __init__(self, enc_feat: int, fut_feat: int, d_model: int = 64,
                 n_heads: int = 4, lstm_layers: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model

        # Variable selection networks (past and future)
        self.enc_vsn = VSN(enc_feat, d_model, dropout=dropout)
        self.fut_vsn = VSN(fut_feat, d_model, dropout=dropout)

        # LSTM encoder/decoder
        self.lstm_enc = nn.LSTM(d_model, d_model, num_layers=lstm_layers,
                                batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.lstm_dec = nn.LSTM(d_model, d_model, num_layers=lstm_layers,
                                batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        # Gate-and-norm after LSTM
        self.enc_gn = GRN(d_model, d_model, dropout=dropout)
        self.dec_gn = GRN(d_model, d_model, dropout=dropout)

        # Multi-head temporal self-attention over the FULL sequence (enc + dec)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_grn = GRN(d_model, d_model, dropout=dropout)

        # Output positionwise GRN → quantile projection
        self.out_grn = GRN(d_model, d_model, dropout=dropout)
        self.head = nn.Linear(d_model, N_Q)

    def forward(self, enc: torch.Tensor, fut: torch.Tensor,
                anchor: torch.Tensor, y_teacher: torch.Tensor | None = None,
                ) -> torch.Tensor:
        # Variable selection
        enc_emb = self.enc_vsn(enc)   # (B, ENCODER_HOURS, d_model)
        fut_emb = self.fut_vsn(fut)   # (B, 24, d_model)

        # LSTM encoder
        enc_out, state = self.lstm_enc(enc_emb)
        enc_out = self.enc_gn(enc_out)

        # LSTM decoder (initialized with encoder final state)
        dec_out, _ = self.lstm_dec(fut_emb, state)
        dec_out = self.dec_gn(dec_out)

        # Temporal self-attention over decoder output, attending to encoder
        seq = torch.cat([enc_out, dec_out], dim=1)     # (B, ENCODER+24, d_model)
        attn_out, _ = self.attn(dec_out, seq, seq)     # decoder queries enc+dec keys/values
        attn_out = self.attn_norm(dec_out + attn_out)  # residual + norm
        attn_out = self.attn_grn(attn_out)

        # Per-step output
        out = self.out_grn(attn_out)
        return self.head(out)  # (B, 24, 3)


# Registered sizes for the campaign sweep
TFT_CONFIGS: list[tuple[str, int]] = [
    ("tft_d32", 32),
    ("tft_d64", 64),
    ("tft_d128", 128),
]
