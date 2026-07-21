"""PatchTST (Nie et al. 2023) adapted for day-ahead price.
Past series cut into patches -> tokens -> transformer, channel-independent.
Head adds the known-future covariates (RES, TSO, calendar).
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_Q = 3


class PatchTST(nn.Module):
    name = "patchtst"

    def __init__(self, enc_feat: int, fut_feat: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, patch_len: int = 24,
                 stride: int = 12, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.enc_feat = enc_feat

        self.patch_embed = nn.Linear(patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))  # max 512 patches

        self.fut_embed = nn.Sequential(
            nn.Linear(fut_feat, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, N_Q),
        )

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) one channel -> (B, n_patches, patch_len)
        return x.unfold(dimension=1, size=self.patch_len, step=self.stride)

    def forward(self, enc: torch.Tensor, fut: torch.Tensor,
                anchor: torch.Tensor, y_teacher: torch.Tensor | None = None,
                ) -> torch.Tensor:
        B, T, C = enc.shape
        # channel independence: fold channels into the batch dimension
        chans = enc.permute(0, 2, 1).reshape(B * C, T)          # (B*C, T)
        patches = self._patch(chans)                             # (B*C, P, L)
        tok = self.patch_embed(patches)                          # (B*C, P, d)
        tok = tok + self.pos[:, : tok.shape[1]]
        z = self.encoder(tok)                                    # (B*C, P, d)
        pooled = z.mean(dim=1).reshape(B, C, -1).mean(dim=1)     # (B, d)

        fut_emb = self.fut_embed(fut)                            # (B, 24, d)
        ctx = pooled.unsqueeze(1).expand(-1, fut_emb.shape[1], -1)
        return self.head(torch.cat([ctx, fut_emb], dim=-1))      # (B, 24, 3)
