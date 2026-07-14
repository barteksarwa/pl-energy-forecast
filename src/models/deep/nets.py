"""Four LSTM architectures for day-ahead load. All output (batch, 24, 3) quantiles.

Variants (the overnight screening question: which structure fits this task?):
- enc_direct : LSTM encoder -> final state -> MLP -> all 24 hours at once.
               Ignores future covariates. The "does the past alone suffice?" control.
- enc_futmlp : encoder state broadcast to each target hour + that hour's known-future
               covariates -> shared MLP. Simple, parallel, no recurrence in decode.
- enc_dec    : LSTM decoder unrolled over the 24 target hours, fed future covariates,
               initialized from encoder state. The classic seq2seq.
- residual   : enc_dec, but predicts the residual around the seasonal-naive anchor.
               Model only learns the correction, level comes free.
"""

from __future__ import annotations

import torch
from torch import nn

N_Q = 3  # p10, p50, p90


class EncDirect(nn.Module):
    name = "enc_direct"

    def __init__(self, enc_feat: int, fut_feat: int, hidden: int = 96) -> None:
        super().__init__()
        self.encoder = nn.LSTM(enc_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, 24 * N_Q)
        )

    def forward(self, enc: torch.Tensor, fut: torch.Tensor, anchor: torch.Tensor,
                y_teacher: torch.Tensor | None = None):
        _, (h, _) = self.encoder(enc)
        return self.head(h[-1]).view(-1, 24, N_Q)


class EncFutMLP(nn.Module):
    name = "enc_futmlp"

    def __init__(self, enc_feat: int, fut_feat: int, hidden: int = 96) -> None:
        super().__init__()
        self.encoder = nn.LSTM(enc_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + fut_feat, hidden), nn.ReLU(), nn.Linear(hidden, N_Q)
        )

    def forward(self, enc: torch.Tensor, fut: torch.Tensor, anchor: torch.Tensor,
                y_teacher: torch.Tensor | None = None):
        _, (h, _) = self.encoder(enc)
        state = h[-1].unsqueeze(1).expand(-1, fut.shape[1], -1)
        return self.head(torch.cat([state, fut], dim=-1))


class EncDec(nn.Module):
    name = "enc_dec"

    def __init__(self, enc_feat: int, fut_feat: int, hidden: int = 96) -> None:
        super().__init__()
        self.encoder = nn.LSTM(enc_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(fut_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, N_Q)

    def forward(self, enc: torch.Tensor, fut: torch.Tensor, anchor: torch.Tensor,
                y_teacher: torch.Tensor | None = None):
        _, state = self.encoder(enc)
        out, _ = self.decoder(fut, state)
        return self.head(out)


class ResidualEncDec(EncDec):
    """enc_dec predicting the correction to the seasonal-naive anchor."""

    name = "residual"

    def forward(self, enc: torch.Tensor, fut: torch.Tensor, anchor: torch.Tensor,
                y_teacher: torch.Tensor | None = None):
        correction = super().forward(enc, fut, anchor)
        return anchor.unsqueeze(-1) + correction


class DeepARStyle(nn.Module):
    """Literature recreation: DeepAR (Salinas et al. 2020), adapted.

    Paper config: 3 LSTM layers x 40 cells, dropout 0.1, Adam, autoregressive
    decoder that consumes the previous target value.
    Logged deviations from the paper (docs/specs/deepar_spec.md): quantile
    heads instead of a Gaussian likelihood; single series (no cross-series
    training); our encoder covariates; teacher forcing in training with
    p50 fed back at inference.
    """

    name = "deepar_style"

    def __init__(self, enc_feat: int, fut_feat: int, hidden: int = 40, layers: int = 3):
        super().__init__()
        self.encoder = nn.LSTM(enc_feat, hidden, num_layers=layers,
                               batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(fut_feat + 1, hidden, num_layers=layers,
                               batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, N_Q)

    def forward(self, enc: torch.Tensor, fut: torch.Tensor, anchor: torch.Tensor,
                y_teacher: torch.Tensor | None = None):
        _, state = self.encoder(enc)
        last_load = enc[:, -1, 0:1]  # normalized load at the window end
        if self.training and y_teacher is not None:
            prev = torch.cat([last_load, y_teacher[:, :-1]], dim=1).unsqueeze(-1)
            out, _ = self.decoder(torch.cat([fut, prev], dim=-1), state)
            return self.head(out)
        # Inference: feed own p50 back, hour by hour.
        preds = []
        prev = last_load
        for t in range(fut.shape[1]):
            step = torch.cat([fut[:, t : t + 1], prev.unsqueeze(-1)], dim=-1)
            out, state = self.decoder(step, state)
            q = self.head(out)
            preds.append(q)
            prev = q[:, :, 1].detach()  # p50
        return torch.cat(preds, dim=1)


VARIANTS = {
    "enc_direct": EncDirect,
    "enc_futmlp": EncFutMLP,
    "enc_dec": EncDec,
    "residual": ResidualEncDec,
}
