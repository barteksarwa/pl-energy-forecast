"""Shared chart style. Palette validated for colorblind safety (dataviz checks).

Rules applied everywhere:
- one y-axis per chart, units in the axis label,
- recessive grid, direct labels where possible,
- timezone always stated on the time axis.
"""

from __future__ import annotations

import matplotlib as mpl

# Fixed categorical order. Never cycle past it; fold extras into "other".
BLUE = "#0072B2"  # primary series (actuals, our forecast)
ORANGE = "#E69F00"  # benchmark series (TSO)
GREEN = "#009E73"  # third series
PINK = "#CC79A7"  # fourth series
CATEGORICAL = [BLUE, ORANGE, GREEN, PINK]

GRID_COLOR = "#d9d9d9"
BAND_ALPHA = 0.20  # P10-P90 uncertainty band fill


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "lines.linewidth": 2.0,
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )
