"""Shared publication style for the experiment figures."""

import os
from functools import wraps

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # Registers the SciencePlots styles.


GRID_STYLE = {
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
}


def science_figure(draw):
    @wraps(draw)
    def styled(*args, **kwargs):
        styles = ["science"]
        if os.environ.get("BENDFLOW_PLOT_TEX") == "0":
            styles.append("no-latex")
        with plt.style.context([*styles, GRID_STYLE]):
            return draw(*args, **kwargs)

    return styled


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
