"""Shared style module for paper/figures/ -- import and call setup()
once per script, then use the size/color/marker constants below.

Design choices:
- Liberation Serif (Times New Roman metric-compatible, installed via
  fonts-liberation) at 8-9pt, matching typical two-column paper templates.
- Okabe-Ito colorblind-safe 8-color palette. Every series additionally
  gets a distinct marker AND linestyle/hatch so figures stay
  distinguishable when printed in grayscale, not just for colorblind
  readers on a screen.
- Two physical sizes per figure: SINGLE_COL (~3.3in wide, for a figure
  that fits one column) and DOUBLE_COL (~7.0in wide, spanning both
  columns) -- both saved as vector PDF and 300dpi PNG.
"""

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

LIBERATION_SERIF_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
LIBERATION_SERIF_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008).
COLOR_BLACK = "#000000"
COLOR_ORANGE = "#E69F00"
COLOR_SKY_BLUE = "#56B4E9"
COLOR_GREEN = "#009E73"
COLOR_YELLOW = "#F0E442"
COLOR_BLUE = "#0072B2"
COLOR_VERMILLION = "#D55E00"
COLOR_PURPLE = "#CC79A7"

# One consistent (color, marker, linestyle) triple per recurring series
# name used across this project's figures -- reuse these, don't invent
# new ones per script, so the same series always looks the same way
# across every figure in the paper.
SERIES_STYLE = {
    "baseline": {"color": COLOR_BLACK, "marker": "o", "linestyle": "-", "hatch": None},
    "A": {"color": COLOR_BLACK, "marker": "o", "linestyle": "-", "hatch": None},
    "recent_k": {"color": COLOR_BLUE, "marker": "s", "linestyle": "-", "hatch": "//"},
    "B": {"color": COLOR_BLUE, "marker": "s", "linestyle": "-", "hatch": "//"},
    "attention_topk": {"color": COLOR_VERMILLION, "marker": "^", "linestyle": "--", "hatch": "xx"},
    "speculative": {"color": COLOR_GREEN, "marker": "D", "linestyle": "--", "hatch": "\\\\"},
    "C": {"color": COLOR_GREEN, "marker": "D", "linestyle": "--", "hatch": "\\\\"},
    "combined": {"color": COLOR_VERMILLION, "marker": "^", "linestyle": "-.", "hatch": "xx"},
    "D": {"color": COLOR_VERMILLION, "marker": "^", "linestyle": "-.", "hatch": "xx"},
    "degraded": {"color": COLOR_VERMILLION, "marker": "^", "linestyle": None, "hatch": None},
    "improved": {"color": COLOR_BLUE, "marker": None, "linestyle": None, "hatch": "//"},
    "worsened": {"color": COLOR_VERMILLION, "marker": None, "linestyle": None, "hatch": "xx"},
    "true_positive": {"color": COLOR_BLUE, "marker": None, "linestyle": None, "hatch": "//"},
    "false_positive": {"color": COLOR_VERMILLION, "marker": None, "linestyle": None, "hatch": "xx"},
}

# Figure widths in inches. Height is set per-figure (aspect depends on
# content), these are the widths a journal/conference two-column
# template expects.
SINGLE_COL_WIDTH_IN = 3.3
DOUBLE_COL_WIDTH_IN = 7.0

FONT_SIZE_BASE = 8
FONT_SIZE_TITLE = 9
DPI = 300


def setup():
    fm.fontManager.addfont(LIBERATION_SERIF_REGULAR)
    fm.fontManager.addfont(LIBERATION_SERIF_BOLD)
    plt.rcParams.update({
        "font.family": "Liberation Serif",
        "font.size": FONT_SIZE_BASE,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_BASE,
        "xtick.labelsize": FONT_SIZE_BASE - 1,
        "ytick.labelsize": FONT_SIZE_BASE - 1,
        "legend.fontsize": FONT_SIZE_BASE - 1,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,   # embed as real (not Type 3) fonts in the PDF
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
    })


def save_figure(figure, name: str):
    """Saves figure as both {name}.pdf (vector) and {name}.png (300dpi)
    under paper/figures/. Caller sets the figure's own size beforehand
    (single- or double-column) -- this just handles the dual-format
    output contract every paper figure in this directory follows."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=DPI, bbox_inches="tight")
    return pdf_path, png_path
