"""
notes_style.py
Helper for reproducing the figure look of the statistical-mechanics notes.

    from notes_style import PALETTE, use_style, boxed
    use_style()                      # applies rotskoff.mplstyle
    fig, ax = plt.subplots()
    boxed(ax)                        # (optional) re-assert the closed-box look

Colours were measured directly from the lecture-note PDF.
"""
import os
import matplotlib.pyplot as plt

# ---- measured palette ----------------------------------------------------
PALETTE = {
    "navy":   "#1F3A5F",   # section headers / primary dark line
    "orange": "#D8481A",   # accent (links, TOC) / secondary line
    "teal":   "#2E9E7E",   # tertiary line
    "cyan":   "#1C9BD1",   # bright line / URL links
    "gold":   "#C9B23B",   # quaternary line
    "ink":    "#000000",   # reference / asymptote lines (often dashed black)
    "grey":   "#777777",
}
CYCLE = [PALETTE[k] for k in ("navy", "orange", "teal", "cyan", "gold")]

_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotskoff.mplstyle")


def use_style():
    """Apply the bundled .mplstyle (falls back gracefully if path differs)."""
    try:
        plt.style.use(_STYLE)
    except OSError:
        plt.style.use("rotskoff.mplstyle")


def boxed(ax):
    """Re-assert the closed-box / outward-tick look on a given Axes."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.1)
    ax.tick_params(direction="out", length=5, width=1.1)
    return ax
