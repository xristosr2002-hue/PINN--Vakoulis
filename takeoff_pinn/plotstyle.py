"""Shared figure style (run plots and thesis figures).

Validated reference palette, used in fixed slot order (never cycled; colour
follows the entity). Slots 3-4 sit below 3:1 contrast on the light surface,
so every multi-series figure also carries a legend, direct labels and
distinct line styles (which also survive black-and-white printing).
Correlations use a blue-gray-red diverging map with a neutral midpoint.
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
LINESTYLES = ["-", "--", "-.", ":"]
INK = "#0b0b0b"         # primary ink (also the "total"/reference line)
INK_2 = "#52514e"       # secondary ink
MUTED = "#898781"       # axis labels / ticks
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
TRUTH = INK_2           # reference/truth curves: neutral, dashed

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "legend.frameon": False, "font.size": 9.5,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def label_line_end(ax, x, y, text, color=INK_2) -> None:
    """Direct label at the right end of a line (text in ink, not series colour)."""
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=color)
