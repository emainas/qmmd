#!/usr/bin/env python3
"""
plot_temp_and_density_grid.py

Run:
  python plotting/plot_temp_and_density_grid.py

Reads:
  <root>/data/<SYSTEM>/solv_*_temp.dat
  <root>/data/<SYSTEM>/solv_*_density.dat

Writes:
  <root>/reports/temp_timeseries_grid.png
  <root>/reports/density_timeseries_grid.png
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


SYSTEM_CMAP = {
    "BV": "Blues",
    "TPS": "YlOrBr",   # yellow-ish but visible
    "MEA": "Greens",
    "HIST": "Reds",
}


def load_two_col_dat(path: Path):
    try:
        arr = np.loadtxt(path, comments="#")
    except Exception:
        return None, None

    if arr.size == 0:
        return None, None
    if arr.ndim == 1:
        if arr.shape[0] < 2:
            return None, None
        arr = arr.reshape(1, -1)

    return arr[:, 0], arr[:, 1]


def second_half_mean(y: np.ndarray) -> float:
    if y is None or y.size == 0:
        return float("nan")
    half = y.size // 2
    return float(np.mean(y[half:]))


def parse_buffer_from_file(path: Path) -> float | None:
    # expects: solv_5.0_temp.dat or solv_5.0_density.dat
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def plot_system_timeseries(ax, system: str, files: list[Path], ylabel: str, legend_loc="upper right"):
    """
    Draw all buffers for one system on a provided Axes.
    """
    series = []  # (buffer, t, y, mean)
    for f in sorted(files):
        buf = parse_buffer_from_file(f)
        if buf is None:
            continue
        t, y = load_two_col_dat(f)
        if t is None or y is None:
            continue
        m = second_half_mean(y)
        series.append((buf, t, y, m))

    ax.set_title(system)

    if not series:
        # Keep the axes but indicate missing
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    buffers = np.array([s[0] for s in series], dtype=float)
    norm = mpl.colors.Normalize(vmin=float(buffers.min()), vmax=float(buffers.max()))

    cmap_name = SYSTEM_CMAP.get(system, "Greys")
    cmap = plt.get_cmap(cmap_name)

    # plot in ascending buffer so shade ordering is consistent
    for buf, t, y, m in sorted(series, key=lambda x: x[0]):
        u = float(norm(buf))
        shade = 0.35 + 0.60 * u  # avoid too-light colors
        color = cmap(shade)
        ax.plot(t, y, lw=1.4, color=color, label=f"{buf:.1f}") #, {m:.3g}")

    #ax.legend(frameon=False, fontsize=7, title="buffer", title_fontsize=7)
    ax.legend(
    frameon=False,
    fontsize=7,
    title="buffer",
    title_fontsize=7,
    ncol=2,
    loc=legend_loc,
    columnspacing=0.8,
    handlelength=2.5,
    handletextpad=0.6,
    )


def make_grid_figure(systems: list[str], data_dir: Path, pattern: str, ylabel: str, out_png: Path, force_yticks=None, legend_loc: str = "upper right"):
    """
    pattern examples:
      "solv_*_temp.dat"
      "solv_*_density.dat"
    """
    n = len(systems)
    if n == 0:
        return

    # square-ish grid
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)

    for idx, system in enumerate(systems):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        sys_dir = data_dir / system
        files = sorted(sys_dir.glob(pattern)) if sys_dir.exists() else []
        plot_system_timeseries(ax, system, files, ylabel, legend_loc)

    # turn off unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")
    
    # One global label for the whole figure
    fig.supxlabel("Time (ps)")
    fig.supylabel(ylabel)

    if force_yticks is not None:
        # Apply shared ticks to all axes
        for ax in axes.flat:
            if ax.axison:
                ax.set_yticks(force_yticks)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    plt.style.use("prl.mplstyle")

    script_path = Path(__file__).resolve()
    root = script_path.parents[1]

    data_dir = root / "data"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    systems = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    make_grid_figure(
        systems=systems,
        data_dir=data_dir,
        pattern="solv_*_temp.dat",
        ylabel="Temperature (K)",
        out_png=reports_dir / "temp_timeseries_grid.png",
        force_yticks=[0, 100, 200, 300, 400],
        legend_loc="lower right",
    )

    make_grid_figure(
        systems=systems,
        data_dir=data_dir,
        pattern="solv_*_density.dat",
        ylabel=r"Density (g/cm$^3$)",
        out_png=reports_dir / "density_timeseries_grid.png",
        legend_loc="center right",
    )

    print(f"Wrote:\n  {reports_dir/'temp_timeseries_grid.png'}\n  {reports_dir/'density_timeseries_grid.png'}")


if __name__ == "__main__":
    main()

