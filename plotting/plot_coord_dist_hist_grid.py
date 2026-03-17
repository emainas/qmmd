#!/usr/bin/env python3
"""Plot 2D histogram grids of coord vs dist (prod only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def read_series(path: Path) -> np.ndarray:
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                vals.append(float(parts[-1]))
    return np.array(vals, dtype=float)


def discover_runs(runs_path: Path, cv_dir: str) -> List[Tuple[int, Path, Path]]:
    out: List[Tuple[int, Path, Path]] = []
    for p in sorted(runs_path.iterdir()):
        if not (p.is_dir() and p.name.startswith("run-")):
            continue
        try:
            run_id = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        run_cv = p / cv_dir
        coord = run_cv / "manual-cv" / "coord_prod.dat"
        dist = run_cv / "manual-cv" / "dist_prod.dat"
        if coord.exists() and dist.exists():
            out.append((run_id, coord, dist))
    return out


def grid_shape(n: int) -> Tuple[int, int]:
    if n == 20:
        return 4, 5
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    return rows, cols


def infer_system(runs_path: Path) -> str:
    parts = runs_path.resolve().parts
    if "systems" in parts:
        idx = parts.index("systems")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "system"


def main() -> None:
    p = argparse.ArgumentParser(description="Plot coord vs dist 2D histogram grids.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--run1-frac", type=float, default=0.15, help="Fraction of run-1 data to plot")
    p.add_argument("--ymax", type=float, default=10.0, help="Max distance (y-axis)")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No coord_prod.dat + dist_prod.dat found.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220, sharex=True, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # pre-load to get global ranges
    series = []
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    for run_id, coord_path, dist_path in runs:
        x = read_series(coord_path)
        y = read_series(dist_path)
        if x.size == 0 or y.size == 0:
            series.append((run_id, x, y))
            continue
        n_min = min(x.size, y.size)
        if run_id == 1:
            n_min = max(1, int(n_min * args.run1_frac))
        x = x[:n_min]
        y = y[:n_min]
        series.append((run_id, x, y))
        x_min = x.min() if x_min is None else min(x_min, x.min())
        x_max = x.max() if x_max is None else max(x_max, x.max())
        y_min = y.min() if y_min is None else min(y_min, y.min())
        y_max = y.max() if y_max is None else max(y_max, y.max())

    if x_min is None or x_max is None or y_min is None or y_max is None:
        raise SystemExit("No valid coord/dist data found.")

    xedges = np.linspace(x_min, x_max, args.bins + 1)
    yedges = np.linspace(y_min, min(y_max, args.ymax), args.bins + 1)

    for i, (run_id, x, y) in enumerate(series):
        ax = axes_flat[i]
        if x.size == 0 or y.size == 0:
            ax.set_axis_off()
            continue
        h, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        h = np.ma.masked_equal(h, 0)
        ax.imshow(
            h.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="Blues",
            norm=LogNorm(vmin=1),
        )
        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    for r in range(rows):
        if cols > 1:
            axes[r, 0].set_ylabel("d (Å)", fontsize=10)
        else:
            axes.set_ylabel("d (Å)", fontsize=10)
    for c in range(cols):
        if rows > 1:
            axes[rows - 1, c].set_xlabel("s", fontsize=10)
        else:
            axes.set_xlabel("s", fontsize=10)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_coord_dist_hist_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
