#!/usr/bin/env python3
"""Make a GIF of coord vs dist histogram grids over time."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import imageio.v2 as imageio


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
    p = argparse.ArgumentParser(description="GIF of coord vs dist histogram grids.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--run1-frac", type=float, default=0.15, help="Fraction of run-1 data to plot")
    p.add_argument("--ymax", type=float, default=10.0)
    p.add_argument("--step", type=int, default=100, help="Add this many points per frame")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No coord_prod.dat + dist_prod.dat found.")

    if args.style.exists():
        plt.style.use(args.style)

    # preload series and global ranges
    series = []
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    max_len = 0
    for run_id, coord_path, dist_path in runs:
        x = read_series(coord_path)
        y = read_series(dist_path)
        n_min = min(x.size, y.size)
        if run_id == 1:
            n_min = max(1, int(n_min * args.run1_frac))
        x = x[:n_min]
        y = y[:n_min]
        max_len = max(max_len, n_min)
        series.append((run_id, x, y))
        if n_min > 0:
            x_min = x.min() if x_min is None else min(x_min, x.min())
            x_max = x.max() if x_max is None else max(x_max, x.max())
            y_min = y.min() if y_min is None else min(y_min, y.min())
            y_max = y.max() if y_max is None else max(y_max, y.max())

    if x_min is None or x_max is None or y_min is None or y_max is None:
        raise SystemExit("No valid coord/dist data found.")

    y_max = min(y_max, args.ymax)
    xedges = np.linspace(x_min, x_max, args.bins + 1)
    yedges = np.linspace(y_min, y_max, args.bins + 1)

    rows, cols = grid_shape(len(runs))
    frames: List[np.ndarray] = []

    total_frames = max(1, (max_len + args.step - 1) // args.step)
    for frame_idx in range(1, total_frames + 1):
        end = min(frame_idx * args.step, max_len)
        frac = end / max_len if max_len else 1.0
        print(f"Frame {frame_idx}/{total_frames} (progress {frac:.1%})")
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=160, sharex=True, sharey=True)
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (run_id, x, y) in enumerate(series):
            ax = axes_flat[i]
            if x.size == 0 or y.size == 0:
                ax.set_axis_off()
                continue
            n = min(x.size, y.size, max(1, int(np.ceil(frac * min(x.size, y.size)))))
            h, _, _ = np.histogram2d(x[:n], y[:n], bins=[xedges, yedges])
            h = np.ma.masked_equal(h, 0)
            ax.imshow(
                h.T,
                origin="lower",
                aspect="auto",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="Blues",
                norm=LogNorm(vmin=1),
            )
            ax.set_title(f"run-{run_id}", fontsize=9)
            ax.tick_params(labelsize=7)

        for ax in axes_flat[len(runs):]:
            ax.set_axis_off()

        for r in range(rows):
            if cols > 1:
                axes[r, 0].set_ylabel("d (Å)", fontsize=9)
            else:
                axes.set_ylabel("d (Å)", fontsize=9)
        for c in range(cols):
            if rows > 1:
                axes[rows - 1, c].set_xlabel("s", fontsize=9)
            else:
                axes.set_xlabel("s", fontsize=9)

        fig.suptitle(f"Progress: {frac:.1%}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(image)
        plt.close(fig)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_coord_dist_hist.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=4)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
