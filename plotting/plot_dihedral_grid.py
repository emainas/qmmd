#!/usr/bin/env python3
"""Plot chi1/chi2 2D density contour for each run in a grid."""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

POINTS = [
    (90, 180, "tp"),
    (-90, 180, "tm"),
    (90, 60, "gpp"),
    (-90, 60, "gpm"),
    (90, -60, "gmp"),
    (-90, -60, "gmm"),
]


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def read_series(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Bad dihedral file: {path}")
    return data[:, 1]


def discover_runs(runs_path: Path, rel_dir: Path) -> List[Tuple[int, Path, Path]]:
    out: List[Tuple[int, Path, Path]] = []
    for run_dir in sorted(runs_path.glob("run-*")):
        if not run_dir.is_dir():
            continue
        try:
            run_id = int(run_dir.name.split("-")[-1])
        except Exception:
            continue
        analysis_dir = run_dir / rel_dir
        chi1 = analysis_dir / "dih_chi1.dat"
        chi2 = analysis_dir / "dih_chi2.dat"
        if not chi1.exists() or not chi2.exists():
            print(f"WARN: missing dihedral files in {analysis_dir}")
            out.append((run_id, None, None))
            continue
        out.append((run_id, chi1, chi2))
    return out


def compute_hist(chi1: np.ndarray, chi2: np.ndarray, bins: int) -> np.ndarray:
    h, _, _ = np.histogram2d(
        chi1, chi2, bins=bins, range=[[-180, 180], [-180, 180]], density=True
    )
    return h.T  # transpose for plotting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    ap.add_argument("--rel-dir", default="equil/analysis", help="Relative path under each run")
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--out", type=Path, default=Path("reports/chi1_chi2_grid.png"))
    args = ap.parse_args()

    runs = discover_runs(args.runs_path, Path(args.rel_dir))
    if not runs:
        raise SystemExit("No runs found")

    # Precompute histograms and max density for consistent scaling
    hists = []
    max_val = 0.0
    for run_id, chi1_path, chi2_path in runs:
        if chi1_path is None:
            hists.append((run_id, None))
            continue
        chi1 = read_series(chi1_path)
        chi2 = read_series(chi2_path)
        h = compute_hist(chi1, chi2, args.bins)
        max_val = max(max_val, float(np.max(h)))
        hists.append((run_id, h))

    n = len(hists)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows), sharex=True, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    xs = np.linspace(-180, 180, args.bins)
    ys = np.linspace(-180, 180, args.bins)

    for i, (run_id, h) in enumerate(hists):
        ax = axes_flat[i]
        if h is None:
            ax.set_axis_off()
            continue
        cf = ax.contourf(xs, ys, h, levels=12, vmin=0.0, vmax=max_val, cmap="viridis")
        ax.set_title(f"run-{run_id}", fontsize=8)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        for x, y, label in POINTS:
            ax.plot(x, y, marker="o", ms=3, color="white", mec="black", mew=0.3)
            ax.text(x + 4, y + 4, label, fontsize=6, color="white")

    # turn off unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.text(0.5, 0.04, "chi1 (deg)", ha="center")
    fig.text(0.04, 0.5, "chi2 (deg)", va="center", rotation="vertical")
    fig.colorbar(cf, ax=axes_flat[:n], fraction=0.02, pad=0.02, label="density")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
