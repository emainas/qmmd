#!/usr/bin/env python3
"""Plot equil+prod distance time-series grids."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
        series_all = run_cv / "manual-cv" / "dist.dat"
        series_prod = run_cv / "manual-cv" / "dist_prod.dat"
        if series_all.exists() and series_prod.exists():
            out.append((run_id, series_all, series_prod))
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
    p = argparse.ArgumentParser(description="Plot equil+prod distance grids.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No dist.dat + dist_prod.dat found.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    eq_window_ps = 20.0
    prod_window_ps = 15.0
    run1_prod_ps = 5.0

    special_runs = {1, 3, 4, 9, 12, 16}
    threshold = 4.5
    for i, (run_id, series_all_path, series_prod_path) in enumerate(runs):
        ax = axes_flat[i]
        all_vals = read_series(series_all_path)
        prod_vals = read_series(series_prod_path)
        if all_vals.size == 0:
            ax.set_axis_off()
            continue
        prod_vals = prod_vals.copy()
        if run_id == 1:
            prod_vals = prod_vals[: max(1, int(prod_vals.size * (run1_prod_ps / prod_window_ps)))]
            prod_ps = run1_prod_ps
        else:
            prod_ps = prod_window_ps
        if prod_vals.size > 1:
            t_prod = np.linspace(eq_window_ps, eq_window_ps + prod_ps, prod_vals.size)
        elif prod_vals.size == 1:
            t_prod = np.array([eq_window_ps])
        else:
            t_prod = np.array([])
        if prod_vals.size:
            color = "lightgreen" if run_id in special_runs else "#1f77b4"
            ax.plot(t_prod, prod_vals, color=color, lw=1.6)
        ymax = 8.0
        ax.axhspan(threshold, ymax, color="salmon", alpha=0.25, zorder=0)
        ax.axhline(threshold, color="#d62728", lw=2.2)
        ax.set_xlim(eq_window_ps, eq_window_ps + prod_ps)
        ax.set_ylim(0.0, ymax)
        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.grid(alpha=0.25)
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
            axes[rows - 1, c].set_xlabel("t (ps)", fontsize=10)
        else:
            axes.set_xlabel("t (ps)", fontsize=10)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_dist_prod_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
