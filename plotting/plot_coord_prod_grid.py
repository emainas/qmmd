#!/usr/bin/env python3
"""Plot prod-only coordination time-series grids."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")


def parse_biaspot_times(path: Path) -> np.ndarray:
    times: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)
    t = np.array(times, dtype=float)
    if t.size:
        t = t - t[0]
    return t


def read_series(path: Path) -> np.ndarray:
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                vals.append(float(parts[-1]))
    return np.array(vals, dtype=float)


def discover_runs(runs_path: Path, cv_dir: str, biaspot_name: str) -> List[Tuple[int, Path, Path]]:
    out: List[Tuple[int, Path, Path]] = []
    for p in sorted(runs_path.iterdir()):
        if not (p.is_dir() and p.name.startswith("run-")):
            continue
        try:
            run_id = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        run_cv = p / cv_dir
        series = run_cv / "manual-cv" / "coord_prod.dat"
        biaspot = run_cv / biaspot_name
        if series.exists() and biaspot.exists():
            out.append((run_id, series, biaspot))
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
    p = argparse.ArgumentParser(description="Plot prod-only coordination grids.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--biaspot-name", default="biaspot")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir, args.biaspot_name)
    if not runs:
        raise SystemExit("No coord_prod.dat + biaspot found.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (run_id, series_path, biaspot) in enumerate(runs):
        ax = axes_flat[i]
        y = read_series(series_path)
        t = parse_biaspot_times(biaspot)
        if y.size == 0:
            ax.set_axis_off()
            continue
        if t.size == y.size:
            x = t
        else:
            x = np.arange(y.size, dtype=float)
        ax.plot(x, y, color="#1f77b4", lw=1.6)
        ax.set_xlim(x.min(), x.max())
        ylo, yhi = np.min(y), np.max(y)
        pad = 0.05 * (yhi - ylo) if yhi > ylo else 0.05
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    for r in range(rows):
        if cols > 1:
            axes[r, 0].set_ylabel("s", fontsize=10)
        else:
            axes.set_ylabel("s", fontsize=10)
    for c in range(cols):
        if rows > 1:
            axes[rows - 1, c].set_xlabel("t (ps)", fontsize=10)
        else:
            axes.set_xlabel("t (ps)", fontsize=10)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_coord_prod_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
