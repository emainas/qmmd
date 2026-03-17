#!/usr/bin/env python3
"""Plot CV time-series grids from biaspot files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
COORD_RE = re.compile(r"Coordinate\s*=\s*([+-]?[0-9.]+)")


def parse_biaspot(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    coords: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)  # fsec -> ps
                continue
            m = COORD_RE.search(line)
            if m:
                coords.append(float(m.group(1)))
    n = min(len(times), len(coords))
    if n == 0:
        return np.array([]), np.array([])
    t = np.array(times[:n])
    return t, np.array(coords[:n])


def discover_runs(runs_path: Path, cv_dir: str) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in sorted(runs_path.iterdir()):
        if not (p.is_dir() and p.name.startswith("run-")):
            continue
        try:
            run_id = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        biaspot = p / cv_dir / "biaspot"
        if biaspot.exists():
            out.append((run_id, biaspot))
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
    p = argparse.ArgumentParser(description="Plot CV grids from biaspot files.")
    p.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    p.add_argument("--cv-dir", required=True, help="CV directory under each run (e.g., all-meta-hid)")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No biaspot files found.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    all_y = []
    for i, (run_id, biaspot) in enumerate(runs):
        ax = axes_flat[i]
        t, y = parse_biaspot(biaspot)
        if y.size == 0:
            ax.set_axis_off()
            continue
        t0 = float(t[0])
        t_rel = t - t0
        ax.plot(t_rel, y, color="#1f77b4", lw=1.6)
        ax.set_xlim(0.0, float(t_rel[-1]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + t0:.1f}"))
        ylo, yhi = np.min(y), np.max(y)
        pad = 0.05 * (yhi - ylo) if yhi > ylo else 0.05
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        all_y.append(y)

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    # per-run scaling already applied via each panel's data range

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
        out = Path("reports") / f"{system}_{args.cv_dir}_cv_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
