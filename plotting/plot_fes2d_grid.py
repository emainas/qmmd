#!/usr/bin/env python3
"""Plot 2D FES contour grids (last block only) from cached fes.last.dat."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

RUN_RE = re.compile(r"^run-(\d+)$")


def reshape_fes(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data[:, 0]
    y = data[:, 1]
    f = data[:, 2]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = len(x_unique)
    ny = len(y_unique)
    z = f.reshape(ny, nx)
    return x_unique, y_unique, z


def find_minimum_near(x: np.ndarray, y: np.ndarray, z: np.ndarray, target: Tuple[float, float], window: float) -> Tuple[float, float, float]:
    tx, ty = target
    mask = (np.abs(x[None, :] - tx) <= window) & (np.abs(y[:, None] - ty) <= window)
    if not np.any(mask):
        xi = np.argmin(np.abs(x - tx))
        yi = np.argmin(np.abs(y - ty))
        return x[xi], y[yi], z[yi, xi]
    sub = np.where(mask)
    vals = z[sub]
    idx = np.argmin(vals)
    yi = sub[0][idx]
    xi = sub[1][idx]
    return x[xi], y[yi], z[yi, xi]


def discover_runs(runs_path: Path, cv_dir: str) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in sorted(runs_path.iterdir()):
        if not p.is_dir():
            continue
        m = RUN_RE.match(p.name)
        if not m:
            continue
        run_id = int(m.group(1))
        fes = p / cv_dir / "fes.last.dat"
        if fes.exists():
            out.append((run_id, fes))
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
    p = argparse.ArgumentParser(description="Plot 2D FES contour grids from cached last block.")
    p.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    p.add_argument("--cv-dir", required=True, help="CV directory under each run")
    p.add_argument("--style", type=Path, default=Path("plotting/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--window", type=float, default=0.1)
    args = p.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No fes.last.dat files found. Run precompute_fes2d.py first.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    targets = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]

    for i, (run_id, fes_path) in enumerate(runs):
        ax = axes_flat[i]
        data = np.loadtxt(fes_path)
        x, y, z = reshape_fes(data)
        z = z - np.nanmin(z)
        X, Y = np.meshgrid(x, y)
        levels = 30
        ax.contourf(X, Y, z, levels=levels, cmap="viridis")
        ax.contour(X, Y, z, levels=levels, colors="k", linewidths=0.3, alpha=0.4)

        m11 = find_minimum_near(x, y, z, targets[0], args.window)
        m10 = find_minimum_near(x, y, z, targets[1], args.window)
        m01 = find_minimum_near(x, y, z, targets[2], args.window)
        for mx, my, _ in (m11, m10, m01):
            ax.plot(mx, my, "r.")

        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.set_xlabel("CV1")
        ax.set_ylabel("CV2")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_fes2d_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
