#!/usr/bin/env python3
"""Plot pKa1/pKa2 vs time grids from cached minima.dat."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RUN_RE = re.compile(r"^run-(\d+)$")
PKA_FACTOR = 0.004576
EH_TO_KCALMOL = 627.509474


def discover_runs(runs_path: Path, cv_dir: str) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in sorted(runs_path.iterdir()):
        if not p.is_dir():
            continue
        m = RUN_RE.match(p.name)
        if not m:
            continue
        run_id = int(m.group(1))
        minima = p / cv_dir / "minima.dat"
        if minima.exists():
            out.append((run_id, minima))
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


def load_minima(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 10:
        raise ValueError(f"Unexpected minima format in {path}")
    t = data[:, 0]
    m11_f = data[:, 3]
    m10_f = data[:, 6]
    m01_f = data[:, 9]
    if np.nanmax(np.abs(np.concatenate([m11_f, m10_f, m01_f]))) < 0.1:
        m11_f = m11_f * EH_TO_KCALMOL
        m10_f = m10_f * EH_TO_KCALMOL
        m01_f = m01_f * EH_TO_KCALMOL
    return t, m11_f, m10_f, m01_f


def main() -> None:
    p = argparse.ArgumentParser(description="Plot pKa grids from cached minima.dat.")
    p.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    p.add_argument("--cv-dir", required=True, help="CV directory under each run")
    p.add_argument("--style", type=Path, default=Path("plotting/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--temp", type=float, default=313.15)
    p.add_argument("--exp-pkas", type=str, default=None, help="Comma-separated exp pKas for pKa1,pKa2")
    args = p.parse_args()

    exp1: Optional[float] = None
    exp2: Optional[float] = None
    if args.exp_pkas:
        parts = [p.strip() for p in args.exp_pkas.split(",") if p.strip()]
        if len(parts) >= 1:
            exp1 = float(parts[0])
        if len(parts) >= 2:
            exp2 = float(parts[1])

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No minima.dat files found. Run precompute_fes2d.py first.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (run_id, minima_path) in enumerate(runs):
        ax = axes_flat[i]
        t, m11_f, m10_f, m01_f = load_minima(minima_path)
        t0 = float(t[0])
        t_rel = t - t0

        df_11_01 = m11_f - m01_f
        df_11_10 = m11_f - m10_f
        pka1 = df_11_01 / (PKA_FACTOR * args.temp)
        pka2 = df_11_10 / (PKA_FACTOR * args.temp)

        ax.plot(t_rel, pka1, color="red", lw=1.6, label="pKa1 (11-01)")
        ax.plot(t_rel, pka2, color="blue", lw=1.6, label="pKa2 (11-10)")
        if exp1 is not None:
            ax.axhline(exp1, color="red", ls="--", lw=1.2)
        if exp2 is not None:
            ax.axhline(exp2, color="blue", ls="--", lw=1.2)
        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.set_xlim(0.0, float(t_rel[-1]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + t0:.1f}"))
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    for r in range(rows):
        if cols > 1:
            axes[r, 0].set_ylabel("pKa", fontsize=10)
        else:
            axes.set_ylabel("pKa", fontsize=10)
    for c in range(cols):
        if rows > 1:
            axes[rows - 1, c].set_xlabel("t (ps)", fontsize=10)
        else:
            axes.set_xlabel("t (ps)")

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_pka2d_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
