#!/usr/bin/env python3
"""Plot density slab profiles with mean ± std for x/y/z."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


TV_RE = re.compile(r"^TV\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)")


def read_density(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Unexpected format in {path}")
    x = data[:, 0]
    mean = data[:, 1]
    std = data[:, 2]
    return x, mean, std


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_pad = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def read_tv_lengths(dftb_inp: Path) -> Dict[str, float]:
    lengths: Dict[str, float] = {}
    with dftb_inp.open("r", encoding="utf-8") as f:
        for line in f:
            m = TV_RE.match(line.strip())
            if not m:
                continue
            vec = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            if abs(vec[0]) > 0 and abs(vec[1]) < 1e-6 and abs(vec[2]) < 1e-6:
                lengths["x"] = abs(vec[0])
            elif abs(vec[1]) > 0 and abs(vec[0]) < 1e-6 and abs(vec[2]) < 1e-6:
                lengths["y"] = abs(vec[1])
            elif abs(vec[2]) > 0 and abs(vec[0]) < 1e-6 and abs(vec[1]) < 1e-6:
                lengths["z"] = abs(vec[2])
    return lengths


def infer_run_id(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("run-"):
        return name
    return "run"


def infer_cv_dir(run_dir: Path) -> str:
    # For equil, use its parent as cv-dir label
    parent = run_dir.parent
    if parent.name.startswith("run-"):
        return "equil"
    return parent.name


def _plot_run(ax: plt.Axes, run_dir: Path, smooth_window: int) -> None:
    run_dir = run_dir.resolve()
    dens_files = {
        "x": run_dir / "density_x_slab.dat",
        "y": run_dir / "density_y_slab.dat",
        "z": run_dir / "density_z_slab.dat",
    }
    for axis, path in dens_files.items():
        if not path.exists():
            raise SystemExit(f"Missing {axis} slab file: {path}")

    dftb_inp = run_dir / "dftb.inp"
    if not dftb_inp.exists():
        raise SystemExit(f"Missing dftb.inp: {dftb_inp}")

    lengths = read_tv_lengths(dftb_inp)
    if not lengths:
        raise SystemExit(f"Could not parse TV lengths from {dftb_inp}")

    colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
    global_min = None
    global_max = None
    for axis in ("x", "y", "z"):
        x, mean, std = read_density(dens_files[axis])
        if axis in lengths:
            half = lengths[axis] / 2.0
            mask = (x >= -half) & (x <= half)
            x = x[mask]
            mean = mean[mask]
            std = std[mask]
        if x.size > 0:
            local_min = float(np.min(x))
            local_max = float(np.max(x))
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)
        mean_s = moving_average(mean, smooth_window)
        std_s = moving_average(std, smooth_window)
        ax.plot(x, mean_s, color=colors[axis], lw=1.6, label=f"{axis}-slab")
        ax.fill_between(x, mean_s - std_s, mean_s + std_s, color=colors[axis], alpha=0.2, lw=0)

    if global_min is not None and global_max is not None:
        ax.set_xlim(global_min, global_max)
    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Water O count (mean ± std)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(run_dir.name)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot density slab profiles.")
    p.add_argument("--run-dir", type=Path, required=True, help="Path to run dir (e.g., .../run-1/equil)")
    p.add_argument("--run-dir-2", type=Path, default=None, help="Second run dir for side-by-side plot")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    p.add_argument("--smooth-window", type=int, default=20, help="Moving average window size")
    p.add_argument("--style", type=Path, default=Path("plotting/prl.mplstyle"))
    args = p.parse_args()

    if args.style.exists():
        plt.style.use(args.style)
    if args.run_dir_2 is None:
        fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=220)
        _plot_run(ax, args.run_dir, args.smooth_window)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=220, sharey=True)
        _plot_run(axes[0], args.run_dir, args.smooth_window)
        _plot_run(axes[1], args.run_dir_2, args.smooth_window)

    out = args.out
    if out is None:
        run_id = infer_run_id(args.run_dir)
        cv_dir = infer_cv_dir(args.run_dir)
        out = Path("reports") / f"{run_id}_{cv_dir}_density_slabs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
