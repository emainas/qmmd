#!/usr/bin/env python3
"""Plot chi1 and chi2 time series with all runs overlaid."""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
            continue
        out.append((run_id, chi1, chi2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    ap.add_argument("--rel-dir", default="equil/analysis", help="Relative path under each run")
    ap.add_argument("--out", type=Path, default=Path("reports/chi_timeseries.png"))
    args = ap.parse_args()

    runs = discover_runs(args.runs_path, Path(args.rel_dir))
    if not runs:
        raise SystemExit("No runs found")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    cmap = plt.get_cmap("tab20")

    for idx, (run_id, chi1_path, chi2_path) in enumerate(runs):
        color = cmap(idx % cmap.N)
        chi1 = read_series(chi1_path)
        chi2 = read_series(chi2_path)
        t1 = np.arange(len(chi1))
        t2 = np.arange(len(chi2))
        axes[0].plot(t1, chi1, color=color, lw=0.5, alpha=0.9, label=f"run-{run_id}")
        axes[1].plot(t2, chi2, color=color, lw=0.5, alpha=0.9, label=f"run-{run_id}")

    axes[0].set_title("chi1")
    axes[1].set_title("chi2")
    axes[0].set_xlabel("frame")
    axes[1].set_xlabel("frame")
    axes[0].set_ylabel("dihedral (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
