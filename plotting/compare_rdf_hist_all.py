#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt

PLOT_R_MAX = 8.0


def read_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected data format in {path}")
    return data[:, 0], data[:, 1]


def collect_label_files(root: Path, label: str, suffix: str, exclude: Set[str]) -> List[Path]:
    run_dirs = sorted(root.glob("run-*/equil/analysis"))
    files: List[Path] = []
    fname = f"{label}.{suffix}"
    for rdir in run_dirs:
        run_name = rdir.parent.parent.name
        if run_name in exclude:
            continue
        target = rdir / fname
        if not target.exists():
            print(f"WARN: missing {fname} in {rdir}")
            continue
        files.append(target)
    return files


def _trim_xy(x, y, lo, hi):
    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    m = (x >= lo) & (x <= hi)
    return x[m], y[m]


def plot_all(root: Path, left_label: str, right_label: str, out_path: Path, exclude: Set[str]) -> None:
    labels = [left_label, right_label]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for col, label in enumerate(labels):
        rdf_paths = collect_label_files(root, label, "dat", exclude)
        ax = axes[col]
        n = 0
        cmap = plt.get_cmap("tab20")
        for idx, p in enumerate(rdf_paths):
            r, g = read_xy(p)
            r, g = _trim_xy(r, g, 0.0, PLOT_R_MAX)
            color = cmap(idx % cmap.N)
            ax.plot(r, g, color=color, alpha=0.85, lw=0.6, label=p.parent.parent.parent.name)
            run_name = p.parent.parent.parent.name
            ax.text(r[-1], g[-1], run_name, fontsize=7, alpha=0.7, color=color,
                    ha="left", va="center")
            n += 1
        ax.set_title(f"{label} (n={n})")
        ax.set_xlabel("r (Å)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("g(r)")
        ax.legend(fontsize=7, ncol=2, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Root containing run-*/equil/analysis files")
    ap.add_argument("--left-label", default="rdf_11_WAT_O", help="Left subplot RDF stem (without .dat)")
    ap.add_argument("--right-label", default="rdf_15_WAT_O", help="Right subplot RDF stem (without .dat)")
    ap.add_argument("--out", default="reports/rdf_hist_all.png", type=Path, help="Output PNG path")
    ap.add_argument("--exclude", default="", help="Comma-separated run numbers to exclude (e.g. 1,3,7)")
    args = ap.parse_args()

    exclude = {f"run-{x.strip()}" for x in args.exclude.split(",") if x.strip()}
    plot_all(args.root, args.left_label, args.right_label, args.out, exclude)


if __name__ == "__main__":
    main()
