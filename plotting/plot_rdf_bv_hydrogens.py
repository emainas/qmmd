#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PLOT_R_MAX = 8.0


def read_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected data format in {path}")
    return data[:, 0], data[:, 1]


def collect_label_files(root: Path, label: str) -> List[Path]:
    run_dirs = sorted(root.glob("run-*/equil/analysis"))
    files: List[Path] = []
    fname = f"{label}.dat"
    for rdir in run_dirs:
        target = rdir / fname
        if not target.exists():
            print(f"WARN: missing {fname} in {rdir}")
            continue
        files.append(target)
    return files


def mean_rdf(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, int]:
    if not paths:
        raise RuntimeError("No data files found")
    x_ref, y_ref = read_xy(paths[0])
    ys = [y_ref]
    for p in paths[1:]:
        x, y = read_xy(p)
        if len(x) != len(x_ref) or not np.allclose(x, x_ref, rtol=1e-6, atol=1e-8):
            y = np.interp(x_ref, x, y)
        ys.append(y)
    arr = np.vstack(ys)
    mean = np.mean(arr, axis=0)
    return x_ref, mean, len(paths)


def trim_xy(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    m = (x >= lo) & (x <= hi)
    return x[m], y[m]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("systems/BV/solv_4.0/dftb/N1T64C1"),
                    help="Root containing run-*/equil/analysis files")
    ap.add_argument("--out", type=Path, default=Path("reports/BV_solv_4.0_rdf_HA_HB_HC_HD.png"),
                    help="Output PNG path")
    ap.add_argument("--r-max", type=float, default=PLOT_R_MAX, help="Max r (Å) to plot")
    args = ap.parse_args()

    style_path = Path(__file__).resolve().parent / "prl.mplstyle"
    if style_path.exists():
        plt.style.use(style_path)

    labels: Dict[str, str] = {
        "rdf_1_HA_WAT_O": "HA-O(H$_2$O)",
        "rdf_1_HB_WAT_O": "HB-O(H$_2$O)",
        "rdf_1_HC_WAT_O": "HC-O(H$_2$O)",
        "rdf_1_HD_WAT_O": "HD-O(H$_2$O)",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("tab10")

    for i, (stem, legend_label) in enumerate(labels.items()):
        paths = collect_label_files(args.root, stem)
        r0, g0, n = mean_rdf(paths)
        r, g = trim_xy(r0, g0, 0.0, args.r_max)
        ax.plot(r, g, lw=2.2, color=cmap(i), label=f"{legend_label} (n={n})")

    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0.0, args.r_max)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
