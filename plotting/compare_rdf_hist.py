#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected data format in {path}")
    return data[:, 0], data[:, 1]


def collect_label_files(root: Path, label: str, suffix: str) -> List[Path]:
    run_dirs = sorted(root.glob("run-*/equil/analysis"))
    files: List[Path] = []
    fname = f"{label}.{suffix}"
    for rdir in run_dirs:
        target = rdir / fname
        if not target.exists():
            print(f"WARN: missing {fname} in {rdir}")
            continue
        files.append(target)
    return files


def aggregate_curve(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
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
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    return x_ref, mean, lo, hi, len(paths)


def first_n_minima(x: np.ndarray, y: np.ndarray, n: int) -> List[float]:
    mins: List[int] = []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] < y[i + 1]:
            mins.append(i)
    r_mins = [x[i] for i in mins]
    return r_mins[:n]


def annotate_minima(ax, r_mins: List[float], y_vals: np.ndarray, x_vals: np.ndarray, label_fmt: str):
    for r in r_mins:
        y = np.interp(r, x_vals, y_vals)
        ax.axvline(r, color="gray", ls="--", lw=1, alpha=0.6)
        ax.annotate(label_fmt.format(r=r, y=y), xy=(r, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color="gray")


def plot_compare(dftb_root: Path, orb_root: Path, left_label: str, right_label: str, out_path: Path) -> None:
    labels = [left_label, right_label]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    for col, label in enumerate(labels):
        dftb_rdf = collect_label_files(dftb_root, label, "dat")
        orb_rdf = collect_label_files(orb_root, label, "dat")
        dftb_int = collect_label_files(dftb_root, label, "int.dat")
        orb_int = collect_label_files(orb_root, label, "int.dat")

        r_dftb, g_dftb, g_dftb_lo, g_dftb_hi, n_dftb = aggregate_curve(dftb_rdf)
        r_orb, g_orb, g_orb_lo, g_orb_hi, n_orb = aggregate_curve(orb_rdf)
        r_int_dftb, cn_dftb, cn_dftb_lo, cn_dftb_hi, _ = aggregate_curve(dftb_int)
        r_int_orb, cn_orb, cn_orb_lo, cn_orb_hi, _ = aggregate_curve(orb_int)

        r_mins = first_n_minima(r_dftb, g_dftb, 3)

        ax_rdf = axes[0, col]
        ax_rdf.fill_between(r_dftb, g_dftb_lo, g_dftb_hi, color="tab:blue", alpha=0.2)
        ax_rdf.fill_between(r_orb, g_orb_lo, g_orb_hi, color="tab:orange", alpha=0.2)
        ax_rdf.plot(r_dftb, g_dftb, color="tab:blue", lw=2, label=f"DFTB (n={n_dftb})")
        ax_rdf.plot(r_orb, g_orb, color="tab:orange", lw=2, label=f"ORB (n={n_orb})")
        ax_rdf.set_title(label)
        ax_rdf.set_ylabel("g(r)")
        ax_rdf.grid(True, alpha=0.3)
        ax_rdf.legend()

        annotate_minima(ax_rdf, r_mins, g_dftb, r_dftb, "r={r:.2f}")

        ax_int = axes[1, col]
        ax_int.fill_between(r_int_dftb, cn_dftb_lo, cn_dftb_hi, color="tab:blue", alpha=0.2)
        ax_int.fill_between(r_int_orb, cn_orb_lo, cn_orb_hi, color="tab:orange", alpha=0.2)
        ax_int.plot(r_int_dftb, cn_dftb, color="tab:blue", lw=2, label=f"DFTB (n={n_dftb})")
        ax_int.plot(r_int_orb, cn_orb, color="tab:orange", lw=2, label=f"ORB (n={n_orb})")
        ax_int.set_xlabel("r (Å)")
        ax_int.set_ylabel("Coordination N(r)")
        ax_int.grid(True, alpha=0.3)
        ax_int.legend()

        for r in r_mins:
            cn_val = np.interp(r, r_int_dftb, cn_dftb)
            ax_int.axvline(r, color="gray", ls="--", lw=1, alpha=0.6)
            ax_int.annotate(f"r={r:.2f}\nN={cn_val:.2f}", xy=(r, cn_val), xytext=(5, 5),
                            textcoords="offset points", fontsize=8, color="gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dftb-root",
        default="systems/HIST/solv_4.5/dftb/N1T64C1",
        help="Root containing run-*/equil/analysis/*.dat",
    )
    ap.add_argument(
        "--orb-root",
        default="systems/HIST/solv_4.5/orb/N1T1C4",
        help="Root containing run-*/equil/analysis/*.dat",
    )
    ap.add_argument(
        "--left-label",
        default="rdf_11_WAT_O",
        help="Left subplot RDF stem (without .dat/.int.dat)",
    )
    ap.add_argument(
        "--right-label",
        default="rdf_15_WAT_O",
        help="Right subplot RDF stem (without .dat/.int.dat)",
    )
    ap.add_argument(
        "--out",
        default="reports/rdf_compare_hist_solv_4.5.png",
        help="Output PNG path",
    )
    args = ap.parse_args()

    plot_compare(Path(args.dftb_root), Path(args.orb_root), args.left_label, args.right_label, Path(args.out))


if __name__ == "__main__":
    main()
