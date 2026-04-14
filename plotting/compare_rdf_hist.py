#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
import re

PLOT_R_MAX = 8.0
SMOOTH_WINDOW = 21  # must be odd
MIN_SEARCH_START = 1.5
MIN_DEDUP_EPS = 0.08
TARGET_MINIMA = [2.7, 5.0, 6.8]
TARGET_WINDOW = 0.6


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


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def find_minima_quadratic(x: np.ndarray, y: np.ndarray, n: int) -> List[float]:
    y_s = smooth(y, SMOOTH_WINDOW)
    mins = []
    for i in range(1, len(x) - 1):
        if x[i] < MIN_SEARCH_START:
            continue
        if y_s[i] < y_s[i - 1] and y_s[i] < y_s[i + 1]:
            mins.append(i)
    if not mins:
        return []
    refined = []
    for i in mins:
        lo = max(i - 2, 0)
        hi = min(i + 3, len(x))
        xf = x[lo:hi]
        yf = y_s[lo:hi]
        if len(xf) < 3:
            continue
        a, b, _ = np.polyfit(xf, yf, 2)
        if a == 0:
            continue
        rmin = -b / (2 * a)
        if rmin < xf[0] or rmin > xf[-1]:
            rmin = x[i]
        refined.append(rmin)
    refined = sorted(refined)
    return refined[:n]


def find_minima_fit(x: np.ndarray, y: np.ndarray, n: int) -> List[float]:
    y_s = smooth(y, SMOOTH_WINDOW)
    if len(x) < 6:
        return []
    deg = 8 if len(x) >= 12 else 5
    try:
        p = np.polynomial.Polynomial.fit(x, y_s, deg)
    except np.linalg.LinAlgError:
        return []
    dp = p.deriv()
    ddp = dp.deriv()
    roots = dp.roots()
    mins = []
    for r in roots:
        if not np.isreal(r):
            continue
        r = float(np.real(r))
        if r < MIN_SEARCH_START or r > x.max():
            continue
        if ddp(r) > 0:
            mins.append(r)
    if not mins:
        return []
    mins = sorted(mins)
    dedup = []
    for r in mins:
        if not dedup or abs(r - dedup[-1]) > MIN_DEDUP_EPS:
            dedup.append(r)
    return dedup[:n]


def find_minima_targets(x: np.ndarray, y: np.ndarray, targets: List[float]) -> List[float]:
    y_s = smooth(y, SMOOTH_WINDOW)
    minima = []
    for r0 in targets:
        lo = r0 - TARGET_WINDOW
        hi = r0 + TARGET_WINDOW
        mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        xw = x[mask]
        yw = y_s[mask]
        idx = int(np.argmin(yw))
        minima.append(float(xw[idx]))
    return minima


def annotate_minima(ax, r_mins: List[float], y_vals: np.ndarray, x_vals: np.ndarray,
                    y_lo: np.ndarray, y_hi: np.ndarray, label_fmt: str):
    for r in r_mins:
        y = np.interp(r, x_vals, y_vals)
        y_min = np.interp(r, x_vals, y_lo)
        y_max = np.interp(r, x_vals, y_hi)
        ax.axvline(r, color="gray", ls="--", lw=1, alpha=0.6)
        ax.annotate(label_fmt.format(r=r, y=y, y_min=y_min, y_max=y_max), xy=(r, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color="gray")


def _trim_xy(x, y, lo, hi):
    if len(x) != len(y):
        n = min(len(x), len(y))
        print(f"WARN: length mismatch x={len(x)} y={len(y)}; truncating to {n}")
        x = x[:n]
        y = y[:n]
    m = (x >= lo) & (x <= hi)
    return x[m], y[m]


def _parse_mask2_count(cpptraj_out: Path) -> int:
    text = cpptraj_out.read_text()
    m = re.search(r"(\d+)\s+atoms in Mask2", text)
    if not m:
        m = re.search(r"Mask2:\s*(\d+)", text)
    if not m:
        raise RuntimeError(f"Could not find Mask2 count in {cpptraj_out}")
    return int(m.group(1))


def _parse_box_volume(traj_path: Path) -> float:
    with traj_path.open("r") as fh:
        # First line: natoms, second line: comment with box
        fh.readline()
        line = fh.readline()
    if "Box" not in line:
        raise RuntimeError(f"No box info in {traj_path}")
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
    if len(nums) >= 9:
        a = nums[0]
        b = nums[4]
        c = nums[8]
    elif len(nums) >= 3:
        # Fallback: assume orthorhombic box with a, b, c as last three numbers
        a, b, c = nums[-3:]
    else:
        raise RuntimeError(f"Unexpected box line in {traj_path}: {line.strip()}")
    return a * b * c


def _coordination_from_gr(r: np.ndarray, g: np.ndarray, rho: float) -> np.ndarray:
    integrand = g * r**2
    dr = np.diff(r)
    avg = 0.5 * (integrand[:-1] + integrand[1:])
    integral = np.concatenate([[0.0], np.cumsum(avg * dr)])
    return 4.0 * np.pi * rho * integral


def aggregate_rdf_and_cn(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray, int]:
    if not paths:
        raise RuntimeError("No data files found")
    x_ref, y_ref = read_xy(paths[0])
    ys = [y_ref]
    cns = []
    for p in paths:
        x, y = read_xy(p)
        if len(x) != len(x_ref) or not np.allclose(x, x_ref, rtol=1e-6, atol=1e-8):
            y = np.interp(x_ref, x, y)
        ys.append(y)
        analysis_dir = p.parent
        cpptraj_out = analysis_dir / "cpptraj.out"
        traj_path = analysis_dir / "traj.cpptraj.xyz"
        mask2 = _parse_mask2_count(cpptraj_out)
        vol = _parse_box_volume(traj_path)
        rho = mask2 / vol
        cns.append(_coordination_from_gr(x_ref, y, rho))
    arr = np.vstack(ys)
    mean = np.mean(arr, axis=0)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    cn_arr = np.vstack(cns)
    cn_mean = np.mean(cn_arr, axis=0)
    cn_lo = np.percentile(cn_arr, 2.5, axis=0)
    cn_hi = np.percentile(cn_arr, 97.5, axis=0)
    return x_ref, mean, lo, hi, cn_mean, cn_lo, cn_hi, len(paths)


def plot_single(root: Path, left_label: str, right_label: str, out_path: Path, exclude: Set[str]) -> None:
    labels = [left_label, right_label]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    for col, label in enumerate(labels):
        rdf_paths = collect_label_files(root, label, "dat", exclude)

        r0, g0, g0_lo, g0_hi, cn0, cn0_lo, cn0_hi, n = aggregate_rdf_and_cn(rdf_paths)

        r, g = _trim_xy(r0, g0, 0.0, PLOT_R_MAX)
        r_lo, g_lo = _trim_xy(r0, g0_lo, 0.0, PLOT_R_MAX)
        r_hi, g_hi = _trim_xy(r0, g0_hi, 0.0, PLOT_R_MAX)
        r_int, cn = _trim_xy(r0, cn0, 0.0, PLOT_R_MAX)
        r_int_lo, cn_lo = _trim_xy(r0, cn0_lo, 0.0, PLOT_R_MAX)
        r_int_hi, cn_hi = _trim_xy(r0, cn0_hi, 0.0, PLOT_R_MAX)

        r_mins = find_minima_targets(r, g, TARGET_MINIMA)
        if len(r_mins) < 3:
            r_mins = find_minima_fit(r, g, 3)

        ax_rdf = axes[0, col]
        ax_rdf.fill_between(r, g_lo, g_hi, color="tab:blue", alpha=0.2)
        ax_rdf.plot(r, g, color="tab:blue", lw=2, label=f"n={n}")
        ax_rdf.set_title(label)
        ax_rdf.set_ylabel("g(r)")
        ax_rdf.grid(True, alpha=0.3)
        ax_rdf.legend()
        annotate_minima(ax_rdf, r_mins, g, r, g_lo, g_hi, "r={r:.2f}\nmin={y_min:.2f}\nmax={y_max:.2f}")

        ax_int = axes[1, col]
        ax_int.fill_between(r_int, cn_lo, cn_hi, color="tab:blue", alpha=0.2)
        ax_int.plot(r_int, cn, color="tab:blue", lw=2, label=f"n={n}")
        ax_int.set_xlabel("r (Å)")
        ax_int.set_ylabel("Coordination N(r)")
        ax_int.grid(True, alpha=0.3)
        ax_int.legend()
        for rmin in r_mins:
            cn_val = np.interp(rmin, r_int, cn)
            cn_min = np.interp(rmin, r_int, cn_lo)
            cn_max = np.interp(rmin, r_int, cn_hi)
            ax_int.axvline(rmin, color="gray", ls="--", lw=1, alpha=0.6)
            ax_int.annotate(f"r={rmin:.2f}\nN={cn_val:.2f}\nmin={cn_min:.2f}\nmax={cn_max:.2f}",
                            xy=(rmin, cn_val), xytext=(5, 5),
                            textcoords="offset points", fontsize=8, color="gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def plot_compare(dftb_root: Path, orb_root: Path, left_label: str, right_label: str,
                 out_path: Path, exclude: Set[str]) -> None:
    labels = [left_label, right_label]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    for col, label in enumerate(labels):
        dftb_rdf = collect_label_files(dftb_root, label, "dat", exclude)
        orb_rdf = collect_label_files(orb_root, label, "dat", exclude)

        r_d0, g_d0, g_d0_lo, g_d0_hi, cn_d0, cn_d0_lo, cn_d0_hi, n_dftb = aggregate_rdf_and_cn(dftb_rdf)
        r_o0, g_o0, g_o0_lo, g_o0_hi, cn_o0, cn_o0_lo, cn_o0_hi, n_orb = aggregate_rdf_and_cn(orb_rdf)

        r_dftb, g_dftb = _trim_xy(r_d0, g_d0, 0.0, PLOT_R_MAX)
        r_dftb_lo, g_dftb_lo = _trim_xy(r_d0, g_d0_lo, 0.0, PLOT_R_MAX)
        r_dftb_hi, g_dftb_hi = _trim_xy(r_d0, g_d0_hi, 0.0, PLOT_R_MAX)
        r_orb, g_orb = _trim_xy(r_o0, g_o0, 0.0, PLOT_R_MAX)
        r_orb_lo, g_orb_lo = _trim_xy(r_o0, g_o0_lo, 0.0, PLOT_R_MAX)
        r_orb_hi, g_orb_hi = _trim_xy(r_o0, g_o0_hi, 0.0, PLOT_R_MAX)

        r_int_dftb, cn_dftb = _trim_xy(r_d0, cn_d0, 0.0, PLOT_R_MAX)
        r_int_dftb_lo, cn_dftb_lo = _trim_xy(r_d0, cn_d0_lo, 0.0, PLOT_R_MAX)
        r_int_dftb_hi, cn_dftb_hi = _trim_xy(r_d0, cn_d0_hi, 0.0, PLOT_R_MAX)
        r_int_orb, cn_orb = _trim_xy(r_o0, cn_o0, 0.0, PLOT_R_MAX)
        r_int_orb_lo, cn_orb_lo = _trim_xy(r_o0, cn_o0_lo, 0.0, PLOT_R_MAX)
        r_int_orb_hi, cn_orb_hi = _trim_xy(r_o0, cn_o0_hi, 0.0, PLOT_R_MAX)

        r_mins = find_minima_targets(r_dftb, g_dftb, TARGET_MINIMA)
        if len(r_mins) < 3:
            r_mins = find_minima_fit(r_dftb, g_dftb, 3)

        ax_rdf = axes[0, col]
        ax_rdf.fill_between(r_dftb, g_dftb_lo, g_dftb_hi, color="tab:blue", alpha=0.2)
        ax_rdf.fill_between(r_orb, g_orb_lo, g_orb_hi, color="tab:orange", alpha=0.2)
        ax_rdf.plot(r_dftb, g_dftb, color="tab:blue", lw=2, label=f"DFTB (n={n_dftb})")
        ax_rdf.plot(r_orb, g_orb, color="tab:orange", lw=2, label=f"ORB (n={n_orb})")
        ax_rdf.set_title(label)
        ax_rdf.set_ylabel("g(r)")
        ax_rdf.grid(True, alpha=0.3)
        ax_rdf.legend()

        annotate_minima(ax_rdf, r_mins, g_dftb, r_dftb, g_dftb_lo, g_dftb_hi,
                        "r={r:.2f}\nmin={y_min:.2f}\nmax={y_max:.2f}")

        ax_int = axes[1, col]
        ax_int.fill_between(r_int_dftb, cn_dftb_lo, cn_dftb_hi, color="tab:blue", alpha=0.2)
        ax_int.fill_between(r_int_orb, cn_orb_lo, cn_orb_hi, color="tab:orange", alpha=0.2)
        ax_int.plot(r_int_dftb, cn_dftb, color="tab:blue", lw=2, label=f"DFTB (n={n_dftb})")
        ax_int.plot(r_int_orb, cn_orb, color="tab:orange", lw=2, label=f"ORB (n={n_orb})")
        ax_int.set_xlabel("r (Å)")
        ax_int.set_ylabel("Coordination N(r)")
        ax_int.grid(True, alpha=0.3)
        ax_int.legend()

        for rmin in r_mins:
            cn_val = np.interp(rmin, r_int_dftb, cn_dftb)
            cn_min = np.interp(rmin, r_int_dftb, cn_dftb_lo)
            cn_max = np.interp(rmin, r_int_dftb, cn_dftb_hi)
            ax_int.axvline(rmin, color="gray", ls="--", lw=1, alpha=0.6)
            ax_int.annotate(f"r={rmin:.2f}\nN={cn_val:.2f}\nmin={cn_min:.2f}\nmax={cn_max:.2f}",
                            xy=(rmin, cn_val), xytext=(5, 5),
                            textcoords="offset points", fontsize=8, color="gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dftb-root", required=True, type=Path, help="Root containing run-*/equil/analysis files")
    ap.add_argument("--orb-root", type=Path, default=None, help="Optional ORB root (omit for single-system plot)")
    ap.add_argument("--left-label", default="rdf_11_WAT_O", help="Left subplot RDF stem (without .dat/.int.dat)")
    ap.add_argument("--right-label", default="rdf_15_WAT_O", help="Right subplot RDF stem (without .dat/.int.dat)")
    ap.add_argument("--out", default="reports/rdf_compare_hist_solv_4.5.png", type=Path, help="Output PNG path")
    ap.add_argument("--exclude", default="", help="Comma-separated run numbers to exclude (e.g. 1,3,7)")
    args = ap.parse_args()

    exclude = {f"run-{x.strip()}" for x in args.exclude.split(",") if x.strip()}

    if args.orb_root is None:
        plot_single(args.dftb_root, args.left_label, args.right_label, args.out, exclude)
    else:
        plot_compare(args.dftb_root, args.orb_root, args.left_label, args.right_label, args.out, exclude)


if __name__ == "__main__":
    main()
