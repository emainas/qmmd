#!/usr/bin/env python3
"""Plot FES curves for specific (run, time) pairs on one plot."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
try:
    from scipy.signal import savgol_filter, find_peaks
except Exception:  # pragma: no cover
    savgol_filter = None
    find_peaks = None

# Edit this list to choose which (run_id, time_ps) to overlay
PAIRS: List[Tuple[int, float]] = [
    (14, 24.0),
    (15, 22.0),
    (20, 24.0),
    (22, 20.0),
    (27, 25.0),
    (30, 22.0),
    (31, 21.0),
    (36, 25.0),
    (49, 25.0),
    (51, 25.0),
    (56, 22.0),
    (57, 22.0),
    (58, 20.0),
    (6, 17.0),
    (63, 22.0),
    (65, 22.0),
    (67, 25.0),
    (70, 25.0),
    (71, 25.0),
    (72, 25.0),
    (73, 22.0),
    (75, 24.0),
    (77, 25.0),
    (78, 17.0),
    (81, 25.0),
    (82, 25.0),
    (83, 17.0),
    (86, 25.0),
    (90, 19.0),
    (91, 20.0),
    (96, 17.0)
]

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576
SMOOTH_WINDOW = 9  # odd
MINIMA_GLOBAL_RANGE = (0.0, 1.25)
PROMINENCE = 0.1


def parse_biaspot_times_ps(path: Path) -> np.ndarray:
    times: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)
    return np.array(times, dtype=float)


def read_fes_blocks(path: Path) -> List[np.ndarray]:
    blocks: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("### FREE ENERGY SURFACE"):
                if cur:
                    blocks.append(np.array(cur, dtype=float))
                    cur = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            cur.append((x, y))
    if cur:
        blocks.append(np.array(cur, dtype=float))
    return blocks


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def derivative_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if savgol_filter is None or find_peaks is None:
        raise RuntimeError("scipy is required for Savitzky-Golay smoothing (pip/conda install scipy).")
    ys = smooth(y, SMOOTH_WINDOW)
    if len(x) < 6:
        return x, np.zeros_like(x)
    # Restrict to global range for fitting stability
    gmin, gmax = MINIMA_GLOBAL_RANGE
    gmask = (x >= gmin) & (x <= gmax)
    x = x[gmask]
    ys = ys[gmask]
    if len(x) < 6:
        return x, np.zeros_like(x)
    win = min(9, len(x) // 2 * 2 + 1)
    if win < 5:
        win = 5
    ys_sg = savgol_filter(ys, window_length=win, polyorder=3, mode="interp")
    dy = np.gradient(ys_sg, x)
    return x, dy


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


def find_minimum_near_target(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
    half_window: float,
    xmin: float,
    xmax: float,
) -> Tuple[float, float]:
    lo = max(xmin, target - half_window)
    hi = min(xmax, target + half_window)
    m = (x >= lo) & (x <= hi)
    xw = x[m]
    yw = y[m]
    if xw.size == 0:
        raise ValueError("Empty FES window while searching minima")
    o = np.argsort(xw)
    xw = xw[o]
    yw = yw[o]
    idxs = []
    if len(yw) >= 2 and yw[0] <= yw[1]:
        idxs.append(0)
    for i in range(1, len(yw) - 1):
        if yw[i] < yw[i - 1] and yw[i] < yw[i + 1]:
            idxs.append(i)
    if len(yw) >= 2 and yw[-1] <= yw[-2]:
        idxs.append(len(yw) - 1)
    if idxs:
        b = min(idxs, key=lambda i: (abs(xw[i] - target), yw[i]))
    else:
        b = min(range(len(xw)), key=lambda i: (abs(xw[i] - target), yw[i]))
    return float(xw[b]), float(yw[b])


def deltaf(
    block: np.ndarray,
    min1_x: float,
    min2_x: float,
    half_window: float,
    xmin: float,
    xmax: float,
) -> float:
    m1 = find_minimum_near_target(block[:, 0], block[:, 1], min1_x, half_window, xmin, xmax)
    m2 = find_minimum_near_target(block[:, 0], block[:, 1], min2_x, half_window, xmin, xmax)
    return m1[1] - m2[1]


def main() -> None:
    if not PAIRS:
        raise SystemExit("PAIRS is empty; edit plotting/plot_fes_pairs.py to add (run_id, time_ps) pairs.")

    p = argparse.ArgumentParser(description="Plot FES curves for specific (run,time) pairs.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--fes-name", default="fes.dat")
    p.add_argument("--biaspot-name", default="biaspot")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--temp", type=float, default=313.15)
    p.add_argument("--min1-x", type=float, default=0.0)
    p.add_argument("--min2-x", type=float, default=1.0)
    p.add_argument("--half-window", type=float, default=0.1)
    p.add_argument("--fes-xmin", type=float, default=0.0)
    p.add_argument("--fes-xmax", type=float, default=1.25)
    args = p.parse_args()

    if args.style.exists():
        plt.style.use(args.style)

    fig = plt.figure(figsize=(10.5, 8.0), dpi=220)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2], wspace=0.3, hspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax_df = fig.add_subplot(gs[0, 1])
    ax_dfds = fig.add_subplot(gs[1, 0])
    ax_pka = fig.add_subplot(gs[1, 1])

    pka_vals: List[float] = []
    df_vals: List[float] = []
    pka_labels: List[str] = []
    pka_colors: List[str] = []
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_idx = 0

    ref_min_x = None
    for run_id, time_ps in PAIRS:
        run_dir = args.runs_path / f"run-{run_id}" / args.cv_dir
        biaspot = run_dir / args.biaspot_name
        fes = run_dir / args.fes_name
        if not biaspot.exists() or not fes.exists():
            print(f"SKIP: run-{run_id} missing {args.biaspot_name} or {args.fes_name}")
            continue

        times = parse_biaspot_times_ps(biaspot)
        if times.size == 0:
            print(f"SKIP: run-{run_id} has no biaspot times")
            continue

        idx = int(np.argmin(np.abs(times - time_ps)))
        blocks = read_fes_blocks(fes)
        if idx >= len(blocks):
            print(f"SKIP: run-{run_id} fes has {len(blocks)} blocks, need {idx+1}")
            continue

        data = blocks[idx]
        data[:, 1] *= EH_TO_KCALMOL

        try:
            df = deltaf(data, args.min1_x, args.min2_x, args.half_window, args.fes_xmin, args.fes_xmax)
            df_vals.append(df)
            pka = df / (PKA_FACTOR * args.temp)
            pka_vals.append(pka)
        except Exception as e:
            print(f"SKIP: run-{run_id} deltaF/pKa failed: {e}")

        try:
            min_x, min_y = find_minimum_near_target(
                data[:, 0], data[:, 1], args.min2_x, args.half_window, args.fes_xmin, args.fes_xmax
            )
            if ref_min_x is None:
                ref_min_x = min_y
            y_shift = (ref_min_x - min_y) if ref_min_x is not None else 0.0
        except Exception as e:
            print(f"SKIP: run-{run_id} shift failed: {e}")
            y_shift = 0.0

        data_plot = data.copy()
        data_plot[:, 1] = data_plot[:, 1] + y_shift
        color = color_cycle[color_idx % len(color_cycle)] if color_cycle else None
        color_idx += 1
        ax.plot(
            data_plot[:, 0],
            data_plot[:, 1],
            lw=0.8,
            color=color,
            label=f"run-{run_id} @ {time_ps:.2f} ps",
        )
        ax.text(
            data_plot[-1, 0],
            data_plot[-1, 1],
            f"run-{run_id}",
            fontsize=7,
            color=color,
            alpha=0.8,
            ha="left",
            va="center",
        )
        if len(pka_vals) > len(pka_colors):
            pka_colors.append(color or "#4c72b0")
            pka_labels.append(f"run-{run_id}")
        try:
            x_d, dy = derivative_curve(data[:, 0], data[:, 1])
            ax_dfds.plot(x_d, dy, lw=0.8, color=color, alpha=0.8)
        except Exception as e:
            print(f"SKIP: run-{run_id} dF/ds failed: {e}")

    ax.set_xlabel("s")
    ax.set_ylabel("F (kcal mol$^{-1}$)")
    ax.set_ylim(-40.0, 0.0)
    ax.set_xlim(args.fes_xmin, args.fes_xmax)
    ax.grid(alpha=0.25)
    # No legend; too many curves

    if df_vals:
        vals = np.array(df_vals, dtype=float)
        x = np.arange(len(vals))
        ax_df.bar(x, vals, color=pka_colors, alpha=0.85, edgecolor="white")
        ax_df.set_title("ΔF (kcal/mol)", fontsize=10)
        ax_df.set_ylabel("ΔF")
        ax_df.set_xticks(x)
        ax_df.set_xticklabels(pka_labels, rotation=90, ha="center", fontsize=5)
        ax_df.grid(alpha=0.25, axis="y")
        for i, v in enumerate(vals):
            ax_df.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax_df.set_axis_off()

    ax_dfds.set_title("dF/ds (smoothed)", fontsize=10)
    ax_dfds.set_xlabel("s")
    ax_dfds.set_ylabel("dF/ds")
    ax_dfds.set_xlim(args.fes_xmin, args.fes_xmax)
    ax_dfds.grid(alpha=0.25)

    if pka_vals:
        vals = np.array(pka_vals, dtype=float)
        x = np.arange(len(vals))
        ax_pka.bar(x, vals, color=pka_colors, alpha=0.85, edgecolor="white")
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax_pka.set_title("pKa", fontsize=10)
        ax_pka.set_ylabel("pKa")
        ax_pka.set_xticks(x)
        ax_pka.set_xticklabels(pka_labels, rotation=90, ha="center", fontsize=5)
        ax_pka.grid(alpha=0.25, axis="y")
        for i, v in enumerate(vals):
            ax_pka.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax_pka.plot([], [], label=f"mean={mean:.2f}", color="none")
        ax_pka.plot([], [], label=f"±std={std:.2f}", color="none")
        ax_pka.legend(loc="upper right", frameon=False, fontsize=9)
    else:
        ax_pka.set_axis_off()

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_fes_pairs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
