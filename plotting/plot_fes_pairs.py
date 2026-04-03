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

# Edit this list to choose which (run_id, time_ps) to overlay
PAIRS: List[Tuple[int, float]] = [
    (11, 25.0),
    (19, 25.0),
    (20, 24.0),
    (6, 25.0),
    # (9, 22.0),
]

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576


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

    fig = plt.figure(figsize=(9.0, 4.5), dpi=220)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    pka_vals: List[float] = []
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
            pka = df / (PKA_FACTOR * args.temp)
            pka_vals.append(pka)
        except Exception as e:
            print(f"SKIP: run-{run_id} pKa failed: {e}")

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
            lw=1.6,
            color=color,
            label=f"run-{run_id} @ {time_ps:.2f} ps",
        )
        if len(pka_vals) > len(pka_colors):
            pka_colors.append(color or "#4c72b0")
            pka_labels.append(f"run-{run_id}")

    ax.set_xlabel("s")
    ax.set_ylabel("F (kcal mol$^{-1}$)")
    ax.set_ylim(-40.0, 0.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    if pka_vals:
        vals = np.array(pka_vals, dtype=float)
        x = np.arange(len(vals))
        ax_hist.bar(x, vals, color=pka_colors, alpha=0.85, edgecolor="white")
        mean = float(np.mean(vals))
        ax_hist.set_title("pKa", fontsize=10)
        ax_hist.set_ylabel("pKa")
        ax_hist.set_xticks(x)
        ax_hist.set_xticklabels(pka_labels, rotation=45, ha="right", fontsize=8)
        ax_hist.grid(alpha=0.25, axis="y")
        for i, v in enumerate(vals):
            ax_hist.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax_hist.plot([], [], label=f"mean={mean:.2f}", color="none")
        ax_hist.legend(loc="upper right", frameon=False, fontsize=9)
    else:
        ax_hist.set_axis_off()

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
