#!/usr/bin/env python3
"""Plot selected FES pairs in a multi-panel grid (one curve per panel)."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
EH_TO_KCALMOL = 627.509474
SMOOTH_WINDOW = 9


def load_pairs(pairs_script: Path) -> List[Tuple[int, float]]:
    text = pairs_script.read_text()
    m = re.search(r"PAIRS[^\n=]*=\s*(\[.*?\])", text, flags=re.S | re.M)
    if not m:
        raise RuntimeError(f"Could not find PAIRS in {pairs_script}")
    raw = m.group(1)
    # strip comments inside list
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"#.*$", "", line)
        lines.append(line)
    cleaned = "\n".join(lines)
    return ast.literal_eval(cleaned)


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


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def main() -> None:
    p = argparse.ArgumentParser(description="Plot selected FES pairs in multi-panel grid.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--pairs-script", type=Path, default=Path("plotting/plot_fes_pairs.py"))
    p.add_argument("--fes-name", default="fes.dat")
    p.add_argument("--biaspot-name", default="biaspot")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min2-x", type=float, default=1.0)
    p.add_argument("--half-window", type=float, default=0.1)
    p.add_argument("--fes-xmin", type=float, default=0.0)
    p.add_argument("--fes-xmax", type=float, default=1.25)
    args = p.parse_args()

    pairs = load_pairs(args.pairs_script)
    if not pairs:
        raise SystemExit("PAIRS is empty in the pairs script")

    if args.style.exists():
        plt.style.use(args.style)

    rows, cols = grid_shape(len(pairs))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), dpi=220)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_idx = 0

    for i, (run_id, time_ps) in enumerate(pairs):
        ax = axes_flat[i]
        run_dir = args.runs_path / f"run-{run_id}" / args.cv_dir
        biaspot = run_dir / args.biaspot_name
        fes = run_dir / args.fes_name
        if not biaspot.exists() or not fes.exists():
            ax.set_axis_off()
            continue

        times = parse_biaspot_times_ps(biaspot)
        if times.size == 0:
            ax.set_axis_off()
            continue

        idx = int(np.argmin(np.abs(times - time_ps)))
        blocks = read_fes_blocks(fes)
        if idx >= len(blocks):
            ax.set_axis_off()
            continue

        data = blocks[idx]
        data[:, 1] *= EH_TO_KCALMOL

        try:
            min_x, min_y = find_minimum_near_target(
                data[:, 0], data[:, 1], args.min2_x, args.half_window, args.fes_xmin, args.fes_xmax
            )
            # Shift so minima align
            y_shift = -min_y
        except Exception:
            y_shift = 0.0

        data_plot = data.copy()
        data_plot[:, 1] = data_plot[:, 1] + y_shift

        color = color_cycle[color_idx % len(color_cycle)] if color_cycle else None
        color_idx += 1

        # F(s) + dF/ds in same panel (twin axis)
        ax.plot(data_plot[:, 0], data_plot[:, 1], lw=1.2, color=color)
        if savgol_filter is not None:
            x = data_plot[:, 0]
            y = data_plot[:, 1]
            win = min(9, len(x) // 2 * 2 + 1)
            if win < 5:
                win = 5
            y_sg = savgol_filter(y, window_length=win, polyorder=3, mode="interp")
            dy = np.gradient(y_sg, x)
            ax2 = ax.twinx()
            ax2.plot(x, dy, lw=0.8, color=color, alpha=0.7, ls="--")
            ax2.axhline(0.0, color="black", lw=0.6, alpha=0.5)
            # Mark zero crossings of dF/ds
            signs = np.sign(dy)
            zidx = np.where(np.diff(signs) != 0)[0]
            star_count = 0
            for zi in zidx:
                x0, x1 = x[zi], x[zi + 1]
                if x0 > 1.4 and x1 > 1.4:
                    continue
                y0, y1 = dy[zi], dy[zi + 1]
                if y1 == y0:
                    xc = x0
                else:
                    xc = x0 - y0 * (x1 - x0) / (y1 - y0)
                ax2.plot(xc, 0.0, marker="*", ms=5, color=color, mec="black", mew=0.3)
                star_count += 1
            ax2.tick_params(axis="y", labelsize=6)
            ax.text(0.98, 0.02, f"n*={star_count}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=color)

        ax.set_title(f"run-{run_id} @ {time_ps:.2f} ps", fontsize=8)
        ax.grid(alpha=0.25)

    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.text(0.5, 0.04, "s", ha="center")
    fig.text(0.04, 0.5, "F (kcal mol$^{-1}$)", va="center", rotation="vertical")

    out = args.out
    if out is None:
        out = Path("reports") / f"fes_pairs_panels_{args.cv_dir}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
