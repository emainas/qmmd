#!/usr/bin/env python3
"""Plot FES panels, then filter by star count and make 2x2 summary plot."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
COORD_RE = re.compile(r"Coordinate\s*=\s*([-0-9.]+)")
EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576
SMOOTH_WINDOW = 9
STAR_XMAX = 1.2


def load_pairs(pairs_script: Path) -> List[Tuple[int, float]]:
    text = pairs_script.read_text()
    m = re.search(r"PAIRS[^\n=]*=\s*(\[.*?\])", text, flags=re.S | re.M)
    if not m:
        raise RuntimeError(f"Could not find PAIRS in {pairs_script}")
    raw = m.group(1)
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


def parse_biaspot_times_coords_ps(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    coords: List[float] = []
    cur_time = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                cur_time = float(m.group(1)) / 1000.0
                continue
            m = COORD_RE.search(line)
            if m and cur_time is not None:
                times.append(cur_time)
                coords.append(float(m.group(1)))
                cur_time = None
    return np.array(times, dtype=float), np.array(coords, dtype=float)


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


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def derivative_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if savgol_filter is None:
        raise RuntimeError("scipy is required for Savitzky-Golay smoothing.")
    ys = smooth(y, SMOOTH_WINDOW)
    win = min(9, len(x) // 2 * 2 + 1)
    if win < 5:
        win = 5
    y_sg = savgol_filter(ys, window_length=win, polyorder=3, mode="interp")
    dy = np.gradient(y_sg, x)
    ddy = np.gradient(dy, x)
    dddy = np.gradient(ddy, x)
    return x, dy, ddy, dddy


def red_star_positions(x: np.ndarray, dy: np.ndarray) -> List[float]:
    """Return x positions where dF/ds crosses zero (red stars logic)."""
    xs: List[float] = []
    for zi in range(len(dy) - 1):
        x0, x1 = x[zi], x[zi + 1]
        if x0 > STAR_XMAX and x1 > STAR_XMAX:
            continue
        if x0 > 1.1 and x1 > 1.1:
            continue
        y0, y1 = dy[zi], dy[zi + 1]
        # segment crosses zero
        if not (y0 <= 0.0 <= y1 or y1 <= 0.0 <= y0):
            continue
        if y1 == y0:
            xc = x0
        else:
            xc = x0 - y0 * (x1 - x0) / (y1 - y0)
        xs.append(xc)
    return xs


def count_stars(x: np.ndarray, dy: np.ndarray) -> int:
    signs = np.sign(dy)
    zidx = np.where(np.diff(signs) != 0)[0]
    count = 0
    for zi in zidx:
        x0, x1 = x[zi], x[zi + 1]
        if x0 > STAR_XMAX and x1 > STAR_XMAX:
            continue
        count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="Panels + filtered 2x2 summary based on star count.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--pairs-script", type=Path, default=Path("plotting/plot_fes_pairs.py"))
    p.add_argument("--fes-name", default="fes.dat")
    p.add_argument("--biaspot-name", default="biaspot")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out-panels", type=Path, default=None)
    p.add_argument("--out-summary", type=Path, default=None)
    p.add_argument("--out-diffs", type=Path, default=None)
    p.add_argument("--out-minima", type=Path, default=None)
    p.add_argument("--out-transitions", type=Path, default=None)
    p.add_argument("--out-st", type=Path, default=None)
    p.add_argument("--min1-x", type=float, default=0.0)
    p.add_argument("--min2-x", type=float, default=1.0)
    p.add_argument("--half-window", type=float, default=0.1)
    p.add_argument("--fes-xmin", type=float, default=0.0)
    p.add_argument("--fes-xmax", type=float, default=1.2)
    p.add_argument("--star-count", type=int, default=5)
    p.add_argument("--temp", type=float, default=313.15)
    args = p.parse_args()

    pairs = load_pairs(args.pairs_script)
    if not pairs:
        raise SystemExit("PAIRS is empty in the pairs script")

    if args.style.exists():
        plt.style.use(args.style)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    # Precompute data per pair
    pair_data: Dict[Tuple[int, float], Dict[str, np.ndarray]] = {}
    star_counts: Dict[Tuple[int, float], int] = {}

    for i, (run_id, time_ps) in enumerate(pairs):
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
            min_x, min_y = find_minimum_near_target(
                data[:, 0], data[:, 1], args.min2_x, args.half_window, args.fes_xmin, args.fes_xmax
            )
            y_shift = -min_y
        except Exception:
            y_shift = 0.0

        data_plot = data.copy()
        data_plot[:, 1] = data_plot[:, 1] + y_shift

        # dF/ds and stars
        try:
            x_d, dy, ddy, dddy = derivative_curve(data_plot[:, 0], data_plot[:, 1])
            stars = count_stars(x_d, dy)
        except Exception as e:
            print(f"SKIP: run-{run_id} dF/ds failed: {e}")
            x_d, dy, ddy, dddy, stars = (
                data_plot[:, 0],
                np.zeros_like(data_plot[:, 0]),
                np.zeros_like(data_plot[:, 0]),
                np.zeros_like(data_plot[:, 0]),
                0,
            )

        pair_data[(run_id, time_ps)] = {
            "raw_x": data[:, 0],
            "raw_y": data[:, 1],
            "x": data_plot[:, 0],
            "y": data_plot[:, 1],
            "x_d": x_d,
            "dy": dy,
            "ddy": ddy,
            "dddy": dddy,
        }
        star_counts[(run_id, time_ps)] = stars

    # Panels plot
    rows, cols = grid_shape(len(pair_data))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), dpi=220)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ((run_id, time_ps), data) in enumerate(pair_data.items()):
        ax = axes_flat[i]
        color = color_cycle[i % len(color_cycle)] if color_cycle else None
        ax.plot(data["x"], data["y"], lw=1.6, color="black")
        ax2 = ax.twinx()
        ax2.plot(data["x_d"], data["dy"], lw=1.4, color="red", ls="-")
        ax2.plot(data["x_d"], data["ddy"], lw=1.4, color="blue", ls="-")
        ax2.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        # no tolerance band
        # Star markers at zero crossings (x <= STAR_XMAX)
        x = data["x_d"]
        dy = data["dy"]
        dddy = data["dddy"]
        # dF/ds stars (red) at zero crossings
        n2 = 0
        for zi in range(len(dy) - 1):
            x0, x1 = x[zi], x[zi + 1]
            if x0 > STAR_XMAX and x1 > STAR_XMAX:
                continue
            if x0 > 1.1 and x1 > 1.1:
                continue
            y0, y1 = dy[zi], dy[zi + 1]
            if not (y0 <= 0.0 <= y1 or y1 <= 0.0 <= y0):
                continue
            if y1 == y0:
                xc = x0
            else:
                # intersect with y=0 for marker placement
                xc = x0 - y0 * (x1 - x0) / (y1 - y0)
            ax2.plot(xc, 0.0, marker="*", ms=5, color="red", mec="black", mew=0.3)
        # d2F/ds2 stars (blue)
        for zi in range(len(data["ddy"]) - 1):
            x0, x1 = x[zi], x[zi + 1]
            if x0 > 1.1 and x1 > 1.1:
                continue
            if x0 < 0.05 and x1 < 0.05:
                continue
            y0, y1 = data["ddy"][zi], data["ddy"][zi + 1]
            if not (y0 <= 0.0 <= y1 or y1 <= 0.0 <= y0):
                continue
            if y1 == y0:
                xc = x0
            else:
                xc = x0 - y0 * (x1 - x0) / (y1 - y0)
            if xc > 1.1:
                continue
            ax2.plot(xc, 0.0, marker="*", ms=5, color="blue", mec="black", mew=0.3)
            n2 += 1
        # d3F/ds3 stars (green)
        n3 = 0
        for zi in range(len(dddy) - 1):
            x0, x1 = x[zi], x[zi + 1]
            if x0 > STAR_XMAX and x1 > STAR_XMAX:
                continue
            if x0 < 0.05 and x1 < 0.05:
                continue
            y0, y1 = dddy[zi], dddy[zi + 1]
            if not (y0 <= 0.0 <= y1 or y1 <= 0.0 <= y0):
                continue
            if y1 == y0:
                xc = x0
            else:
                xc = x0 - y0 * (x1 - x0) / (y1 - y0)
            if xc > 1.1:
                continue
            ax2.plot(xc, 0.0, marker="*", ms=5, color="green", mec="black", mew=0.3)
            n3 += 1
        ax.tick_params(axis="both", labelsize=8, colors="black")
        ax2.tick_params(axis="y", labelsize=8, colors="red")
        # right-side ticks for second derivative in green
        # no extra ticks for second derivative
        ax.set_title(f"run-{run_id} @ {time_ps:.2f} ps", fontsize=8)
        ax.set_xlim(args.fes_xmin, args.fes_xmax)
        ax.grid(alpha=0.25)
        # annotate first/third/fifth red-star positions if available
        red_stars = red_star_positions(x, dy)
        if len(red_stars) >= 5:
            s1, s2, s3 = red_stars[0], red_stars[2], red_stars[4]
            ax.text(0.98, 0.95, f"n*df(s={s1:.2f},s={s2:.2f},s={s3:.2f})={len(red_stars)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="red", fontweight="bold")
        else:
            ax.text(0.98, 0.95, f"n*df={star_counts[(run_id, time_ps)]}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="red", fontweight="bold")
        ax.text(0.98, 0.91, f"n*d2={n2}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="blue", fontweight="bold")
        ax.text(0.98, 0.87, f"n*d3={n3}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="green", fontweight="bold")

    for j in range(len(pair_data), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.text(0.5, 0.04, "s", ha="center")
    fig.text(0.04, 0.5, "F (kcal mol$^{-1}$)", va="center", rotation="vertical")

    out_panels = args.out_panels
    if out_panels is None:
        out_panels = Path("reports") / f"fes_pairs_panels_{args.cv_dir}.png"
    out_panels.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
    fig.savefig(out_panels)
    plt.close(fig)
    print(f"Wrote {out_panels}")

    # Extra plot: ΔF(0-0.5) and ΔF(1-0.5) time evolution up to selected time for each pair
    diff_rows, diff_cols = grid_shape(len(pair_data))
    fig = plt.figure(figsize=(3.2 * diff_cols, 2.8 * diff_rows), dpi=220)
    axes = fig.subplots(diff_rows, diff_cols, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (run_id, time_ps) in enumerate(pair_data.keys()):
        ax = axes_flat[i]
        run_dir = args.runs_path / f"run-{run_id}" / args.cv_dir
        biaspot = run_dir / args.biaspot_name
        fes = run_dir / args.fes_name
        if not biaspot.exists() or not fes.exists():
            ax.set_axis_off()
            continue

        times, coords = parse_biaspot_times_coords_ps(biaspot)
        if times.size == 0:
            ax.set_axis_off()
            continue
        blocks = read_fes_blocks(fes)
        nblocks = min(len(blocks), len(times))
        times = times[:nblocks]
        blocks = blocks[:nblocks]

        idx = int(np.argmin(np.abs(times - time_ps)))
        times = times[: idx + 1]
        coords = coords[: idx + 1]
        blocks = blocks[: idx + 1]

        dfr = []
        dfb = []
        dfg = []
        s0_series = []
        smid_series = []
        s1_series = []
        last_s0 = None
        last_smid = None
        last_s1 = None
        max_jump = 0.15
        for b in blocks:
            b = b.copy()
            b[:, 1] *= EH_TO_KCALMOL
            try:
                # use red-star positions from dF/ds for minima proxies
                x_d, dy, _, _ = derivative_curve(b[:, 0], b[:, 1])
                stars = red_star_positions(x_d, dy)
                # track by position: ~0, ~0.4±0.1, ~1
                # pick candidates by window, then enforce continuity vs last values
                s0_cands = stars
                smid_cands = [s for s in stars if 0.3 <= s <= 0.5]
                s1_cands = stars

                def pick_continuous(cands, target, last):
                    if not cands:
                        return None
                    if last is None:
                        return min(cands, key=lambda s: abs(s - target))
                    best = min(cands, key=lambda s: abs(s - last))
                    if abs(best - last) > max_jump:
                        return None
                    return best

                s0 = pick_continuous(s0_cands, 0.0, last_s0)
                s_mid = pick_continuous(smid_cands, 0.4, last_smid)
                s1 = pick_continuous(s1_cands, 1.0, last_s1)

                if s0 is None or s_mid is None or s1 is None:
                    # carry forward last known minima positions if available
                    if last_s0 is None or last_smid is None or last_s1 is None:
                        dfr.append(np.nan)
                        dfb.append(np.nan)
                        dfg.append(np.nan)
                        s0_series.append(np.nan)
                        smid_series.append(np.nan)
                        s1_series.append(np.nan)
                        continue
                    s0, s_mid, s1 = last_s0, last_smid, last_s1
                f0 = float(np.interp(s0, b[:, 0], b[:, 1]))
                fmid = float(np.interp(s_mid, b[:, 0], b[:, 1]))
                f1 = float(np.interp(s1, b[:, 0], b[:, 1]))
                dfr.append(f0 - fmid)
                dfb.append(f1 - fmid)
                dfg.append(f0 - f1)
                s0_series.append(s0)
                smid_series.append(s_mid)
                s1_series.append(s1)
                last_s0 = s0
                last_smid = s_mid
                last_s1 = s1
            except Exception:
                dfr.append(np.nan)
                dfb.append(np.nan)
                dfg.append(np.nan)
                s0_series.append(np.nan)
                smid_series.append(np.nan)
                s1_series.append(np.nan)

        # legend labels with last snapshot s positions
        ax.scatter(times, dfr, color="red", s=8,
                   label=f"ΔF(s={last_s0:.2f} - s={last_smid:.2f})")
        ax.scatter(times, dfb, color="blue", s=8,
                   label=f"ΔF(s={last_s1:.2f} - s={last_smid:.2f})")
        ax.scatter(times, dfg, color="green", s=8,
                   label=f"ΔF(s={last_s0:.2f} - s={last_s1:.2f})")
        ax.set_title(f"run-{run_id} @ {time_ps:.2f} ps", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6)

        # store minima time series for later plot
        pair_data[(run_id, time_ps)]["times"] = times
        pair_data[(run_id, time_ps)]["coords"] = coords
        pair_data[(run_id, time_ps)]["s0_series"] = np.array(s0_series, dtype=float)
        pair_data[(run_id, time_ps)]["smid_series"] = np.array(smid_series, dtype=float)
        pair_data[(run_id, time_ps)]["s1_series"] = np.array(s1_series, dtype=float)

    for j in range(len(pair_data), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.text(0.5, 0.04, "time (ps)", ha="center")
    fig.text(0.04, 0.5, "ΔF (kcal/mol)", va="center", rotation="vertical")

    out_diffs = args.out_diffs
    if out_diffs is None:
        out_diffs = Path("reports") / f"fes_pairs_dgdiffs_{args.cv_dir}.png"
    out_diffs.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
    fig.savefig(out_diffs)
    plt.close(fig)
    print(f"Wrote {out_diffs}")

    # Extra plot: time evolution of minima positions
    fig = plt.figure(figsize=(3.2 * diff_cols, 2.8 * diff_rows), dpi=220)
    axes = fig.subplots(diff_rows, diff_cols, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (run_id, time_ps) in enumerate(pair_data.keys()):
        ax = axes_flat[i]
        data = pair_data[(run_id, time_ps)]
        if "times" not in data:
            ax.set_axis_off()
            continue
        t = data["times"]
        ax.scatter(t, data["s0_series"], color="red", s=8, label="s≈0")
        ax.scatter(t, data["smid_series"], color="black", s=8, label="s≈0.4")
        ax.scatter(t, data["s1_series"], color="blue", s=8, label="s≈1")
        ax.set_title(f"run-{run_id} @ {time_ps:.2f} ps", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6)

    for j in range(len(pair_data), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.text(0.5, 0.04, "time (ps)", ha="center")
    fig.text(0.04, 0.5, "s (minima)", va="center", rotation="vertical")

    out_min = args.out_minima
    if out_min is None:
        out_min = Path("reports") / f"fes_pairs_minima_{args.cv_dir}.png"
    out_min.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
    fig.savefig(out_min)
    plt.close(fig)
    print(f"Wrote {out_min}")

    # Extra plot: s(t) time evolution from biaspot for each pair
    fig = plt.figure(figsize=(3.2 * diff_cols, 2.8 * diff_rows), dpi=220)
    axes = fig.subplots(diff_rows, diff_cols, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (run_id, time_ps) in enumerate(pair_data.keys()):
        ax = axes_flat[i]
        data = pair_data[(run_id, time_ps)]
        if "times" not in data or "coords" not in data:
            ax.set_axis_off()
            continue
        t = data["times"]
        s = data["coords"]
        if len(t) == 0:
            ax.set_axis_off()
            continue
        # trim to the selected timestamp
        idx = int(np.argmin(np.abs(t - time_ps)))
        t = t[: idx + 1]
        s = s[: idx + 1]
        ax.plot(t, s, color="black", lw=1.0)
        ax.set_title(f"run-{run_id} @ {time_ps:.2f} ps", fontsize=8)
        ax.grid(alpha=0.25)

    for j in range(len(pair_data), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.text(0.5, 0.04, "time (ps)", ha="center")
    fig.text(0.04, 0.5, "s(t)", va="center", rotation="vertical")

    out_st = args.out_st
    if out_st is None:
        out_st = Path("reports") / f"fes_pairs_st_{args.cv_dir}.png"
    out_st.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
    fig.savefig(out_st)
    plt.close(fig)
    print(f"Wrote {out_st}")

    # Extra plot: transition counts over time (intermediate<->0 and intermediate<->1)
    fig = plt.figure(figsize=(3.2 * diff_cols, 2.8 * diff_rows), dpi=220)
    axes = fig.subplots(diff_rows, diff_cols, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (run_id, time_ps) in enumerate(pair_data.keys()):
        ax = axes_flat[i]
        data = pair_data[(run_id, time_ps)]
        if "times" not in data:
            ax.set_axis_off()
            continue
        t = data["times"]
        s0 = data["s0_series"].copy()
        sm = data["smid_series"].copy()
        s1 = data["s1_series"].copy()
        cv = data["coords"]

        # forward-fill minima positions to avoid missing transitions
        def ffill(arr: np.ndarray) -> np.ndarray:
            out = arr.copy()
            last = np.nan
            for i, v in enumerate(out):
                if np.isnan(v):
                    out[i] = last
                else:
                    last = v
            return out

        s0 = ffill(s0)
        sm = ffill(sm)
        s1 = ffill(s1)

        # assign state based on closest minimum to the actual s(t) coordinate
        states = []
        for a, b, c, s in zip(s0, sm, s1, cv):
            if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(s):
                states.append(None)
                continue
            d0 = abs(s - a)
            d1 = abs(s - b)
            d2 = abs(s - c)
            st = int(np.argmin([d0, d1, d2]))  # 0,1,2
            states.append(st)

        # track transitions between 1<->0 and 1<->2
        trans_10 = []
        trans_12 = []
        c10 = 0
        c12 = 0
        prev = None
        for st in states:
            if st is None:
                trans_10.append(c10)
                trans_12.append(c12)
                continue
            if prev is not None and st != prev:
                if (prev == 1 and st == 0) or (prev == 0 and st == 1):
                    c10 += 1
                if (prev == 1 and st == 2) or (prev == 2 and st == 1):
                    c12 += 1
            prev = st
            trans_10.append(c10)
            trans_12.append(c12)

        ax.plot(t, trans_10, color="red", lw=1.2, label="int↔0")
        ax.plot(t, trans_12, color="blue", lw=1.2, label="int↔1")
        ax.set_title(f"run-{run_id} @ {time_ps:.2f} ps", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6)

    for j in range(len(pair_data), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.text(0.5, 0.04, "time (ps)", ha="center")
    fig.text(0.04, 0.5, "transition count", va="center", rotation="vertical")

    out_tr = args.out_transitions
    if out_tr is None:
        out_tr = Path("reports") / f"fes_pairs_transitions_{args.cv_dir}.png"
    out_tr.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
    fig.savefig(out_tr)
    plt.close(fig)
    print(f"Wrote {out_tr}")

    # Filter by star counts (df and d2)
    filtered = []
    for p in pair_data.keys():
        if star_counts.get(p, 0) != args.star_count:
            continue
        # recompute blue-star count for this pair (zero crossings)
        data = pair_data[p]
        x = data["x_d"]
        ddy = data["ddy"]
        n2 = 0
        for zi in range(len(ddy) - 1):
            x0, x1 = x[zi], x[zi + 1]
            if x0 > 1.1 and x1 > 1.1:
                continue
            if x0 < 0.05 and x1 < 0.05:
                continue
            y0, y1 = ddy[zi], ddy[zi + 1]
            if not (y0 <= 0.0 <= y1 or y1 <= 0.0 <= y0):
                continue
            if y1 == y0:
                xc = x0
            else:
                xc = x0 - y0 * (x1 - x0) / (y1 - y0)
            if xc > 1.1:
                continue
            n2 += 1
        if n2 == 4:
            filtered.append(p)
    if not filtered:
        print(f"No pairs with n*={args.star_count}; skipping summary plot")
        return

    # 2x2 summary plot
    fig = plt.figure(figsize=(10.5, 8.0), dpi=220)
    gs = plt.GridSpec(2, 2, width_ratios=[3, 2], wspace=0.3, hspace=0.3)
    ax_fes = fig.add_subplot(gs[0, 0])
    ax_df = fig.add_subplot(gs[0, 1])
    ax_dfds = fig.add_subplot(gs[1, 0])
    ax_pka = fig.add_subplot(gs[1, 1])

    df_vals = []
    pka_vals = []
    labels = []
    colors = []

    for i, key in enumerate(filtered):
        run_id, time_ps = key
        data = pair_data[key]
        color = color_cycle[i % len(color_cycle)] if color_cycle else None
        ax_fes.plot(data["x"], data["y"], lw=0.8, color=color)
        ax_dfds.plot(data["x_d"], data["dy"], lw=0.8, color="red", alpha=0.8, ls="-")
        ax_dfds.plot(data["x_d"], data["ddy"], lw=1.2, color="blue", alpha=0.7, ls="-")

        # deltaF and pKa (same definition as plot_fes_pairs.py)
        raw = data["y"]
        # Use stored FES arrays (x, y) with the same deltaf definition as plot_fes_pairs.py
        df = deltaf(np.column_stack([data["x"], data["y"]]), args.min1_x, args.min2_x,
                    args.half_window, args.fes_xmin, args.fes_xmax)
        df_vals.append(df)
        pka_vals.append(df / (PKA_FACTOR * args.temp))
        labels.append(f"run-{run_id}")
        colors.append(color or "#4c72b0")

    ax_fes.set_xlabel("s")
    ax_fes.set_ylabel("F (kcal mol$^{-1}$)")
    ax_fes.set_xlim(args.fes_xmin, args.fes_xmax)
    ax_fes.grid(alpha=0.25)

    if df_vals:
        x = np.arange(len(df_vals))
        ax_df.bar(x, df_vals, color=colors, alpha=0.85, edgecolor="white")
        ax_df.set_title("ΔF (kcal/mol)", fontsize=10)
        ax_df.set_ylabel("ΔF")
        ax_df.set_xticks(x)
        ax_df.set_xticklabels(labels, rotation=90, ha="center", fontsize=5)
        ax_df.grid(alpha=0.25, axis="y")

    ax_dfds.set_title("dF/ds (smoothed)", fontsize=10)
    ax_dfds.set_xlabel("s")
    ax_dfds.set_ylabel("dF/ds")
    ax_dfds.set_xlim(args.fes_xmin, args.fes_xmax)
    ax_dfds.grid(alpha=0.25)

    if pka_vals:
        x = np.arange(len(pka_vals))
        ax_pka.bar(x, pka_vals, color=colors, alpha=0.85, edgecolor="white")
        mean = float(np.mean(pka_vals))
        std = float(np.std(pka_vals, ddof=1)) if len(pka_vals) > 1 else 0.0
        ax_pka.set_title("pKa", fontsize=10)
        ax_pka.set_ylabel("pKa")
        ax_pka.set_xticks(x)
        ax_pka.set_xticklabels(labels, rotation=90, ha="center", fontsize=5)
        ax_pka.grid(alpha=0.25, axis="y")
        ax_pka.plot([], [], label=f"mean={mean:.2f}", color="none")
        ax_pka.plot([], [], label=f"±std={std:.2f}", color="none")
        ax_pka.legend(loc="upper right", frameon=False, fontsize=9)

    out_summary = args.out_summary
    if out_summary is None:
        out_summary = Path("reports") / f"fes_pairs_summary_{args.cv_dir}.png"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_summary)
    plt.close(fig)
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
