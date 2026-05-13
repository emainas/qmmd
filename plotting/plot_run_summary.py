#!/usr/bin/env python3
"""Plot CV time series, FES snapshots, and pKa time series for a single run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
COORD_RE = re.compile(r"Coordinate\s*=\s*([+-]?[0-9.]+)")
EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576

T_INTERVAL = 1.0


def parse_biaspot(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    coords: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)  # fsec -> ps
                continue
            m = COORD_RE.search(line)
            if m:
                coords.append(float(m.group(1)))
    n = min(len(times), len(coords))
    if n == 0:
        return np.array([]), np.array([])
    t = np.array(times[:n])
    return t, np.array(coords[:n])


def parse_biaspot_times(path: Path) -> np.ndarray:
    times: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)
    return np.array(times, dtype=float)


def load_fes_blocks(path: Path) -> List[np.ndarray]:
    blocks: List[np.ndarray] = []
    current: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("###"):
                if current:
                    arr = np.array(current, dtype=float)
                    arr[:, 1] *= EH_TO_KCALMOL
                    blocks.append(arr)
                current = []
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                current.append([float(parts[0]), float(parts[1])])
            except Exception:
                continue
    if current:
        arr = np.array(current, dtype=float)
        arr[:, 1] *= EH_TO_KCALMOL
        blocks.append(arr)
    return blocks


def load_biaspot_with_restart(run_dir: Path, cv_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    base_biaspot = run_dir / cv_dir / "biaspot"
    t0, y0 = parse_biaspot(base_biaspot)
    if t0.size == 0:
        return t0, y0

    restart_dir = run_dir / cv_dir / "metad-restart"
    restart_biaspot = restart_dir / "biaspot"
    if not restart_biaspot.exists():
        return t0, y0

    t1, y1 = parse_biaspot(restart_biaspot)
    if t1.size == 0:
        return t0, y0

    last_base = float(t0[-1])
    if float(t1[-1]) > last_base:
        keep = t1 > last_base
        t1 = t1[keep]
        y1 = y1[keep]
        return np.concatenate([t0, t1]), np.concatenate([y0, y1])

    t1 = t1 + last_base
    return np.concatenate([t0, t1]), np.concatenate([y0, y1])


def load_fes_with_restart(run_dir: Path, cv_dir: str) -> List[np.ndarray]:
    base_fes = run_dir / cv_dir / "fes.dat"
    blocks = load_fes_blocks(base_fes)

    restart_dir = run_dir / cv_dir / "metad-restart"
    restart_fes = restart_dir / "fes.dat"
    if not restart_fes.exists():
        return blocks

    restart_blocks = load_fes_blocks(restart_fes)
    if len(restart_blocks) >= 2:
        restart_blocks = restart_blocks[1:]
    elif restart_blocks:
        restart_blocks = []
    return blocks + restart_blocks


def load_biaspot_times_with_restart(run_dir: Path, cv_dir: str) -> np.ndarray:
    base_biaspot = run_dir / cv_dir / "biaspot"
    t0 = parse_biaspot_times(base_biaspot)
    if t0.size == 0:
        return t0

    restart_dir = run_dir / cv_dir / "metad-restart"
    restart_biaspot = restart_dir / "biaspot"
    if not restart_biaspot.exists():
        return t0

    t1 = parse_biaspot_times(restart_biaspot)
    if t1.size == 0:
        return t0

    last_base = float(t0[-1])
    if float(t1[-1]) > last_base:
        keep = t1 > last_base
        t1 = t1[keep]
        return np.concatenate([t0, t1])

    t1 = t1 + last_base
    return np.concatenate([t0, t1])


def select_indices_aligned(times: np.ndarray, t_fill: float, t_interval: float, t_max: float | None) -> List[int]:
    if times.size == 0 or t_interval <= 0:
        return []
    start = np.ceil(t_fill / t_interval) * t_interval
    target = start
    out: List[int] = []
    i = 0
    n = len(times)
    t_end = float(times[-1])
    if t_max is not None:
        t_end = min(t_end, t_max)
    while i < n and target <= t_end:
        while i < n and times[i] < target:
            i += 1
        if i >= n:
            break
        out.append(i)
        target += t_interval
    return out




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
    p = argparse.ArgumentParser(description="Plot CV, FES snapshots, and pKa for one run.")
    p.add_argument("--runs-path", required=True, type=str,
                   help="Path(s) containing run-* directories (comma-separated for overlay)")
    p.add_argument("--run-id", required=True, type=str,
                   help="Run id(s), comma-separated for overlay (e.g., 67,99)")
    p.add_argument("--cv-dir", required=True, type=str,
                   help="CV dir(s), comma-separated for overlay (e.g., all-meta-hid,all-meta-hie)")
    p.add_argument("--style", type=Path, default=Path("plotting/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--temp", type=float, default=313.15)
    p.add_argument("--t-fill", type=str, default="15.0", help="t_fill(s) in ps, comma-separated")
    p.add_argument("--t-max", type=str, default="", help="t_max(s) in ps, comma-separated (blank = no max)")
    p.add_argument("--min1-x", type=float, default=0.0)
    p.add_argument("--min2-x", type=float, default=1.0)
    p.add_argument("--half-window", type=float, default=0.1)
    p.add_argument("--fes-xmin", type=float, default=0.0)
    p.add_argument("--fes-xmax", type=float, default=1.25)
    p.add_argument("--exp-pka", type=float, default=None)
    args = p.parse_args()

    if args.style.exists():
        plt.style.use(args.style)

    def _split_list(raw: str) -> List[str]:
        return [x.strip() for x in raw.split(",") if x.strip()]

    run_ids = [int(x) for x in _split_list(args.run_id)]
    runs_paths = [Path(x) for x in _split_list(args.runs_path)]
    cv_dirs = _split_list(args.cv_dir)
    t_fills = [float(x) for x in _split_list(args.t_fill)]
    t_max_raw = _split_list(args.t_max)
    t_maxs = [float(x) for x in t_max_raw] if t_max_raw else []

    n = len(run_ids)
    if n not in (1, 2):
        raise SystemExit("Provide 1 or 2 run IDs")

    def _expand(lst: List, name: str) -> List:
        if len(lst) == 1:
            return lst * n
        if len(lst) == n:
            return lst
        raise SystemExit(f"{name} must have 1 or {n} values")

    runs_paths = _expand(runs_paths, "runs-path")
    cv_dirs = _expand(cv_dirs, "cv-dir")
    t_fills = _expand(t_fills, "t-fill")
    t_maxs = _expand(t_maxs, "t-max") if t_maxs else [None] * n

    series_data = []
    for run_id, runs_path, cv_dir, t_fill, t_max in zip(run_ids, runs_paths, cv_dirs, t_fills, t_maxs):
        run_dir = runs_path / f"run-{run_id}"

        t_cv, y_cv = load_biaspot_with_restart(run_dir, cv_dir)

        times_fes = load_biaspot_times_with_restart(run_dir, cv_dir)
        fes_blocks = load_fes_with_restart(run_dir, cv_dir)
        nblocks = min(len(times_fes), len(fes_blocks))
        times_fes = times_fes[:nblocks]
        fes_blocks = fes_blocks[:nblocks]
        idxs = select_indices_aligned(times_fes, t_fill, T_INTERVAL, t_max)

        df_vals = np.array(
            [
                deltaf(b, args.min1_x, args.min2_x, args.half_window, args.fes_xmin, args.fes_xmax)
                for b in fes_blocks
            ],
            dtype=float,
        )
        pka_vals = df_vals / (PKA_FACTOR * args.temp)

        series_data.append((run_id, cv_dir, t_fill, t_max, t_cv, y_cv, times_fes, fes_blocks, idxs, pka_vals))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), dpi=220, sharex=False, sharey=False)
    ax_cv = axes[0, 0]
    ax_fes_a = axes[0, 1]
    ax_pka = axes[1, 0]
    ax_fes_b = axes[1, 1]

    # Panel tags for paper
    ax_cv.text(0.02, 0.98, "A)", transform=ax_cv.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_fes_a.text(0.02, 0.98, "B)", transform=ax_fes_a.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_pka.text(0.02, 0.98, "C)", transform=ax_pka.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_fes_b.text(0.02, 0.98, "D)", transform=ax_fes_b.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

    colors = ["#1f77b4", "#d62728"]

    # CV plot
    for i, (run_id, cv_dir, t_fill, t_max, t_cv, y_cv, times_fes, fes_blocks, idxs, pka_vals) in enumerate(series_data):
        if t_cv.size == 0:
            continue
        t_rel = t_cv
        pre_mask = t_cv < t_fill
        mid_mask = (t_cv >= t_fill) & ((t_cv <= t_max) if t_max is not None else True)
        post_mask = t_cv > t_max if t_max is not None else np.zeros_like(t_cv, dtype=bool)
        if np.any(pre_mask):
            ax_cv.plot(t_rel[pre_mask], y_cv[pre_mask], color="#ff7f0e", lw=1.2, alpha=0.9)
        if np.any(mid_mask):
            ax_cv.plot(t_rel[mid_mask], y_cv[mid_mask], color="#2ca02c", lw=1.3, alpha=0.9)
        ax_cv.plot([], [], color=colors[i], lw=1.2, label=f"run-{run_id} ({cv_dir})")
    ax_cv.set_xlabel("t (ps)")
    ax_cv.set_ylabel("CV")
    ax_cv.set_title("CV time series")
    ax_cv.grid(alpha=0.25)
    # Legend removed to avoid overlap with inset histogram
    t_start_vals = [float(t_cv[0]) for _, _, _, _, t_cv, *_ in series_data if t_cv.size > 0]
    if t_start_vals:
        tmin_val = min(t_start_vals)
        if any(t_max is not None for _, _, _, t_max, *_ in series_data):
            tmax_val = max(t_max for _, _, _, t_max, *_ in series_data if t_max is not None)
            ax_cv.set_xlim(tmin_val, tmax_val)
        else:
            ax_cv.set_xlim(tmin_val, max(float(t_cv[-1]) for _, _, _, _, t_cv, *_ in series_data if t_cv.size > 0))

    # Inset histogram of CV values (t_fill to t_max)
    inset_cv = inset_axes(ax_cv, width="35%", height="35%", loc="upper right", borderpad=1.0)
    filtered_cv = []
    for _, _, t_fill, t_max, t_cv, y_cv, *_ in series_data:
        if y_cv.size == 0:
            continue
        keep = (t_cv >= t_fill) & ((t_cv <= t_max) if t_max is not None else True)
        if np.any(keep):
            filtered_cv.append(y_cv[keep])
    all_cv = np.concatenate(filtered_cv) if filtered_cv else np.array([])
    if all_cv.size > 0:
        inset_cv.hist(all_cv, bins=100, color="#666666", alpha=0.35, edgecolor="black", linewidth=0.4)
    inset_cv.set_xlabel("CV", fontsize=7)
    inset_cv.set_ylabel("count", fontsize=7)
    inset_cv.tick_params(labelsize=6)

    # FES snapshots (separate panels)
    cmap = plt.get_cmap("viridis")
    if len(series_data) >= 1:
        run_id, cv_dir, t_fill, t_max, _, _, times_fes, fes_blocks, idxs, _ = series_data[0]
        for j, idx in enumerate(idxs):
            b = fes_blocks[idx]
            x = b[:, 0]
            y = b[:, 1]
            frac = j / (len(idxs) - 1) if len(idxs) > 1 else 1.0
            ax_fes_a.plot(x, y, color=cmap(frac), lw=1.0, alpha=0.85, label=f"{times_fes[idx]:.2f} ps")
        ax_fes_a.set_xlim(args.fes_xmin, args.fes_xmax)
        ax_fes_a.set_xlabel("CV")
        ax_fes_a.set_ylabel("F (kcal/mol)")
        title = f"FES run-{run_id} (t_fill={t_fill} ps"
        if t_max is not None:
            title += f", t_max={t_max} ps"
        title += ")"
        ax_fes_a.set_title(title)
        ax_fes_a.grid(alpha=0.25)
        if idxs:
            ax_fes_a.legend(loc="lower right", fontsize=7, frameon=False)
    else:
        ax_fes_a.set_axis_off()

    if len(series_data) >= 2:
        run_id, cv_dir, t_fill, t_max, _, _, times_fes, fes_blocks, idxs, _ = series_data[1]
        for j, idx in enumerate(idxs):
            b = fes_blocks[idx]
            x = b[:, 0]
            y = b[:, 1]
            frac = j / (len(idxs) - 1) if len(idxs) > 1 else 1.0
            ax_fes_b.plot(x, y, color=cmap(frac), lw=1.0, alpha=0.85, label=f"{times_fes[idx]:.2f} ps")
        ax_fes_b.set_xlim(args.fes_xmin, args.fes_xmax)
        ax_fes_b.set_xlabel("CV")
        ax_fes_b.set_ylabel("F (kcal/mol)")
        title = f"FES run-{run_id} (t_fill={t_fill} ps"
        if t_max is not None:
            title += f", t_max={t_max} ps"
        title += ")"
        ax_fes_b.set_title(title)
        ax_fes_b.grid(alpha=0.25)
        if idxs:
            ax_fes_b.legend(loc="lower right", fontsize=7, frameon=False)
    else:
        # Single-run mode: use 4th panel for mean-centered FES stats
        run_id, cv_dir, t_fill, t_max, _, _, times_fes, fes_blocks, idxs, _ = series_data[0]
        if not idxs:
            ax_fes_b.set_axis_off()
        else:
            x_ref = fes_blocks[idxs[0]][:, 0]
            curves = []
            for j, idx in enumerate(idxs):
                b = fes_blocks[idx]
                x = b[:, 0]
                y = b[:, 1]
                y = y - float(np.mean(y))
                if len(x) != len(x_ref) or not np.allclose(x, x_ref, rtol=1e-6, atol=1e-8):
                    y = np.interp(x_ref, x, y)
                curves.append(y)
                color = cmap(j / (len(idxs) - 1) if len(idxs) > 1 else 1.0)
                ax_fes_b.plot(x_ref, y, color=color, lw=0.9, alpha=0.25)
            arr = np.vstack(curves)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            ax_fes_b.errorbar(
                x_ref,
                mean,
                yerr=std,
                color="black",
                ecolor="black",
                elinewidth=0.8,
                lw=2.0,
                capsize=0,
                label="mean ±1σ",
            )
            ax_fes_b.set_xlim(args.fes_xmin, args.fes_xmax)
            ax_fes_b.set_xlabel("CV")
            ax_fes_b.set_ylabel("F - mean(F)")
            ax_fes_b.set_title(f"Mean-centered FES (run-{run_id})")
            ax_fes_b.grid(alpha=0.25)
            ax_fes_b.legend(loc="lower right", fontsize=7, frameon=False)

    # pKa plot
    for i, (run_id, cv_dir, t_fill, t_max, t_cv, y_cv, times_fes, fes_blocks, idxs, pka_vals) in enumerate(series_data):
        if times_fes.size == 0:
            continue
        t_rel = times_fes
        pre_mask = times_fes < t_fill
        mid_mask = (times_fes >= t_fill) & ((times_fes <= t_max) if t_max is not None else True)
        post_mask = times_fes > t_max if t_max is not None else np.zeros_like(times_fes, dtype=bool)
        if np.any(pre_mask):
            ax_pka.plot(t_rel[pre_mask], pka_vals[pre_mask], color="#ff7f0e", lw=1.0, alpha=0.9)
            ax_pka.scatter(t_rel[pre_mask], pka_vals[pre_mask], color="#ff7f0e", edgecolor=(0, 0, 0, 0.15), s=12, alpha=0.7)
        if np.any(mid_mask):
            ax_pka.plot(t_rel[mid_mask], pka_vals[mid_mask], color="#2ca02c", lw=1.2, alpha=0.9)
            ax_pka.scatter(t_rel[mid_mask], pka_vals[mid_mask], color="#2ca02c", edgecolor=(0, 0, 0, 0.2), s=14, alpha=0.7)
        ax_pka.plot([], [], color=colors[i], lw=1.0, label=f"run-{run_id} ({cv_dir})")
    if args.exp_pka is not None:
        ax_pka.axhline(args.exp_pka, color="#0000FF", lw=1.4, ls="--")
    ax_pka.set_xlabel("t (ps)")
    ax_pka.set_ylabel("pKa")
    ax_pka.set_title("pKa time series")
    ax_pka.grid(alpha=0.25)
    # Legend removed to avoid overlap with inset histogram
    t_start_vals = [float(times_fes[0]) for *_, times_fes, _, _, _ in series_data if times_fes.size > 0]
    if t_start_vals:
        tmin_val = min(t_start_vals)
        if any(t_max is not None for _, _, _, t_max, *_ in series_data):
            tmax_val = max(t_max for _, _, _, t_max, *_ in series_data if t_max is not None)
            ax_pka.set_xlim(tmin_val, tmax_val)
        else:
            ax_pka.set_xlim(tmin_val, max(float(times_fes[-1]) for *_, times_fes, _, _, _ in series_data if times_fes.size > 0))

    # Inset histogram of pKa values
    inset = inset_axes(ax_pka, width="35%", height="35%", loc="upper right", borderpad=1.0)
    filtered = []
    for _, _, t_fill, t_max, _, _, times_fes, _, _, pka_vals in series_data:
        if pka_vals.size == 0 or times_fes.size == 0:
            continue
        keep = (times_fes >= t_fill) & ((times_fes <= t_max) if t_max is not None else True)
        if np.any(keep):
            filtered.append(pka_vals[keep])
    all_pka = np.concatenate(filtered) if filtered else np.array([])
    if all_pka.size > 0:
        counts, bins, _ = inset.hist(all_pka, bins=100, range=(5.0, 8.0), color="#666666", alpha=0.35, edgecolor="black", linewidth=0.4)
        mean_pka = float(np.mean(all_pka))
        std_pka = float(np.std(all_pka))
        inset.axvline(mean_pka, color="red", lw=1.0, ls="--", label="mean")
        inset.axvline(mean_pka - std_pka, color="red", lw=0.8, ls=":", label="±1σ")
        inset.axvline(mean_pka + std_pka, color="red", lw=0.8, ls=":")
        inset.annotate(
            f"{mean_pka:.2f}",
            xy=(mean_pka, max(counts)),
            xytext=(2, -2),
            textcoords="offset points",
            color="red",
            fontsize=7,
            ha="left",
            va="top",
            rotation=90,
        )
    inset.set_xlabel("pKa", fontsize=7)
    inset.set_ylabel("count", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.set_xlim(5.0, 8.0)
    if all_pka.size > 0 or args.exp_pka is not None:
        legend_items = []
        if all_pka.size > 0:
            legend_items.append(f"mean ±σ: {mean_pka:.2f} ± {std_pka:.2f}")
        if args.exp_pka is not None:
            legend_items.append(f"exp pKa: {args.exp_pka:.2f}")
        ax_pka.text(
            0.98,
            0.02,
            "\n".join(legend_items),
            transform=ax_pka.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, linewidth=0.5),
        )
    if all_pka.size > 0:
        inset.legend(title=f"{mean_pka:.2f} ± {std_pka:.2f}", fontsize=6, title_fontsize=6, frameon=False, loc="upper left")

    out = args.out
    if out is None:
        out = Path("reports") / f"run_{'_'.join(str(r) for r in run_ids)}_{'_'.join(cv_dirs)}_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
