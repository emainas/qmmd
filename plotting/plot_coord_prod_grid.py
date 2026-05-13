#!/usr/bin/env python3
"""Plot coordination time series + histogram + log-prob panels for a single run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, exp, ceil, floor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
import sys

PLOTTING_DIR = Path(__file__).resolve().parent
if str(PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(PLOTTING_DIR))
from defect_identification import parse_mulliken

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
COORD_RE = re.compile(r"Coordinate\s*=\s*([0-9Ee+\\-\\.]+)")
LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+([spdf])\s+"
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?)"
)
EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576
PKA_TEMP = 300.0
PKA_MIN1_X = 0.0
PKA_MIN2_X = 1.0
PKA_HALF_WINDOW = 0.1
PKA_FES_XMIN = 0.0
PKA_FES_XMAX = 1.25


def read_series(path: Path) -> np.ndarray:
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                vals.append(float(parts[-1]))
    return np.array(vals, dtype=float)


def infer_system(runs_path: Path) -> str:
    parts = runs_path.resolve().parts
    if "systems" in parts:
        idx = parts.index("systems")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "system"

def parse_extra_ids(raw: str | None) -> List[int]:
    if not raw:
        return []
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def parse_ids(raw: str | None) -> List[int]:
    return parse_extra_ids(raw)

def parse_fes_times(raw: str | None) -> List[float]:
    if not raw:
        return []
    items = re.split(r"[,\s]+", raw.strip())
    out: List[float] = []
    for item in items:
        if not item:
            continue
        out.append(float(item))
    return out

def compute_time_window(*time_arrays: np.ndarray) -> Tuple[float, float] | None:
    candidates = [arr for arr in time_arrays if arr is not None and arr.size > 0]
    if not candidates:
        return None
    mins = [float(np.min(arr)) for arr in candidates]
    maxs = [float(np.max(arr)) for arr in candidates]
    start = max(mins)
    end = min(maxs)
    if end < start:
        start = min(mins)
        end = max(maxs)
    return float(start), float(end)

def snap_time_window(start: float, end: float, step: float) -> Tuple[float, float]:
    if step <= 0:
        return start, end
    start_adj = floor(start / step) * step
    end_adj = floor(end / step) * step
    if end_adj < start_adj:
        return start, end
    return float(start_adj), float(end_adj)

def apply_tmax(times: np.ndarray, tmax: float | None) -> Tuple[np.ndarray, np.ndarray]:
    if tmax is None or times.size == 0:
        return times, np.arange(times.size, dtype=int)
    mask = times <= tmax
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return times[:0], idx
    return times[mask], idx


def iter_xyz_coords(path: Path):
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError:
                raise ValueError(f"Invalid XYZ atom-count line: {line}")
            _ = f.readline()  # comment
            coords = np.empty((n_atoms, 3), dtype=float)
            for i in range(n_atoms):
                atom_line = f.readline()
                if not atom_line:
                    raise ValueError("Unexpected EOF while reading XYZ frame")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid XYZ atom line: {atom_line.strip()}")
                coords[i, 0] = float(parts[1])
                coords[i, 1] = float(parts[2])
                coords[i, 2] = float(parts[3])
            yield coords


def read_box_lengths_from_dftb_inp(path: Path) -> np.ndarray:
    tv = []
    for line in path.read_text().splitlines():
        if line.startswith("TV"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    tv.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
    if len(tv) != 3:
        raise RuntimeError(f"Expected 3 TV lines in {path}, got {len(tv)}")
    # assume orthorhombic box with TV on diagonal
    return np.array([tv[0][0], tv[1][1], tv[2][2]], dtype=float)

def parse_biaspot_times_ps(path: Path, tmax: float | None = None) -> np.ndarray:
    times: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                t = float(m.group(1)) / 1000.0
                if tmax is not None and t > tmax:
                    break
                times.append(t)
    return np.array(times, dtype=float)

def parse_biaspot_coord_series(path: Path, tmax: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    coords: List[float] = []
    pending_time = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                pending_time = float(m.group(1)) / 1000.0
                if tmax is not None and pending_time > tmax:
                    break
                continue
            if pending_time is None:
                continue
            m = COORD_RE.search(line)
            if m:
                try:
                    coord = float(m.group(1))
                except ValueError:
                    pending_time = None
                    continue
                times.append(pending_time)
                coords.append(coord)
                pending_time = None
    return np.array(times, dtype=float), np.array(coords, dtype=float)

def _append_frame(target_ids: List[int], charges: List[List[float]], orb_sums: dict[int, float]) -> None:
    for idx, atom_id in enumerate(target_ids):
        charges[idx].append(orb_sums.get(atom_id, float("nan")))

def _infer_oxygen_ids(elements: dict[int, str], solute_atoms: int, natoms: int) -> List[int]:
    oxygen_ids: List[int] = []
    start = solute_atoms + 1
    for atom_id in range(start, natoms + 1):
        if elements.get(atom_id) == "O":
            oxygen_ids.append(atom_id)
    if not oxygen_ids:
        raise SystemExit("No oxygen atoms found after solute atoms.")
    return oxygen_ids

def parse_mulliken_limited(
    path: Path,
    solute_atoms: int,
    tmax: float,
    natoms: int | None = None,
    extra_ids: List[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], dict[int, str]]:
    times: List[float] = []
    charges: List[List[float]] = []
    oxygen_ids: List[int] | None = None
    target_ids: List[int] | None = None

    current_time: float | None = None
    current_orb_sums: dict[int, float] = {}
    current_max_id = 0
    current_elements: dict[int, str] = {}
    base_elements: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m_time = TIME_RE.search(line)
            if m_time:
                if current_time is not None:
                    if oxygen_ids is None:
                        if natoms is None:
                            natoms = current_max_id
                        if natoms is None:
                            raise SystemExit("Could not determine total atom count.")
                        oxygen_ids = _infer_oxygen_ids(current_elements, solute_atoms, natoms)
                        extra_ids = extra_ids or []
                        seen = set()
                        target_ids = []
                        for atom_id in oxygen_ids + extra_ids:
                            if atom_id not in seen:
                                seen.add(atom_id)
                                target_ids.append(atom_id)
                        charges = [[] for _ in target_ids]
                        base_elements = dict(current_elements)
                    _append_frame(target_ids or [], charges, current_orb_sums)

                current_time = float(m_time.group(1)) / 1000.0
                if current_time > tmax:
                    break
                times.append(current_time)
                current_orb_sums = {}
                current_elements = {}
                current_max_id = 0
                continue

            m_line = LINE_RE.match(line)
            if not m_line:
                if natoms is None:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        try:
                            natoms = int(parts[0])
                        except ValueError:
                            pass
                continue

            atom_id = int(m_line.group(1))
            elem = m_line.group(2)
            orb = m_line.group(3)
            val = float(m_line.group(4))

            if atom_id > current_max_id:
                current_max_id = atom_id

            if atom_id not in current_elements:
                current_elements[atom_id] = elem

            if orb in ("s", "p"):
                current_orb_sums[atom_id] = current_orb_sums.get(atom_id, 0.0) + val

        if current_time is not None and (tmax is None or current_time <= tmax):
            if oxygen_ids is None:
                if natoms is None:
                    natoms = current_max_id
                if natoms is None:
                    raise SystemExit("Could not determine total atom count.")
                oxygen_ids = _infer_oxygen_ids(current_elements, solute_atoms, natoms)
                extra_ids = extra_ids or []
                seen = set()
                target_ids = []
                for atom_id in oxygen_ids + extra_ids:
                    if atom_id not in seen:
                        seen.add(atom_id)
                        target_ids.append(atom_id)
                charges = [[] for _ in target_ids]
                base_elements = dict(current_elements)
            _append_frame(target_ids or [], charges, current_orb_sums)

    if not times:
        raise SystemExit("No frames found in mulliken file.")

    charges_array = np.array(charges, dtype=float)
    return np.array(times, dtype=float), charges_array, (target_ids or []), base_elements

def read_fes_blocks(path: Path, max_blocks: int | None = None) -> List[np.ndarray]:
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
                    if max_blocks is not None and len(blocks) >= max_blocks:
                        return blocks
                    cur = []
                continue
            if line.startswith("#"):
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

def main() -> None:
    p = argparse.ArgumentParser(description="Plot coordination report for a single run.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--run-dir", required=True, help="Run subdirectory (e.g., equil, prod)")
    p.add_argument("--phase", choices=["equil", "prod"], default="equil", help="Plot mode")
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--solute-atoms", type=int, default=None)
    p.add_argument("--extra-ids", type=str, default=None)
    p.add_argument("--n-ids", type=str, default=None, help="Comma-separated N atom IDs for N-O distance")
    p.add_argument("--traj-name", type=str, default="traject")
    p.add_argument("--fes-name", type=str, default="fes.dat", help="FES file name (prod)")
    p.add_argument("--biaspot-name", type=str, default="biaspot", help="Biaspot file name (prod)")
    p.add_argument("--fes-times", type=str, default=None, help="Comma-separated FES times in ps (prod)")
    p.add_argument("--t-fill", type=float, default=None, help="Start time (ps) for prod histograms")
    p.add_argument("--T-max", type=float, default=None, help="Cap time series at this time (ps)")
    p.add_argument("--exp-pka", type=float, default=None, help="Experimental pKa (prod)")
    p.add_argument("--style", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--html-out", type=Path, default=None)
    args = p.parse_args()

    style = Path(__file__).resolve().parent / "prl.mplstyle"
    if style.exists():
        plt.style.use(style)

    is_prod = args.phase == "prod"
    run_dir = args.runs_path / f"run-{args.run_id}" / args.run_dir / "manual-cv"
    series_all = run_dir / "coord.dat"
    vals = np.array([], dtype=float)
    if series_all.exists():
        vals = read_series(series_all)
        if vals.size == 0:
            raise SystemExit("coord.dat is empty.")
    elif not is_prod:
        raise SystemExit(f"Missing coord.dat: {series_all}")

    dt_ps = 0.01
    t = np.arange(vals.size, dtype=float) * dt_ps  # ps
    if vals.size == 0:
        mu = 0.0
        sd = 0.0
    else:
        mu = float(np.mean(vals))
        sd = float(np.std(vals))

    fig, axes = plt.subplots(2, 4, figsize=(17, 7), dpi=220)
    fes_times = parse_fes_times(args.fes_times) if is_prod else []
    fes_cmap = plt.get_cmap("tab10")
    fes_colors = [fes_cmap(i % 10) for i, _ in enumerate(fes_times)]
    time_window = None
    t_bias_abs = None
    t_pka_abs = None
    t0_ts = 0.0
    t0_pka = 0.0
    ax_mull = None
    ax_neg = None
    ax_dist = None

    # Panel labels
    panel_labels = list("abcdefgh")
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            0.02,
            0.98,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    # Top-left: time series
    ax_ts = axes[0, 0]
    ts_color = "black"
    if is_prod:
        biaspot_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / args.biaspot_name
        if biaspot_path.exists():
            t_bias, vals_bias = parse_biaspot_coord_series(biaspot_path, args.T_max)
            if t_bias.size > 0 and vals_bias.size > 0:
                if args.T_max is not None:
                    t_bias, idx_keep = apply_tmax(t_bias, args.T_max)
                    vals_bias = vals_bias[idx_keep] if idx_keep.size > 0 else vals_bias[:0]
                t_abs = t_bias
                t_plot = t_abs - t_abs[0]
                ts_line = ax_ts.plot(t_plot, vals_bias, color=ts_color, lw=1.4)[0]
                t = t_abs
                vals = vals_bias
                t_bias_abs = t_abs
                t0_ts = float(t_abs[0])
                ax_ts.set_xlim(0.0, float(t_plot[-1]))
                ax_ts.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + t_abs[0]:.1f}"))
            else:
                ts_line = ax_ts.plot(t, vals, color=ts_color, lw=1.4)[0]
        else:
            ts_line = ax_ts.plot(t, vals, color=ts_color, lw=1.4)[0]
    else:
        ts_line = ax_ts.plot(t, vals, color=ts_color, lw=1.4)[0]
    ax_ts.set_xlabel("t (ps)")
    ax_ts.set_ylabel("s(t)")
    ax_ts.set_xlim(left=0.0)
    if is_prod:
        ax_ts.set_ylim(0.0, 1.0)
        ax_ts.set_yticks(np.arange(0.0, 1.01, 0.2))
    else:
        ax_ts.set_ylim(0.0, 2.0)
        ax_ts.set_yticks(np.arange(0.0, 2.01, 0.25))
    if not is_prod:
        ax_ts.axhline(mu, color=ts_color, lw=1.0, ls="--", alpha=0.7)
        ref_levels = [1.10, 1.15]
        ref_colors = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
        mu_sigma = mu + sd
        mu_2sigma = mu + 2 * sd
        mu_sigma_color = "#1f77b4"
        mu_2sigma_color = "#ff00ff"
        ref_lines = []
        for y, c in zip(ref_levels, ref_colors):
            ref_lines.append(ax_ts.axhline(y, color=c, lw=1.8, ls=":", alpha=0.9))
        ref_lines.append(ax_ts.axhline(mu_sigma, color=mu_sigma_color, lw=1.8, ls=":", alpha=0.9))
        ref_lines.append(ax_ts.axhline(mu_2sigma, color=mu_2sigma_color, lw=1.8, ls="--", alpha=0.9))
        ax_ts_right = ax_ts.twinx()
        ax_ts_right.set_ylim(0.0, 2.0)
        sigma_steps = [-28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28]
        tick_pairs = []
        for k in sigma_steps:
            val = mu + k * sd
            if 0.0 <= val <= 2.0:
                tick_pairs.append((val, k))
        if tick_pairs:
            ax_ts_right.set_yticks([v for v, _ in tick_pairs])
            labels = []
            for _, k in tick_pairs:
                if k == 0:
                    labels.append("μ")
                elif k > 0:
                    labels.append(f"μ+{k}σ")
                else:
                    labels.append(f"μ{k}σ")
            ax_ts_right.set_yticklabels(labels)
        if sd > 0:
            ref_labels = []
            for y in ref_levels:
                n = (y - mu) / sd
                ref_labels.append(f"{y:.2f} = μ{n:+.2f}σ")
            ref_labels.append(f"{mu_sigma:.4f} = μ+σ")
            ref_labels.append(f"{mu_2sigma:.4f} = μ+2σ")
        else:
            ref_labels = [f"{y:.2f} = μ+NaNσ" for y in ref_levels]
            ref_labels.append(f"{mu_sigma:.4f} = μ+σ")
            ref_labels.append(f"{mu_2sigma:.4f} = μ+2σ")
        ax_ts.legend(
            [ts_line, *ref_lines],
            [f"μ={mu:.4f}, σ={sd:.4f}", *ref_labels],
            frameon=False,
            fontsize=8,
            loc="lower right",
        )

    # Top-right: histogram (equil: horizontal; prod: vertical overlays)
    ax_hist = axes[0, 1]
    if is_prod:
        fes_times = parse_fes_times(args.fes_times)
        if args.t_fill is None:
            ax_hist.text(0.5, 0.5, "missing --t-fill", ha="center", va="center")
        elif not fes_times:
            ax_hist.text(0.5, 0.5, "missing --fes-times", ha="center", va="center")
        else:
            bins = np.linspace(0.0, 1.0, 41)
            valid = []
            for t_ps in fes_times:
                mask = (t > args.t_fill) & (t <= t_ps)
                if np.any(mask):
                    valid.append((t_ps, mask))
            n_valid = len(valid)
            if n_valid == 0:
                ax_hist.text(0.5, 0.5, "no data after --t-fill", ha="center", va="center")
            else:
                ax_hist.set_axis_off()
                strip_h = 1.0 / n_valid
                for i, (t_ps, mask) in enumerate(valid):
                    y0 = 1.0 - (i + 1) * strip_h
                    ax_strip = ax_hist.inset_axes([0.0, y0, 1.0, strip_h])
                    ax_strip.hist(
                        vals[mask],
                        bins=bins,
                        density=True,
                        color=fes_colors[i % len(fes_colors)] if fes_colors else fes_cmap(i % 10),
                        alpha=0.55,
                        edgecolor=(0, 0, 0, 0.35),
                        linewidth=0.4,
                    )
                    ax_strip.set_xlim(0.0, 1.0)
                    ax_strip.set_ylim(bottom=0.0)
                    ax_strip.tick_params(labelsize=6)
                    if i < n_valid - 1:
                        ax_strip.set_xticklabels([])
                    else:
                        ax_strip.set_xlabel("s")
                    ax_strip.set_ylabel("P(s)", fontsize=7)
                    patch = Patch(color=fes_colors[i % len(fes_colors)] if fes_colors else fes_cmap(i % 10), alpha=0.55)
                    ax_strip.legend(
                        [patch],
                        [f"t≤{t_ps:g} ps"],
                        frameon=False,
                        fontsize=7,
                        loc="center",
                    )
    else:
        counts, bins, _ = ax_hist.hist(
            vals,
            bins=40,
            density=True,
            color="lightgreen",
            alpha=0.75,
            label="Histogram",
            orientation="horizontal",
        )
        hist_mu = float(np.sum(0.5 * (bins[:-1] + bins[1:]) * counts) / np.sum(counts)) if np.sum(counts) > 0 else float("nan")
        hist_var = float(np.sum(((0.5 * (bins[:-1] + bins[1:]) - hist_mu) ** 2) * counts) / np.sum(counts)) if np.sum(counts) > 0 else float("nan")
        hist_sd = float(np.sqrt(hist_var)) if np.isfinite(hist_var) else float("nan")
        xs = np.linspace(float(np.min(vals)), float(np.max(vals)), 300)
        xs_hist = np.linspace(0.0, 2.0, 400)
        if sd > 0:
            ys = (1.0 / (sd * sqrt(2 * pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
            ax_hist.plot(ys, xs, color="black", lw=1.5, label=f"Gaussian μ={mu:.4f}, σ={sd:.4f}")
        # removed auxiliary Gaussian from probability panel
        ax_hist.set_xlabel("Probability density")
        ax_hist.set_ylabel("")
        ax_hist.set_ylim(0.0, 2.0)
        ax_hist.set_xlim(left=0.0)
        for y, c in zip(ref_levels, ref_colors):
            ax_hist.axhline(y, color=c, lw=1.8, ls=":", alpha=0.9)
        ax_hist.axhline(mu_sigma, color=mu_sigma_color, lw=1.8, ls=":", alpha=0.9)
        ax_hist.axhline(mu_2sigma, color=mu_2sigma_color, lw=1.8, ls="--", alpha=0.9)
        hist_handle = Line2D([0], [0], color="lightgreen", lw=1.5)
        gauss_handle = Line2D([0], [0], color="black", lw=1.5)
        ax_hist.legend(
            [hist_handle, gauss_handle],
            [f"hist μ={hist_mu:.4f}, σ={hist_sd:.4f}", f"Gaussian μ={mu:.4f}, σ={sd:.4f}"],
            frameon=False,
            fontsize=8,
            loc="upper right",
        )

    # Lower-left: -log(hist) and -log(Gaussian)
    ax_in = axes[0, 2]
    if is_prod:
        fes_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / args.fes_name
        biaspot_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / args.biaspot_name
        if not fes_path.exists() or not biaspot_path.exists():
            ax_in.text(0.5, 0.5, "fes.dat or biaspot not found", ha="center", va="center")
        else:
            times = parse_biaspot_times_ps(biaspot_path, args.T_max)
            blocks = read_fes_blocks(fes_path, max_blocks=len(times) if times.size > 0 else None)
            if times.size == 0 or not blocks:
                ax_in.text(0.5, 0.5, "fes blocks not found", ha="center", va="center")
            else:
                nblocks = min(len(times), len(blocks))
                times = times[:nblocks]
                blocks = blocks[:nblocks]
                if args.T_max is not None:
                    times, idx_keep = apply_tmax(times, args.T_max)
                    blocks = [blocks[i] for i in idx_keep] if idx_keep.size > 0 else []
                fes_times = parse_fes_times(args.fes_times)
                if not fes_times:
                    ax_in.text(0.5, 0.5, "missing --fes-times", ha="center", va="center")
                else:
                    curves = []
                    for i, t_ps in enumerate(fes_times):
                        idx = int(np.argmin(np.abs(times - t_ps)))
                        data = blocks[idx].copy()
                        data[:, 1] *= EH_TO_KCALMOL
                        color = fes_colors[i % len(fes_colors)] if fes_colors else fes_cmap(i % 10)
                        curves.append((times[idx], data, color))
                    if not curves:
                        ax_in.text(0.5, 0.5, "no FES curves found", ha="center", va="center")
                    else:
                        global_min = min(float(np.min(d[:, 1])) for _, d, _ in curves)
                        for t_ps, data, color in curves:
                            ax_in.plot(data[:, 0], data[:, 1] - global_min, lw=1.2, color=color, label=f"{t_ps:.2f} ps")
                        ax_in.set_xlabel("s")
                        ax_in.set_ylabel("F(s) [kcal/mol]")
                        ax_in.set_xlim(0.0, 1.0)
                        ax_in.grid(alpha=0.25)
                        ax_in.legend(frameon=False, fontsize=7, loc="best")
    else:
        eps = 1e-12
        centers = 0.5 * (bins[:-1] + bins[1:])
        neglog_hist = -np.log(np.clip(counts, eps, None))
        mask = counts > 0
        # Ratio legend data (computed before plotting legends)
        ratio_handles = []
        ratio_labels = []
        ratio_items = [
            (f"{mu_sigma:.4f}", mu_sigma, mu_sigma_color),
            ("1.10", 1.10, ref_colors[0]),
            (f"{mu_2sigma:.4f}", mu_2sigma, mu_2sigma_color),
        ]
        if counts.size and centers.size:
            idx_mu = int(np.argmin(np.abs(centers - mu)))
            p_mu_hist = max(counts[idx_mu], eps)
            for label, y, c in ratio_items:
                idx_y = int(np.argmin(np.abs(centers - y)))
                p_y_hist = max(counts[idx_y], eps)
                ratio = -np.log(p_y_hist / p_mu_hist)
                ratio_kcal = ratio * 0.59
                ratio_labels.append(f"{label}: {ratio:.3f} k$_B$T | {ratio_kcal:.3f} kcal/mol")
                ratio_handles.append(Line2D([0], [0], color=c, lw=1.5))
        else:
            for label, _, c in ratio_items:
                ratio_labels.append(f"{label}: NaN k$_B$T | NaN kcal/mol")
                ratio_handles.append(Line2D([0], [0], color=c, lw=1.5))

        if np.any(mask):
            min_hist = float(np.min(neglog_hist[mask]))
        else:
            min_hist = 0.0
        ax_in.plot(centers[mask], (neglog_hist[mask] - min_hist), color="lightgreen", lw=1.0, label="-log(hist)")
        if sd > 0:
            ys = (1.0 / (sd * sqrt(2 * pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
            neglog_g = -np.log(np.clip(ys, eps, None))
            min_g = float(np.min(neglog_g))
            ax_in.plot(xs, neglog_g - min_g, color="black", lw=1.0, label="-log(Gaussian)")
        # add Gaussian in free energy panel centered at min of -log(hist)
        if np.any(mask):
            min_idx = int(np.argmin(neglog_hist[mask]))
            x_center = float(centers[mask][min_idx])
            x_gauss = np.linspace(0.9, 1.25, 200)
            amp = 0.314  # kcal/mol
            sigma_fe = 0.1
            y_gauss = amp * np.exp(-0.5 * ((x_gauss - x_center) / sigma_fe) ** 2)
            gauss_label = "Gauss A=0.314, σ=0.1"
            ax_in.plot(
                x_gauss,
                y_gauss,
                color="gray",
                lw=1.5,
                alpha=0.7,
                label=gauss_label,
            )
        for y, c in zip(ref_levels, ref_colors):
            ax_in.axvline(y, color=c, lw=0.9, ls=":", alpha=0.9)
        ax_in.axvline(mu_sigma, color=mu_sigma_color, lw=0.9, ls=":", alpha=0.9)
        ax_in.axvline(mu_2sigma, color=mu_2sigma_color, lw=0.9, ls="--", alpha=0.9)
        # vertical markers for free energy levels
        ax_in.axvline(mu, color="black", lw=0.8, alpha=0.3)
        ax_in.vlines(mu, 0, 10, colors="red", linestyles="-", lw=1.2)
        ax_in.plot([mu], [10], marker="*", color="red", markersize=6)
        ax_in.vlines(mu, 0, 15, colors="green", linestyles="-", lw=1.2)
        ax_in.plot([mu], [15], marker="*", color="green", markersize=6)
        ax_in.text(
            0.02,
            0.98,
            "10 kcal/mol ~ 17 k$_B$T\n15 kcal/mol ~ 25 k$_B$T\n(from Anil et al.)",
            transform=ax_in.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax_in.set_xlim(left=0.0)
        ax_in.set_xlabel("Coordination")
        ax_in.set_ylabel("Free energy")
        ax_in.tick_params(labelsize=8)
        if "gauss_label" in locals():
            ratio_labels.append(gauss_label)
            ratio_handles.append(Line2D([0], [0], color="gray", lw=1.5))
        leg2 = ax_in.legend(
            ratio_handles,
            ratio_labels,
            frameon=False,
            loc="lower left",
            title="-log(P(y)/P(μ))\n(k$_B$T | kcal/mol)",
            title_fontsize=7,
            fontsize=7,
        )

    # Top-right: autocorrelation function / pKa time series (prod)
    ax_acf = axes[0, 3]
    if is_prod:
        ax_acf.set_axis_on()
        fes_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / args.fes_name
        biaspot_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / args.biaspot_name
        if not fes_path.exists() or not biaspot_path.exists():
            ax_acf.text(0.5, 0.5, "fes.dat or biaspot not found", ha="center", va="center")
        else:
            times = parse_biaspot_times_ps(biaspot_path, args.T_max)
            blocks = read_fes_blocks(fes_path, max_blocks=len(times) if times.size > 0 else None)
            if times.size == 0 or not blocks:
                ax_acf.text(0.5, 0.5, "fes blocks not found", ha="center", va="center")
            else:
                nblocks = min(len(times), len(blocks))
                times = times[:nblocks]
                blocks = blocks[:nblocks]
                if args.T_max is not None:
                    times, idx_keep = apply_tmax(times, args.T_max)
                    blocks = [blocks[i] for i in idx_keep] if idx_keep.size > 0 else []
                blocks_kcal = []
                for b in blocks:
                    b_kcal = b.copy()
                    b_kcal[:, 1] *= EH_TO_KCALMOL
                    blocks_kcal.append(b_kcal)
                df_vals = np.array(
                    [
                        deltaf(b, PKA_MIN1_X, PKA_MIN2_X, PKA_HALF_WINDOW, PKA_FES_XMIN, PKA_FES_XMAX)
                        for b in blocks_kcal
                    ],
                    dtype=float,
                )
                pka_vals = df_vals / (PKA_FACTOR * PKA_TEMP)
                t0 = float(times[0])
                times_rel = times - t0
                t_pka_abs = times
                t0_pka = t0
                ax_acf.plot(times_rel, pka_vals, color="black", lw=1.6)
                ax_acf.scatter(times_rel, pka_vals, color="#FFA500", edgecolor=(0.0, 0.0, 0.0, 0.35), s=18)
                legend_handles = []
                legend_labels = []
                if fes_times:
                    for i, t_req in enumerate(fes_times):
                        idx = int(np.argmin(np.abs(times - t_req)))
                        t_rel = times_rel[idx]
                        pka_val = pka_vals[idx]
                        color = fes_colors[i % len(fes_colors)] if fes_colors else fes_cmap(i % 10)
                        ax_acf.scatter([t_rel], [pka_val], marker="*", s=90, color=color, zorder=4)
                        legend_handles.append(
                            Line2D([0], [0], marker="*", color="none", markerfacecolor=color, markersize=7)
                        )
                        legend_labels.append(f"{times[idx]:.2f} ps: pKa={pka_val:.2f}")
                if args.exp_pka is not None:
                    ax_acf.axhline(args.exp_pka, color="#0000FF", lw=1.6, ls="--")
                ax_acf.set_xlim(0.0, float(times_rel[-1]))
                ax_acf.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + t0:.1f}"))
                ax_acf.set_xlabel("t (ps)")
                ax_acf.set_ylabel("pKa")
                ax_acf.grid(alpha=0.25)
                if legend_handles:
                    ax_acf.legend(
                        legend_handles,
                        legend_labels,
                        frameon=False,
                        fontsize=7,
                        loc="best",
                    )
    else:
        ax_acf.set_axis_on()
        x = vals - np.mean(vals)
        N = len(x)
        if N > 1:
            acf = np.correlate(x, x, mode="full")
            acf = acf[N - 1 :]
            acf = acf / (N - np.arange(N))
            if acf[0] != 0:
                acf = acf / acf[0]
            lags = np.arange(N) * dt_ps
            ax_acf.plot(lags, acf, color="#1f77b4", lw=1.2)
            ax_acf.axhline(1 / np.e, linestyle="--", color="#7f7f7f", lw=1.0)
            ax_acf.axhline(0, linestyle="--", color="#7f7f7f", lw=1.0)
            ax_acf.axvline(0.04, linestyle="--", color="#7f7f7f", lw=1.0)
            idx_004 = int(np.argmin(np.abs(lags - 0.04)))
            acf_004 = float(acf[idx_004])
            ax_acf.set_xlabel("lag time (ps)")
            ax_acf.set_ylabel("normalized autocorrelation")
            ax_acf.set_xlim(0, 5)
            ax_acf.set_ylim(0, 1)
            idx_1e = np.where(acf <= 1 / np.e)[0]
            x_1e = float(lags[idx_1e[0]]) if len(idx_1e) > 0 else float("nan")
            ax_acf.legend(
                [
                    f"C(0.04 ps) = {acf_004:.3f}",
                    f"C({x_1e:.3f} ps) = 1/e = {1/np.e:.3f}",
                ],
                frameon=False,
                fontsize=8,
                loc="upper right",
            )
        else:
            ax_acf.text(0.5, 0.5, "not enough data", ha="center", va="center")

    axes[1, 3].axis("off")

    mull_data = None
    if args.solute_atoms is not None:
        mulliken_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / "mulliken"
        if mulliken_path.exists():
            extra_ids = parse_extra_ids(args.extra_ids)
            if args.T_max is not None:
                times, charges, target_ids, elements = parse_mulliken_limited(
                    mulliken_path, args.solute_atoms, args.T_max, natoms=None, extra_ids=extra_ids
                )
            else:
                times, charges, target_ids, elements = parse_mulliken(
                    mulliken_path, args.solute_atoms, natoms=None, extra_ids=extra_ids
                )
            oxygen_idx = [i for i, atom_id in enumerate(target_ids) if elements.get(atom_id) == "O"]
            sel_ids = None
            least_ids = None
            if oxygen_idx:
                oxy_ids = [target_ids[i] for i in oxygen_idx]
                oxy_chg = charges[oxygen_idx, :]
                sel = []
                sel_least = []
                for j in range(oxy_chg.shape[1]):
                    vals_j = oxy_chg[:, j]
                    mask = (vals_j >= -1.4) & (vals_j <= -1.1)
                    if not np.any(mask):
                        sel.append(np.nan)
                    else:
                        idx = int(np.argmin(vals_j[mask]))
                        cand_indices = np.where(mask)[0]
                        sel.append(oxy_ids[cand_indices[idx]])
                    mask_least = (vals_j >= -0.625) & (vals_j <= -0.525)
                    if not np.any(mask_least):
                        sel_least.append(np.nan)
                    else:
                        idx_least = int(np.argmax(vals_j[mask_least]))
                        cand_least = np.where(mask_least)[0]
                        sel_least.append(oxy_ids[cand_least[idx_least]])
                sel_ids = np.array(sel, dtype=float)
                least_ids = np.array(sel_least, dtype=float)
            mull_data = (times, charges, target_ids, elements, sel_ids, least_ids)
        else:
            mull_data = None

    # Second row (col 1): Mulliken time series
    if args.solute_atoms is not None:
        ax_mull = axes[1, 0]
        ax_mull.set_axis_on()
        if mull_data is not None:
            times, charges, target_ids, elements, _sel_ids, _least_ids = mull_data
            if args.T_max is not None:
                times, idx_keep = apply_tmax(times, args.T_max)
                charges = charges[:, idx_keep] if idx_keep.size > 0 else charges[:, :0]
            colors = plt.cm.coolwarm(np.linspace(0, 1, max(1, len(target_ids))))
            for i, atom_id in enumerate(target_ids):
                ax_mull.plot(times, charges[i], lw=1.0, color=colors[i])
            ax_mull.set_xlabel("t (ps)")
            ax_mull.set_ylabel("q(t) [e^-]")
        else:
            ax_mull.text(0.5, 0.5, "mulliken not found", ha="center", va="center")
    # Second row (col 2): Mulliken horizontal probability densities
    if args.solute_atoms is not None:
        ax_mull_hist = axes[1, 1]
        ax_mull_hist.set_axis_on()
        if mull_data is not None:
            _times, charges, target_ids, _elements, _sel_ids, _least_ids = mull_data
            if args.T_max is not None:
                _times, idx_keep = apply_tmax(_times, args.T_max)
                charges = charges[:, idx_keep] if idx_keep.size > 0 else charges[:, :0]
            colors = plt.cm.coolwarm(np.linspace(0, 1, max(1, len(target_ids))))
            for i, atom_id in enumerate(target_ids):
                y = charges[i]
                hist_counts, bin_edges = np.histogram(y, bins=60, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax_mull_hist.plot(hist_counts, bin_centers, lw=1.0, color=colors[i])
            ax_mull_hist.set_xlabel("P(q)")
            ax_mull_hist.set_ylabel("q(t) [e^-]")
            ax_mull_hist.axhspan(-1.4, -1.1, color="gray", alpha=0.15)
            ax_mull_hist.axhline(-1.25, color="black", lw=1.0, ls="--", alpha=0.5)
            ax_mull_hist.axhspan(-0.625, -0.525, color="gray", alpha=0.15)
            ax_mull_hist.axhline(-0.575, color="black", lw=1.0, ls="--", alpha=0.5)
        else:
            ax_mull_hist.text(0.5, 0.5, "mulliken not found", ha="center", va="center")

    # Second row (col 3): most-negative oxygen ID over time
    if args.solute_atoms is not None:
        ax_neg = axes[1, 2]
        ax_neg.set_axis_on()
        if mull_data is not None:
            times, _charges, target_ids, elements, sel_ids, least_ids = mull_data
            if args.T_max is not None:
                times, idx_keep = apply_tmax(times, args.T_max)
                sel_ids = sel_ids[idx_keep] if idx_keep.size > 0 else sel_ids[:0]
                least_ids = least_ids[idx_keep] if least_ids is not None and idx_keep.size > 0 else (least_ids[:0] if least_ids is not None else None)
            oxygen_idx = [i for i, atom_id in enumerate(target_ids) if elements.get(atom_id) == "O"]
            if oxygen_idx and sel_ids is not None:
                oxy_ids = [target_ids[i] for i in oxygen_idx]
                # piecewise-constant step function (compress repeated IDs)
                t_arr = np.array(times, dtype=float)
                y_arr = np.array(sel_ids, dtype=float)
                if t_arr.size > 0:
                    t_step = [t_arr[0]]
                    y_step = [y_arr[0]]
                    for i in range(1, t_arr.size):
                        if y_arr[i] != y_arr[i - 1]:
                            t_step.append(t_arr[i])
                            y_step.append(y_arr[i - 1])
                            t_step.append(t_arr[i])
                            y_step.append(y_arr[i])
                    t_step.append(t_arr[-1])
                    y_step.append(y_arr[-1])
                    ax_neg.plot(t_step, y_step, lw=1.0, color="#1f77b4")
                if least_ids is not None:
                    y_arr2 = np.array(least_ids, dtype=float)
                    if t_arr.size > 0:
                        t_step2 = [t_arr[0]]
                        y_step2 = [y_arr2[0]]
                        for i in range(1, t_arr.size):
                            if y_arr2[i] != y_arr2[i - 1]:
                                t_step2.append(t_arr[i])
                                y_step2.append(y_arr2[i - 1])
                                t_step2.append(t_arr[i])
                                y_step2.append(y_arr2[i])
                        t_step2.append(t_arr[-1])
                        y_step2.append(y_arr2[-1])
                        ax_neg.plot(t_step2, y_step2, lw=1.0, color="#d62728")
                ax_neg.set_xlabel("Time (ps)")
                ax_neg.set_ylabel("Oxygen ID")
                ax_neg.set_ylim(1, max(oxy_ids) + 1)
                unique_ids = sorted({int(v) for v in y_arr if np.isfinite(v)})
                if unique_ids:
                    if len(unique_ids) <= 20:
                        yticks = unique_ids
                    else:
                        step = max(1, len(unique_ids) // 20)
                        yticks = unique_ids[::step]
                    ax_neg.set_yticks(yticks)
                    ax_neg.tick_params(axis="y", labelsize=6)
            else:
                ax_neg.text(0.5, 0.5, "no oxygen IDs found", ha="center", va="center")
        else:
            ax_neg.text(0.5, 0.5, "mulliken not found", ha="center", va="center")

    # Second row (col 4): empty
    ax_dist = axes[1, 3]
    ax_dist.set_axis_on()
    if args.n_ids and mull_data is not None:
        n_ids = parse_ids(args.n_ids)
        times, _charges, _target_ids, _elements, sel_ids, least_ids = mull_data
        if args.T_max is not None:
            times, idx_keep = apply_tmax(times, args.T_max)
            sel_ids = sel_ids[idx_keep] if idx_keep.size > 0 else sel_ids[:0]
            least_ids = least_ids[idx_keep] if least_ids is not None and idx_keep.size > 0 else (least_ids[:0] if least_ids is not None else None)
        traj_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / args.traj_name
        if traj_path.exists():
            dftb_inp = args.runs_path / f"run-{args.run_id}" / args.run_dir / "dftb.inp"
            box = read_box_lengths_from_dftb_inp(dftb_inp) if dftb_inp.exists() else None
            n_series = {nid: [] for nid in n_ids}
            n_series_least = {nid: [] for nid in n_ids}
            t_series: List[float] = []
            max_frames = len(times)
            for idx, coords in enumerate(iter_xyz_coords(traj_path)):
                if idx >= max_frames:
                    break
                o_id = sel_ids[idx] if sel_ids is not None else np.nan
                o2_id = least_ids[idx] if least_ids is not None else np.nan
                if not np.isfinite(o_id):
                    for nid in n_ids:
                        n_series[nid].append(np.nan)
                        n_series_least[nid].append(np.nan)
                    t_series.append(times[idx])
                    continue
                o_idx = int(o_id) - 1
                for nid in n_ids:
                    n_idx = int(nid) - 1
                    d = coords[n_idx] - coords[o_idx]
                    if box is not None:
                        d = d - box * np.round(d / box)
                    dist = float(np.linalg.norm(d))
                    n_series[nid].append(dist)
                if np.isfinite(o2_id):
                    o2_idx = int(o2_id) - 1
                    for nid in n_ids:
                        n_idx = int(nid) - 1
                        d2 = coords[n_idx] - coords[o2_idx]
                        if box is not None:
                            d2 = d2 - box * np.round(d2 / box)
                        dist2 = float(np.linalg.norm(d2))
                        n_series_least[nid].append(dist2)
                else:
                    for nid in n_ids:
                        n_series_least[nid].append(np.nan)
                t_series.append(times[idx])
            colors = ["green", "orange"]
            least_colors = ["red", "purple"]
            for i, nid in enumerate(n_ids):
                color = colors[i % len(colors)]
                ax_dist.scatter(t_series, n_series[nid], s=6, color=color, label=f"N{nid}")
                least_color = least_colors[i % len(least_colors)]
                ax_dist.scatter(t_series, n_series_least[nid], s=6, color=least_color, alpha=0.6, label=f"N{nid} (least)")
            ax_dist.set_xlabel("Time (ps)")
            ax_dist.set_ylabel("N–O distance (Å)")
            ax_dist.legend(frameon=False, fontsize=6, loc="upper right")
        else:
            ax_dist.text(0.5, 0.5, "traject not found", ha="center", va="center")
    else:
        ax_dist.set_axis_on()
        ax_dist.text(0.5, 0.5, "missing N IDs or mulliken data", ha="center", va="center")

    # Text box explaining lines (since legend is used for ratios)

    system = infer_system(args.runs_path)
    if is_prod:
        time_window = compute_time_window(
            t_bias_abs if t_bias_abs is not None else None,
            t_pka_abs if t_pka_abs is not None else None,
            (mull_data[0] if mull_data is not None else None),
        )
        if time_window is not None:
            x_start, x_end = time_window
            if args.T_max is not None:
                x_end = min(x_end, args.T_max)
            x_start, _ = snap_time_window(x_start, x_end, 2.5)
            xticks = np.arange(x_start, x_end + 1e-6, 2.5)
            ax_ts.set_xlim(x_start - t0_ts, x_end - t0_ts)
            ax_ts.set_xticks(xticks - t0_ts)
            if ax_acf is not None:
                ax_acf.set_xlim(x_start - t0_pka, x_end - t0_pka)
                ax_acf.set_xticks(xticks - t0_pka)
            if ax_mull is not None:
                ax_mull.set_xlim(x_start, x_end)
                ax_mull.set_xticks(xticks)
            if ax_neg is not None:
                ax_neg.set_xlim(x_start, x_end)
                ax_neg.set_xticks(xticks)
            if ax_dist is not None:
                ax_dist.set_xlim(x_start, x_end)
                ax_dist.set_xticks(xticks)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.run_dir}_report_{args.phase}_run{args.run_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")

    # Optional interactive HTML for Mulliken time series
    if args.solute_atoms is not None and mull_data is not None:
        if go is None:
            print("NOTE: plotly not installed; skipping HTML output")
            return
        times, charges, target_ids, elements, _sel_ids, _least_ids = mull_data
        fig_html = go.Figure()
        for i, atom_id in enumerate(target_ids):
            label = f"{elements.get(atom_id, 'X')}-{atom_id}"
            fig_html.add_trace(
                go.Scatter(
                    x=times,
                    y=charges[i],
                    mode="lines",
                    name=label,
                    hovertemplate=f"{label}<br>t=%{{x:.3f}} ps<br>q=%{{y:.5f}}<extra></extra>",
                )
            )
        fig_html.update_layout(
            title="Mulliken charges",
            xaxis_title="Time (ps)",
            yaxis_title="Mulliken (s+p)",
        )
        html_out = args.html_out
        if html_out is None:
            html_out = Path("reports") / f"{system}_{args.run_dir}_mulliken_run{args.run_id}.html"
        html_out.parent.mkdir(parents=True, exist_ok=True)
        fig_html.write_html(html_out, include_plotlyjs="cdn")
        print(f"Wrote {html_out}")


if __name__ == "__main__":
    main()
