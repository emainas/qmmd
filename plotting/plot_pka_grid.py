#!/usr/bin/env python3
"""Plot pKa-vs-time grids from biaspot + fes.dat files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
RUN_RE = re.compile(r"^run-(\d+)$")
EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576


def parse_run_ids(value: str) -> set[int]:
    """Parse comma-separated run IDs and inclusive ranges."""
    run_ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("run IDs must not contain empty entries")

        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"invalid run ID range: {item!r}")
            try:
                start, end = (int(part.strip()) for part in parts)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid run ID range: {item!r}") from exc
            if start < 1 or end < 1:
                raise argparse.ArgumentTypeError("run IDs must be positive integers")
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"run ID range must be ascending: {item!r}"
                )
            run_ids.update(range(start, end + 1))
            continue

        try:
            run_id = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid run ID: {item!r}") from exc
        if run_id < 1:
            raise argparse.ArgumentTypeError("run IDs must be positive integers")
        run_ids.add(run_id)

    return run_ids


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


def load_biaspot_with_restart(run_dir: Path, cv_dir: str) -> np.ndarray:
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


def discover_runs(
    runs_path: Path,
    cv_dir: str,
    run_ids: Optional[set[int]] = None,
) -> List[Tuple[int, Path, Path]]:
    out: List[Tuple[int, Path, Path]] = []
    for p in sorted(runs_path.iterdir()):
        if not p.is_dir():
            continue
        m = RUN_RE.match(p.name)
        if not m:
            continue
        run_id = int(m.group(1))
        if run_ids is not None and run_id not in run_ids:
            continue
        biaspot = p / cv_dir / "biaspot"
        fes = p / cv_dir / "fes.dat"
        if biaspot.exists() and fes.exists():
            out.append((run_id, biaspot, fes))
    return sorted(out, key=lambda run: run[0])


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

def parse_exp_pkas(raw: str | None) -> List[float]:
    if raw is None:
        return []
    items = re.split(r"[,\s]+", raw.strip())
    out: List[float] = []
    for item in items:
        if not item:
            continue
        out.append(float(item))
    return out

def main() -> None:
    p = argparse.ArgumentParser(description="Plot pKa grids from biaspot + fes.dat.")
    p.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    p.add_argument("--cv-dir", required=True, help="CV directory under each run (e.g., all-meta-hid)")
    p.add_argument(
        "--run-ids",
        type=parse_run_ids,
        default=None,
        metavar="IDS",
        help="Run IDs to plot, as comma-separated IDs or inclusive ranges (for example, 51-100)",
    )
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--temp", type=float, default=313.15)
    p.add_argument("--min1-x", type=float, default=0.0)
    p.add_argument("--min2-x", type=float, default=1.0)
    p.add_argument("--half-window", type=float, default=0.1)
    p.add_argument("--fes-xmin", type=float, default=0.0)
    p.add_argument("--fes-xmax", type=float, default=1.25)
    p.add_argument(
        "--exp-pka",
        type=str,
        default=None,
        help="Experimental pKa(s), comma or space separated (horizontal dashed lines)",
    )
    args = p.parse_args()
    exp_pkas = parse_exp_pkas(args.exp_pka)

    runs = discover_runs(args.runs_path, args.cv_dir, args.run_ids)
    if not runs:
        raise SystemExit("No biaspot/fes.dat pairs found.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    all_y = []
    for i, (run_id, biaspot, fes_path) in enumerate(runs):
        ax = axes_flat[i]
        run_dir = biaspot.parent.parent
        times = load_biaspot_with_restart(run_dir, args.cv_dir)
        blocks = load_fes_with_restart(run_dir, args.cv_dir)
        if times.size == 0 or not blocks:
            ax.set_axis_off()
            continue

        nblocks = min(len(blocks), len(times))
        times = times[:nblocks]
        blocks = blocks[:nblocks]
        t0 = float(times[0])
        times_rel = times - t0

        df_vals = np.array(
            [
                deltaf(b, args.min1_x, args.min2_x, args.half_window, args.fes_xmin, args.fes_xmax)
                for b in blocks
            ],
            dtype=float,
        )
        pka_vals = df_vals / (PKA_FACTOR * args.temp)
        ax.plot(times_rel, pka_vals, color="black", lw=1.6)
        ax.scatter(times_rel, pka_vals, color="#FFA500", edgecolor=(0.0, 0.0, 0.0, 0.35), s=18)
        if exp_pkas:
            for j, pka in enumerate(exp_pkas):
                color = f"C{j % 10}"
                ax.axhline(pka, color=color, lw=1.2, ls="--", alpha=0.85)
        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.set_xlim(0.0, float(times_rel[-1]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + t0:.1f}"))
        ylo, yhi = np.min(pka_vals), np.max(pka_vals)
        pad = 0.05 * (yhi - ylo) if yhi > ylo else 0.5
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_ylim(-20,20)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        all_y.append(pka_vals)

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    # per-run scaling already applied via each panel's data range

    for r in range(rows):
        if cols > 1:
            axes[r, 0].set_ylabel("pKa", fontsize=10)
        else:
            axes.set_ylabel("pKa", fontsize=10)
    for c in range(cols):
        if rows > 1:
            axes[rows - 1, c].set_xlabel("t (ps)", fontsize=10)
        else:
            axes.set_xlabel("t (ps)", fontsize=10)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_pka_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
