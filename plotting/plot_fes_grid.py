#!/usr/bin/env python3
"""Plot FES grids from biaspot + fes.dat files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
RUN_RE = re.compile(r"^run-(\d+)$")
EH_TO_KCALMOL = 627.509474


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


def discover_runs(runs_path: Path, cv_dir: str) -> List[Tuple[int, Path, Path]]:
    out: List[Tuple[int, Path, Path]] = []
    for p in sorted(runs_path.iterdir()):
        if not p.is_dir():
            continue
        m = RUN_RE.match(p.name)
        if not m:
            continue
        run_id = int(m.group(1))
        biaspot = p / cv_dir / "biaspot"
        fes = p / cv_dir / "fes.dat"
        if biaspot.exists() and fes.exists():
            out.append((run_id, biaspot, fes))
    return out


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


def main() -> None:
    p = argparse.ArgumentParser(description="Plot FES grids from biaspot + fes.dat.")
    p.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    p.add_argument("--cv-dir", required=True, help="CV directory under each run (e.g., all-meta-hid)")
    p.add_argument("--n-snapshots", type=int, default=5, help="Number of time snapshots to plot")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No biaspot/fes.dat pairs found.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    cmap = plt.get_cmap("tab10")

    all_y = []
    for i, (run_id, biaspot, fes_path) in enumerate(runs):
        ax = axes_flat[i]
        times = parse_biaspot_times(biaspot)
        blocks = load_fes_blocks(fes_path)
        if times.size == 0 or not blocks:
            ax.set_axis_off()
            continue

        nblocks = min(len(blocks), len(times))
        times = times[:nblocks]
        blocks = blocks[:nblocks]
        t0 = float(times[0])
        times_rel = times - t0
        pick_idx = np.linspace(0, nblocks - 1, args.n_snapshots, dtype=int)

        for j, idx in enumerate(pick_idx):
            fes = blocks[idx]
            c = cmap(j % 10)
            ax.plot(fes[:, 0], fes[:, 1], color=c, lw=1.6, label=f"{times[idx]:.1f} ps")
            all_y.append(fes[:, 1])

        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.set_xlim(fes[:, 0].min(), fes[:, 0].max())
        ylo, yhi = np.min(fes[:, 1]), np.max(fes[:, 1])
        pad = 0.05 * (yhi - ylo) if yhi > ylo else 0.5
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    # per-run scaling already applied via each panel's data range

    for r in range(rows):
        if cols > 1:
            axes[r, 0].set_ylabel("F (kcal mol$^{-1}$)", fontsize=10)
        else:
            axes.set_ylabel("F (kcal mol$^{-1}$)", fontsize=10)
    for c in range(cols):
        if rows > 1:
            axes[rows - 1, c].set_xlabel("s", fontsize=10)
        else:
            axes.set_xlabel("s", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label = {}
        for h, l in zip(handles, labels):
            by_label[l] = h
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=min(args.n_snapshots, 6),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
            title="Time",
        )

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_fes_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
