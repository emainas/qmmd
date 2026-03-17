#!/usr/bin/env python3
"""Plot 4x5 COLVAR grids for HIST ND/NE 20-run datasets."""

from __future__ import annotations

import re
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLE_FILE = PROJECT_ROOT / "src" / "prl.mplstyle"

ND_DIR = PROJECT_ROOT / "data/molecules/HIST/colvars/all_ND_coord/colvars_hid"
NE_DIR = PROJECT_ROOT / "data/molecules/HIST/colvars/all_NE_coord/colvars_hie"

ND_OUT = PROJECT_ROOT / "reports/molecules/HIST/colvars/all_ND_coord/summaries/colvar_20run_grid.png"
NE_OUT = PROJECT_ROOT / "reports/molecules/HIST/colvars/all_NE_coord/summaries/colvar_20run_grid.png"

RUN_FILE_RE = re.compile(r"^(\d+)_colvar\.dat$")
FLOAT_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
REF_FRAMES = 375
# Keep physical frame->time conversion fixed; plot-window changes must not
# re-scale the data axis, only clip it.
# Colvar sampled every 80 MD steps; timestep = 0.5 fs => 40 fs = 0.04 ps.
FRAME_TO_PS = 0.04
RUN1_MAX_FRAMES = 700


def parse_colvar_file(path: Path) -> np.ndarray:
    vals: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            matches = FLOAT_RE.findall(line)
            if not matches:
                continue
            vals.append(float(matches[-1]))
    return np.array(vals, dtype=float)


def discover_runs(data_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        m = RUN_FILE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def make_grid(data_dir: Path, out_png: Path, panel_title: str, tmax_ps: float) -> None:
    runs = discover_runs(data_dir)
    if len(runs) < 20:
        raise ValueError(f"Expected 20 run files in {data_dir}, found {len(runs)}")

    plt.style.use(STYLE_FILE)
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for i, (run_id, path) in enumerate(runs[:20]):
        ax = axes_flat[i]
        y = parse_colvar_file(path)
        if run_id == 1:
            y = y[:RUN1_MAX_FRAMES]
        if y.size == 0:
            ax.set_axis_off()
            continue
        x = np.arange(y.size, dtype=float) * FRAME_TO_PS
        ax.plot(x, y, color="#1f77b4", lw=1.8)
        ax.set_title(f"run-{run_id}", fontsize=11)
        ax.set_ylim(0.0, 1.5)
        ax.set_xlim(0.0, tmax_ps)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[len(runs[:20]):]:
        ax.set_axis_off()

    for r in range(4):
        axes[r, 0].set_ylabel(r"$s$", fontsize=11)
    for c in range(5):
        axes[3, c].set_xlabel(r"$t$ (ps)", fontsize=10)

    fig.suptitle(panel_title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Wrote {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4x5 HIST COLVAR grids")
    p.add_argument("--tmax-ps", type=float, default=15.0, help="Displayed time window in ps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_grid(ND_DIR, ND_OUT, "HIST single_ND_coord: 20-run COLVAR time series", args.tmax_ps)
    make_grid(NE_DIR, NE_OUT, "HIST single_NE_coord: 20-run COLVAR time series", args.tmax_ps)


if __name__ == "__main__":
    main()
