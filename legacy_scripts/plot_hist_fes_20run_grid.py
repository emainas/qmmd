#!/usr/bin/env python3
"""Plot 4x5 FES panel grids for HIST ND/NE using 4-8 ps snapshots."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLE_FILE = PROJECT_ROOT / "src" / "prl.mplstyle"

ND_COLVAR_DIR = PROJECT_ROOT / "data/molecules/HIST/colvars/all_ND_coord/colvars_hid"
ND_FES_DIR = PROJECT_ROOT / "data/molecules/HIST/colvars/all_ND_coord/fes_hid"
NE_COLVAR_DIR = PROJECT_ROOT / "data/molecules/HIST/colvars/all_NE_coord/colvars_hie"
NE_FES_DIR = PROJECT_ROOT / "data/molecules/HIST/colvars/all_NE_coord/fes_hie"

ND_OUT = PROJECT_ROOT / "reports/molecules/HIST/colvars/all_ND_coord/summaries/fes_20run_grid_4to8ps.png"
NE_OUT = PROJECT_ROOT / "reports/molecules/HIST/colvars/all_NE_coord/summaries/fes_20run_grid_4to8ps.png"

RUN_COLVAR_RE = re.compile(r"^(\d+)_colvar\.dat$")
RUN_FES_RE = re.compile(r"^(\d+)_fes\.dat$")

TIME_MARKS_PS = [4.0, 5.0, 6.0, 7.0, 8.0]
FES_XMIN = 0.0
FES_XMAX = 1.25
EH_TO_KCALMOL = 627.509474
REF_FRAMES = 375
REF_TMAX_PS = 9.0
FRAME_TO_PS = REF_TMAX_PS / REF_FRAMES


def parse_colvar_file(path: Path) -> np.ndarray:
    vals: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s.split()[-1]))
            except Exception:
                continue
    return np.array(vals, dtype=float)


def load_all_fes_blocks(path: Path) -> list[tuple[int, np.ndarray]]:
    blocks: list[tuple[int, np.ndarray]] = []
    current_block: list[list[float]] = []
    current_ng: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("###"):
                if current_block and current_ng is not None:
                    arr = np.array(current_block, dtype=float)
                    arr[:, 1] *= EH_TO_KCALMOL
                    blocks.append((current_ng, arr))
                current_block = []
                current_ng = None
                parts = s.split()
                try:
                    current_ng = int(parts[-2])
                except Exception:
                    current_ng = None
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                current_block.append([float(parts[0]), float(parts[1])])
            except Exception:
                continue

    if current_block and current_ng is not None:
        arr = np.array(current_block, dtype=float)
        arr[:, 1] *= EH_TO_KCALMOL
        blocks.append((current_ng, arr))

    return blocks


def discover_ids(colvar_dir: Path, fes_dir: Path) -> list[int]:
    colvar_ids = {
        int(m.group(1))
        for p in colvar_dir.iterdir()
        if p.is_file()
        for m in [RUN_COLVAR_RE.match(p.name)]
        if m
    }
    fes_ids = {
        int(m.group(1))
        for p in fes_dir.iterdir()
        if p.is_file()
        for m in [RUN_FES_RE.match(p.name)]
        if m
    }
    return sorted(colvar_ids & fes_ids)


def selected_blocks_for_times(colvar_len: int, available_ng: np.ndarray) -> list[tuple[float, int]]:
    if colvar_len <= 1:
        return []
    t_col = np.arange(colvar_len, dtype=float) * FRAME_TO_PS
    out: list[tuple[float, int]] = []
    for t_mark in TIME_MARKS_PS:
        idx = int(np.argmin(np.abs(t_col - t_mark)))
        ng_raw = idx + 1
        j = int(np.argmin(np.abs(available_ng - ng_raw)))
        out.append((t_mark, int(available_ng[j])))
    return out


def make_fes_grid(colvar_dir: Path, fes_dir: Path, out_png: Path, panel_title: str) -> None:
    run_ids = discover_ids(colvar_dir, fes_dir)
    if len(run_ids) < 20:
        raise ValueError(f"Expected at least 20 matching runs in {colvar_dir} and {fes_dir}, found {len(run_ids)}")

    plt.style.use(STYLE_FILE)
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), dpi=220, sharex=True, sharey=True)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for i, run_id in enumerate(run_ids[:20]):
        ax = axes_flat[i]
        colvar_file = colvar_dir / f"{run_id}_colvar.dat"
        fes_file = fes_dir / f"{run_id}_fes.dat"

        colvar = parse_colvar_file(colvar_file)
        blocks = load_all_fes_blocks(fes_file)
        if colvar.size == 0 or not blocks:
            ax.set_axis_off()
            continue

        block_by_ng = {ng: arr for ng, arr in blocks}
        available_ng = np.array(sorted(block_by_ng.keys()), dtype=int)
        picks = selected_blocks_for_times(colvar.size, available_ng)

        for t_mark, ng in picks:
            # Keep the same color logic as the recent nice FES plots:
            # absolute timestamp maps to tab10 index (t=1 -> c0, ..., t=8 -> c7).
            color_idx = (int(round(t_mark)) - 1) % 10
            c = cmap(color_idx)
            fes = block_by_ng[ng]
            ax.plot(fes[:, 0], fes[:, 1], color=c, lw=1.8, label=f"{t_mark:.0f} ps")

        ax.set_title(f"run-{run_id}", fontsize=11)
        ax.set_xlim(FES_XMIN, FES_XMAX)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[len(run_ids[:20]):]:
        ax.set_axis_off()

    for r in range(4):
        axes[r, 0].set_ylabel(r"$F$ (kcal mol$^{-1}$)", fontsize=10)
    for c in range(5):
        axes[3, c].set_xlabel(r"$s$", fontsize=10)

    # Shared legend from first panel's handles
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label: dict[str, object] = {}
        for h, l in zip(handles, labels):
            by_label[l] = h
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            fontsize=10,
            title="Time",
        )

    fig.suptitle(panel_title, fontsize=16, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    make_fes_grid(
        ND_COLVAR_DIR,
        ND_FES_DIR,
        ND_OUT,
        "HIST single_ND_coord: 20-run FES snapshots at 4-8 ps",
    )
    make_fes_grid(
        NE_COLVAR_DIR,
        NE_FES_DIR,
        NE_OUT,
        "HIST single_NE_coord: 20-run FES snapshots at 4-8 ps",
    )


if __name__ == "__main__":
    main()
