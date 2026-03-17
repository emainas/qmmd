#!/usr/bin/env python3
"""Plot 4x5 pKa-vs-time grids for HIST ND/NE 20-run datasets."""

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

ND_OUT = PROJECT_ROOT / "reports/molecules/HIST/colvars/all_ND_coord/summaries/pka_20run_grid.png"
NE_OUT = PROJECT_ROOT / "reports/molecules/HIST/colvars/all_NE_coord/summaries/pka_20run_grid.png"

RUN_COLVAR_RE = re.compile(r"^(\d+)_colvar\.dat$")
RUN_FES_RE = re.compile(r"^(\d+)_fes\.dat$")

EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576
TEMPERATURE_K = 313.15
# Colvar/FES snapshots are sampled every 80 MD steps at 0.5 fs/step:
# 40 fs per point = 0.04 ps per point.
FRAME_TO_PS = 0.04
FES_XMIN = 0.0
FES_XMAX = 1.25
TARGET_MIN1_X = 0.0
TARGET_MIN2_X = 1.0
MIN_WINDOW_HALF_WIDTH = 0.1
PKA_YMIN = -10.0
PKA_YMAX = 20.0


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


def find_minimum_near_target(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
    half_window: float,
    xmin: float,
    xmax: float,
) -> tuple[float, float]:
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


def deltaf(block: np.ndarray) -> float:
    m1 = find_minimum_near_target(
        block[:, 0], block[:, 1], TARGET_MIN1_X, MIN_WINDOW_HALF_WIDTH, FES_XMIN, FES_XMAX
    )
    m2 = find_minimum_near_target(
        block[:, 0], block[:, 1], TARGET_MIN2_X, MIN_WINDOW_HALF_WIDTH, FES_XMIN, FES_XMAX
    )
    return m1[1] - m2[1]


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


def make_pka_grid(
    colvar_dir: Path,
    fes_dir: Path,
    out_png: Path,
    panel_title: str,
    experimental_pka: float,
) -> None:
    run_ids = discover_ids(colvar_dir, fes_dir)
    if len(run_ids) < 20:
        raise ValueError(f"Expected at least 20 matching runs in {colvar_dir} and {fes_dir}, found {len(run_ids)}")

    plt.style.use(STYLE_FILE)
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), dpi=220, sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for i, run_id in enumerate(run_ids[:20]):
        ax = axes_flat[i]
        colvar_file = colvar_dir / f"{run_id}_colvar.dat"
        fes_file = fes_dir / f"{run_id}_fes.dat"

        colvar = parse_colvar_file(colvar_file)
        blocks = load_all_fes_blocks(fes_file)
        if colvar.size == 0 or not blocks:
            ax.set_axis_off()
            continue

        ng_vals = np.array([ng for ng, _ in blocks], dtype=int)
        df_vals = np.array([deltaf(arr) for _, arr in blocks], dtype=float)
        pka_vals = df_vals / (PKA_FACTOR * TEMPERATURE_K)

        t_col = np.arange(colvar.size, dtype=float) * FRAME_TO_PS
        ng_idx = np.clip(ng_vals - 1, 0, t_col.size - 1)
        t_pka = t_col[ng_idx]
        t_plot_max = 15.0
        keep = t_pka <= t_plot_max
        t_pka = t_pka[keep]
        pka_vals = pka_vals[keep]
        if t_pka.size == 0:
            ax.set_axis_off()
            continue

        # Report pKa at the maximum displayed timestamp (12 ps), not simply
        # the dataset's final value.
        if np.any(np.isclose(t_pka, t_plot_max)):
            pka_at_tmax = float(pka_vals[np.where(np.isclose(t_pka, t_plot_max))[0][-1]])
        elif t_pka.size >= 2 and t_pka[0] <= t_plot_max <= t_pka[-1]:
            pka_at_tmax = float(np.interp(t_plot_max, t_pka, pka_vals))
        else:
            # If a run does not reach 12 ps, fall back to last available point.
            pka_at_tmax = float(pka_vals[-1])

        # Match recent TEA pKa aesthetics:
        # black line + yellow scatter, blue dashed experimental line.
        ax.plot(t_pka, pka_vals, color="black", lw=2.2, zorder=2)
        ax.scatter(
            t_pka,
            pka_vals,
            color="#FFA500",
            edgecolor=(0.0, 0.0, 0.0, 0.35),
            linewidth=0.4,
            s=28,
            zorder=3,
        )
        ax.axhline(experimental_pka, color="#0000FF", lw=2.0, ls="--", label=f"exp={experimental_pka:.2f}")
        ax.plot([], [], " ", label=f"pKa@{t_plot_max:.0f}ps={pka_at_tmax:.2f}")

        ax.set_title(f"run-{run_id}", fontsize=11)
        ax.set_ylim(PKA_YMIN, PKA_YMAX)
        ax.set_xlim(0.0, 15.0)
        ax.grid(True, which="both", alpha=0.35, linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.legend(loc="lower right", fontsize=7, frameon=False, handlelength=1.6)

    for ax in axes_flat[len(run_ids[:20]):]:
        ax.set_axis_off()

    for r in range(4):
        axes[r, 0].set_ylabel(r"$pK_a$", fontsize=10)
    for c in range(5):
        axes[3, c].set_xlabel(r"$t$ (ps)", fontsize=10)

    fig.suptitle(panel_title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    make_pka_grid(
        ND_COLVAR_DIR,
        ND_FES_DIR,
        ND_OUT,
        "HIST single_ND_coord: 20-run pKa time series",
        experimental_pka=7.1,
    )
    make_pka_grid(
        NE_COLVAR_DIR,
        NE_FES_DIR,
        NE_OUT,
        "HIST single_NE_coord: 20-run pKa time series",
        experimental_pka=6.5,
    )


if __name__ == "__main__":
    main()
