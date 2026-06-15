#!/usr/bin/env python3
"""Plot two FES curves (two systems, two timestamps) on one figure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from notes_style import CYCLE, boxed, use_style

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
EH_TO_KCALMOL = 627.509474
PKA_FACTOR = 0.004576
PKA_TEMP = 300.0
PKA_MIN1_X = 0.0
PKA_MIN2_X = 1.0
PKA_HALF_WINDOW = 0.1
PKA_FES_XMIN = 0.0
PKA_FES_XMAX = 1.25


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


def infer_system_name(path: Path) -> str:
    parts = path.resolve().parts
    if "systems" in parts:
        idx = parts.index("systems")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.name


def load_fes_at_time(run_path: Path, biaspot_name: str, fes_name: str, time_ps: float) -> Tuple[float, np.ndarray]:
    biaspot_path = run_path / biaspot_name
    fes_path = run_path / fes_name
    if not biaspot_path.exists():
        raise SystemExit(f"Missing biaspot: {biaspot_path}")
    if not fes_path.exists():
        raise SystemExit(f"Missing fes.dat: {fes_path}")

    times = parse_biaspot_times_ps(biaspot_path)
    blocks = read_fes_blocks(fes_path)
    if times.size == 0 or not blocks:
        raise SystemExit(f"No FES blocks found in {fes_path}")

    nblocks = min(len(times), len(blocks))
    times = times[:nblocks]
    blocks = blocks[:nblocks]
    idx = int(np.argmin(np.abs(times - time_ps)))
    data = blocks[idx].copy()
    data[:, 1] *= EH_TO_KCALMOL
    return float(times[idx]), data


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
    p = argparse.ArgumentParser(description="Plot two FES curves from two systems/time points.")
    p.add_argument("--sys1-path", required=True, type=Path, help="Path containing biaspot + fes.dat")
    p.add_argument("--sys1-time", required=True, type=float, help="Target time (ps) for system 1")
    p.add_argument("--sys2-path", required=True, type=Path, help="Path containing biaspot + fes.dat")
    p.add_argument("--sys2-time", required=True, type=float, help="Target time (ps) for system 2")
    p.add_argument("--biaspot-name", type=str, default="biaspot")
    p.add_argument("--fes-name", type=str, default="fes.dat")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    use_style()

    t1, fes1 = load_fes_at_time(args.sys1_path, args.biaspot_name, args.fes_name, args.sys1_time)
    t2, fes2 = load_fes_at_time(args.sys2_path, args.biaspot_name, args.fes_name, args.sys2_time)

    fes1[:, 1] -= float(np.min(fes1[:, 1]))
    fes2[:, 1] -= float(np.min(fes2[:, 1]))

    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=300)
    boxed(ax)

    sys1_name = infer_system_name(args.sys1_path)
    sys2_name = infer_system_name(args.sys2_path)

    c_blue = CYCLE[0] if CYCLE else "C0"
    c_orange = CYCLE[1] if len(CYCLE) > 1 else "C1"

    df_orange = deltaf(fes2, PKA_MIN1_X, PKA_MIN2_X, PKA_HALF_WINDOW, PKA_FES_XMIN, PKA_FES_XMAX)
    pka_orange = df_orange / (PKA_FACTOR * PKA_TEMP)
    df_blue = deltaf(fes1, PKA_MIN1_X, PKA_MIN2_X, PKA_HALF_WINDOW, PKA_FES_XMIN, PKA_FES_XMAX)
    pka_blue = df_blue / (PKA_FACTOR * PKA_TEMP)
    delta_pka = pka_blue - pka_orange

    line_orange = ax.plot(
        fes2[:, 0],
        fes2[:, 1],
        color=c_orange,
        lw=1.4,
        label=fr"$N_{{\delta}}$, p$K_a$={pka_orange:.2f}",
    )[0]
    line_blue = ax.plot(
        fes1[:, 0],
        fes1[:, 1],
        color=c_blue,
        lw=1.4,
        label=fr"$N_{{\epsilon}}$, p$K_a$={pka_blue:.2f}",
    )[0]

    ax.set_xlabel("s")
    ax.set_ylabel("F(s) [kcal/mol]")
    ax.set_xlim(0.0, 1.0)
    delta_handle = plt.Line2D([], [], color="none")
    delta_label = fr"$\Delta pK_a$={delta_pka:.2f}"
    ax.legend(
        handles=[line_orange, line_blue, delta_handle],
        labels=[line_orange.get_label(), line_blue.get_label(), delta_label],
        frameon=False,
        fontsize=8,
        loc="upper left",
        handlelength=1.6,
    )

    out = args.out
    if out is None:
        out = Path("reports") / f"fes_compare_{sys1_name}_{sys2_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
