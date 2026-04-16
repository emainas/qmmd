#!/usr/bin/env python3
"""Make a GIF of FES curves over time for selected (run,time) pairs in a grid."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
EH_TO_KCALMOL = 627.509474


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


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def main() -> None:
    p = argparse.ArgumentParser(description="GIF of FES curves for selected pairs.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--pairs-script", type=Path, default=Path("plotting/plot_fes_pairs.py"))
    p.add_argument("--fes-name", default="fes.dat")
    p.add_argument("--biaspot-name", default="biaspot")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--step", type=int, default=1, help="Use every Nth frame")
    args = p.parse_args()

    pairs = load_pairs(args.pairs_script)
    if not pairs:
        raise SystemExit("PAIRS is empty in the pairs script")

    if args.style.exists():
        plt.style.use(args.style)

    # preload blocks/time for each pair
    series = []
    max_frames = 0
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    for run_id, time_ps in pairs:
        run_dir = args.runs_path / f"run-{run_id}" / args.cv_dir
        biaspot = run_dir / args.biaspot_name
        fes = run_dir / args.fes_name
        if not biaspot.exists() or not fes.exists():
            print(f"SKIP: run-{run_id} missing {args.biaspot_name} or {args.fes_name}")
            series.append((run_id, time_ps, [], []))
            continue
        times = parse_biaspot_times_ps(biaspot)
        blocks = read_fes_blocks(fes)
        n = min(len(times), len(blocks))
        times = times[:n]
        blocks = blocks[:n]
        idx = int(np.argmin(np.abs(times - time_ps)))
        times = times[: idx + 1]
        blocks = blocks[: idx + 1]
        max_frames = max(max_frames, len(blocks))
        # track global ranges for consistent axes
        if blocks:
            all_x = np.concatenate([b[:, 0] for b in blocks])
            all_y = np.concatenate([b[:, 1] for b in blocks]) * EH_TO_KCALMOL
            x_min = all_x.min() if x_min is None else min(x_min, all_x.min())
            x_max = all_x.max() if x_max is None else max(x_max, all_x.max())
            y_min = all_y.min() if y_min is None else min(y_min, all_y.min())
            y_max = all_y.max() if y_max is None else max(y_max, all_y.max())
        series.append((run_id, time_ps, times, blocks))

    rows, cols = grid_shape(len(pairs))
    out = args.out
    if out is None:
        out = Path("reports") / f"fes_pairs_{args.cv_dir}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out, mode="I", fps=args.fps) as writer:
        for frame_idx in range(0, max_frames, args.step):
            frac = frame_idx / (max_frames - 1) if max_frames > 1 else 1.0
            fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), dpi=160, sharex=True, sharey=True)
            axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, (run_id, time_ps, times, blocks) in enumerate(series):
                ax = axes_flat[i]
                if not blocks:
                    ax.set_axis_off()
                    continue
                idx = int(round(frac * (len(blocks) - 1)))
                idx = max(0, min(len(blocks) - 1, idx))
                b = blocks[idx].copy()
                b[:, 1] *= EH_TO_KCALMOL
                ax.plot(b[:, 0], b[:, 1], color="black", lw=1.2)
                t = times[idx] if idx < len(times) else None
                if t is not None:
                    ax.set_title(f"run-{run_id} @ {t:.2f} ps", fontsize=8)
                ax.grid(alpha=0.25)
                if x_min is not None and x_max is not None:
                    ax.set_xlim(x_min, x_max)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)

            for ax in axes_flat[len(pairs):]:
                ax.set_axis_off()

        fig.text(0.5, 0.04, "s", ha="center")
        fig.text(0.04, 0.5, "F (kcal mol$^{-1}$)", va="center", rotation="vertical")
        fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(image)
        plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
