#!/usr/bin/env python3
"""Precompute last FES block and minima time series for 2D metad runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
RUN_RE = re.compile(r"^run-(\d+)$")


def parse_biaspot_times(path: Path) -> List[float]:
    times: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)
    return times


def reshape_fes(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data[:, 0]
    y = data[:, 1]
    f = data[:, 2]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = len(x_unique)
    ny = len(y_unique)
    z = f.reshape(ny, nx)
    return x_unique, y_unique, z


def find_minimum_near(x: np.ndarray, y: np.ndarray, z: np.ndarray, target: Tuple[float, float], window: float) -> Tuple[float, float, float]:
    tx, ty = target
    mask = (np.abs(x[None, :] - tx) <= window) & (np.abs(y[:, None] - ty) <= window)
    if not np.any(mask):
        xi = np.argmin(np.abs(x - tx))
        yi = np.argmin(np.abs(y - ty))
        return x[xi], y[yi], z[yi, xi]
    sub = np.where(mask)
    vals = z[sub]
    idx = np.argmin(vals)
    yi = sub[0][idx]
    xi = sub[1][idx]
    return x[xi], y[yi], z[yi, xi]


def iter_fes_blocks(path: Path):
    current = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(" ### FREE ENERGY SURFACE"):
                if current:
                    yield np.array(current, dtype=float)
                    current = []
                continue
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            current.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if current:
        yield np.array(current, dtype=float)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-path", required=True, type=Path)
    ap.add_argument("--cv-dir", required=True)
    ap.add_argument("--window", type=float, default=0.1)
    args = ap.parse_args()

    runs = discover_runs(args.runs_path, args.cv_dir)
    if not runs:
        raise SystemExit("No biaspot/fes.dat pairs found.")

    targets = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]

    for run_id, biaspot, fes_path in runs:
        times = parse_biaspot_times(biaspot)
        if not times:
            print(f"WARN: no times in {biaspot}, skipping run-{run_id}")
            continue

        minima_rows: List[str] = []
        last_block = None
        nblocks = 0

        for block in iter_fes_blocks(fes_path):
            last_block = block
            nblocks += 1
            if nblocks > len(times):
                break
            x, y, z = reshape_fes(block)
            z = z - np.nanmin(z)
            m11 = find_minimum_near(x, y, z, targets[0], args.window)
            m10 = find_minimum_near(x, y, z, targets[1], args.window)
            m01 = find_minimum_near(x, y, z, targets[2], args.window)
            t = times[nblocks - 1]
            minima_rows.append(
                f"{t:.6f} {m11[0]:.6f} {m11[1]:.6f} {m11[2]:.6f} "
                f"{m10[0]:.6f} {m10[1]:.6f} {m10[2]:.6f} "
                f"{m01[0]:.6f} {m01[1]:.6f} {m01[2]:.6f}"
            )

        if last_block is None:
            print(f"WARN: no blocks in {fes_path}, skipping run-{run_id}")
            continue

        out_dir = biaspot.parent
        last_path = out_dir / "fes.last.dat"
        np.savetxt(last_path, last_block, fmt="%.10f")

        minima_path = out_dir / "minima.dat"
        with minima_path.open("w", encoding="utf-8") as f:
            f.write("# t_ps m11_x m11_y m11_f m10_x m10_y m10_f m01_x m01_y m01_f\n")
            f.write("\n".join(minima_rows) + "\n")

        print(f"run-{run_id}: wrote {last_path} and {minima_path}")


if __name__ == "__main__":
    main()
