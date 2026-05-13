#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from qmmd.gif import parse_biaspot_times_ps, read_fes_blocks, normalize_units_to_kcal, find_repo_root


@dataclass(frozen=True)
class RunSpec:
    run_id: int
    t_fill: float
    t_interval: float
    t_max: Optional[float] = None
    restart_t_fill: Optional[float] = None
    restart_max_curves: Optional[int] = None


@dataclass(frozen=True)
class StationConfig:
    system: str
    buffer: float
    prefix: str
    method_dir: str
    bench_tag: Optional[str]
    cv_dir: str
    fes_name: str
    biaspot_name: str
    runs: List[RunSpec]
    out: Path
    style: Optional[Path]
    dpi: int
    label_size: int
    time_unit: str


def load_config(path: Path) -> StationConfig:
    data = yaml.safe_load(path.read_text())
    runs = []
    for item in data.get("runs", []):
        runs.append(
            RunSpec(
                run_id=int(item["run_id"]),
                t_fill=float(item["t_fill"]),
                t_interval=float(item["t_interval"]),
                t_max=float(item["t_max"]) if item.get("t_max") is not None else None,
                restart_t_fill=float(item["restart_t_fill"]) if item.get("restart_t_fill") is not None else None,
                restart_max_curves=int(item["restart_max_curves"]) if item.get("restart_max_curves") is not None else None,
            )
        )
    if not runs:
        raise RuntimeError("runs list is required")

    return StationConfig(
        system=str(data["system"]),
        buffer=float(data["buffer"]),
        prefix=str(data.get("prefix", "solv")),
        method_dir=str(data.get("method_dir", "dftb")),
        bench_tag=str(data.get("bench_tag")) if data.get("bench_tag") is not None else None,
        cv_dir=str(data.get("cv_dir", "equil")),
        fes_name=str(data.get("fes_name", "fes.dat")),
        biaspot_name=str(data.get("biaspot_name", "biaspot")),
        runs=runs,
        out=Path(str(data.get("out", "reports/station_grid.png"))),
        style=Path(data["style"]) if data.get("style") else None,
        dpi=int(data.get("dpi", 160)),
        label_size=int(data.get("label_size", 9)),
        time_unit=str(data.get("time_unit", "ps")),
    )


def runs_root(cfg: StationConfig, repo_root: Path) -> Path:
    base = repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}" / cfg.method_dir
    if cfg.bench_tag:
        base = base / cfg.bench_tag
    return base


def select_indices_aligned(
    times: np.ndarray, t_fill: float, t_interval: float, t_max: Optional[float], phase: float = 0.0
) -> List[int]:
    if t_interval <= 0:
        return []
    out: List[int] = []
    # Align targets to a global grid (phase) instead of the first available time
    start = np.ceil((t_fill - phase) / t_interval) * t_interval + phase
    if t_max is not None:
        t_end = min(float(times[-1]), float(t_max))
    else:
        t_end = float(times[-1])
    if start > t_end:
        return []
    target = start
    i = 0
    n = len(times)
    while i < n and target <= t_end:
        while i < n and times[i] < target:
            i += 1
        if i >= n:
            break
        out.append(i)
        target += t_interval
    return out


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def read_fes_blocks_complete(path: Path) -> List[np.ndarray]:
    blocks: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("### FREE ENERGY SURFACE"):
                if cur:
                    blocks.append(cur)
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
    # Do not append trailing partial block; only keep complete blocks
    return [np.array(b, dtype=float) for b in blocks]


def run_station(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    runs_base = runs_root(cfg, repo_root)

    series = []
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    global_t_min = None
    global_t_max = None

    for spec in cfg.runs:
        run_dir = runs_base / f"run-{spec.run_id}" / cfg.cv_dir
        biaspot = run_dir / cfg.biaspot_name
        fes = run_dir / cfg.fes_name
        restart_dir = run_dir / "metad-restart"
        restart_biaspot = restart_dir / cfg.biaspot_name
        restart_fes = restart_dir / cfg.fes_name
        if not biaspot.exists() or not fes.exists():
            print(f"WARN: missing {cfg.biaspot_name} or {cfg.fes_name} in {run_dir}")
            series.append((spec.run_id, [], [], []))
            continue

        orig_times = parse_biaspot_times_ps(biaspot)
        orig_blocks = read_fes_blocks(fes)

        idxs: List[int] = []
        times: np.ndarray = np.array([], dtype=float)
        blocks: List[np.ndarray] = []

        n0 = min(len(orig_times), len(orig_blocks))
        orig_times = orig_times[:n0]
        orig_blocks = orig_blocks[:n0]
        orig_idxs = select_indices_aligned(orig_times, spec.t_fill, spec.t_interval, spec.t_max)
        blocks = list(orig_blocks)
        times = orig_times
        idxs.extend(orig_idxs)

        if restart_dir.exists() and restart_fes.exists():
            restart_blocks = read_fes_blocks_complete(restart_fes)
            if len(restart_blocks) >= 2:
                restart_blocks = restart_blocks[1:]
            elif restart_blocks:
                print(f"WARN: restart fes in {restart_dir} has only 1 block; skipping append")
                restart_blocks = []
            else:
                print(f"WARN: restart fes in {restart_dir} has no complete blocks")
                restart_blocks = []

            if restart_blocks:
                if restart_biaspot.exists():
                    restart_times_full = parse_biaspot_times_ps(restart_biaspot)
                    if len(restart_times_full) >= len(restart_blocks):
                        restart_times = restart_times_full[-len(restart_blocks):]
                    else:
                        restart_times = restart_times_full
                        restart_blocks = restart_blocks[: len(restart_times)]
                else:
                    restart_times = np.array([], dtype=float)

                if restart_blocks and restart_times.size > 0:
                    start_t = float(restart_times[0])
                    restart_fill = float(spec.restart_t_fill) if spec.restart_t_fill is not None else start_t
                    r_idxs_local = select_indices_aligned(restart_times, restart_fill, spec.t_interval, spec.t_max)
                    if spec.restart_max_curves is not None:
                        r_idxs_local = r_idxs_local[: spec.restart_max_curves]
                    base = len(blocks)
                    for ridx in r_idxs_local:
                        idxs.append(base + ridx)
                    blocks.extend(restart_blocks)
                    times = np.concatenate([times, restart_times])

        n = min(len(times), len(blocks))
        times = times[:n]
        blocks = blocks[:n]
        idxs = [i for i in idxs if i < n]
        if not idxs:
            print(f"WARN: run-{spec.run_id} has no frames after t_fill")
            series.append((spec.run_id, [], [], []))
            continue

        t_sel = times[idxs]
        for i in idxs:
            b = blocks[i]
            x = b[:, 0]
            y = normalize_units_to_kcal(b[:, 1])
            x_min = x.min() if x_min is None else min(x_min, x.min())
            x_max = x.max() if x_max is None else max(x_max, x.max())
            y_min = y.min() if y_min is None else min(y_min, y.min())
            y_max = y.max() if y_max is None else max(y_max, y.max())

        t_min = float(t_sel.min())
        t_max = float(t_sel.max())
        global_t_min = t_min if global_t_min is None else min(global_t_min, t_min)
        global_t_max = t_max if global_t_max is None else max(global_t_max, t_max)
        series.append((spec.run_id, times, blocks, idxs))

    if x_min is None or x_max is None or y_min is None or y_max is None:
        raise RuntimeError("No valid FES data found")

    if cfg.style and cfg.style.exists():
        plt.style.use(cfg.style)

    rows, cols = grid_shape(len(series))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), sharex=True, sharey=True, dpi=cfg.dpi)
    fig.set_layout_engine(None)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    cmap = plt.get_cmap("viridis")

    for i, (run_id, times, blocks, idxs) in enumerate(series):
        ax = axes_list[i]
        if not idxs:
            ax.set_axis_off()
            continue
        t_sel = times[idxs]
        t0 = float(t_sel.min())
        t1 = float(t_sel.max())
        denom = (t1 - t0) if t1 > t0 else 1.0
        for j, idx in enumerate(idxs):
            b = blocks[idx]
            x = b[:, 0]
            y = normalize_units_to_kcal(b[:, 1])
            frac = (times[idx] - t0) / denom
            color = cmap(frac)
            label = f"{times[idx]:.2f} {cfg.time_unit}"
            ax.plot(x, y, color=color, lw=1.2, label=label)
        ax.set_title(f"run-{run_id} (n={len(idxs)})", fontsize=cfg.label_size)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc="lower right", fontsize=max(6, cfg.label_size - 2), frameon=False)

    for j in range(len(series), len(axes_list)):
        axes_list[j].set_axis_off()

    fig.text(0.5, 0.04, "CV", ha="center", fontsize=cfg.label_size)
    fig.text(0.04, 0.5, "F (kcal/mol)", va="center", rotation="vertical", fontsize=cfg.label_size)

    # Colorbar not used; timestamps are listed in per-panel legends.

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92, wspace=0.2, hspace=0.2)
    fig.savefig(cfg.out, dpi=cfg.dpi)
    print(f"Wrote {cfg.out}")
