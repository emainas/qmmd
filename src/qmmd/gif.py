#!/usr/bin/env python3

from __future__ import annotations

import math
import subprocess
import shutil
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64
import io
import json

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
COORD_RE = re.compile(r"Coordinate\s*=\s*([0-9Ee+\\-\\.]+)")
EH_TO_KCALMOL = 627.509474
KB_KCALMOL = 0.0019872041  # kcal/mol/K


@dataclass(frozen=True, slots=True)
class SlurmJobConfig:
    name: str
    partition: Optional[str]
    nodes: int
    ntasks: int
    mem: str
    time: str
    stdout: str
    stderr: str
    cpus_per_task: Optional[int] = None
    flags: Optional[List[str]] = None


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    job: SlurmJobConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    python: str
    env: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class PairSpec:
    run_id: int
    ceiling_time_ps: float


@dataclass(frozen=True, slots=True)
class MinimaConfig:
    with_minima: bool
    use_minima_from_end: bool
    precise: bool
    nbins: int
    temp: float
    energy_unit: str


@dataclass(frozen=True, slots=True)
class FloodConfig:
    system: str
    buffer: float
    prefix: str
    dftb_dirname: str
    bench_tag: str
    cv_dir: str
    pairs: List[PairSpec]
    biaspot_name: str
    fes_name: str
    out_dir: Path
    style: Optional[Path]
    fps: int
    step: int
    time_min: Optional[float]
    time_unit: str
    tu: str
    use_vmax_from_end: bool
    vmin: Optional[float]
    vmax: Optional[float]
    xlim: Optional[Tuple[float, float]]
    ylim: Optional[Tuple[float, float]]
    cmap: str
    xlabel: str
    ylabel: str
    label_size: int
    image_size: Optional[Tuple[float, float]]
    image_size_unit: str
    dpi: int
    enable_loop: bool
    clear_temporary_folder: bool
    temporary_folder_name: str
    contours_spacing: Optional[float]
    levels: Optional[int]
    opacity: Optional[float]
    show_sampling: bool
    sampling_color: str
    sampling_size: float
    sampling_offset: float
    basin_stats: bool
    minima: MinimaConfig
    per_run_gifs: bool
    grid_gif: bool
    grid_gif_name: Optional[str]
    mode: str
    runtime: RuntimeConfig
    slurm: Optional[SlurmConfig]


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def system_base_dir(system: str, prefix: str, buffer: float, repo_root: Path) -> Path:
    return repo_root / "systems" / system / f"{prefix}_{buffer:.1f}"


def runs_path(cfg: FloodConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg.system, cfg.prefix, cfg.buffer, repo_root) / cfg.dftb_dirname / cfg.bench_tag


def parse_biaspot_times_ps(path: Path) -> np.ndarray:
    times: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)
    return np.array(times, dtype=float)


def parse_biaspot_series(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    coords: List[float] = []
    pending_time: Optional[float] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                pending_time = float(m.group(1)) / 1000.0
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


def normalize_units_to_kcal(y: np.ndarray) -> np.ndarray:
    return y * EH_TO_KCALMOL


def maybe_plot_sampling(ax: plt.Axes, x: np.ndarray, y: np.ndarray, cv_value: float, cfg: FloodConfig) -> None:
    if not cfg.show_sampling:
        return
    if cv_value is None or not np.isfinite(cv_value):
        return
    if x.size == 0 or y.size == 0:
        return
    if cv_value < x.min() or cv_value > x.max():
        return
    idx = int(np.argmin(np.abs(x - cv_value)))
    ax.scatter(
        [x[idx]],
        [y[idx] + cfg.sampling_offset],
        s=cfg.sampling_size,
        c=cfg.sampling_color,
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
    )


def basin_time_from_minima(
    times: np.ndarray, coords: np.ndarray, minima_series: List[List[Tuple[float, float]]]
) -> Optional[Tuple[List[int], List[float]]]:
    if times.size == 0 or coords.size == 0 or not minima_series:
        return None
    n = min(times.size, coords.size, len(minima_series))
    if n < 2:
        return None
    times = times[:n]
    coords = coords[:n]
    dt = np.diff(times)
    if dt.size == 0:
        return None
    basin_ids: List[int] = []
    for mins in minima_series[:n - 1]:
        if not mins:
            basin_ids.append(-1)
            continue
        xs = np.array([m[0] for m in mins], dtype=float)
        if xs.size == 0:
            basin_ids.append(-1)
            continue
        basin_ids.append(int(np.argmin(np.abs(xs - coords[len(basin_ids)]))))
    max_id = max(basin_ids) if basin_ids else -1
    if max_id < 0:
        return None
    times_by_basin = [0.0 for _ in range(max_id + 1)]
    for i, b in enumerate(basin_ids):
        if b >= 0:
            times_by_basin[b] += float(dt[i])
    return list(range(len(times_by_basin))), times_by_basin


def time_scale_factor(unit: str) -> float:
    unit = unit.strip().lower()
    if unit in ("fs", "fsec"):
        return 1e3
    if unit in ("ps", "psec"):
        return 1.0
    if unit in ("ns", "nsec"):
        return 1e-3
    if unit in ("us", "usec"):
        return 1e-6
    if unit in ("ms", "msec"):
        return 1e-9
    if unit in ("s", "sec"):
        return 1e-12
    return 1.0


def energy_unit_to_kcal(unit: str) -> float:
    unit = unit.strip().lower()
    if unit in ("kcal/mol", "kcal"):
        return 1.0
    if unit in ("kj/mol", "kj"):
        return 0.239005736
    if unit in ("ev", "electronvolt"):
        return 23.060549
    if unit in ("au", "hartree"):
        return EH_TO_KCALMOL
    return 1.0


def local_minima_indices(y: np.ndarray) -> List[int]:
    if y.size == 0:
        return []
    mins: List[int] = []
    if y.size == 1:
        return [0]
    if y[0] <= y[1]:
        mins.append(0)
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            mins.append(i)
    if y[-1] <= y[-2]:
        mins.append(len(y) - 1)
    return mins


def bin_minima(x: np.ndarray, y: np.ndarray, nbins: int) -> List[Tuple[float, float]]:
    if nbins <= 0 or x.size == 0:
        return []
    xmin = float(x.min())
    xmax = float(x.max())
    if xmin == xmax:
        return [(xmin, float(y.min()))]
    edges = np.linspace(xmin, xmax, nbins + 1)
    bin_mins: List[Tuple[float, float]] = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (x >= lo) & (x <= hi if i == nbins - 1 else x < hi)
        if not np.any(m):
            bin_mins.append((math.nan, math.inf))
            continue
        idx = np.argmin(y[m])
        xw = x[m]
        yw = y[m]
        bin_mins.append((float(xw[idx]), float(yw[idx])))
    # local minima among bins
    mins: List[Tuple[float, float]] = []
    for i, (_, yi) in enumerate(bin_mins):
        if math.isinf(yi):
            continue
        left = bin_mins[i - 1][1] if i > 0 else math.inf
        right = bin_mins[i + 1][1] if i + 1 < len(bin_mins) else math.inf
        if yi <= left and yi <= right:
            mins.append(bin_mins[i])
    mins.sort(key=lambda t: t[1])
    return mins


def precise_minima(x: np.ndarray, y_kcal: np.ndarray, temp: float, energy_unit: str) -> List[Tuple[float, float]]:
    mins_idx = local_minima_indices(y_kcal)
    if not mins_idx:
        return []
    # gradient-descent basin assignment
    n = len(y_kcal)
    basin_for = [-1] * n
    minima_set = set(mins_idx)

    def descend(i: int) -> int:
        steps = 0
        while i not in minima_set and steps < n:
            left = i - 1 if i > 0 else i
            right = i + 1 if i + 1 < n else i
            if y_kcal[left] < y_kcal[i] and y_kcal[left] <= y_kcal[right]:
                i = left
            elif y_kcal[right] < y_kcal[i]:
                i = right
            else:
                break
            steps += 1
        return i

    for i in range(n):
        basin_for[i] = descend(i)

    unit_to_kcal = energy_unit_to_kcal(energy_unit)
    if unit_to_kcal <= 0:
        raise RuntimeError("Invalid energy unit for precise minima")
    kT_unit = (KB_KCALMOL * temp) / unit_to_kcal
    if kT_unit <= 0:
        raise RuntimeError("Invalid temperature for precise minima")

    basin_weights: Dict[int, float] = {m: 0.0 for m in mins_idx}
    for i, m in enumerate(basin_for):
        if m in basin_weights:
            y_unit = y_kcal[i] / unit_to_kcal
            basin_weights[m] += math.exp(-y_unit / kT_unit)

    mins: List[Tuple[float, float]] = []
    for m in mins_idx:
        w = basin_weights[m]
        if w <= 0:
            continue
        fmin_unit = -kT_unit * math.log(w)
        fmin_kcal = fmin_unit * unit_to_kcal
        mins.append((float(x[m]), float(fmin_kcal)))
    mins.sort(key=lambda t: t[1])
    return mins


def select_frame_indices(times: np.ndarray, time_min: Optional[float], time_max: Optional[float]) -> np.ndarray:
    if times.size == 0:
        return np.array([], dtype=int)
    mask = np.ones_like(times, dtype=bool)
    if time_min is not None:
        mask &= times >= time_min
    if time_max is not None:
        mask &= times <= time_max
    return np.where(mask)[0]


def apply_x_window(x: np.ndarray, y: np.ndarray, xlim: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if xlim is None:
        return x, y
    lo, hi = xlim
    mask = (x >= lo) & (x <= hi)
    return x[mask], y[mask]


def format_time_ps(t: float) -> str:
    s = f"{t:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def label_for_index(i: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if i < 26:
        return letters[i]
    return f"{letters[i % 26]}{i // 26}"


def figure_size(image_size: Optional[Tuple[float, float]], unit: str) -> Optional[Tuple[float, float]]:
    if image_size is None:
        return None
    unit = unit.strip().lower()
    w, h = image_size
    if unit in ("in", "inch", "inches"):
        return (w, h)
    if unit in ("cm", "centimeter", "centimeters"):
        return (w / 2.54, h / 2.54)
    return (w, h)


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def render_gif_for_pair(cfg: FloodConfig, repo_root: Path, pair: PairSpec) -> None:
    run_dir = runs_path(cfg, repo_root) / f"run-{pair.run_id}" / cfg.cv_dir
    biaspot = run_dir / cfg.biaspot_name
    fes = run_dir / cfg.fes_name
    if not biaspot.exists() or not fes.exists():
        print(f"SKIP: run-{pair.run_id} missing {cfg.biaspot_name} or {cfg.fes_name}", flush=True)
        return

    times, coords = parse_biaspot_series(biaspot)
    blocks = read_fes_blocks(fes)
    n = min(len(times), len(coords), len(blocks))
    times = times[:n]
    coords = coords[:n]
    blocks = blocks[:n]
    if n == 0:
        print(f"SKIP: run-{pair.run_id} has no frames", flush=True)
        return

    idxs = select_frame_indices(times, cfg.time_min, pair.ceiling_time_ps)
    if idxs.size == 0:
        print(f"SKIP: run-{pair.run_id} has no frames within time range", flush=True)
        return
    idxs = idxs[:: max(1, cfg.step)]
    print(f"RUN: run-{pair.run_id} frames={len(idxs)}", flush=True)

    # Apply limits for plotting
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    for i in idxs:
        b = blocks[i]
        x = b[:, 0]
        y = normalize_units_to_kcal(b[:, 1])
        x_min = x.min() if x_min is None else min(x_min, x.min())
        x_max = x.max() if x_max is None else max(x_max, x.max())
        y_min = y.min() if y_min is None else min(y_min, y.min())
        y_max = y.max() if y_max is None else max(y_max, y.max())

    if cfg.use_vmax_from_end and idxs.size > 0:
        b = blocks[idxs[-1]]
        y_end = normalize_units_to_kcal(b[:, 1])
        y_max = y_end.max()

    if cfg.xlim is not None:
        x_min, x_max = cfg.xlim
    if cfg.ylim is not None:
        y_min, y_max = cfg.ylim
    if cfg.vmin is not None:
        y_min = cfg.vmin
    if cfg.vmax is not None:
        y_max = cfg.vmax

    if x_min is None or x_max is None or y_min is None or y_max is None:
        print(f"SKIP: run-{pair.run_id} could not determine axis limits", flush=True)
        return

    if cfg.style and cfg.style.exists():
        plt.style.use(cfg.style)

    minima_from_end: Optional[List[Tuple[float, float]]] = None
    if cfg.minima.with_minima and cfg.minima.use_minima_from_end:
        b_end = blocks[idxs[-1]]
        x_end = b_end[:, 0]
        y_end = normalize_units_to_kcal(b_end[:, 1])
        xw_end, yw_end = apply_x_window(x_end, y_end, cfg.xlim)
        if cfg.minima.precise:
            minima_from_end = precise_minima(xw_end, yw_end, cfg.minima.temp, cfg.minima.energy_unit)
        else:
            minima_from_end = bin_minima(xw_end, yw_end, cfg.minima.nbins)

    out = cfg.out_dir / f"run-{pair.run_id}_tmax-{format_time_ps(pair.ceiling_time_ps)}ps.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    loop_val = 0 if cfg.enable_loop else 1
    fig_size = figure_size(cfg.image_size, cfg.image_size_unit)
    tscale = time_scale_factor(cfg.tu or cfg.time_unit)
    tunit = (cfg.tu or cfg.time_unit or "ps").strip()
    cmap = plt.get_cmap(cfg.cmap)

    temp_dir = None
    if not cfg.clear_temporary_folder:
        temp_dir = cfg.out_dir / cfg.temporary_folder_name / f"run-{pair.run_id}_tmax-{format_time_ps(pair.ceiling_time_ps)}ps"
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = cfg.out_dir / cfg.temporary_folder_name / f"run-{pair.run_id}_tmax-{format_time_ps(pair.ceiling_time_ps)}ps"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    with imageio.get_writer(out, mode="I", fps=cfg.fps, loop=loop_val) as writer:
        for j, idx in enumerate(idxs):
            b = blocks[idx]
            x = b[:, 0]
            y = normalize_units_to_kcal(b[:, 1])
            frac = j / (len(idxs) - 1) if len(idxs) > 1 else 1.0
            color = cmap(frac)

            fig, ax = plt.subplots(figsize=fig_size, dpi=cfg.dpi)
            ax.plot(x, y, color=color, lw=2.0)
            maybe_plot_sampling(ax, x, y, coords[idx], cfg)
            ax.set_xlim(x_min, x_max)
            pad = 0.04 * (y_max - y_min) if y_max > y_min else 0.5
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(alpha=0.25)
            ax.set_xlabel(cfg.xlabel, fontsize=cfg.label_size)
            ax.set_ylabel(cfg.ylabel, fontsize=cfg.label_size)
            t_scaled = times[idx] * tscale
            ax.set_title(f"run-{pair.run_id}  t = {t_scaled:.3f} {tunit}", fontsize=cfg.label_size)

            if cfg.minima.with_minima:
                if minima_from_end is not None:
                    minima = minima_from_end
                else:
                    xw, yw = apply_x_window(x, y, cfg.xlim)
                    if cfg.minima.precise:
                        minima = precise_minima(xw, yw, cfg.minima.temp, cfg.minima.energy_unit)
                    else:
                        minima = bin_minima(xw, yw, cfg.minima.nbins)
                for k, (mx, my) in enumerate(minima):
                    # snap y to curve if using minima from end
                    if minima_from_end is not None:
                        jx = int(np.argmin(np.abs(x - mx)))
                        my = float(y[jx])
                    ax.text(mx, my, label_for_index(k), color="black", fontsize=cfg.label_size, ha="center", va="bottom")

            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(image)
            if temp_dir is not None and not cfg.clear_temporary_folder:
                frame_path = temp_dir / f"frame_{j:04d}.png"
                fig.savefig(frame_path, dpi=cfg.dpi)
            plt.close(fig)

    print(f"Wrote {out}", flush=True)


def write_html_slider(out: Path, frames: List[bytes], times: List[float], tunit: str, title: str) -> None:
    data_uris = [f"data:image/png;base64,{base64.b64encode(f).decode('ascii')}" for f in frames]
    time_labels = [f"{t:.3f}" for t in times]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
    .controls {{ display: flex; align-items: center; gap: 12px; margin-top: 12px; }}
    input[type=range] {{ flex: 1; }}
    .label {{ min-width: 140px; text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>{title}</h2>
    <img id="frame" src="{data_uris[0]}" alt="frame">
    <div class="controls">
      <input id="slider" type="range" min="0" max="{len(data_uris) - 1}" value="0" step="1">
      <div class="label" id="tlabel">t = {time_labels[0]} {tunit}</div>
    </div>
  </div>
  <script>
    const frames = {json.dumps(data_uris)};
    const times = {json.dumps(time_labels)};
    const tunit = "{tunit}";
    const img = document.getElementById("frame");
    const slider = document.getElementById("slider");
    const label = document.getElementById("tlabel");
    slider.addEventListener("input", () => {{
      const i = Number(slider.value);
      img.src = frames[i];
      label.textContent = `t = ${{times[i]}} ${{tunit}}`;
    }});
  </script>
</body>
</html>
"""
    out.write_text(html)
    print(f"Wrote {out}", flush=True)


def render_html_for_pair(cfg: FloodConfig, repo_root: Path, pair: PairSpec) -> None:
    run_dir = runs_path(cfg, repo_root) / f"run-{pair.run_id}" / cfg.cv_dir
    biaspot = run_dir / cfg.biaspot_name
    fes = run_dir / cfg.fes_name
    if not biaspot.exists() or not fes.exists():
        print(f"SKIP: run-{pair.run_id} missing {cfg.biaspot_name} or {cfg.fes_name}", flush=True)
        return

    times, coords = parse_biaspot_series(biaspot)
    blocks = read_fes_blocks(fes)
    n = min(len(times), len(coords), len(blocks))
    times = times[:n]
    coords = coords[:n]
    blocks = blocks[:n]
    if n == 0:
        print(f"SKIP: run-{pair.run_id} has no frames", flush=True)
        return

    idxs = select_frame_indices(times, cfg.time_min, pair.ceiling_time_ps)
    if idxs.size == 0:
        print(f"SKIP: run-{pair.run_id} has no frames within time range", flush=True)
        return
    idxs = idxs[:: max(1, cfg.step)]
    print(f"RUN: run-{pair.run_id} frames={len(idxs)}", flush=True)

    x_min = None
    x_max = None
    y_min = None
    y_max = None
    for i in idxs:
        b = blocks[i]
        x = b[:, 0]
        y = normalize_units_to_kcal(b[:, 1])
        x_min = x.min() if x_min is None else min(x_min, x.min())
        x_max = x.max() if x_max is None else max(x_max, x.max())
        y_min = y.min() if y_min is None else min(y_min, y.min())
        y_max = y.max() if y_max is None else max(y_max, y.max())

    if cfg.use_vmax_from_end and idxs.size > 0:
        b = blocks[idxs[-1]]
        y_end = normalize_units_to_kcal(b[:, 1])
        y_max = y_end.max()

    if cfg.xlim is not None:
        x_min, x_max = cfg.xlim
    if cfg.ylim is not None:
        y_min, y_max = cfg.ylim
    if cfg.vmin is not None:
        y_min = cfg.vmin
    if cfg.vmax is not None:
        y_max = cfg.vmax

    if x_min is None or x_max is None or y_min is None or y_max is None:
        print(f"SKIP: run-{pair.run_id} could not determine axis limits", flush=True)
        return

    if cfg.style and cfg.style.exists():
        plt.style.use(cfg.style)

    minima_from_end: Optional[List[Tuple[float, float]]] = None
    if cfg.minima.with_minima and cfg.minima.use_minima_from_end:
        b_end = blocks[idxs[-1]]
        x_end = b_end[:, 0]
        y_end = normalize_units_to_kcal(b_end[:, 1])
        xw_end, yw_end = apply_x_window(x_end, y_end, cfg.xlim)
        if cfg.minima.precise:
            minima_from_end = precise_minima(xw_end, yw_end, cfg.minima.temp, cfg.minima.energy_unit)
        else:
            minima_from_end = bin_minima(xw_end, yw_end, cfg.minima.nbins)

    out = cfg.out_dir / f"run-{pair.run_id}_tmax-{format_time_ps(pair.ceiling_time_ps)}ps.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig_size = figure_size(cfg.image_size, cfg.image_size_unit)
    tscale = time_scale_factor(cfg.tu or cfg.time_unit)
    tunit = (cfg.tu or cfg.time_unit or "ps").strip()
    cmap = plt.get_cmap(cfg.cmap)

    frames: List[bytes] = []
    frame_times: List[float] = []
    for j, idx in enumerate(idxs):
        b = blocks[idx]
        x = b[:, 0]
        y = normalize_units_to_kcal(b[:, 1])
        frac = j / (len(idxs) - 1) if len(idxs) > 1 else 1.0
        color = cmap(frac)

        fig, ax = plt.subplots(figsize=fig_size, dpi=cfg.dpi)
        ax.plot(x, y, color=color, lw=2.0)
        maybe_plot_sampling(ax, x, y, coords[idx], cfg)
        ax.set_xlim(x_min, x_max)
        pad = 0.04 * (y_max - y_min) if y_max > y_min else 0.5
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(alpha=0.25)
        ax.set_xlabel(cfg.xlabel, fontsize=cfg.label_size)
        ax.set_ylabel(cfg.ylabel, fontsize=cfg.label_size)
        t_scaled = times[idx] * tscale
        ax.set_title(f"run-{pair.run_id}  t = {t_scaled:.3f} {tunit}", fontsize=cfg.label_size)

        if cfg.minima.with_minima:
            if minima_from_end is not None:
                minima = minima_from_end
            else:
                xw, yw = apply_x_window(x, y, cfg.xlim)
                if cfg.minima.precise:
                    minima = precise_minima(xw, yw, cfg.minima.temp, cfg.minima.energy_unit)
                else:
                    minima = bin_minima(xw, yw, cfg.minima.nbins)
            for k, (mx, my) in enumerate(minima):
                if minima_from_end is not None:
                    jx = int(np.argmin(np.abs(x - mx)))
                    my = float(y[jx])
                ax.text(mx, my, label_for_index(k), color="black", fontsize=cfg.label_size, ha="center", va="bottom")

        fig.tight_layout()
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=cfg.dpi)
        frames.append(buf.getvalue())
        frame_times.append(t_scaled)
        plt.close(fig)

    write_html_slider(out, frames, frame_times, tunit, f"run-{pair.run_id}")


def minima_for_pair(cfg: FloodConfig, repo_root: Path, pair: PairSpec) -> Optional[List[Tuple[float, float]]]:
    run_dir = runs_path(cfg, repo_root) / f"run-{pair.run_id}" / cfg.cv_dir
    biaspot = run_dir / cfg.biaspot_name
    fes = run_dir / cfg.fes_name
    if not biaspot.exists() or not fes.exists():
        return None

    times, _ = parse_biaspot_series(biaspot)
    blocks = read_fes_blocks(fes)
    n = min(len(times), len(blocks))
    times = times[:n]
    blocks = blocks[:n]
    if n == 0:
        return None

    idxs = select_frame_indices(times, cfg.time_min, pair.ceiling_time_ps)
    if idxs.size == 0:
        return None
    idxs = idxs[:: max(1, cfg.step)]
    idx = idxs[-1]

    b = blocks[idx]
    x = b[:, 0]
    y = normalize_units_to_kcal(b[:, 1])
    xw, yw = apply_x_window(x, y, cfg.xlim)
    if cfg.minima.precise:
        return precise_minima(xw, yw, cfg.minima.temp, cfg.minima.energy_unit)
    return bin_minima(xw, yw, cfg.minima.nbins)


def minima_series_for_pair(
    cfg: FloodConfig, repo_root: Path, pair: PairSpec
) -> Optional[Tuple[np.ndarray, List[List[Tuple[float, float]]]]]:
    run_dir = runs_path(cfg, repo_root) / f"run-{pair.run_id}" / cfg.cv_dir
    biaspot = run_dir / cfg.biaspot_name
    fes = run_dir / cfg.fes_name
    if not biaspot.exists() or not fes.exists():
        return None

    times, _ = parse_biaspot_series(biaspot)
    blocks = read_fes_blocks(fes)
    n = min(len(times), len(blocks))
    times = times[:n]
    blocks = blocks[:n]
    if n == 0:
        return None

    idxs = select_frame_indices(times, cfg.time_min, pair.ceiling_time_ps)
    if idxs.size == 0:
        return None
    idxs = idxs[:: max(1, cfg.step)]

    minima_series: List[List[Tuple[float, float]]] = []
    for idx in idxs:
        b = blocks[idx]
        x = b[:, 0]
        y = normalize_units_to_kcal(b[:, 1])
        xw, yw = apply_x_window(x, y, cfg.xlim)
        if cfg.minima.precise:
            mins = precise_minima(xw, yw, cfg.minima.temp, cfg.minima.energy_unit)
        else:
            mins = bin_minima(xw, yw, cfg.minima.nbins)
        minima_series.append(mins)

    tscale = time_scale_factor(cfg.tu or cfg.time_unit)
    tunit_times = times[idxs] * tscale
    return tunit_times, minima_series


def write_basin_stats(out_path: Path, basin_ids: List[int], times_by_basin: List[float], tunit: str) -> None:
    lines = ["basin_id,time_" + tunit]
    for b, t in zip(basin_ids, times_by_basin):
        lines.append(f"{b},{t:.6f}")
    out_path.write_text("\n".join(lines) + "\n")


def prepare_run_series(
    cfg: FloodConfig, repo_root: Path, pair: PairSpec
) -> Optional[Tuple[int, np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]]:
    run_dir = runs_path(cfg, repo_root) / f"run-{pair.run_id}" / cfg.cv_dir
    biaspot = run_dir / cfg.biaspot_name
    fes = run_dir / cfg.fes_name
    if not biaspot.exists() or not fes.exists():
        return None

    times, coords = parse_biaspot_series(biaspot)
    blocks = read_fes_blocks(fes)
    n = min(len(times), len(coords), len(blocks))
    times = times[:n]
    coords = coords[:n]
    blocks = blocks[:n]
    if n == 0:
        return None

    idxs = select_frame_indices(times, cfg.time_min, pair.ceiling_time_ps)
    if idxs.size == 0:
        return None
    idxs = idxs[:: max(1, cfg.step)]
    return pair.run_id, times, coords, blocks, idxs


def render_grid_gif(cfg: FloodConfig, repo_root: Path, pairs: List[PairSpec]) -> None:
    series: List[Tuple[int, np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]] = []
    for pair in pairs:
        item = prepare_run_series(cfg, repo_root, pair)
        if item is not None:
            series.append(item)
        else:
            print(f"SKIP: run-{pair.run_id} missing data for grid GIF")

    if not series:
        print("SKIP: no runs available for grid GIF")
        return

    rows, cols = grid_shape(len(series))
    max_frames = max(len(idxs) for _, _, _, _, idxs in series)

    x_min = None
    x_max = None
    y_min = None
    y_max = None
    for _, _, _, blocks, idxs in series:
        for i in idxs:
            b = blocks[i]
            x = b[:, 0]
            y = normalize_units_to_kcal(b[:, 1])
            x_min = x.min() if x_min is None else min(x_min, x.min())
            x_max = x.max() if x_max is None else max(x_max, x.max())
            y_min = y.min() if y_min is None else min(y_min, y.min())
            y_max = y.max() if y_max is None else max(y_max, y.max())

    if cfg.use_vmax_from_end:
        for _, _, _, blocks, idxs in series:
            if len(idxs) == 0:
                continue
            b_end = blocks[idxs[-1]]
            y_end = normalize_units_to_kcal(b_end[:, 1])
            y_max = y_end.max() if y_max is None else max(y_max, y_end.max())

    if cfg.xlim is not None:
        x_min, x_max = cfg.xlim
    if cfg.ylim is not None:
        y_min, y_max = cfg.ylim
    if cfg.vmin is not None:
        y_min = cfg.vmin
    if cfg.vmax is not None:
        y_max = cfg.vmax

    if x_min is None or x_max is None or y_min is None or y_max is None:
        print("SKIP: grid GIF could not determine axis limits")
        return

    if cfg.style and cfg.style.exists():
        plt.style.use(cfg.style)

    minima_from_end: Dict[int, List[Tuple[float, float]]] = {}
    if cfg.minima.with_minima and cfg.minima.use_minima_from_end:
        for run_id, _, _, blocks, idxs in series:
            if len(idxs) == 0:
                continue
            b_end = blocks[idxs[-1]]
            x_end = b_end[:, 0]
            y_end = normalize_units_to_kcal(b_end[:, 1])
            xw_end, yw_end = apply_x_window(x_end, y_end, cfg.xlim)
            if cfg.minima.precise:
                minima_from_end[run_id] = precise_minima(xw_end, yw_end, cfg.minima.temp, cfg.minima.energy_unit)
            else:
                minima_from_end[run_id] = bin_minima(xw_end, yw_end, cfg.minima.nbins)

    if cfg.grid_gif_name:
        out = cfg.out_dir / cfg.grid_gif_name
    else:
        out = cfg.out_dir / "flooding_grid.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    loop_val = 0 if cfg.enable_loop else 1
    fig_size = figure_size(cfg.image_size, cfg.image_size_unit)
    tscale = time_scale_factor(cfg.tu or cfg.time_unit)
    tunit = (cfg.tu or cfg.time_unit or "ps").strip()
    cmap = plt.get_cmap(cfg.cmap)

    with imageio.get_writer(out, mode="I", fps=cfg.fps, loop=loop_val) as writer:
        for frame_idx in range(max_frames):
            fig, axes = plt.subplots(rows, cols, figsize=fig_size or (3.6 * cols, 2.8 * rows), dpi=cfg.dpi, sharex=True, sharey=True)
            axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, (run_id, times, coords, blocks, idxs) in enumerate(series):
                ax = axes_flat[i]
                if len(idxs) == 0:
                    ax.set_axis_off()
                    continue
                frac = frame_idx / (max_frames - 1) if max_frames > 1 else 1.0
                idx = int(round(frac * (len(idxs) - 1)))
                idx = max(0, min(len(idxs) - 1, idx))
                block_idx = idxs[idx]
                b = blocks[block_idx]
                x = b[:, 0]
                y = normalize_units_to_kcal(b[:, 1])
                color = cmap(frac)

                ax.plot(x, y, color=color, lw=1.6)
                maybe_plot_sampling(ax, x, y, coords[block_idx], cfg)
                ax.set_xlim(x_min, x_max)
                pad = 0.04 * (y_max - y_min) if y_max > y_min else 0.5
                ax.set_ylim(y_min - pad, y_max + pad)
                ax.grid(alpha=0.25)
                t_scaled = times[block_idx] * tscale
                ax.set_title(f"run-{run_id}  t = {t_scaled:.2f} {tunit}", fontsize=cfg.label_size)

                if cfg.minima.with_minima:
                    if run_id in minima_from_end:
                        minima = minima_from_end[run_id]
                    else:
                        xw, yw = apply_x_window(x, y, cfg.xlim)
                        if cfg.minima.precise:
                            minima = precise_minima(xw, yw, cfg.minima.temp, cfg.minima.energy_unit)
                        else:
                            minima = bin_minima(xw, yw, cfg.minima.nbins)
                    for k, (mx, my) in enumerate(minima):
                        if run_id in minima_from_end:
                            jx = int(np.argmin(np.abs(x - mx)))
                            my = float(y[jx])
                        ax.text(mx, my, label_for_index(k), color="black", fontsize=max(6, cfg.label_size - 2), ha="center", va="bottom")

            for ax in axes_flat[len(series):]:
                ax.set_axis_off()

            fig.text(0.5, 0.04, cfg.xlabel, ha="center", fontsize=cfg.label_size)
            fig.text(0.04, 0.5, cfg.ylabel, va="center", rotation="vertical", fontsize=cfg.label_size)
            fig.tight_layout(rect=[0.06, 0.06, 0.98, 0.98])
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(image)
            plt.close(fig)

    print(f"Wrote {out}")


def render_grid_html(cfg: FloodConfig, repo_root: Path, pairs: List[PairSpec]) -> None:
    series: List[Tuple[int, np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]] = []
    for pair in pairs:
        item = prepare_run_series(cfg, repo_root, pair)
        if item is not None:
            series.append(item)
        else:
            print(f"SKIP: run-{pair.run_id} missing data for grid HTML")

    if not series:
        print("SKIP: no runs available for grid HTML")
        return

    rows, cols = grid_shape(len(series))
    max_frames = max(len(idxs) for _, _, _, _, idxs in series)

    x_min = None
    x_max = None
    y_min = None
    y_max = None
    for _, _, _, blocks, idxs in series:
        for i in idxs:
            b = blocks[i]
            x = b[:, 0]
            y = normalize_units_to_kcal(b[:, 1])
            x_min = x.min() if x_min is None else min(x_min, x.min())
            x_max = x.max() if x_max is None else max(x_max, x.max())
            y_min = y.min() if y_min is None else min(y_min, y.min())
            y_max = y.max() if y_max is None else max(y_max, y.max())

    if cfg.use_vmax_from_end:
        for _, _, _, blocks, idxs in series:
            if len(idxs) == 0:
                continue
            b_end = blocks[idxs[-1]]
            y_end = normalize_units_to_kcal(b_end[:, 1])
            y_max = y_end.max() if y_max is None else max(y_max, y_end.max())

    if cfg.xlim is not None:
        x_min, x_max = cfg.xlim
    if cfg.ylim is not None:
        y_min, y_max = cfg.ylim
    if cfg.vmin is not None:
        y_min = cfg.vmin
    if cfg.vmax is not None:
        y_max = cfg.vmax

    if x_min is None or x_max is None or y_min is None or y_max is None:
        print("SKIP: grid HTML could not determine axis limits")
        return

    if cfg.style and cfg.style.exists():
        plt.style.use(cfg.style)

    minima_from_end: Dict[int, List[Tuple[float, float]]] = {}
    if cfg.minima.with_minima and cfg.minima.use_minima_from_end:
        for run_id, _, _, blocks, idxs in series:
            if len(idxs) == 0:
                continue
            b_end = blocks[idxs[-1]]
            x_end = b_end[:, 0]
            y_end = normalize_units_to_kcal(b_end[:, 1])
            xw_end, yw_end = apply_x_window(x_end, y_end, cfg.xlim)
            if cfg.minima.precise:
                minima_from_end[run_id] = precise_minima(xw_end, yw_end, cfg.minima.temp, cfg.minima.energy_unit)
            else:
                minima_from_end[run_id] = bin_minima(xw_end, yw_end, cfg.minima.nbins)

    if cfg.grid_gif_name:
        out = cfg.out_dir / cfg.grid_gif_name
        out = out.with_suffix(".html")
    else:
        out = cfg.out_dir / "flooding_grid.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig_size = figure_size(cfg.image_size, cfg.image_size_unit)
    tscale = time_scale_factor(cfg.tu or cfg.time_unit)
    tunit = (cfg.tu or cfg.time_unit or "ps").strip()
    cmap = plt.get_cmap(cfg.cmap)

    frames: List[bytes] = []
    frame_times: List[float] = []
    base_run_id, base_times, _, _, base_idxs = series[0]

    for frame_idx in range(max_frames):
        fig, axes = plt.subplots(rows, cols, figsize=fig_size or (3.6 * cols, 2.8 * rows), dpi=cfg.dpi, sharex=True, sharey=True)
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (run_id, times, coords, blocks, idxs) in enumerate(series):
            ax = axes_flat[i]
            if len(idxs) == 0:
                ax.set_axis_off()
                continue
            frac = frame_idx / (max_frames - 1) if max_frames > 1 else 1.0
            idx = int(round(frac * (len(idxs) - 1)))
            idx = max(0, min(len(idxs) - 1, idx))
            block_idx = idxs[idx]
            b = blocks[block_idx]
            x = b[:, 0]
            y = normalize_units_to_kcal(b[:, 1])
            color = cmap(frac)

            ax.plot(x, y, color=color, lw=1.6)
            maybe_plot_sampling(ax, x, y, coords[block_idx], cfg)
            ax.set_xlim(x_min, x_max)
            pad = 0.04 * (y_max - y_min) if y_max > y_min else 0.5
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(alpha=0.25)
            t_scaled = times[block_idx] * tscale
            ax.set_title(f"run-{run_id}  t = {t_scaled:.2f} {tunit}", fontsize=cfg.label_size)

            if cfg.minima.with_minima:
                if run_id in minima_from_end:
                    minima = minima_from_end[run_id]
                else:
                    xw, yw = apply_x_window(x, y, cfg.xlim)
                    if cfg.minima.precise:
                        minima = precise_minima(xw, yw, cfg.minima.temp, cfg.minima.energy_unit)
                    else:
                        minima = bin_minima(xw, yw, cfg.minima.nbins)
                for k, (mx, my) in enumerate(minima):
                    if run_id in minima_from_end:
                        jx = int(np.argmin(np.abs(x - mx)))
                        my = float(y[jx])
                    ax.text(mx, my, label_for_index(k), color="black", fontsize=max(6, cfg.label_size - 2), ha="center", va="bottom")

        for ax in axes_flat[len(series):]:
            ax.set_axis_off()

        fig.text(0.5, 0.04, cfg.xlabel, ha="center", fontsize=cfg.label_size)
        fig.text(0.04, 0.5, cfg.ylabel, va="center", rotation="vertical", fontsize=cfg.label_size)
        fig.tight_layout(rect=[0.06, 0.06, 0.98, 0.98])
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=cfg.dpi)
        frames.append(buf.getvalue())

        frac = frame_idx / (max_frames - 1) if max_frames > 1 else 1.0
        base_idx = int(round(frac * (len(base_idxs) - 1)))
        base_idx = max(0, min(len(base_idxs) - 1, base_idx))
        t_scaled = base_times[base_idxs[base_idx]] * tscale
        frame_times.append(t_scaled)
        plt.close(fig)

    write_html_slider(out, frames, frame_times, tunit, "flooding_grid")

def load_config(path: Path) -> FloodConfig:
    data = yaml.safe_load(path.read_text())

    pairs: List[PairSpec] = []
    for item in data.get("pairs", []):
        pairs.append(PairSpec(run_id=int(item["run_id"]), ceiling_time_ps=float(item["ceiling_time_ps"])))

    min_data = data.get("minima", {})
    minima_cfg = MinimaConfig(
        with_minima=bool(min_data.get("with_minima", True)),
        use_minima_from_end=bool(min_data.get("use_minima_from_end", False)),
        precise=bool(min_data.get("precise", True)),
        nbins=int(min_data.get("nbins", 20)),
        temp=float(min_data.get("temp", 300.0)),
        energy_unit=str(min_data.get("energy_unit", "kcal/mol")),
    )

    opts = data.get("options", {})
    runtime_data = data.get("runtime", {})
    runtime_cfg = RuntimeConfig(
        python=str(runtime_data.get("python", "python")),
        env=dict(runtime_data.get("env", {})),
    )

    slurm_cfg = None
    if data.get("slurm") is not None:
        job = data["slurm"]["job"]
        slurm_cfg = SlurmConfig(
            job=SlurmJobConfig(
                name=job["name"],
                partition=job.get("partition"),
                nodes=int(job["nodes"]),
                ntasks=int(job["ntasks"]),
                mem=str(job["mem"]),
                time=str(job["time"]),
                stdout=str(job["stdout"]),
                stderr=str(job["stderr"]),
                cpus_per_task=int(job["cpus_per_task"]) if "cpus_per_task" in job else None,
                flags=list(job.get("flags", [])) if job.get("flags") is not None else None,
            )
        )

    image_size = None
    if "image_size" in opts and opts["image_size"] is not None:
        image_size = (float(opts["image_size"][0]), float(opts["image_size"][1]))
    xlim = None
    if "xlim" in opts and opts["xlim"] is not None:
        xlim = (float(opts["xlim"][0]), float(opts["xlim"][1]))
    ylim = None
    if "ylim" in opts and opts["ylim"] is not None:
        ylim = (float(opts["ylim"][0]), float(opts["ylim"][1]))

    if "out_dir" in data and data["out_dir"] is not None:
        out_dir = Path(str(data["out_dir"]))
    else:
        solv = f"{data.get('prefix', 'solv')}_{float(data['buffer']):.1f}"
        out_dir = Path("reports") / f"flooding_{data['system']}_{solv}_{data['cv_dir']}"

    return FloodConfig(
        system=str(data["system"]),
        buffer=float(data["buffer"]),
        prefix=str(data.get("prefix", "solv")),
        dftb_dirname=str(data.get("dftb_dirname", "dftb")),
        bench_tag=str(data["bench_tag"]),
        cv_dir=str(data["cv_dir"]),
        pairs=pairs,
        biaspot_name=str(data.get("biaspot_name", "biaspot")),
        fes_name=str(data.get("fes_name", "fes.dat")),
        out_dir=out_dir,
        style=Path(data["style"]) if data.get("style") else None,
        fps=int(opts.get("fps", 4)),
        step=int(opts.get("step", 1)),
        time_min=float(opts["time_min"]) if opts.get("time_min") is not None else None,
        time_unit=str(opts.get("time_unit", "ps")),
        tu=str(opts.get("tu", "ps")),
        use_vmax_from_end=bool(opts.get("use_vmax_from_end", True)),
        vmin=float(opts["vmin"]) if opts.get("vmin") is not None else None,
        vmax=float(opts["vmax"]) if opts.get("vmax") is not None else None,
        xlim=xlim,
        ylim=ylim,
        cmap=str(opts.get("cmap", "RdYlBu_r")),
        xlabel=str(opts.get("xlabel", "s")),
        ylabel=str(opts.get("ylabel", "F (kcal mol$^{-1}$)")),
        label_size=int(opts.get("label_size", 10)),
        image_size=image_size,
        image_size_unit=str(opts.get("image_size_unit", "in")),
        dpi=int(opts.get("dpi", 160)),
        enable_loop=bool(opts.get("enable_loop", True)),
        clear_temporary_folder=bool(opts.get("clear_temporary_folder", True)),
        temporary_folder_name=str(opts.get("temporary_folder_name", "temp_fes_gif")),
        contours_spacing=float(opts["contours_spacing"]) if opts.get("contours_spacing") is not None else None,
        levels=int(opts["levels"]) if opts.get("levels") is not None else None,
        opacity=float(opts["opacity"]) if opts.get("opacity") is not None else None,
        show_sampling=bool(opts.get("show_sampling", False)),
        sampling_color=str(opts.get("sampling_color", "black")),
        sampling_size=float(opts.get("sampling_size", 28.0)),
        sampling_offset=float(opts.get("sampling_offset", 0.0)),
        basin_stats=bool(opts.get("basin_stats", False)),
        minima=minima_cfg,
        per_run_gifs=bool(opts.get("per_run_gifs", True)),
        grid_gif=bool(opts.get("grid_gif", False)),
        grid_gif_name=str(opts["grid_gif_name"]) if opts.get("grid_gif_name") else None,
        mode=str(opts.get("mode", "gif")).lower(),
        runtime=runtime_cfg,
        slurm=slurm_cfg,
    )


def write_slurm_script(cfg: FloodConfig, yaml_path: Path, out_path: Path, repo_root: Path) -> None:
    if cfg.slurm is None:
        raise RuntimeError("slurm config missing in YAML")
    job = cfg.slurm.job
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export PYTHONUNBUFFERED=1",
        f"#SBATCH -J {job.name}",
    ]
    if job.partition:
        lines.append(f"#SBATCH -p {job.partition}")
    lines += [
        f"#SBATCH -N {job.nodes}",
        f"#SBATCH --ntasks {job.ntasks}",
        f"#SBATCH --mem={job.mem}",
        f"#SBATCH -t {job.time}",
        f"#SBATCH -o {job.stdout}",
        f"#SBATCH -e {job.stderr}",
    ]
    if job.cpus_per_task is not None:
        lines.append(f"#SBATCH --cpus-per-task {job.cpus_per_task}")
    if job.flags:
        for flag in job.flags:
            lines.append(f"#SBATCH {flag}")
    lines.append("")
    for k, v in cfg.runtime.env.items():
        lines.append(f"export {k}={v}")
    lines += [
        f"cd {repo_root}",
        "echo \"PWD=$(pwd)\"",
        f"echo \"PYTHON={cfg.runtime.python}\"",
        f"{cfg.runtime.python} -V",
        f"ls -l {yaml_path}",
        f"{cfg.runtime.python} -u -m qmmd.cli gif-run {yaml_path}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def submit_slurm(script: Path) -> None:
    subprocess.run(["sbatch", str(script)], check=True)


def run_gif_submit(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_name = f"gif-submit-{cfg.system}-{cfg.cv_dir}.slurm"
    script_path = scripts_dir / script_name
    write_slurm_script(cfg, yaml_path.resolve(), script_path, repo_root)
    print(f"OK: wrote {script_path}")
    resp = input(f"About to submit 1 job: {script_path}. Proceed? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        print("Cancelled by user.")
        return
    print(f"Submitting job via sbatch for {script_path}...")
    submit_slurm(script_path)
    print("OK: job submitted")


def run_gif_run(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    if not cfg.pairs:
        raise RuntimeError("pairs is empty in config")
    print(f"GIF-RUN: pairs={len(cfg.pairs)} per_run_gifs={cfg.per_run_gifs} grid_gif={cfg.grid_gif}", flush=True)
    minima_time_series: List[Tuple[int, np.ndarray, List[List[Tuple[float, float]]]]] = []
    tunit = (cfg.tu or cfg.time_unit or "ps").strip()
    for pair in cfg.pairs:
        if cfg.per_run_gifs:
            if cfg.mode == "html":
                render_html_for_pair(cfg, repo_root, pair)
            else:
                render_gif_for_pair(cfg, repo_root, pair)
        if cfg.minima.with_minima:
            series = minima_series_for_pair(cfg, repo_root, pair)
            if series:
                times, mins_series = series
                minima_time_series.append((pair.run_id, times, mins_series))
                if cfg.basin_stats:
                    run_dir = runs_path(cfg, repo_root) / f"run-{pair.run_id}" / cfg.cv_dir
                    biaspot = run_dir / cfg.biaspot_name
                    times_raw, coords = parse_biaspot_series(biaspot)
                    idxs = select_frame_indices(times_raw, cfg.time_min, pair.ceiling_time_ps)
                    if idxs.size > 0:
                        idxs = idxs[:: max(1, cfg.step)]
                        times_sel = times_raw[idxs]
                        coords_sel = coords[idxs]
                        basin = basin_time_from_minima(times_sel, coords_sel, mins_series)
                        if basin:
                            basin_ids, times_by_basin = basin
                            out_path = cfg.out_dir / f"run-{pair.run_id}_basin_times.csv"
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            write_basin_stats(out_path, basin_ids, times_by_basin, tunit)
                            print(f"Wrote {out_path}")

    if cfg.grid_gif:
        if cfg.mode == "html":
            render_grid_html(cfg, repo_root, cfg.pairs)
        else:
            render_grid_gif(cfg, repo_root, cfg.pairs)

    if cfg.minima.with_minima and minima_time_series:
        rows, cols = grid_shape(len(minima_time_series))
        cmap = plt.get_cmap("tab10")
        tunit = (cfg.tu or cfg.time_unit or "ps").strip()

        fig_pos, axes_pos = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=cfg.dpi, sharex=False, sharey=False)
        fig_fe, axes_fe = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), dpi=cfg.dpi, sharex=False, sharey=False)
        axes_pos_flat = axes_pos.flatten() if isinstance(axes_pos, np.ndarray) else [axes_pos]
        axes_fe_flat = axes_fe.flatten() if isinstance(axes_fe, np.ndarray) else [axes_fe]

        for i, (run_id, times, mins_series) in enumerate(minima_time_series):
            ax_p = axes_pos_flat[i]
            ax_f = axes_fe_flat[i]
            max_mins = max((len(m) for m in mins_series), default=0)
            for k in range(max_mins):
                xs = []
                ys_pos = []
                ys_fe = []
                for t, mins in zip(times, mins_series):
                    if k < len(mins):
                        xs.append(t)
                        ys_pos.append(mins[k][0])
                        ys_fe.append(mins[k][1])
                    else:
                        xs.append(t)
                        ys_pos.append(np.nan)
                        ys_fe.append(np.nan)
                color = cmap(k % 10)
                label = label_for_index(k)
                ax_p.plot(xs, ys_pos, color=color, lw=1.4, label=label)
                ax_f.plot(xs, ys_fe, color=color, lw=1.4, label=label)
            ax_p.set_title(f"run-{run_id}", fontsize=cfg.label_size)
            ax_f.set_title(f"run-{run_id}", fontsize=cfg.label_size)
            ax_p.grid(alpha=0.25)
            ax_f.grid(alpha=0.25)
            ax_p.tick_params(labelsize=max(6, cfg.label_size - 2))
            ax_f.tick_params(labelsize=max(6, cfg.label_size - 2))

        for ax in axes_pos_flat[len(minima_time_series):]:
            ax.set_axis_off()
        for ax in axes_fe_flat[len(minima_time_series):]:
            ax.set_axis_off()

        for r in range(rows):
            if cols > 1:
                axes_pos[r, 0].set_ylabel(cfg.xlabel, fontsize=cfg.label_size)
                axes_fe[r, 0].set_ylabel(cfg.ylabel, fontsize=cfg.label_size)
            else:
                axes_pos.set_ylabel(cfg.xlabel, fontsize=cfg.label_size)
                axes_fe.set_ylabel(cfg.ylabel, fontsize=cfg.label_size)
        for c in range(cols):
            if rows > 1:
                axes_pos[rows - 1, c].set_xlabel(f"time ({tunit})", fontsize=cfg.label_size)
                axes_fe[rows - 1, c].set_xlabel(f"time ({tunit})", fontsize=cfg.label_size)
            else:
                axes_pos.set_xlabel(f"time ({tunit})", fontsize=cfg.label_size)
                axes_fe.set_xlabel(f"time ({tunit})", fontsize=cfg.label_size)

        handles, labels = axes_pos_flat[0].get_legend_handles_labels()
        if handles:
            fig_pos.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False, fontsize=cfg.label_size)
            fig_fe.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False, fontsize=cfg.label_size)

        fig_pos.tight_layout(rect=[0, 0, 1, 0.96])
        fig_fe.tight_layout(rect=[0, 0, 1, 0.96])
        out_pos = cfg.out_dir / "minima_positions_timeseries.png"
        out_fe = cfg.out_dir / "minima_free_energy_timeseries.png"
        fig_pos.savefig(out_pos, dpi=cfg.dpi)
        fig_fe.savefig(out_fe, dpi=cfg.dpi)
        plt.close(fig_pos)
        plt.close(fig_fe)
        print(f"Wrote {out_pos}")
        print(f"Wrote {out_fe}")
