#!/usr/bin/env python3
"""Plot CV time-series grids from biaspot files."""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
STEP_RE = re.compile(r"STEP NO\.\s*=\s*([0-9]+)")
COORD_RE = re.compile(r"Coordinate\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)")


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


def parse_biaspot(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Pair each CV with its timestamp in a fixed-size snapshot."""
    times, coords = [], []
    pending_time, pending_coord = None, None
    with path.open("rb") as handle:
        snapshot = handle.read(os.fstat(handle.fileno()).st_size).decode()
    for line in io.StringIO(snapshot):
        if not line.endswith("\n"):
            break
        match = TIME_RE.search(line)
        if match:
            if pending_time is not None:
                if pending_coord is None:
                    raise ValueError(f"Missing CV in interior bias block: {path}")
                times.append(pending_time)
                coords.append(pending_coord)
            pending_time, pending_coord = float(match[1]) / 1000., None
            continue
        match = COORD_RE.search(line)
        if match:
            if pending_time is None or pending_coord is not None:
                raise ValueError(f"Expected one CV per timed bias block: {path}")
            pending_coord = float(match[1].replace("D", "E").replace("d", "e"))
    if pending_time is not None and pending_coord is not None:
        times.append(pending_time)
        coords.append(pending_coord)
    t, y = np.asarray(times, dtype=float), np.asarray(coords, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"Nonfinite or nonmonotonic bias samples: {path}")
    return t, y


def read_last_time_ps(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                last = float(m.group(1)) / 1000.0  # fsec -> ps
    return last


def read_first_step(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "*** AT T=" not in line:
                continue
            m = STEP_RE.search(line)
            if m:
                return int(m.group(1))
            return None
    return None


def read_deltat_fs(dftb_inp: Path) -> Optional[float]:
    if not dftb_inp.exists():
        return None
    for line in dftb_inp.read_text().splitlines():
        if "DELTAT=" not in line:
            continue
        m = re.search(r"DELTAT=([0-9.eEdD+-]+)", line)
        if not m:
            continue
        val = m.group(1).replace("D", "E").replace("d", "e")
        try:
            return float(val) * 1e15  # seconds -> fs
        except ValueError:
            continue
    return None


def read_restart_time_ps(dftb_out: Path) -> Optional[float]:
    if not dftb_out.exists():
        return None
    last_time = None
    last_restart = None
    with dftb_out.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                last_time = float(m.group(1)) / 1000.0
                continue
            if "Restart information dumped to" in line:
                if last_time is not None:
                    last_restart = last_time
    return last_restart


def read_restart_step(restart_path: Path, target_step: Optional[int]) -> Optional[int]:
    if not restart_path.exists():
        return None
    data = restart_path.read_bytes()
    best = None
    best_dist = None
    for i in range(0, len(data) - 4, 4):
        val = int.from_bytes(data[i : i + 4], "little", signed=False)
        if val < 1000 or val > 2000000:
            continue
        if target_step is None:
            if best is None or val > best:
                best = val
            continue
        dist = abs(val - target_step)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = val
    return best


def infer_restart_time_ps(run_dir: Path, cv_dir: str) -> Optional[float]:
    cv_path = run_dir / cv_dir
    spec_path = None
    for name in ("wtmeta_spec.yaml", "meta_spec.yaml"):
        p = cv_path / name
        if p.exists():
            spec_path = p
            break
    if spec_path is None:
        return None

    data = yaml.safe_load(spec_path.read_text())
    replica_dirname = data.get("replica_dirname")
    if not replica_dirname:
        return None
    equil_out = run_dir / replica_dirname / "dftb.out"
    return read_last_time_ps(equil_out)


def first_trajectory_clock(path: Path) -> tuple[int, float] | None:
    """Read initial step and reported ps from the trajectory, not a live restart."""
    if not path.exists():
        return None
    with path.open() as handle:
        count, comment = handle.readline(), handle.readline()
    step, time = STEP_RE.search(comment), TIME_RE.search(comment)
    if not count.strip().isdigit() or not step or not time:
        raise ValueError(f"Invalid initial trajectory clock: {path}")
    return int(step[1]), float(time[1]) / 1000.


def infer_offset_ps(run_dir: Path, cv_dir: str, t0: float) -> float:
    """Map raw CV time to its parent trajectory clock using the initial frame.

    Explicit restart_time_ps anchors the first trajectory frame, not the first
    deposited hill. Never infer historical timing from mutable binary restarts
    or the parent's final restart dump.
    """
    cv_path = run_dir / cv_dir
    data = {}
    for name in ("wtmeta_spec.yaml", "meta_spec.yaml"):
        spec = cv_path / name
        if spec.exists():
            data = yaml.safe_load(spec.read_text()) or {}
            break
    clock = first_trajectory_clock(cv_path / "traject")
    if clock is None:
        raise ValueError(f"No trajectory-start clock in {cv_path}; use --time-axis raw")
    step, raw_start = clock
    if raw_start > t0 + 1.e-8:
        raise ValueError(f"Trajectory starts after first CV sample in {cv_path}")
    explicit = data.get("restart_time_ps")
    restart_time_txt = cv_path / "restart_time_ps.txt"
    if explicit is None and restart_time_txt.exists():
        explicit = restart_time_txt.read_text().strip()
    if explicit is not None:
        value = float(explicit)
        if not np.isfinite(value):
            raise ValueError("restart_time_ps must be finite")
        return value - raw_start
    parent = run_dir / data.get("replica_dirname", "equil")
    dt_parent = read_deltat_fs(parent / "dftb.inp")
    parent_clock = first_trajectory_clock(parent / "traject")
    if dt_parent is None or dt_parent <= 0 or parent_clock is None:
        raise ValueError(f"Cannot determine parent clock for {cv_path}; use --time-axis raw")
    parent_step, parent_time = parent_clock
    if step < parent_step:
        raise ValueError(f"Reset step counter in {cv_path}; provide explicit restart_time_ps")
    parent_start = parent_time + (step - parent_step) * dt_parent / 1000.
    return parent_start - raw_start


def load_biaspot_with_restart(run_dir: Path, cv_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    base_biaspot = run_dir / cv_dir / "biaspot"
    t0, y0 = parse_biaspot(base_biaspot)
    if t0.size == 0:
        return t0, y0

    restart_dir = run_dir / cv_dir / "metad-restart"
    restart_biaspot = restart_dir / "biaspot"
    if not restart_biaspot.exists():
        return t0, y0

    t1, y1 = parse_biaspot(restart_biaspot)
    if t1.size == 0:
        return t0, y0

    last_base = float(t0[-1])
    if float(t1[0]) < float(t0[0]):
        raise ValueError(
            f"Possible reset-clock continuation in {restart_dir}; explicit stitching "
            "is required rather than guessing an offset"
        )
    # If restart times extend beyond base, assume biaspot includes old+new; keep only new.
    if float(t1[-1]) > last_base:
        keep = t1 > last_base
        t1 = t1[keep]
        y1 = y1[keep]
        t = np.concatenate([t0, t1])
        y = np.concatenate([y0, y1])
        return t, y

    # An overlapping absolute-clock file may have no samples newer than base.
    return t0, y0


def discover_runs(
    runs_path: Path,
    cv_dirs: List[str],
    run_ids: Optional[set[int]] = None,
) -> List[Tuple[int, Path, List[str]]]:
    out: List[Tuple[int, Path, List[str]]] = []
    for p in sorted(runs_path.iterdir()):
        if not (p.is_dir() and p.name.startswith("run-")):
            continue
        try:
            run_id = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        if run_ids is not None and run_id not in run_ids:
            continue
        present = []
        for cv_dir in cv_dirs:
            cv_path = p / cv_dir
            if cv_path.exists():
                present.append(cv_dir)
        if not present:
            continue
        out.append((run_id, p, present))
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


def main() -> None:
    p = argparse.ArgumentParser(description="Plot CV grids from biaspot files.")
    p.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    p.add_argument("--cv-dir", required=True, help="CV directory under each run (comma-separated supported)")
    p.add_argument(
        "--run-ids",
        type=parse_run_ids,
        default=None,
        metavar="IDS",
        help="Run IDs to plot, as comma-separated IDs or inclusive ranges (for example, 51-100)",
    )
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--debug", action="store_true", help="Print timing/offset diagnostics")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--time-axis", choices=("restart-aligned", "raw"), default="restart-aligned",
                   help="Parent-trajectory-aligned time (default), or raw DFTB-reported time")
    p.add_argument("--data-out", type=Path, default=None, help="Aligned CSV (default: PNG stem + .csv)")
    p.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"),
                   default=(0.0, 2.0), help="Shared y-axis limits in CV units (default: 0 2)")
    args = p.parse_args()
    if not np.isfinite(args.ylim).all() or args.ylim[0] >= args.ylim[1]:
        p.error("--ylim requires finite MIN < MAX")

    cv_dirs = [s.strip() for s in args.cv_dir.split(",") if s.strip()]
    runs = discover_runs(args.runs_path, cv_dirs, args.run_ids)
    if not runs:
        raise SystemExit("No biaspot files found.")

    if args.style.exists():
        plt.style.use(args.style)

    n = len(runs)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.6 * cols, 2.8 * rows),
        dpi=220,
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    numerical_rows = []
    for i, (run_id, run_dir, present_cv_dirs) in enumerate(runs):
        ax = axes_flat[i]
        plotted = False
        min_t = None
        max_t_rel = 0.0
        ylo = None
        yhi = None
        for cv_dir in present_cv_dirs:
            biaspot = run_dir / cv_dir / "biaspot"
            if not biaspot.exists():
                continue
            t, y = load_biaspot_with_restart(run_dir, cv_dir)
            if y.size == 0:
                continue
            offset_ps = infer_offset_ps(run_dir, cv_dir, float(t[0])) if args.time_axis == "restart-aligned" else 0.
            plotted_times = t + offset_ps
            if args.debug:
                print(f"[debug] run-{run_id} cv={cv_dir} raw=({t[0]:.6f}, {t[-1]:.6f}) "
                      f"offset={offset_ps:.6f} plotted=({plotted_times[0]:.6f}, {plotted_times[-1]:.6f}) ps")
            ax.plot(plotted_times, y, lw=1.6, label=cv_dir)
            numerical_rows.extend(
                (run_id, cv_dir, float(raw), float(shown), float(value), offset_ps, args.time_axis)
                for raw, shown, value in zip(t, plotted_times, y)
            )
            max_t_rel = max(max_t_rel, float(plotted_times[-1]))
            min_t = float(plotted_times[0]) if min_t is None else min(min_t, float(plotted_times[0]))
            ylo = np.min(y) if ylo is None else min(ylo, float(np.min(y)))
            yhi = np.max(y) if yhi is None else max(yhi, float(np.max(y)))
            plotted = True

        if not plotted:
            ax.set_title(f"run-{run_id}", fontsize=10)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="#666666")
            ax.set_xticks([])
            ax.grid(False)
            continue

        if min_t is None:
            min_t = 0.0
        ax.set_xlim(min_t, max_t_rel)
        pad = 0.05 * (yhi - ylo) if (yhi is not None and ylo is not None and yhi > ylo) else 0.05
        #ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_ylim(*args.ylim)
        ax.set_title(f"run-{run_id}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        if len(cv_dirs) > 1:
            ax.legend(fontsize=6, frameon=False, loc="upper right")

    for ax in axes_flat[len(runs):]:
        ax.set_axis_off()

    # per-run scaling already applied via each panel's data range

    for r in range(rows):
        axes[r, 0].set_ylabel("s", fontsize=10)
    for c in range(cols):
        label = "Restart-aligned time (ps)" if args.time_axis == "restart-aligned" else "DFTB reported time (ps)"
        axes[rows - 1, c].set_xlabel(label, fontsize=10)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_cv_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    data_out = args.data_out or out.with_suffix(".csv")
    data_out.parent.mkdir(parents=True, exist_ok=True)
    with data_out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "cv_dir", "raw_time_ps", "plotted_time_ps", "coordination", "offset_ps", "time_axis"])
        writer.writerows(numerical_rows)
    print(f"Wrote {out}")
    print(f"Wrote {data_out}")


if __name__ == "__main__":
    main()
