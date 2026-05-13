#!/usr/bin/env python3
"""Plot CV time-series grids from biaspot files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import yaml


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
STEP_RE = re.compile(r"STEP NO\.\s*=\s*([0-9]+)")
COORD_RE = re.compile(r"Coordinate\s*=\s*([+-]?[0-9.]+)")


def parse_biaspot(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    coords: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)) / 1000.0)  # fsec -> ps
                continue
            m = COORD_RE.search(line)
            if m:
                coords.append(float(m.group(1)))
    n = min(len(times), len(coords))
    if n == 0:
        return np.array([]), np.array([])
    t = np.array(times[:n])
    return t, np.array(coords[:n])


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


def infer_offset_ps(run_dir: Path, cv_dir: str, t0: float) -> float:
    cv_path = run_dir / cv_dir
    spec_path = None
    for name in ("wtmeta_spec.yaml", "meta_spec.yaml"):
        p = cv_path / name
        if p.exists():
            spec_path = p
            break
    if spec_path is None:
        return t0

    data = yaml.safe_load(spec_path.read_text())
    if "restart_time_ps" in data and data["restart_time_ps"] is not None:
        try:
            return float(data["restart_time_ps"]) - t0
        except (TypeError, ValueError):
            pass

    restart_time_txt = cv_path / "restart_time_ps.txt"
    if restart_time_txt.exists():
        try:
            return float(restart_time_txt.read_text().strip()) - t0
        except ValueError:
            pass
    replica_dirname = data.get("replica_dirname")
    if not replica_dirname:
        return t0

    step0 = read_first_step(cv_path / "biaspot")
    if step0 is None:
        return t0

    dt_old_fs = read_deltat_fs(run_dir / replica_dirname / "dftb.inp")
    if dt_old_fs is None:
        return 0.0

    restart_step = read_restart_step(run_dir / cv_dir / "restart", target_step=step0)
    if restart_step is not None:
        return (restart_step * dt_old_fs) / 1000.0 - t0

    # Fallback: use time of last restart dump in equil output.
    restart_time = read_restart_time_ps(run_dir / replica_dirname / "dftb.out")
    if restart_time is not None:
        return restart_time - t0

    return 0.0


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
    # If restart times extend beyond base, assume biaspot includes old+new; keep only new.
    if float(t1[-1]) > last_base:
        keep = t1 > last_base
        t1 = t1[keep]
        y1 = y1[keep]
        t = np.concatenate([t0, t1])
        y = np.concatenate([y0, y1])
        return t, y

    # Otherwise treat restart as its own segment starting near 0 and offset by base end.
    t1 = t1 + last_base
    t = np.concatenate([t0, t1])
    y = np.concatenate([y0, y1])
    return t, y


def discover_runs(runs_path: Path, cv_dirs: List[str]) -> List[Tuple[int, Path, List[str]]]:
    out: List[Tuple[int, Path, List[str]]] = []
    for p in sorted(runs_path.iterdir()):
        if not (p.is_dir() and p.name.startswith("run-")):
            continue
        try:
            run_id = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        present = []
        for cv_dir in cv_dirs:
            cv_path = p / cv_dir
            if cv_path.exists():
                present.append(cv_dir)
        if not present:
            continue
        out.append((run_id, p, present))
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
    p = argparse.ArgumentParser(description="Plot CV grids from biaspot files.")
    p.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    p.add_argument("--cv-dir", required=True, help="CV directory under each run (comma-separated supported)")
    p.add_argument("--style", type=Path, default=Path("src/prl.mplstyle"))
    p.add_argument("--debug", action="store_true", help="Print timing/offset diagnostics")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cv_dirs = [s.strip() for s in args.cv_dir.split(",") if s.strip()]
    runs = discover_runs(args.runs_path, cv_dirs)
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

    all_y = []
    for i, (run_id, run_dir, present_cv_dirs) in enumerate(runs):
        ax = axes_flat[i]
        plotted = False
        min_t = None
        max_t_rel = 0.0
        ylo = None
        yhi = None
        t0_by_cv: dict[str, float] = {}
        for cv_dir in present_cv_dirs:
            biaspot = run_dir / cv_dir / "biaspot"
            if not biaspot.exists():
                continue
            t, y = load_biaspot_with_restart(run_dir, cv_dir)
            if y.size == 0:
                continue
            t0 = float(t[0])
            t0_by_cv[cv_dir] = t0
            if args.debug:
                print(f"[debug] run-{run_id} cv={cv_dir} biaspot t0={t0:.4f} ps t_last={float(t[-1]):.4f} ps")
            # Use absolute run time (from start of this run); show gaps before first Gaussian.
            t_rel = t
            ax.plot(t_rel, y, lw=1.6, label=cv_dir)
            max_t_rel = max(max_t_rel, float(t_rel[-1]))
            min_t = t0 if min_t is None else min(min_t, t0)
            ylo = np.min(y) if ylo is None else min(ylo, float(np.min(y)))
            yhi = np.max(y) if yhi is None else max(yhi, float(np.max(y)))
            all_y.append(y)
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
        offset_ps = None
        for cv_dir in present_cv_dirs:
            offset_ps = infer_offset_ps(run_dir, cv_dir, t0=t0_by_cv.get(cv_dir, 0.0))
            if args.debug:
                step0 = read_first_step(run_dir / cv_dir / "biaspot")
                dt_old_fs = read_deltat_fs(run_dir / cv_dir / "dftb.inp")
                restart_step = read_restart_step(run_dir / cv_dir / "restart", target_step=step0)
                print(
                    f"[debug] run-{run_id} cv={cv_dir} offset_ps={offset_ps:.4f} "
                    f"(step0={step0} dt_old_fs={dt_old_fs} restart_step={restart_step})"
                )
            if offset_ps is not None:
                break
        if offset_ps is None:
            offset_ps = 0.0
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x + offset_ps:.1f}"))
        if args.debug:
            x0, x1 = ax.get_xlim()
            print(f"[debug] run-{run_id} xlim=({x0:.4f}, {x1:.4f}) label_start={offset_ps + x0:.4f} label_end={offset_ps + x1:.4f}")
        pad = 0.05 * (yhi - ylo) if (yhi is not None and ylo is not None and yhi > ylo) else 0.05
        #ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_ylim(0.0, 2.0)
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
        axes[rows - 1, c].set_xlabel("t (ps)", fontsize=10)

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.cv_dir}_cv_{n}runs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
