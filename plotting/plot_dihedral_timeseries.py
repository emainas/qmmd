#!/usr/bin/env python3
"""Plot dihedral time series either overlaid or in a run grid."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_series(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Bad dihedral file: {path}")
    return data[:, 1]


def _dih_filename(label: str) -> str:
    label = label.strip()
    if label.startswith("dih_") and label.endswith(".dat"):
        return label
    if label.startswith("dih_"):
        return f"{label}.dat"
    return f"dih_{label}.dat"


def discover_runs(
    runs_path: Path, rel_dir: Path, labels: List[str]
) -> List[Tuple[int, List[Path]]]:
    out: List[Tuple[int, List[Path]]] = []
    for run_dir in sorted(runs_path.glob("run-*")):
        if not run_dir.is_dir():
            continue
        try:
            run_id = int(run_dir.name.split("-")[-1])
        except Exception:
            continue
        analysis_dir = run_dir / rel_dir
        paths: List[Path] = []
        missing = False
        for label in labels:
            p = analysis_dir / _dih_filename(label)
            if not p.exists():
                print(f"WARN: missing {p.name} in {analysis_dir}")
                missing = True
                break
            paths.append(p)
        if missing:
            continue
        out.append((run_id, paths))
    return out


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-path", required=True, type=Path, help="Path containing run-* directories")
    ap.add_argument("--rel-dir", default="equil/analysis", help="Relative path under each run")
    ap.add_argument("--labels", default="chi1,chi2",
                    help="Comma-separated dihedral labels (e.g., chi1,chi2 or single5,single10,single15)")
    ap.add_argument("--mode", choices=["overlay", "grid"], default="overlay",
                    help="overlay: all runs overlaid; grid: one panel per run")
    ap.add_argument("--out", type=Path, default=Path("reports/chi_timeseries.png"))
    ap.add_argument("--plot", choices=["line", "scatter"], default="line",
                    help="Plot style for time series")
    ap.add_argument("--scatter-size", type=float, default=6.0, help="Marker size for scatter")
    ap.add_argument("--style", type=Path, default=Path("plotting/prl.mplstyle"),
                    help="Matplotlib style file")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if not labels:
        raise SystemExit("No labels provided")

    if args.style.exists():
        plt.style.use(args.style)

    runs = discover_runs(args.runs_path, Path(args.rel_dir), labels)
    if not runs:
        raise SystemExit("No runs found")

    if args.mode == "overlay":
        fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 4), sharey=True)
        axes_list = axes if isinstance(axes, np.ndarray) else [axes]
        cmap = plt.get_cmap("tab20")
        for idx, (run_id, paths) in enumerate(runs):
            color = cmap(idx % cmap.N)
            for ax, label, path in zip(axes_list, labels, paths):
                series = read_series(path)
                t = np.arange(len(series))
                if args.plot == "scatter":
                    ax.scatter(t, series, s=args.scatter_size, color=color, alpha=0.7, label=f"run-{run_id}")
                else:
                    ax.plot(t, series, color=color, lw=0.5, alpha=0.9, label=f"run-{run_id}")
                ax.set_title(label)
                ax.set_xlabel("frame")
                ax.grid(True, alpha=0.3)
        axes_list[0].set_ylabel("dihedral (deg)")
    else:
        rows, cols = grid_shape(len(runs))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), sharex=True, sharey=True)
        axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        color_map: Dict[str, str] = {
            labels[0]: "tab:blue",
            labels[1] if len(labels) > 1 else "": "tab:red",
            labels[2] if len(labels) > 2 else "": "tab:green",
        }
        for i, (run_id, paths) in enumerate(runs):
            ax = axes_list[i]
            for label, path in zip(labels, paths):
                series = read_series(path)
                t = np.arange(len(series))
                color = color_map.get(label, None)
                if args.plot == "scatter":
                    ax.scatter(t, series, s=args.scatter_size, alpha=0.7, color=color, label=label)
                else:
                    ax.plot(t, series, lw=0.7, alpha=0.9, color=color, label=label)
            ax.set_title(f"run-{run_id}", fontsize=8)
            ax.grid(True, alpha=0.3)
        for j in range(len(runs), len(axes_list)):
            axes_list[j].set_axis_off()
        axes_list[0].set_ylabel("dihedral (deg)")
        axes_list[-1].set_xlabel("frame")
        handles, labels_used = axes_list[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels_used, loc="upper center", ncol=min(4, len(labels_used)), frameon=False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
