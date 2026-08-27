#!/usr/bin/env python3
"""Plot dihedral time series either overlaid or in a run grid."""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")


def read_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Bad dihedral file: {path}")
    return data[:, 0], data[:, 1]


def _dih_filename(label: str) -> str:
    label = label.strip()
    if label.startswith("dih_") and label.endswith(".dat"):
        return label
    if label.startswith("dih_"):
        return f"{label}.dat"
    return f"dih_{label}.dat"


def discover_runs(
    runs_path: Path, rel_dir: Path, labels: List[str]
) -> List[Tuple[str, List[Path]]]:
    out: List[Tuple[str, List[Path]]] = []
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
        out.append((f"run-{run_id}", paths))
    return out


def discover_analysis_dirs(
    analysis_dirs: List[Path], labels: List[str], titles: List[str] | None
) -> List[Tuple[str, List[Path]]]:
    """Resolve explicitly supplied analysis directories in the given order."""
    if titles is not None and len(titles) != len(analysis_dirs):
        raise ValueError(
            f"Expected {len(analysis_dirs)} --titles entries, got {len(titles)}"
        )

    out: List[Tuple[str, List[Path]]] = []
    for index, analysis_dir in enumerate(analysis_dirs):
        paths = [analysis_dir / _dih_filename(label) for label in labels]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing dihedral files in {analysis_dir}: {', '.join(missing)}"
            )
        if titles is not None:
            title = titles[index]
        else:
            cv_name = analysis_dir.parent.name
            run_name = analysis_dir.parent.parent.name
            title = f"{cv_name}, {run_name}"
        out.append((title, paths))
    return out


def read_trajectory_times(path: Path) -> np.ndarray:
    """Read DFTB XYZ comment timestamps in ps without loading coordinates."""
    times: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = TIME_RE.search(line)
            if match:
                times.append(float(match.group(1)) / 1000.0)
    if not times:
        raise ValueError(f"No DFTB timestamps found in {path}")
    return np.asarray(times, dtype=float)


def x_values(
    frames: np.ndarray,
    time_step_ps: float | None,
    trajectory_times: np.ndarray | None = None,
) -> np.ndarray:
    """Return CPPTRAJ frame numbers or elapsed time starting at zero."""
    if trajectory_times is not None:
        indices = np.rint(frames).astype(int) - 1
        if np.any(indices < 0) or np.any(indices >= trajectory_times.size):
            raise ValueError(
                "CPPTRAJ frame indices fall outside the supplied trajectory timestamps"
            )
        return trajectory_times[indices]
    if time_step_ps is None:
        return frames
    return (frames - frames[0]) * time_step_ps


def display_angles(angles: np.ndarray, convention: str) -> np.ndarray:
    """Convert angles to the convention used in the rendered figure and CSV."""
    if convention == "signed":
        return (angles + 180.0) % 360.0 - 180.0
    return angles


def save_aligned_data(
    path: Path,
    runs: List[Tuple[str, List[Path]]],
    labels: List[str],
    time_step_ps: float | None,
    trajectory_times: List[np.ndarray] | None,
    angle_convention: str,
    t_deprot: List[float] | None,
    t_diffuse: List[float] | None,
) -> None:
    """Save the numerical series used to render the figure in long format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "panel",
                "dihedral",
                "frame",
                "time_ps",
                "angle_deg",
                "t_deprot_ps",
                "t_diffuse_ps",
            ]
        )
        for run_index, (title, paths) in enumerate(runs):
            run_times = None if trajectory_times is None else trajectory_times[run_index]
            deprot_value = "" if t_deprot is None else f"{t_deprot[run_index]:g}"
            diffuse_value = "" if t_diffuse is None else f"{t_diffuse[run_index]:g}"
            for label, series_path in zip(labels, paths):
                frames, angles = read_series(series_path)
                angles = display_angles(angles, angle_convention)
                times = x_values(frames, time_step_ps, run_times)
                for frame, time_value, angle in zip(frames, times, angles):
                    writer.writerow(
                        [
                            title,
                            label,
                            f"{frame:g}",
                            (
                                f"{time_value:g}"
                                if time_step_ps is not None or trajectory_times is not None
                                else ""
                            ),
                            f"{angle:.10g}",
                            deprot_value,
                            diffuse_value,
                        ]
                    )


def grid_shape(n: int) -> Tuple[int, int]:
    root = int(np.floor(np.sqrt(n)))
    rows = max(1, root)
    cols = int(np.ceil(n / rows))
    if rows * cols < n:
        rows += 1
    return rows, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--runs-path", type=Path, help="Path containing homogeneous run-* directories"
    )
    source.add_argument(
        "--analysis-dirs",
        nargs="+",
        type=Path,
        help="Explicit analysis directories, useful when runs have different CV paths",
    )
    ap.add_argument("--rel-dir", default="equil/analysis", help="Relative path under each run")
    ap.add_argument(
        "--titles",
        default=None,
        help="Optional comma-separated panel titles for --analysis-dirs",
    )
    ap.add_argument("--labels", default="chi1,chi2",
                    help="Comma-separated dihedral labels (e.g., chi1,chi2 or single5,single10,single15)")
    ap.add_argument("--mode", choices=["overlay", "grid"], default="overlay",
                    help="overlay: all runs overlaid; grid: one panel per run")
    ap.add_argument("--out", type=Path, default=Path("reports/chi_timeseries.png"))
    ap.add_argument(
        "--data-out",
        type=Path,
        default=None,
        help="Aligned long-form CSV (default: --out with .csv suffix)",
    )
    ap.add_argument(
        "--time-step-ps",
        type=float,
        default=None,
        help="Time between saved trajectory frames in ps; omit to plot frame number",
    )
    ap.add_argument(
        "--trajectory-files",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "DFTB XYZ trajectories corresponding one-to-one with --analysis-dirs; "
            "uses their actual timestamps"
        ),
    )
    ap.add_argument(
        "--angle-convention",
        choices=["raw", "signed"],
        default="raw",
        help="Plot raw CPPTRAJ angles or map them to [-180, 180) (default: raw)",
    )
    ap.add_argument(
        "--t-deprot",
        nargs="+",
        type=float,
        default=None,
        help="Per-panel sustained-deprotonation times in ps",
    )
    ap.add_argument(
        "--t-diffuse",
        nargs="+",
        type=float,
        default=None,
        help="Per-panel diffusion-boundary times in ps",
    )
    ap.add_argument("--plot", choices=["line", "scatter"], default="line",
                    help="Plot style for time series")
    ap.add_argument("--scatter-size", type=float, default=6.0, help="Marker size for scatter")
    ap.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Shared y-axis limits",
    )
    ap.add_argument(
        "--independent-x",
        action="store_true",
        help="Use an independent x-axis range for each grid panel",
    )
    ap.add_argument("--style", type=Path, default=Path("plotting/prl.mplstyle"),
                    help="Matplotlib style file")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if not labels:
        raise SystemExit("No labels provided")

    if args.style.exists():
        plt.style.use(args.style)

    if args.time_step_ps is not None and args.time_step_ps <= 0.0:
        ap.error("--time-step-ps must be positive")
    if args.ylim is not None and args.ylim[0] >= args.ylim[1]:
        ap.error("--ylim requires YMIN < YMAX")
    if args.time_step_ps is not None and args.trajectory_files is not None:
        ap.error("--time-step-ps and --trajectory-files are mutually exclusive")
    if args.trajectory_files is not None and args.analysis_dirs is None:
        ap.error("--trajectory-files requires --analysis-dirs")

    if args.analysis_dirs is not None:
        titles = None
        if args.titles is not None:
            titles = [title.strip() for title in args.titles.split(",")]
        try:
            runs = discover_analysis_dirs(args.analysis_dirs, labels, titles)
        except (FileNotFoundError, ValueError) as exc:
            ap.error(str(exc))
    else:
        runs = discover_runs(args.runs_path, Path(args.rel_dir), labels)
    if not runs:
        raise SystemExit("No runs found")

    for option_name, values in (
        ("--t-deprot", args.t_deprot),
        ("--t-diffuse", args.t_diffuse),
    ):
        if values is not None and len(values) != len(runs):
            ap.error(f"{option_name} must contain one value for every panel")

    trajectory_times = None
    if args.trajectory_files is not None:
        if len(args.trajectory_files) != len(runs):
            ap.error(
                "--trajectory-files must contain one path for every analysis directory"
            )
        try:
            trajectory_times = [read_trajectory_times(path) for path in args.trajectory_files]
        except (OSError, ValueError) as exc:
            ap.error(str(exc))

    has_time = args.time_step_ps is not None or trajectory_times is not None
    x_label = "time (ps)" if has_time else "frame"

    if args.mode == "overlay":
        fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 4), sharey=True)
        axes_list = axes if isinstance(axes, np.ndarray) else [axes]
        cmap = plt.get_cmap("tab20")
        for idx, (title, paths) in enumerate(runs):
            run_times = None if trajectory_times is None else trajectory_times[idx]
            color = cmap(idx % cmap.N)
            for ax, label, path in zip(axes_list, labels, paths):
                frames, series = read_series(path)
                series = display_angles(series, args.angle_convention)
                t = x_values(frames, args.time_step_ps, run_times)
                if args.plot == "scatter":
                    ax.scatter(t, series, s=args.scatter_size, color=color, alpha=0.7, label=title)
                else:
                    ax.plot(t, series, color=color, lw=0.5, alpha=0.9, label=title)
                ax.set_title(label)
                ax.set_xlabel(x_label)
                ax.grid(True, alpha=0.3)
        axes_list[0].set_ylabel("dihedral (deg)")
    else:
        rows, cols = grid_shape(len(runs))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(6.2 * cols, 4.2 * rows),
            sharex=not args.independent_x,
            sharey=True,
            squeeze=False,
        )
        axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        cmap = plt.get_cmap("tab10")
        color_map = {label: cmap(index % cmap.N) for index, label in enumerate(labels)}
        for i, (title, paths) in enumerate(runs):
            ax = axes_list[i]
            run_times = None if trajectory_times is None else trajectory_times[i]
            for label, path in zip(labels, paths):
                frames, series = read_series(path)
                series = display_angles(series, args.angle_convention)
                t = x_values(frames, args.time_step_ps, run_times)
                color = color_map[label]
                if args.plot == "scatter":
                    ax.scatter(t, series, s=args.scatter_size, alpha=0.7, color=color, label=label)
                else:
                    ax.plot(t, series, lw=0.7, alpha=0.9, color=color, label=label)
            if args.independent_x:
                finite_x = t[np.isfinite(t)]
                if finite_x.size:
                    ax.set_xlim(float(np.min(finite_x)), float(np.max(finite_x)))
            if args.t_deprot is not None:
                ax.axvline(
                    args.t_deprot[i],
                    color="black",
                    linestyle=":",
                    linewidth=2.0,
                    label=r"$t_{\mathrm{deprot}}$" if i == 0 else None,
                    zorder=10,
                )
            if args.t_diffuse is not None:
                ax.axvline(
                    args.t_diffuse[i],
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    label=r"$t_{\mathrm{diffuse}}$" if i == 0 else None,
                    zorder=10,
                )
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if args.ylim is not None:
                ax.set_ylim(args.ylim[0], args.ylim[1])
        for j in range(len(runs), len(axes_list)):
            axes_list[j].set_axis_off()
        fig.supxlabel(x_label)
        fig.supylabel("dihedral (deg)")
        handles, labels_used = axes_list[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels_used,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=min(6, len(labels_used)),
                frameon=False,
            )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91) if args.mode == "grid" else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    data_out = args.data_out or args.out.with_suffix(".csv")
    save_aligned_data(
        data_out,
        runs,
        labels,
        args.time_step_ps,
        trajectory_times,
        args.angle_convention,
        args.t_deprot,
        args.t_diffuse,
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {data_out}")


if __name__ == "__main__":
    main()
