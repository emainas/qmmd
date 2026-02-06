#!/usr/bin/env python3
"""
Plot DFTB performance summaries from data/dftb.

Outputs (in reports/):
  - dftb_stats_grid.png
  - dftb_cumtime_grid.png
  - dftb_predictions_grid.png
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt


def parse_float(s: str):
    try:
        return float(s)
    except Exception:
        return None


def load_step_perf(path: Path):
    try:
        arr = np.loadtxt(path, comments="#")
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] < 4:
            return None
        arr = arr.reshape(1, -1)
    # columns: step, subsystems, iterations, seconds
    return arr


def load_summary(path: Path):
    try:
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    except Exception:
        return None, None
    if not lines:
        return None, None
    parts = lines[-1].split()
    if len(parts) < 2:
        return None, None
    steps = parse_float(parts[0])
    seconds = parse_float(parts[1])
    if steps is None or seconds is None:
        return None, None
    return steps, seconds


def grid_shape(n: int):
    if n <= 0:
        return 1, 1
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def list_systems(data_dir: Path):
    if not data_dir.exists():
        return []
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


def list_solv_dirs(system_dir: Path):
    if not system_dir.exists():
        return []
    return sorted([p for p in system_dir.glob("solv_*") if p.is_dir()])


def list_step_perf_files(solv_dir: Path):
    return sorted(solv_dir.glob("*_step_perf.dat"))


def run_tag_from_file(path: Path) -> str:
    # N1T64C1_step_perf.dat -> N1T64C1
    return path.stem.replace("_step_perf", "")


def plot_stats_grid(data_dir: Path, reports_dir: Path):
    systems = list_systems(data_dir)
    if not systems:
        return

    ncols = len(systems)
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=(4 * ncols, 9))

    base_colors = ["blue", "red", "green", "orange", "purple", "black"]

    metrics = [
        ("avg subsys", 2),
        ("avg iters", 3),
        ("avg sec/step", 4),
    ]

    for col, system in enumerate(systems):
        sys_dir = data_dir / system
        rows = []  # (solv_tag, run_tag, subsys, iters, sec_per_step)
        for solv_dir in list_solv_dirs(sys_dir):
            solv_tag = solv_dir.name
            files = list_step_perf_files(solv_dir)
            if not files:
                continue
            for f in files:
                arr = load_step_perf(f)
                if arr is None:
                    continue
                subsys = float(np.mean(arr[:, 1]))
                iters = float(np.mean(arr[:, 2]))
                secs = float(np.mean(arr[:, 3]))
                run_tag = run_tag_from_file(f)
                rows.append((solv_tag, run_tag, subsys, iters, secs))

        if not rows:
            for row in range(nrows):
                ax = axes[row][col]
                ax.axis("off")
                ax.text(0.5, 0.5, f"{system}\nno data", ha="center", va="center", transform=ax.transAxes)
            continue

        solv_tags = sorted({r[0] for r in rows})
        run_tags = sorted({r[1] for r in rows})
        color_map = {rt: base_colors[i % len(base_colors)] for i, rt in enumerate(run_tags)}

        x = np.arange(len(solv_tags))
        nrun = max(len(run_tags), 1)
        group_width = 0.8
        bar_w = group_width / nrun

        for row, (ylabel, idx_col) in enumerate(metrics):
            ax = axes[row][col]
            for i, run_tag in enumerate(run_tags):
                centers = x - group_width / 2 + (i + 0.5) * bar_w
                vals = []
                for solv_tag in solv_tags:
                    match = [r for r in rows if r[0] == solv_tag and r[1] == run_tag]
                    if match:
                        vals.append(match[0][idx_col])
                    else:
                        vals.append(np.nan)
                ax.bar(
                    centers,
                    vals,
                    width=bar_w * 0.95,
                    label=run_tag,
                    color=color_map[run_tag],
                    edgecolor="black",
                    linewidth=0.6,
                )
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(system)
            if row == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(solv_tags, rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("Solvation")
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            if run_tags:
                ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(reports_dir / "dftb_stats_grid.png", dpi=300)
    plt.close(fig)


def plot_cumtime_grid(data_dir: Path, out_png: Path):
    systems = list_systems(data_dir)
    if not systems:
        return

    nrows, ncols = grid_shape(len(systems))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=False, sharey=False)

    for idx, system in enumerate(systems):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.set_title(system)
        sys_dir = data_dir / system

        solv_dirs = list_solv_dirs(sys_dir)
        if not solv_dirs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(solv_dirs), 1)))
        legend_items = []

        for i, solv_dir in enumerate(solv_dirs):
            solv_tag = solv_dir.name
            files = list_step_perf_files(solv_dir)
            if not files:
                continue
            color = colors[i]
            for f in files:
                arr = load_step_perf(f)
                if arr is None:
                    continue
                steps = arr[:, 0]
                secs = arr[:, 3]
                cum = np.cumsum(secs)
                idx = np.arange(1, len(cum) + 1)
                speed = cum / idx
                run_tag = run_tag_from_file(f)
                label = f"{solv_tag}:{run_tag}"
                ax.plot(steps, speed, lw=1.2, color=color)
                ax.scatter(steps, speed, s=6, color=color, alpha=0.6)
                legend_items.append(label)

        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative avg seconds/step")
        if legend_items:
            ax.legend(legend_items, fontsize=6, frameon=False)

    for idx in range(len(systems), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def linear_slope_second_half(steps: np.ndarray, cum_seconds: np.ndarray):
    n = len(steps)
    if n < 2:
        return None
    half = n // 2
    x = steps[half:]
    y = cum_seconds[half:]
    if len(x) < 2:
        return None
    # linear fit y = m*x + b
    m, _b = np.polyfit(x, y, 1)
    return float(m)


def plot_predictions_grid(data_dir: Path, out_png: Path):
    systems = list_systems(data_dir)
    if not systems:
        return

    nrows, ncols = grid_shape(len(systems))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)

    for idx, system in enumerate(systems):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.set_title(system)
        sys_dir = data_dir / system

        plotted = False
        for solv_dir in list_solv_dirs(sys_dir):
            solv_tag = solv_dir.name
            summaries = sorted(solv_dir.glob("*_summary.dat"))
            if not summaries:
                continue

            for summary in summaries:
                run_tag = summary.stem.replace("_summary", "")
                step_perf = solv_dir / f"{run_tag}_step_perf.dat"
                if not step_perf.exists():
                    continue

                steps_total, seconds_total = load_summary(summary)
                if steps_total is None or seconds_total is None:
                    continue

                arr = load_step_perf(step_perf)
                if arr is None:
                    continue

                steps = arr[:, 0]
                secs = arr[:, 3]
                cum = np.cumsum(secs)
                slope = linear_slope_second_half(steps, cum)
                if slope is None:
                    continue

                max_steps = steps_total * 100.0
                line_x = np.linspace(steps_total, max_steps, 50)
                line_y = slope * line_x

                pred_x = np.array([steps_total * 10.0, steps_total * 100.0])
                pred_y = slope * pred_x

                label = f"{solv_tag}:{run_tag}"
                ax.plot(line_x, line_y / 3600.0, lw=1.2, label=label)
                ax.scatter(pred_x, pred_y / 3600.0, s=30, alpha=0.8)
                plotted = True

        if not plotted:
            ax.axis("off")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.set_xlabel("Predicted steps")
        ax.set_ylabel("Predicted total hours")
        ax.legend(frameon=False, fontsize=6)

    for idx in range(len(systems), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    script_path = Path(__file__).resolve()
    root = script_path.parents[1]

    data_dir = root / "data" / "dftb"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # plt.style.use("prl.mplstyle")

    plot_stats_grid(data_dir, reports_dir)
    plot_cumtime_grid(data_dir, reports_dir / "dftb_cumtime_grid.png")
    plot_predictions_grid(data_dir, reports_dir / "dftb_predictions_grid.png")

    print("Wrote plots to reports/")


if __name__ == "__main__":
    main()
