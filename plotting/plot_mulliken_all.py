#!/usr/bin/env python3
"""Plot Mulliken (s+p) time series for all atoms in a single run."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go

TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+([spdf])\s+"
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?)"
)


def parse_mulliken_all(path: Path, solute_atoms: int, tmax: float | None = None) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, str]]:
    times: List[float] = []
    target_ids: List[int] | None = None
    charges: List[List[float]] = []
    elements_map: Dict[int, str] = {}

    current_time: float | None = None
    current_orb_sums: Dict[int, float] = {}
    current_elements: Dict[int, str] = {}

    def append_frame() -> None:
        if target_ids is None:
            return
        for i, atom_id in enumerate(target_ids):
            charges[i].append(current_orb_sums.get(atom_id, float("nan")))

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m_time = TIME_RE.search(line)
            if m_time:
                if current_time is not None:
                    if target_ids is None:
                        target_ids = sorted(k for k in current_elements.keys() if k <= solute_atoms)
                        charges = [[] for _ in target_ids]
                        elements_map = dict(current_elements)
                    append_frame()
                current_time = float(m_time.group(1)) / 1000.0
                if tmax is not None and current_time > tmax:
                    break
                times.append(current_time)
                current_orb_sums = {}
                current_elements = {}
                continue

            m_line = LINE_RE.match(line)
            if not m_line:
                continue
            atom_id = int(m_line.group(1))
            elem = m_line.group(2)
            orb = m_line.group(3)
            val = float(m_line.group(4))
            if atom_id not in current_elements:
                current_elements[atom_id] = elem
            if orb in ("s", "p"):
                current_orb_sums[atom_id] = current_orb_sums.get(atom_id, 0.0) + val

        if current_time is not None and (tmax is None or current_time <= tmax):
            if target_ids is None:
                target_ids = sorted(k for k in current_elements.keys() if k <= solute_atoms)
                charges = [[] for _ in target_ids]
                elements_map = dict(current_elements)
            append_frame()

    if not times or target_ids is None:
        raise SystemExit("No frames found in mulliken file.")

    return np.array(times, dtype=float), np.array(charges, dtype=float), target_ids, elements_map


def infer_system(runs_path: Path) -> str:
    parts = runs_path.resolve().parts
    if "systems" in parts:
        idx = parts.index("systems")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "system"


def parse_atom_ids(raw: str | None) -> tuple[int, ...]:
    """Parse a comma-separated list of one-based atom IDs."""
    if not raw:
        return ()
    atom_ids = tuple(dict.fromkeys(int(token.strip()) for token in raw.split(",") if token.strip()))
    if any(atom_id < 1 for atom_id in atom_ids):
        raise ValueError("highlight atom IDs must be positive")
    return atom_ids


def save_csv(
    path: Path,
    times: np.ndarray,
    charges: np.ndarray,
    atom_ids: List[int],
    elements: Dict[int, str],
) -> None:
    """Save the exact time-aligned charge matrix represented in the plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_ps", *[f"q_{elements.get(atom_id, 'X')}{atom_id}_e" for atom_id in atom_ids]]
        )
        for frame_index, time_ps in enumerate(times):
            writer.writerow(
                [f"{time_ps:.10g}", *[f"{value:.10g}" for value in charges[:, frame_index]]]
            )


def plot_png(
    path: Path,
    times: np.ndarray,
    charges: np.ndarray,
    atom_ids: List[int],
    elements: Dict[int, str],
    marker_time: float | None,
    marker_label: str | None,
    highlight_ids: tuple[int, ...],
    title: str,
    style: Path | None,
) -> None:
    """Plot all atomic Mulliken traces, emphasizing selected atoms."""
    import matplotlib.pyplot as plt

    if style is not None and style.is_file():
        plt.style.use(style)
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(atom_ids)))
    highlight_colors = {
        atom_id: color
        for atom_id, color in zip(
            highlight_ids,
            plt.get_cmap("tab10")(np.arange(max(1, len(highlight_ids)))),
        )
    }
    fig, axis = plt.subplots(figsize=(13.5, 7.5), dpi=240, layout="constrained")
    highlighted = set(highlight_ids)
    for series, atom_id, color in zip(charges, atom_ids, colors):
        is_highlighted = atom_id in highlighted
        axis.plot(
            times,
            series,
            color=highlight_colors.get(atom_id, color),
            linewidth=2.4 if is_highlighted else 0.65,
            alpha=1.0 if is_highlighted else 0.38,
            zorder=4 if is_highlighted else 1,
            label=(f"{elements.get(atom_id, 'X')}{atom_id}" if is_highlighted else None),
        )
    if marker_time is not None:
        axis.axvline(marker_time, color="black", linestyle="--", linewidth=1.8, zorder=5)
        if marker_label:
            axis.text(
                marker_time,
                0.98,
                marker_label,
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                rotation=90,
                fontsize=10,
            )
    if highlighted:
        axis.legend(title="highlighted atoms", frameon=False, loc="upper right")
    axis.set(
        xlim=(float(times[0]), float(times[-1])),
        xlabel="DFTB equilibration time (ps)",
        ylabel="summed Mulliken charge, q (e)",
        title=title,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Mulliken (s+p) time series for all atoms.")
    p.add_argument("--mulliken", type=Path, default=None, help="Direct Mulliken file path")
    p.add_argument("--runs-path", type=Path, default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--solute-atoms", required=True, type=int, help="Plot atoms 1..solute_atoms")
    p.add_argument("--t-max", type=float, default=None, help="Cap time series at this time (ps)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--data-out", type=Path, default=None)
    p.add_argument("--marker-time", type=float, default=None)
    p.add_argument("--marker-label", default=None)
    p.add_argument("--highlight-ids", default=None)
    p.add_argument("--title", default="Solute Mulliken charges")
    p.add_argument(
        "--style",
        type=Path,
        default=Path(__file__).resolve().with_name("lefteris.mplstyle"),
    )
    args = p.parse_args()

    if args.mulliken is not None:
        mulliken_path = args.mulliken
    else:
        if args.runs_path is None or args.run_dir is None or args.run_id is None:
            p.error("provide --mulliken or all of --runs-path, --run-dir, and --run-id")
        mulliken_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / "mulliken"
    if not mulliken_path.exists():
        raise SystemExit(f"Missing mulliken file: {mulliken_path}")

    times, charges, target_ids, elements_map = parse_mulliken_all(
        mulliken_path, args.solute_atoms, tmax=args.t_max
    )

    out = args.out
    if out is None:
        system = infer_system(args.runs_path or mulliken_path)
        run_dir = args.run_dir or mulliken_path.parent.name
        run_id = args.run_id if args.run_id is not None else 0
        out = Path("reports") / f"{system}_{run_dir}_mulliken_all_run{run_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    data_out = args.data_out or out.with_name(f"{out.stem}.csv")
    save_csv(data_out, times, charges, target_ids, elements_map)
    if out.suffix.lower() == ".png":
        plot_png(
            out,
            times,
            charges,
            target_ids,
            elements_map,
            args.marker_time,
            args.marker_label,
            parse_atom_ids(args.highlight_ids),
            args.title,
            args.style,
        )
    else:
        fig = go.Figure()
        n = max(1, len(target_ids))
        colors = [f"hsla({int(360 * i / n)}, 70%, 45%, 0.6)" for i in range(n)]
        for i, atom_id in enumerate(target_ids):
            elem = elements_map.get(atom_id, "X")
            label = f"{elem}-{atom_id}"
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=charges[i],
                    mode="lines",
                    name=label,
                    line=dict(color=colors[i], width=1),
                    hovertemplate=f"{label}<br>t=%{{x:.3f}} ps<br>q=%{{y:.5f}}<extra></extra>",
                )
            )
        fig.update_layout(
            xaxis_title="t (ps)",
            yaxis_title="q(t) [e^-]",
            template="simple_white",
            showlegend=False,
        )
        fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    print(f"Wrote {data_out}")


if __name__ == "__main__":
    main()
