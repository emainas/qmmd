#!/usr/bin/env python3
"""Extract water oxygen Mulliken charges and plot time series grid."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.colors as pc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("plotly is required. Install with: pip install plotly") from exc


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+([spdf])\s+"
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?)"
)


def grid_shape(n: int) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    root = int(math.floor(math.sqrt(n)))
    rows = max(1, root)
    cols = int(math.ceil(n / rows))
    return rows, cols


def _append_frame(
    target_ids: List[int],
    charges: List[List[float]],
    orb_sums: Dict[int, float],
) -> None:
    for idx, atom_id in enumerate(target_ids):
        charges[idx].append(orb_sums.get(atom_id, float("nan")))


def _infer_oxygen_ids(
    elements: Dict[int, str],
    solute_atoms: int,
    natoms: int,
) -> List[int]:
    oxygen_ids: List[int] = []
    start = solute_atoms + 1
    for atom_id in range(start, natoms + 1):
        if elements.get(atom_id) == "O":
            oxygen_ids.append(atom_id)
    if not oxygen_ids:
        raise SystemExit("No oxygen atoms found after solute atoms.")
    return oxygen_ids


def _parse_extra_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise SystemExit(f"Invalid extra-id '{part}'. Use comma-separated integers.") from exc
    return ids


def parse_mulliken(
    path: Path,
    solute_atoms: int,
    natoms: Optional[int] = None,
    extra_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, str]]:
    times: List[float] = []
    charges: List[List[float]] = []
    oxygen_ids: Optional[List[int]] = None
    target_ids: Optional[List[int]] = None

    current_time: Optional[float] = None
    current_orb_sums: Dict[int, float] = {}
    current_max_id = 0
    current_elements: Dict[int, str] = {}
    base_elements: Dict[int, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m_time = TIME_RE.search(line)
            if m_time:
                if current_time is not None:
                    if oxygen_ids is None:
                        if natoms is None:
                            natoms = current_max_id
                        if natoms is None:
                            raise SystemExit("Could not determine total atom count.")
                        oxygen_ids = _infer_oxygen_ids(current_elements, solute_atoms, natoms)
                        extra_ids = extra_ids or []
                        seen = set()
                        target_ids = []
                        for atom_id in oxygen_ids + extra_ids:
                            if atom_id not in seen:
                                seen.add(atom_id)
                                target_ids.append(atom_id)
                        charges = [[] for _ in target_ids]
                        base_elements = dict(current_elements)
                    _append_frame(target_ids or [], charges, current_orb_sums)

                current_time = float(m_time.group(1)) / 1000.0  # fsec -> ps
                times.append(current_time)
                current_orb_sums = {}
                current_elements = {}
                current_max_id = 0
                continue

            m_line = LINE_RE.match(line)
            if not m_line:
                # Try to capture natoms from the header if present.
                if natoms is None:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        try:
                            natoms = int(parts[0])
                        except ValueError:
                            pass
                continue

            atom_id = int(m_line.group(1))
            elem = m_line.group(2)
            orb = m_line.group(3)
            val = float(m_line.group(4))

            if atom_id > current_max_id:
                current_max_id = atom_id

            if atom_id not in current_elements:
                current_elements[atom_id] = elem

            if orb in ("s", "p"):
                current_orb_sums[atom_id] = current_orb_sums.get(atom_id, 0.0) + val

        if current_time is not None:
            if oxygen_ids is None:
                if natoms is None:
                    natoms = current_max_id
                if natoms is None:
                    raise SystemExit("Could not determine total atom count.")
                oxygen_ids = _infer_oxygen_ids(current_elements, solute_atoms, natoms)
                extra_ids = extra_ids or []
                seen = set()
                target_ids = []
                for atom_id in oxygen_ids + extra_ids:
                    if atom_id not in seen:
                        seen.add(atom_id)
                        target_ids.append(atom_id)
                charges = [[] for _ in target_ids]
                base_elements = dict(current_elements)
            _append_frame(target_ids or [], charges, current_orb_sums)

    if not times:
        raise SystemExit("No frames found in mulliken file.")

    charges_array = np.array(charges, dtype=float)
    return np.array(times, dtype=float), charges_array, (target_ids or []), base_elements


def _collect_runs(
    runs_path: Path, cv_dir: str, run_start: int, run_end: int
) -> List[Tuple[int, Path]]:
    runs: List[Tuple[int, Path]] = []
    for run_id in range(run_start, run_end + 1):
        run_dir = runs_path / f"run-{run_id}" / cv_dir
        mulliken = run_dir / "mulliken"
        if mulliken.exists():
            runs.append((run_id, mulliken))
    return runs


def _collect_run_inputs(
    runs_path: Path,
    cv_dir: str,
    run_start: int,
    run_end: int,
    include_equil: bool,
    equil_dir: str,
) -> List[Tuple[int, Path, Optional[Path]]]:
    runs: List[Tuple[int, Path, Optional[Path]]] = []
    for run_id in range(run_start, run_end + 1):
        run_dir = runs_path / f"run-{run_id}"
        mulliken = run_dir / cv_dir / "mulliken"
        if not mulliken.exists():
            continue
        equil = None
        if include_equil:
            equil = run_dir / equil_dir / "mulliken"
            if not equil.exists():
                raise SystemExit(f"Equil mulliken not found: {equil}")
        runs.append((run_id, mulliken, equil))
    return runs


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def _peak_indices(values: np.ndarray, min_frac: float) -> List[int]:
    if values.size < 3:
        return []
    vmax = float(np.max(values))
    if vmax <= 0:
        return []
    thresh = vmax * min_frac
    peaks: List[int] = []
    for i in range(1, values.size - 1):
        if values[i] >= thresh and values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    return peaks




def _equil_mulliken_path(mulliken_path: Path, equil_dir: str) -> Path:
    return mulliken_path.parent.parent / equil_dir / "mulliken"


def _concat_with_equil(
    equil: Tuple[np.ndarray, np.ndarray, List[int], Dict[int, str]],
    prod: Tuple[np.ndarray, np.ndarray, List[int], Dict[int, str]],
    equil_duration_ps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, str]]:
    t_eq, c_eq, ids_eq, elems_eq = equil
    t_pr, c_pr, ids_pr, elems_pr = prod
    if ids_eq != ids_pr:
        raise SystemExit("Equil and production have different atom IDs; cannot combine.")
    if t_eq.size == 0:
        return prod
    if t_pr.size == 0:
        return equil
    t_eq_adj = t_eq
    if equil_duration_ps is not None and t_eq.size > 1:
        t0 = float(t_eq[0])
        t1 = float(t_eq[-1])
        span = t1 - t0
        if span > 0:
            scale = equil_duration_ps / span
            t_eq_adj = (t_eq - t0) * scale
        else:
            t_eq_adj = t_eq - t0

    t_pr_shift = t_pr + float(t_eq_adj[-1] - t_pr[0])
    t = np.concatenate([t_eq_adj, t_pr_shift])
    c = np.concatenate([c_eq, c_pr], axis=1)
    elems = elems_eq or elems_pr
    return t, c, ids_pr, elems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract water oxygen Mulliken charges and plot time series grid."
    )
    parser.add_argument(
        "--mulliken",
        type=Path,
        required=False,
        help="Path to the mulliken file.",
    )
    parser.add_argument(
        "--runs-path",
        type=Path,
        default=None,
        help="Path containing run-* directories (enables multi-run mode).",
    )
    parser.add_argument(
        "--cv-dir",
        type=str,
        default=None,
        help="CV directory under each run (used with --runs-path).",
    )
    parser.add_argument(
        "--run-start",
        type=int,
        default=None,
        help="First run id (inclusive) for multi-run mode.",
    )
    parser.add_argument(
        "--run-end",
        type=int,
        default=None,
        help="Last run id (inclusive) for multi-run mode.",
    )
    parser.add_argument(
        "--solute-atoms",
        type=int,
        required=True,
        help="Number of atoms in the solute (waters start after this index).",
    )
    parser.add_argument(
        "--natoms",
        type=int,
        default=None,
        help="Total number of atoms (optional; inferred if omitted).",
    )
    parser.add_argument(
        "--extra-ids",
        type=str,
        default=None,
        help="Comma-separated atom IDs to include in addition to solvent oxygens (e.g., '10,14').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports") / "water_oxygen_mulliken.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=Path("reports") / "water_oxygen_mulliken.npz",
        help="Output NPZ path for timeseries data.",
    )
    parser.add_argument(
        "--hist-out",
        type=Path,
        default=Path("reports") / "water_oxygen_mulliken_hist.html",
        help="Output HTML path for histogram panel.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--bimodal-min-peak-frac",
        type=float,
        default=0.25,
        help="Minimum peak height fraction (of max) to consider bimodal.",
    )
    parser.add_argument(
        "--include-equil",
        action="store_true",
        help="Prepend equil/mulliken as a continuous prefix.",
    )
    parser.add_argument(
        "--equil-dir",
        type=str,
        default="equil",
        help="Name of the equil directory (sibling of cv-dir).",
    )
    parser.add_argument(
        "--equil-duration-ps",
        type=float,
        default=None,
        help="If set, rescale equil times to end at this duration (ps).",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=Path("reports") / "water_oxygen_mulliken.html",
        help="Output HTML path for the time-series plot.",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=1,
        help="Moving average window size (number of points). Use 1 for raw data.",
    )
    parser.add_argument(
        "--drop-equil-first",
        action="store_true",
        help="Drop the first equil frame (useful if it has a spike).",
    )
    parser.add_argument(
        "--time-debug",
        action="store_true",
        help="Print equil/production time ranges for debugging.",
    )
    parser.add_argument(
        "--enable-multirun-html",
        action="store_true",
        help="Enable multi-run dropdown HTML (disabled by default).",
    )
    args = parser.parse_args()

    extra_ids = _parse_extra_ids(args.extra_ids)
    if args.runs_path is not None:
        if args.cv_dir is None or args.run_start is None or args.run_end is None:
            raise SystemExit("--runs-path requires --cv-dir, --run-start, and --run-end.")
        runs = _collect_run_inputs(
            args.runs_path,
            args.cv_dir,
            args.run_start,
            args.run_end,
            args.include_equil,
            args.equil_dir,
        )
        if not runs:
            raise SystemExit("No mulliken files found for the requested run range.")

        run_data: List[Tuple[int, np.ndarray, np.ndarray, List[int], Dict[int, str]]] = []
        base_ids: Optional[List[int]] = None
        base_elements: Optional[Dict[int, str]] = None
        for run_id, mulliken, equil_path in runs:
            prod = parse_mulliken(
                mulliken, args.solute_atoms, natoms=args.natoms, extra_ids=extra_ids
            )
            if equil_path is not None:
                equil = parse_mulliken(
                    equil_path, args.solute_atoms, natoms=args.natoms, extra_ids=extra_ids
                )
                if args.drop_equil_first:
                    t_eq, c_eq, ids_eq, elems_eq = equil
                    if t_eq.size > 1:
                        equil = (t_eq[1:], c_eq[:, 1:], ids_eq, elems_eq)
                times, charges, target_ids, elements = _concat_with_equil(
                    equil, prod, equil_duration_ps=args.equil_duration_ps
                )
            else:
                times, charges, target_ids, elements = prod

            if base_ids is None:
                base_ids = target_ids
                base_elements = elements
            elif target_ids != base_ids:
                raise SystemExit(
                    f"Run {run_id} has different atom IDs; cannot combine into a single panel."
                )
            run_data.append((run_id, times, charges, target_ids, elements))

        target_ids = base_ids or []
        color_scale = pc.sample_colorscale(
            [
                [0.0, "#0b1d5c"],
                [0.25, "#1e5aa8"],
                [0.5, "#2c8cc9"],
                [0.75, "#24b29a"],
                [1.0, "#d64550"],
            ],
            np.linspace(0, 1, max(1, len(target_ids))),
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            column_widths=[0.78, 0.22],
            horizontal_spacing=0.04,
        )

        traces_per_run = len(target_ids) * 2
        for run_idx, (run_id, times, charges, _ids, _elements) in enumerate(run_data):
            for i, atom_id in enumerate(target_ids):
                label = f"{(base_elements or {}).get(atom_id, 'X')}-{atom_id}"
                y = charges[i]
                y_plot = _moving_average(y, args.ma_window)
                color = color_scale[i % len(color_scale)]
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=y_plot,
                        mode="lines",
                        line=dict(width=1.0, color=color),
                        name=label,
                        showlegend=False,
                        visible=(run_idx == 0),
                        hovertemplate=f"{label}<br>t=%{{x:.3f}} ps<br>q=%{{y:.5f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                hist_counts, bin_edges = np.histogram(y, bins=args.bins, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                peaks = _peak_indices(hist_counts, args.bimodal_min_peak_frac)
                stats_label = ""
                if len(peaks) >= 2:
                    top2 = sorted(peaks, key=lambda idx: hist_counts[idx], reverse=True)[:2]
                    c1, c2 = sorted([bin_centers[top2[0]], bin_centers[top2[1]]])
                    mid = 0.5 * (c1 + c2)
                    left = y[y <= mid]
                    right = y[y > mid]
                    if left.size > 0 and right.size > 0:
                        mu1, sd1 = float(np.mean(left)), float(np.std(left))
                        mu2, sd2 = float(np.mean(right)), float(np.std(right))
                        stats_label = (
                            f"bimodal: μ1={mu1:.4f}, σ1={sd1:.4f}; μ2={mu2:.4f}, σ2={sd2:.4f}"
                        )
                if not stats_label:
                    mu, sd = float(np.mean(y)), float(np.std(y))
                    stats_label = f"unimodal: μ={mu:.4f}, σ={sd:.4f}"

                fig.add_trace(
                    go.Scatter(
                        x=hist_counts,
                        y=bin_centers,
                        mode="markers",
                        marker=dict(size=3, color=color, opacity=0.8),
                        name=label,
                        showlegend=False,
                        visible=(run_idx == 0),
                        hovertemplate=(
                            f"{label}<br>{stats_label}<br>q=%{{y:.5f}}<br>p=%{{x:.5f}}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=2,
                )

        fig.update_layout(
            title=f"Run {run_data[0][0]}: Mulliken charges",
            height=650,
            width=1400,
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=f"run-{run_id}",
                            method="update",
                            args=[
                                {
                                    "visible": [
                                        (run_idx == i)
                                        for i in range(len(run_data))
                                        for _ in range(traces_per_run)
                                    ]
                                },
                                {"title": f"Run {run_id}: Mulliken charges"},
                            ],
                        )
                        for run_idx, (run_id, *_rest) in enumerate(run_data)
                    ],
                    direction="down",
                    x=1.02,
                    y=1.0,
                    xanchor="left",
                    yanchor="top",
                )
            ],
        )
        fig.update_xaxes(title_text="t (ps)", row=1, col=1)
        fig.update_yaxes(title_text="charge (s+p)", row=1, col=1)
        fig.update_xaxes(title_text="probability density", row=1, col=2)
        fig.add_vline(x=10.0, line=dict(color="black", width=1), row=1, col=1)

        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(args.html_out, include_plotlyjs="cdn")
        print(f"Wrote {args.html_out}")
        return

    if args.mulliken is None:
        raise SystemExit("Provide --mulliken for single-run mode or --runs-path for multi-run mode.")

    prod = parse_mulliken(
        args.mulliken, args.solute_atoms, natoms=args.natoms, extra_ids=extra_ids
    )
    if args.include_equil:
        equil_path = _equil_mulliken_path(args.mulliken, args.equil_dir)
        if not equil_path.exists():
            raise SystemExit(f"Equil mulliken not found: {equil_path}")
        equil = parse_mulliken(
            equil_path, args.solute_atoms, natoms=args.natoms, extra_ids=extra_ids
        )
        if args.drop_equil_first:
            t_eq, c_eq, ids_eq, elems_eq = equil
            if t_eq.size > 1:
                equil = (t_eq[1:], c_eq[:, 1:], ids_eq, elems_eq)
        if args.time_debug:
            t_eq, _c_eq, _ids_eq, _elems_eq = equil
            t_pr, _c_pr, _ids_pr, _elems_pr = prod
            print(
                f"Equil: t0={t_eq[0]:.6f} ps, t1={t_eq[-1]:.6f} ps, n={t_eq.size}"
            )
            print(
                f"Prod:  t0={t_pr[0]:.6f} ps, t1={t_pr[-1]:.6f} ps, n={t_pr.size}"
            )
        times, charges, target_ids, base_elements = _concat_with_equil(
            equil, prod, equil_duration_ps=args.equil_duration_ps
        )
    else:
        times, charges, target_ids, base_elements = prod

    args.html_out.parent.mkdir(parents=True, exist_ok=True)

    # Time-series HTML for all selected atoms (hover labels).
    color_scale = pc.sample_colorscale(
        [
            [0.0, "#0b1d5c"],
            [0.25, "#1e5aa8"],
            [0.5, "#2c8cc9"],
            [0.75, "#24b29a"],
            [1.0, "#d64550"],
        ],
        np.linspace(0, 1, max(1, len(target_ids))),
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.78, 0.22],
        horizontal_spacing=0.04,
    )
    for i, atom_id in enumerate(target_ids):
        label = f"{base_elements.get(atom_id, 'X')}-{atom_id}"
        y = charges[i]
        y_plot = _moving_average(y, args.ma_window)
        color = color_scale[i % len(color_scale)]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=y_plot,
                mode="lines",
                line=dict(width=1.0, color=color),
                name=label,
                showlegend=False,
                hovertemplate=f"{label}<br>t=%{{x:.3f}} ps<br>q=%{{y:.5f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        hist_counts, bin_edges = np.histogram(y, bins=args.bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        peaks = _peak_indices(hist_counts, args.bimodal_min_peak_frac)
        stats_label = ""
        if len(peaks) >= 2:
            top2 = sorted(peaks, key=lambda idx: hist_counts[idx], reverse=True)[:2]
            c1, c2 = sorted([bin_centers[top2[0]], bin_centers[top2[1]]])
            mid = 0.5 * (c1 + c2)
            left = y[y <= mid]
            right = y[y > mid]
            if left.size > 0 and right.size > 0:
                mu1, sd1 = float(np.mean(left)), float(np.std(left))
                mu2, sd2 = float(np.mean(right)), float(np.std(right))
                stats_label = f"bimodal: μ1={mu1:.4f}, σ1={sd1:.4f}; μ2={mu2:.4f}, σ2={sd2:.4f}"
        if not stats_label:
            mu, sd = float(np.mean(y)), float(np.std(y))
            stats_label = f"unimodal: μ={mu:.4f}, σ={sd:.4f}"
        fig.add_trace(
            go.Scatter(
                x=hist_counts,
                y=bin_centers,
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.8),
                name=label,
                showlegend=False,
                hovertemplate=(
                    f"{label}<br>{stats_label}<br>q=%{{y:.5f}}<br>p=%{{x:.5f}}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Mulliken charges (all selected atoms)",
        height=650,
        width=1400,
    )
    fig.update_xaxes(title_text="t (ps)", row=1, col=1)
    fig.update_yaxes(title_text="charge (s+p)", row=1, col=1)
    fig.update_xaxes(title_text="probability density", row=1, col=2)
    fig.add_vline(x=10.0, line=dict(color="black", width=1), row=1, col=1)
    fig.write_html(args.html_out, include_plotlyjs="cdn")

    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
