#!/usr/bin/env python3
"""Plot Mulliken (s+p) time series for all atoms in a single run."""

from __future__ import annotations

import argparse
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


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Mulliken (s+p) time series for all atoms.")
    p.add_argument("--runs-path", required=True, type=Path)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--solute-atoms", required=True, type=int, help="Plot atoms 1..solute_atoms")
    p.add_argument("--t-max", type=float, default=None, help="Cap time series at this time (ps)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    mulliken_path = args.runs_path / f"run-{args.run_id}" / args.run_dir / "mulliken"
    if not mulliken_path.exists():
        raise SystemExit(f"Missing mulliken file: {mulliken_path}")

    times, charges, target_ids, elements_map = parse_mulliken_all(
        mulliken_path, args.solute_atoms, tmax=args.t_max
    )

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

    system = infer_system(args.runs_path)
    out = args.out
    if out is None:
        out = Path("reports") / f"{system}_{args.run_dir}_mulliken_all_run{args.run_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
