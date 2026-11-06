#!/usr/bin/env python3
"""Audit bulk-water sampling from saved defect--solute distance series.

The audit separates merely observing a distance above the bulk threshold from
sampling a continuous outward path into bulk-like water.  It reads numerical
data already saved by the joint-summary workflows and never touches raw
simulation inputs or trajectories.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    label: str
    system: str
    path: Path
    start_ps: float
    stop_ps: float
    note: str = ""


@dataclass(frozen=True)
class AuditResult:
    label: str
    system: str
    start_ps: float
    stop_ps: float
    frame_interval_ps: float
    sampled_frames: int
    assigned_frames: int
    max_distance_A: float
    frames_beyond_threshold: int
    fraction_beyond_threshold: float
    longest_residence_ps: float
    longest_start_ps: float
    longest_stop_ps: float
    outward_crossings: int
    continuous_outward_crossings: int
    smallest_crossing_jump_A: float
    largest_crossing_jump_A: float
    threshold_reached: bool
    sustained_bulk: bool
    continuous_outward_path: bool
    qualified_bulk_sampling: bool
    note: str


def default_specs() -> tuple[RunSpec, ...]:
    """Return the named runs and their saved, aligned analysis windows."""
    bv = ROOT / "reports/BV/joint_summary_b_and_c"
    hist = ROOT / "reports/HIST/joint_summary_hie_and_hid"
    prn = ROOT / "reports/PRN/joint_summary_anti_and_syn"
    return (
        RunSpec("C-39", "BV ring C", bv / "C-39_nearest_solute_distance.csv", 46.5425, 52.65),
        RunSpec("C-57", "BV ring C", bv / "C-57_nearest_solute_distance.csv", 46.04, 52.25),
        RunSpec("C-99", "BV ring C", bv / "C-99_nearest_solute_distance.csv", 46.04, 52.50),
        RunSpec("C-142", "BV ring C", bv / "C-142_nearest_solute_distance.csv", 44.885, 49.55),
        RunSpec(
            "B-171",
            "BV ring B",
            bv / "B-171_nearest_solute_distance.csv",
            46.26,
            54.75,
            "Included as a check on the value quoted in the draft.",
        ),
        RunSpec("HIE-20", "HIST", hist / "HIE-20_distance.csv", 47.485, 53.50),
        RunSpec("HID-1", "HIST", hist / "HID-1_distance.csv", 45.20, 48.7695),
        RunSpec("HID-49", "HIST", hist / "HID-49_distance.csv", 45.72, 52.50),
        RunSpec("anti-15", "propionic acid", prn / "anti-15_distance.csv", 45.70, 49.25),
        RunSpec("anti-93", "propionic acid", prn / "anti-93_distance.csv", 44.923, 48.746),
        RunSpec("syn-6", "propionic acid", prn / "syn-6_distance.csv", 43.36, 45.75),
        RunSpec("syn-44", "propionic acid", prn / "syn-44_distance.csv", 43.84, 47.75),
    )


def read_distance_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read aligned time, assigned distance, and display distance."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")

    def values(column: str) -> np.ndarray:
        return np.asarray(
            [float(row[column]) if row.get(column, "") not in {"", "nan", "NaN"} else np.nan for row in rows],
            dtype=float,
        )

    return values("aligned_time_ps"), values("distance_A"), values("display_distance_A")


def true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index pairs for contiguous True segments."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), stops.tolist()))


def audit(
    spec: RunSpec,
    threshold_A: float,
    sustained_ps: float,
    max_crossing_jump_A: float,
) -> AuditResult:
    """Audit one saved distance series over its reaction-to-end window."""
    time, assigned, display = read_distance_csv(spec.path)
    in_window = (time >= spec.start_ps - 1.0e-8) & (time <= spec.stop_ps + 1.0e-8)
    time = time[in_window]
    assigned = assigned[in_window]
    display = display[in_window]
    if time.size < 2:
        raise ValueError(f"Insufficient samples for {spec.label}")
    dt = float(np.median(np.diff(time)))
    beyond = np.isfinite(display) & (display > threshold_A)
    segments = true_segments(beyond)
    segment_rows = [
        (stop - start + 1, time[start], time[stop], (stop - start + 1) * dt)
        for start, stop in segments
    ]
    longest = max(segment_rows, default=(0, np.nan, np.nan, 0.0), key=lambda row: row[3])

    crossing_indices = np.flatnonzero(beyond & ~np.r_[False, beyond[:-1]])
    crossing_indices = crossing_indices[crossing_indices > 0]
    jumps = display[crossing_indices] - display[crossing_indices - 1]
    # A resolved crossing begins in the second layer and does not advance by
    # more than the configurable per-frame displacement.
    continuous = (
        np.isfinite(display[crossing_indices - 1])
        & (display[crossing_indices - 1] >= 3.5)
        & (display[crossing_indices - 1] <= threshold_A)
        & (jumps >= 0.0)
        & (jumps <= max_crossing_jump_A)
    )
    threshold_reached = bool(np.any(beyond))
    sustained_bulk = bool(longest[3] >= sustained_ps - 1.0e-8)
    continuous_outward_path = bool(np.any(continuous))
    qualified = threshold_reached and sustained_bulk and continuous_outward_path

    return AuditResult(
        label=spec.label,
        system=spec.system,
        start_ps=spec.start_ps,
        stop_ps=spec.stop_ps,
        frame_interval_ps=dt,
        sampled_frames=int(time.size),
        assigned_frames=int(np.count_nonzero(np.isfinite(assigned))),
        max_distance_A=float(np.nanmax(display)),
        frames_beyond_threshold=int(np.count_nonzero(beyond)),
        fraction_beyond_threshold=float(np.mean(beyond)),
        longest_residence_ps=float(longest[3]),
        longest_start_ps=float(longest[1]),
        longest_stop_ps=float(longest[2]),
        outward_crossings=int(crossing_indices.size),
        continuous_outward_crossings=int(np.count_nonzero(continuous)),
        smallest_crossing_jump_A=float(np.min(jumps)) if jumps.size else np.nan,
        largest_crossing_jump_A=float(np.max(jumps)) if jumps.size else np.nan,
        threshold_reached=threshold_reached,
        sustained_bulk=sustained_bulk,
        continuous_outward_path=continuous_outward_path,
        qualified_bulk_sampling=qualified,
        note=spec.note,
    )


def save_csv(results: list[AuditResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AuditResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def save_markdown(
    results: list[AuditResult],
    path: Path,
    threshold_A: float,
    sustained_ps: float,
    max_crossing_jump_A: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bulk-water filter audit",
        "",
        "This read-only audit uses the periodic nearest-solute distance series saved by the joint-summary workflows. "
        "The analysis starts at the charge/CV reaction marker and ends at the established run-specific analysis endpoint.",
        "",
        "A run *reaches* bulk-like water when the displayed defect distance exceeds "
        f"{threshold_A:g} Å. A conservative qualified event additionally requires at least {sustained_ps:g} ps "
        "of continuous residence and at least one resolved outward crossing from the 3.5–5.5 Å second layer "
        f"with a one-frame radial increment no larger than {max_crossing_jump_A:g} Å. The latter prevents a "
        "concerted change of defect identity or direct placement in a remote pocket from being mistaken for "
        "sampling of the intervening reaction tube.",
        "",
        "`display_distance_A` is used because the draft residence times use the established carry-forward display "
        "policy: when a charge assignment is absent, the last assigned defect oxygen is retained and its distance "
        "is recomputed in the current trajectory frame. No identity is invented before the first assignment. "
        "The assigned-frame count is reported so this convention remains visible.",
        "",
        "| Run | System | max d (Å) | longest d>5.5 (ps) | crossings | resolved crossings | threshold | qualified |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.label} | {result.system} | {result.max_distance_A:.2f} | "
            f"{result.longest_residence_ps:.2f} | {result.outward_crossings} | "
            f"{result.continuous_outward_crossings} | {yes_no(result.threshold_reached)} | "
            f"{yes_no(result.qualified_bulk_sampling)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- C-57 confirms the proposed concerted/direct-placement interpretation: its only outward crossing is "
        "5.45→8.26 Å in one 0.02 ps frame (Δd=2.82 Å), so it reaches the geometric bulk region but does not "
        "resolve passage through the outer second layer.",
        "- C-142 never crosses 5.5 Å (maximum 5.32 Å).",
        "- C-39 and C-99 both cross the boundary through resolved outward motion and sustain bulk-like residence. "
        "C-99 reproduces the quoted 0.68 ps residence. The current saved C-39 carry-forward series gives 2.12 ps, "
        "not 0.60 ps; assigned frames without carry-forward give 0.24 ps. The 0.60 ps value therefore needs its "
        "original gap-handling/window definition recovered before publication.",
        "- The HIST and propionic-acid conclusions for every requested run are reported in the table; a threshold "
        "crossing and qualified bulk sampling remain separate decisions.",
        "- The present B-171 cache does not reproduce 1.26 ps: it gives 2.22 ps with carry-forward (0.14 ps for "
        "strictly consecutive assigned frames). That draft value also requires provenance reconciliation.",
        "",
        "## Provenance limitation for the 31 ring-C survivors",
        "",
        "The aggregate funnel currently records the revised count (31) but not the revised 31 run IDs. "
        "`run-1-200-classification.csv` is an older run-level classification that still rejects C-39 and C-142 "
        "and names a different set of finalists. Consequently, the claim that exactly two of all 31 qualify cannot "
        "be certified from the current saved provenance without reconstructing or supplying the revised survivor "
        "list. The four named current finalists are fully audited above; among them, exactly C-39 and C-99 qualify.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-threshold", type=float, default=5.5, help="Bulk boundary in Å (default: 5.5).")
    parser.add_argument(
        "--minimum-residence",
        type=float,
        default=0.25,
        help="Minimum continuous bulk residence in ps (default: 0.25).",
    )
    parser.add_argument(
        "--maximum-crossing-jump",
        type=float,
        default=1.0,
        help="Maximum one-frame radial increment for a resolved crossing in Å (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/bulk-water-filter-audit",
        help="Directory for the CSV and Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [
        audit(spec, args.bulk_threshold, args.minimum_residence, args.maximum_crossing_jump)
        for spec in default_specs()
    ]
    save_csv(results, args.output_dir / "named-runs.csv")
    save_markdown(
        results,
        args.output_dir / "README.md",
        args.bulk_threshold,
        args.minimum_residence,
        args.maximum_crossing_jump,
    )
    for result in results:
        print(
            f"{result.label:8s} max={result.max_distance_A:5.2f} Å "
            f"longest={result.longest_residence_ps:5.2f} ps "
            f"resolved={result.continuous_outward_crossings:2d} "
            f"qualified={yes_no(result.qualified_bulk_sampling)}"
        )


if __name__ == "__main__":
    main()
