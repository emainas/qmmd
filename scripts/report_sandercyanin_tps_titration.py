#!/usr/bin/env python3
"""Report every titratable site in the Sandercyanin monomer CpHMD run.

The protein calculation contains eight standard Amber titratable residues and
the custom five-state BLA/TPS residue. Each CPOUT record carries its
instantaneous pH, so populations can be collected directly in fixed-pH
ensembles even when PMEMD aborts during final cleanup.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from qmmd.cphmd_titr_report import TpsDecompositionConfig, _mean_block_sem, _plot_bla_tps


@dataclass(frozen=True)
class SiteDefinition:
    """One titratable residue in CPOUT/CPIN order."""

    residue_name: str
    topology_residue_id: int
    protonated_states: frozenset[int]
    state_names: tuple[str, ...]


TPS_STATE_NAMES = ("ppp", "app", "bpp", "cpp", "dpp")
SITE_DEFINITIONS = (
    SiteDefinition(
        "AS4",
        28,
        frozenset({1, 2, 3, 4}),
        ("deprotonated", "protonated-1", "protonated-2", "protonated-3", "protonated-4"),
    ),
    SiteDefinition("LYS", 38, frozenset({0}), ("protonated", "deprotonated")),
    SiteDefinition("TYR", 46, frozenset({0}), ("protonated", "deprotonated")),
    SiteDefinition(
        "GL4",
        60,
        frozenset({1, 2, 3, 4}),
        ("deprotonated", "protonated-1", "protonated-2", "protonated-3", "protonated-4"),
    ),
    SiteDefinition("LYS", 68, frozenset({0}), ("protonated", "deprotonated")),
    SiteDefinition("HIP", 89, frozenset({0}), ("HIP", "HID", "HIE")),
    SiteDefinition("TYR", 97, frozenset({0}), ("protonated", "deprotonated")),
    SiteDefinition("TYR", 123, frozenset({0}), ("protonated", "deprotonated")),
    SiteDefinition("BLA/TPS", 170, frozenset({0}), TPS_STATE_NAMES),
)
RESIDUE_RE = re.compile(
    r"^Residue\s+(?P<residue>\d+)\s+State:\s+(?P<state>\d+)\s+"
    r"pH:\s+(?P<ph>[-+0-9.eE]+)\s*$"
)


def _finish_record(
    path: Path,
    record_number: int,
    current: dict[int, tuple[float, int]],
    site_definitions: tuple[SiteDefinition, ...],
) -> tuple[int, float, tuple[int, ...]]:
    expected = set(range(len(site_definitions)))
    if set(current) != expected:
        raise ValueError(
            f"Incomplete CpH record in {path}: found residues {sorted(current)}; "
            f"expected {sorted(expected)}"
        )
    ph_values = {value[0] for value in current.values()}
    if len(ph_values) != 1:
        raise ValueError(f"Mixed pH labels in one CpH record in {path}: {ph_values}")
    states = tuple(current[index][1] for index in range(len(site_definitions)))
    for index, (state, site) in enumerate(zip(states, site_definitions)):
        if not 0 <= state < len(site.state_names):
            raise ValueError(
                f"Unexpected state {state} for CpH site {index} "
                f"({site.residue_name} {site.topology_residue_id}) in {path}"
            )
    return record_number, ph_values.pop(), states


def parse_cpout(
    path: Path, site_definitions: tuple[SiteDefinition, ...]
) -> tuple[list[tuple[int, float, tuple[int, ...]]], int]:
    """Return complete ``(record, pH, all-local-states)`` CPOUT records."""
    records: list[tuple[int, float, tuple[int, ...]]] = []
    current: dict[int, tuple[float, int]] = {}
    record_number = 0
    for line in path.read_text(errors="replace").splitlines():
        match = RESIDUE_RE.match(line)
        if match is None:
            continue
        residue = int(match.group("residue"))
        if residue == 0 and current:
            record_number += 1
            records.append(_finish_record(path, record_number, current, site_definitions))
            current = {}
        if residue in current:
            raise ValueError(f"Duplicate CpH residue {residue} in {path}")
        current[residue] = (float(match.group("ph")), int(match.group("state")))

    discarded_partial_records = 0
    if current:
        if len(current) != len(site_definitions):
            discarded_partial_records = 1
        else:
            record_number += 1
            records.append(_finish_record(path, record_number, current, site_definitions))
    return records, discarded_partial_records


def mean_block_sem_live(values: np.ndarray, requested_blocks: int) -> tuple[float, float]:
    """Return a mean/SEM while allowing sparse early live snapshots."""
    if values.size == 0:
        return math.nan, math.nan
    block_count = min(requested_blocks, len(values))
    if block_count == 1:
        return float(values.mean()), math.nan
    return _mean_block_sem(values, block_count)


def tps_population_rows(
    samples: list[tuple[int, int, float, tuple[int, ...]]],
    ph_values: tuple[float, ...],
    blocks: int,
    tps_index: int,
) -> list[dict[str, float]]:
    """Return sampled five-state TPS populations in the shared report format."""
    rows: list[dict[str, float]] = []
    for ph in ph_values:
        states = np.asarray(
            [
                all_states[tps_index]
                for _walker, _record, sample_ph, all_states in samples
                if sample_ph == ph
            ],
            dtype=int,
        )
        state_stats = {
            name: mean_block_sem_live((states == index).astype(float), blocks)
            for index, name in enumerate(TPS_STATE_NAMES)
        }
        deprotonated = states != 0
        deprot_mean, deprot_sem = mean_block_sem_live(deprotonated.astype(float), blocks)
        deprot_count = int(deprotonated.sum())
        row: dict[str, float] = {
            "pH": ph,
            "records": float(states.size),
            "ppp_fraction": state_stats["ppp"][0],
            "ppp_sem": state_stats["ppp"][1],
            "deprotonated_fraction": deprot_mean,
            "deprotonated_sem": deprot_sem,
            "deprotonated_samples": float(deprot_count),
        }
        for index, name in enumerate(TPS_STATE_NAMES[1:], start=1):
            mean, sem = state_stats[name]
            row[f"{name}_fraction_total"] = mean
            row[f"{name}_sem_total"] = sem
            row[f"{name}_fraction_of_deprotonated"] = (
                float(np.count_nonzero(states == index) / deprot_count)
                if deprot_count
                else math.nan
            )
        rows.append(row)
    return rows


def all_residue_population_rows(
    samples: list[tuple[int, int, float, tuple[int, ...]]],
    ph_values: tuple[float, ...],
    blocks: int,
    site_definitions: tuple[SiteDefinition, ...],
) -> list[dict[str, float | int | str]]:
    """Return aligned protonated fractions for every titratable residue."""
    rows: list[dict[str, float | int | str]] = []
    for site_index, site in enumerate(site_definitions):
        for ph in ph_values:
            indicator = np.asarray(
                [
                    int(all_states[site_index] in site.protonated_states)
                    for _walker, _record, sample_ph, all_states in samples
                    if sample_ph == ph
                ],
                dtype=float,
            )
            mean, sem = mean_block_sem_live(indicator, blocks)
            rows.append(
                {
                    "pH": ph,
                    "cpin_residue_index": site_index,
                    "topology_residue_id": site.topology_residue_id,
                    "residue": site.residue_name,
                    "records": int(indicator.size),
                    "protonated_fraction": mean,
                    "protonated_sem": sem,
                }
            )
    return rows


def site_state_population_rows(
    samples: list[tuple[int, int, float, tuple[int, ...]]],
    ph_values: tuple[float, ...],
    blocks: int,
    site_index: int,
    site: SiteDefinition,
) -> list[dict[str, float]]:
    """Return every local-state population for one multistate residue."""
    rows: list[dict[str, float]] = []
    for ph in ph_values:
        states = np.asarray(
            [
                all_states[site_index]
                for _walker, _record, sample_ph, all_states in samples
                if sample_ph == ph
            ],
            dtype=int,
        )
        row: dict[str, float] = {"pH": ph, "records": float(states.size)}
        for state_index, state_name in enumerate(site.state_names):
            mean, sem = mean_block_sem_live(
                (states == state_index).astype(float), blocks
            )
            key = state_name.lower()
            row[f"{key}_fraction"] = mean
            row[f"{key}_sem"] = sem
        rows.append(row)
    return rows


def read_ph_ladder(run_dir: Path) -> tuple[float, ...]:
    """Read and validate the replica/pH ladder prepared for a run."""
    path = run_dir / "ph-ladder.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pH ladder: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    replicas = [int(row["replica"]) for row in rows]
    if replicas != list(range(1, len(rows) + 1)):
        raise ValueError(f"Replica numbering is not contiguous in {path}")
    ph_values = tuple(float(row["pH"]) for row in rows)
    if ph_values != tuple(sorted(set(ph_values))):
        raise ValueError(f"pH ladder must be unique and increasing in {path}")
    return ph_values


def replica_number(path: Path) -> int:
    match = re.fullmatch(r"replica-(\d+)\.cpout", path.name)
    if match is None:
        raise ValueError(f"Unexpected CPOUT filename: {path.name}")
    return int(match.group(1))


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_all_residue_panels(
    path: Path,
    rows: list[dict[str, float | int | str]],
    tps_rows: list[dict[str, float]],
    hip_rows: list[dict[str, float]],
    site_definitions: tuple[SiteDefinition, ...],
    style: Path,
) -> None:
    """Plot sampled titration populations for all nine Sandercyanin sites."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.6, 10.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    for site_index, (site, axis) in enumerate(zip(site_definitions[:-1], axes_flat[:-1])):
        site_rows = [row for row in rows if row["cpin_residue_index"] == site_index]
        ph = np.asarray([float(row["pH"]) for row in site_rows])
        if site.residue_name == "HIP":
            hip_colors = ("#111111", "#D55E00", "#0072B2")
            hip_markers = ("o", "s", "^")
            for state_name, color, marker in zip(
                site.state_names, hip_colors, hip_markers
            ):
                key = state_name.lower()
                axis.errorbar(
                    ph,
                    [row[f"{key}_fraction"] for row in hip_rows],
                    yerr=[row[f"{key}_sem"] for row in hip_rows],
                    marker=marker,
                    ms=3.7,
                    lw=1.5,
                    capsize=1.8,
                    color=color,
                    label=state_name,
                )
            axis.legend(fontsize=7.5, ncol=3, loc="lower left")
        else:
            fraction = np.asarray([float(row["protonated_fraction"]) for row in site_rows])
            sem = np.asarray([float(row["protonated_sem"]) for row in site_rows])
            axis.errorbar(
                ph,
                fraction,
                yerr=sem,
                marker="o",
                ms=4.0,
                lw=1.7,
                capsize=2.0,
                color="#2B6CB0",
                label="Protonated",
            )
        axis.axhline(0.5, color="0.62", ls="--", lw=0.9)
        axis.set_title(f"{site.residue_name} {site.topology_residue_id}")
        axis.grid(alpha=0.18)

    tps_axis = axes_flat[-1]
    ph = np.asarray([row["pH"] for row in tps_rows])
    tps_axis.errorbar(
        ph,
        [row["ppp_fraction"] for row in tps_rows],
        yerr=[row["ppp_sem"] for row in tps_rows],
        marker="o",
        ms=3.7,
        lw=1.6,
        capsize=1.8,
        color="#111111",
        label="PPP",
    )
    tps_axis.errorbar(
        ph,
        [row["deprotonated_fraction"] for row in tps_rows],
        yerr=[row["deprotonated_sem"] for row in tps_rows],
        marker="s",
        ms=3.5,
        lw=1.5,
        capsize=1.8,
        color="#777777",
        label="Total deprot.",
    )
    colors = ("#D55E00", "#009E73", "#0072B2", "#CC79A7")
    markers = ("^", "v", "D", "P")
    for name, color, marker in zip(TPS_STATE_NAMES[1:], colors, markers):
        tps_axis.plot(
            ph,
            [row[f"{name}_fraction_total"] for row in tps_rows],
            marker=marker,
            ms=3.2,
            lw=1.2,
            color=color,
            label=name.upper(),
        )
    tps_axis.axhline(0.5, color="0.62", ls="--", lw=0.9)
    tps_axis.set_title("BLA/TPS 170")
    tps_axis.legend(fontsize=7.0, ncol=2, loc="center left")
    tps_axis.grid(alpha=0.18)

    tick_ph = ph if len(ph) <= 12 else ph[::2]
    for axis in axes_flat:
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(tick_ph)
    for axis in axes[-1, :]:
        axis.set_xlabel("pH")
    for axis in axes[:, 0]:
        axis.set_ylabel("Molar fraction")
    fig.suptitle("Sandercyanin monomer: sampled CpHMD titration populations", fontsize=15)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--style", type=Path, default=Path("plotting/lefteris.mplstyle"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Output directory; defaults to RUN_DIR/post/report",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Label the output as a live, incomplete CPOUT snapshot",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    expected_ph = read_ph_ladder(run_dir)
    cpouts = sorted(run_dir.glob("replica-*.cpout"), key=replica_number)
    if not cpouts:
        raise ValueError(
            f"No CPOUT files exist in {run_dir}; PMEMD has not produced data yet"
        )
    if len(cpouts) != len(expected_ph):
        raise ValueError(
            f"Expected {len(expected_ph)} replica CPOUT files in {run_dir}; "
            f"found {len(cpouts)}"
        )
    samples: list[tuple[int, int, float, tuple[int, ...]]] = []
    discarded_partial_records = 0
    empty_cpouts: list[str] = []
    for cpout in cpouts:
        walker = replica_number(cpout)
        walker_records, discarded = parse_cpout(cpout, SITE_DEFINITIONS)
        if not walker_records:
            empty_cpouts.append(cpout.name)
        discarded_partial_records += discarded
        samples.extend((walker, record, ph, states) for record, ph, states in walker_records)

    if not samples:
        raise ValueError(f"No complete CpH records are available yet in {run_dir}")
    observed_ph = {ph for _walker, _record, ph, _states in samples}
    unexpected_ph = observed_ph.difference(expected_ph)
    if unexpected_ph:
        raise ValueError(f"CPOUT contains pH values outside the ladder: {unexpected_ph}")
    ph_values = expected_ph

    post = run_dir / "post"
    report = args.report_dir.resolve() if args.report_dir else post / "report"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "protonation-states.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_replica",
                "source_record",
                "pH",
                "cpin_residue_index",
                "topology_residue_id",
                "residue",
                "state",
                "state_name",
                "protonated",
            ]
        )
        for walker, record, ph, states in samples:
            for site_index, (state, site) in enumerate(zip(states, SITE_DEFINITIONS)):
                writer.writerow(
                    [
                        walker,
                        record,
                        f"{ph:.6f}",
                        site_index,
                        site.topology_residue_id,
                        site.residue_name,
                        state,
                        site.state_names[state],
                        int(state in site.protonated_states),
                    ]
                )

    tps_index = len(SITE_DEFINITIONS) - 1
    tps_rows = tps_population_rows(samples, ph_values, args.blocks, tps_index)
    write_dict_csv(report / "tps-titration-tautomers.csv", tps_rows)
    all_rows = all_residue_population_rows(samples, ph_values, args.blocks, SITE_DEFINITIONS)
    write_dict_csv(report / "all-residue-titration-panels.csv", all_rows)
    hip_index = next(
        index for index, site in enumerate(SITE_DEFINITIONS) if site.residue_name == "HIP"
    )
    hip_rows = site_state_population_rows(
        samples, ph_values, args.blocks, hip_index, SITE_DEFINITIONS[hip_index]
    )
    write_dict_csv(report / "hip-titration-states.csv", hip_rows)

    style = args.style.resolve()
    if not style.is_file():
        raise FileNotFoundError(f"Missing matplotlib style: {style}")
    plot_all_residue_panels(
        report / "all-residue-titration-panels.png",
        all_rows,
        tps_rows,
        hip_rows,
        SITE_DEFINITIONS,
        style,
    )
    tps = TpsDecompositionConfig(
        residue_id=SITE_DEFINITIONS[tps_index].topology_residue_id,
        residue_name="BLA",
        protonated_name="ppp",
        deprotonated_names=TPS_STATE_NAMES[1:],
        protonated_state=0,
        deprotonated_states=(1, 2, 3, 4),
        microstate_model_pkas=None,
    )
    _plot_bla_tps(
        SimpleNamespace(style=style),
        report / "tps-titration-tautomers.png",
        tps_rows,
        tps,
    )

    (report / "tps-microstate-fit.yaml").write_text(
        yaml.safe_dump(
            {
                "source": str(run_dir),
                "method": "direct fixed-pH aggregation from CPOUT pH labels",
                "samples_per_pH": {f"{row['pH']:.1f}": int(row["records"]) for row in tps_rows},
                "discarded_truncated_eof_records": discarded_partial_records,
                "empty_cpout_files": empty_cpouts,
                "live_snapshot": args.live,
                "microscopic_pKa": None,
                "macroscopic_pKa": None,
                "fit_error": "No deprotonated TPS states were sampled.",
                "theoretical_microstate_overlay": False,
                "limitation": (
                    "Live snapshot from complete CPOUT records only; populations and "
                    "block SEM values will change as sampling continues."
                    if args.live
                    else "Report uses every complete CPOUT state record."
                ),
            },
            sort_keys=False,
        )
    )
    print(f"OK: wrote all-residue Sandercyanin report to {report}")
    print(
        f"OK: {len(samples)} aligned CpH records; "
        f"{min(int(row['records']) for row in tps_rows)}-"
        f"{max(int(row['records']) for row in tps_rows)} per pH"
    )
    print(f"OK: discarded {discarded_partial_records} truncated EOF record(s)")


if __name__ == "__main__":
    main()
