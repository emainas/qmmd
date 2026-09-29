#!/usr/bin/env python3
"""Combine selected BV B/C metadynamics runs without modifying their summaries.

The default profile reproduces the original joint summary.  The focused
``b171-c99-lcod`` profile compares one B and one C trajectory and adds the
independently calculated B/C LCOD PMF.
"""
import argparse
from pathlib import Path
import csv
import runpy
import matplotlib.pyplot as plt
import numpy as np
import json
from acid_base_BV import iter_xyz_frames, read_box_lengths_from_dftb_inp, _minimum_image

from correct_summary import ROOT, load_run, render, save_csv, smooth_helicity

EXCLUDED = {'meta-hib': {56, 98}, 'meta-hic': {57}}
FOCUSED_RUNS = {'meta-hib': {171}, 'meta-hic': {99}}
SITE_LABELS = {'meta-hib': 'B', 'meta-hic': 'C'}
FOCUSED_COLORS = {'B-171': '#2166AC', 'C-99': '#B2182B'}

def nearest_solute_distance(d: dict, out: Path) -> tuple[np.ndarray, np.ndarray]:
    """Match cached defect identities to actual frames; fill distances, not IDs."""
    cached = out / f"{d['label']}_nearest_solute_distance.csv"
    if cached.exists():
        a = np.genfromtxt(cached, delimiter=',', names=True)
        end = d['marker'] + 1.75
        if (np.isclose(a['aligned_time_ps'][0],40.,atol=.03)
                and a['aligned_time_ps'][-1] >= end-.03
                and np.allclose(a['aligned_time_ps']-a['raw_time_ps'],d['display_offset'])):
            keep=a['aligned_time_ps']<=end+1e-8
            return a['aligned_time_ps'][keep],a['display_distance_A'][keep]
    assignments = out / f"{d['label']}_coordination_wires.csv"
    with assignments.open() as handle:
        rows = list(csv.DictReader(handle))
    mapping = {round(float(r["raw_time_ps"]), 8): float(r["defect_oxygen_id"]) for r in rows}
    end = d["raw_marker"] + 1.75
    box = read_box_lengths_from_dftb_inp(Path(d["source"]) / "dftb.inp")
    result, export = [], []
    previous = np.nan
    for time, coords in iter_xyz_frames(Path(d["source"]) / "traject"):
        if time > end + 1e-8:
            break
        key = round(time, 8)
        if key not in mapping:
            continue
        defect = mapping[key]
        distance, atom = np.nan, np.nan
        if np.isfinite(defect):
            distances = np.linalg.norm(_minimum_image(coords[:78] - coords[int(defect)-1], box), axis=1)
            atom = int(np.argmin(distances)) + 1
            distance = float(distances[atom-1])
            previous = distance
        aligned = time + d["display_offset"]
        carried = not np.isfinite(distance) and np.isfinite(previous)
        result.append((aligned, previous))
        export.append((d["label"], time, aligned, defect, atom, distance, previous, int(carried)))
    save_csv(out / f"{d['label']}_nearest_solute_distance.csv",
             ["label", "raw_time_ps", "aligned_time_ps", "defect_oxygen_id", "nearest_solute_atom_id",
              "distance_A", "display_distance_A", "carried_forward"], export)
    a = np.asarray(result)
    if not len(a) or a[-1, 0] < d["marker"] + 1.75 - .03:
        raise ValueError(f"Distance assignments do not cover endpoint: {d['label']}")
    return a[:, 0], a[:, 1]


def load_joint_data(run_filter: dict[str, set[int]] | None = None) -> list[dict]:
    """Load selected runs with the clock and reaction alignment used originally."""
    data: list[dict] = []
    for cv, label in [("meta-hib", "B"), ("meta-hic", "C")]:
        base = ROOT / "reports/BV" / cv / "acid_base"
        combined = base / "correct_summary"
        selected = runpy.run_path(str(combined / "rerun.py"))["SELECTED"]
        with (base / "index.csv").open() as handle:
            entries = {
                int(row["run"]): row
                for row in csv.DictReader(handle)
                if row["bench_tag"] == "N1T48C1"
            }
        with (combined / "reaction_markers.csv").open() as handle:
            markers = {
                int(row["run"]): float(row["treact_raw_ps"])
                for row in csv.DictReader(handle)
            }
        for run, tdiff in selected:
            if run_filter is None:
                keep = run not in EXCLUDED[cv]
            else:
                keep = run in run_filter[cv]
            if not keep:
                continue
            d = load_run(base, entries[run], 10, 1.75, tdiff)
            d["treact"] = markers[run] + d["display_offset"]
            d["label"] = f"{label}-{run}"
            data.append(d)
    return data


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def deprotonation_barrier(block: np.ndarray) -> dict[str, float]:
    """Measure the s≈1 protonated-basin to s≈0 deprotonated-basin barrier."""
    coordinate = np.asarray(block[:, 0], dtype=float)
    energy = np.asarray(block[:, 1], dtype=float)
    finite = np.isfinite(coordinate) & np.isfinite(energy)
    coordinate, energy = coordinate[finite], energy[finite]
    deprotonated = np.flatnonzero((coordinate >= 0.0) & (coordinate <= 0.1))
    protonated = np.flatnonzero((coordinate >= 0.9) & (coordinate <= 1.0))
    if not deprotonated.size or not protonated.size:
        raise ValueError("F(s) does not cover both endpoint basins")
    deprotonated_index = int(deprotonated[np.argmin(energy[deprotonated])])
    protonated_index = int(protonated[np.argmin(energy[protonated])])
    first, last = sorted((deprotonated_index, protonated_index))
    transition_index = first + int(np.argmax(energy[first:last + 1]))
    return {
        "deprotonated_s": float(coordinate[deprotonated_index]),
        "deprotonated_f": float(energy[deprotonated_index]),
        "protonated_s": float(coordinate[protonated_index]),
        "protonated_f": float(energy[protonated_index]),
        "transition_s": float(coordinate[transition_index]),
        "transition_f": float(energy[transition_index]),
        "deprotonation_barrier": float(
            energy[transition_index] - energy[protonated_index]
        ),
    }


def focused_auxiliary_data(
    d: dict, source_dir: Path, output_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read audited distance/wire caches and export the focused aligned subset."""
    end = d["marker"] + 1.75
    distance_rows = read_csv_rows(source_dir / f"{d['label']}_nearest_solute_distance.csv")
    wire_rows = read_csv_rows(source_dir / f"{d['label']}_coordination_wires.csv")

    distance_export: list[tuple] = []
    for row in distance_rows:
        time = float(row["aligned_time_ps"])
        if time <= end + 1e-8:
            distance_export.append((
                d["label"], d["run"], float(row["raw_time_ps"]), time,
                row["defect_oxygen_id"], row["nearest_solute_atom_id"],
                row["distance_A"], row["display_distance_A"], row["carried_forward"],
            ))
    wire_export: list[tuple] = []
    for row in wire_rows:
        time = float(row["aligned_time_ps"])
        if time <= end + 1e-8:
            bridging = float(row["hbond_bridging_waters"])
            bonds = bridging + 1 if np.isfinite(bridging) else np.nan
            wire_export.append((
                d["label"], d["run"], float(row["raw_time_ps"]), time,
                row["coordination_s"], row["defect_oxygen_id"],
                row["binding_nitrogen_id"], bridging, bonds,
                row["oxygen_path_ids"],
            ))

    save_csv(
        output_dir / f"{d['label']}_defect_distance.csv",
        ["label", "run", "raw_time_ps", "aligned_time_ps", "defect_oxygen_id",
         "nearest_solute_atom_id", "distance_A", "display_distance_A", "carried_forward"],
        distance_export,
    )
    save_csv(
        output_dir / f"{d['label']}_hydrogen_bonds.csv",
        ["label", "run", "raw_time_ps", "aligned_time_ps", "coordination_s",
         "defect_oxygen_id", "binding_nitrogen_id", "bridging_water_count",
         "hydrogen_bond_count", "oxygen_path_ids"],
        wire_export,
    )
    distance_time = np.asarray([row[3] for row in distance_export], dtype=float)
    distance = np.asarray([float(row[7]) for row in distance_export], dtype=float)
    wire_time = np.asarray([row[3] for row in wire_export], dtype=float)
    bonds = np.asarray([float(row[8]) for row in wire_export], dtype=float)
    return distance_time, distance, wire_time, bonds


def render_focused() -> Path:
    """Render the six-panel B171/C99 report and its aligned numerical data."""
    data = load_joint_data(FOCUSED_RUNS)
    if {d["label"] for d in data} != set(FOCUSED_COLORS):
        raise ValueError("Focused selection did not resolve exactly B-171 and C-99")

    output_root = ROOT / "reports/tetrapyrrole_lcod_pmf"
    output_dir = output_root / "b171_c99_joint_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot = output_root / "b171_c99_joint_summary.png"
    source_dir = ROOT / "reports/BV/joint_summary_b_and_c"

    pmf_rows = read_csv_rows(output_root / "three_pmf_curves.csv")
    pmf = [row for row in pmf_rows if row["pair"] == "B ⇌ C"]
    if not pmf:
        raise ValueError("B ⇌ C curve is absent from three_pmf_curves.csv")
    lcod = np.asarray([float(row["lcod_A"]) for row in pmf])
    free_energy = np.asarray([float(row["shifted_pmf_kcal_mol"]) for row in pmf])
    summary_rows = read_csv_rows(output_root / "three_pmf_summary.csv")
    pmf_summary = next(row for row in summary_rows if row["pair"] == "B ⇌ C")

    plt.style.use(ROOT / "plotting/lefteris.mplstyle")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), layout="constrained")
    ax_fes, ax_pka, ax_pmf, ax_helix, ax_distance, ax_hbond = axes.flat
    fes_export: list[tuple] = []
    pka_export: list[tuple] = []
    helicity_export: list[tuple] = []
    all_distances: list[tuple] = []
    all_hbonds: list[tuple] = []
    endpoint_export: list[tuple] = []
    barrier_export: list[tuple] = []

    for d in data:
        color = FOCUSED_COLORS[d["label"]]
        endpoint = d["marker"] + 1.75
        features = deprotonation_barrier(d["fes"])
        ax_fes.plot(
            d["fes"][:, 0], d["fes"][:, 1], color=color, lw=3.0,
            label=(f"{d['label']} · "
                   + rf"$\Delta G^\ddagger_{{\rm deprot}}$={features['deprotonation_barrier']:.2f}"),
        )
        ax_fes.scatter(
            [features["transition_s"]], [features["transition_f"]],
            color=color, s=42, edgecolor="white", linewidth=.8, zorder=5,
        )
        pka = d["df"] / d["factor"]
        endpoint_index = int(np.flatnonzero(d["time"] <= endpoint + 1e-8)[-1])
        endpoint_time = float(d["time"][endpoint_index])
        endpoint_pka = float(pka[endpoint_index])
        ax_pka.plot(
            d["time"], pka, color=color, lw=2.2,
            label=(f"{d['label']}: "
                   + rf"p$K_a(t_{{\rm diff}}+t_{{\rm stop}})\simeq${endpoint_pka:.2f}"),
        )
        ax_pka.scatter(
            [endpoint_time], [endpoint_pka], color=color, s=42,
            edgecolor="white", linewidth=.8, zorder=5,
        )
        ax_pka.axvline(d["treact"], color=color, ls="--", lw=1.4, alpha=.8)

        smoothed = smooth_helicity(d["helix_time"], d["helicity"], .2)
        ax_helix.plot(d["helix_time"], smoothed, color=color, lw=2.2, label=d["label"])
        ax_helix.axvline(d["treact"], color=color, ls="--", lw=1.4, alpha=.8)

        distance_time, distance, wire_time, bonds = focused_auxiliary_data(
            d, source_dir, output_dir
        )
        ax_distance.scatter(
            distance_time, distance, color=color, s=10, linewidths=0,
            alpha=.82, label=d["label"],
        )
        ax_hbond.scatter(
            wire_time, bonds, color=color, s=10, linewidths=0,
            alpha=.82, label=d["label"],
        )

        fes_export.extend((d["label"], d["run"], d["snapshot_time"], s, f)
                          for s, f in d["fes"][:, :2])
        pka_export.extend((d["label"], d["run"], t, value, d["factor"])
                          for t, value in zip(d["time"], pka))
        helicity_export.extend((d["label"], d["run"], t, raw, smooth)
                               for t, raw, smooth in zip(d["helix_time"], d["helicity"], smoothed))
        all_distances.extend((d["label"], d["run"], t, value)
                             for t, value in zip(distance_time, distance))
        all_hbonds.extend((d["label"], d["run"], t, value)
                          for t, value in zip(wire_time, bonds))
        endpoint_export.append((
            d["label"], d["run"], d["marker"], 1.75, endpoint,
            endpoint_time, endpoint_pka,
        ))
        barrier_export.append((
            d["label"], d["run"], d["snapshot_time"],
            features["protonated_s"], features["protonated_f"],
            features["transition_s"], features["transition_f"],
            features["deprotonated_s"], features["deprotonated_f"],
            features["deprotonation_barrier"],
        ))

        for ax in (ax_pka, ax_helix, ax_distance, ax_hbond):
            ax.set_xlim(40, max(item["marker"] + 1.75 for item in data))
        if d["time"][-1] < endpoint - .03:
            raise ValueError(f"pKa time series does not cover the endpoint for {d['label']}")

    ax_pmf.plot(lcod, free_energy, color="#1F5AA6", lw=4.0, label="B ⇌ C")
    negative_x = float(pmf_summary["negative_minimum_A"])
    positive_x = float(pmf_summary["positive_minimum_A"])
    negative_state = pmf_summary["negative_state"]
    positive_state = pmf_summary["positive_state"]
    delta_g = float(pmf_summary["delta_g_positive_minus_negative_kcal_mol"])
    barrier = float(pmf_summary["barrier_negative_to_positive_kcal_mol"])
    # The B/C LCOD convention places CPP (C) in the negative basin and
    # BPP (B) in the positive basin; color the minima by that state mapping.
    ax_pmf.axvline(negative_x, color=FOCUSED_COLORS["C-99"], ls="--", lw=1.3, alpha=.75)
    ax_pmf.axvline(positive_x, color=FOCUSED_COLORS["B-171"], ls="--", lw=1.3, alpha=.75)
    ax_pmf.text(
        .04, .96,
        rf"$\Delta G_{{\mathrm{{{positive_state}-{negative_state}}}}}={delta_g:+.2f}$ kcal mol$^{{-1}}$" + "\n"
        + rf"$\Delta G^\ddagger_{{\mathrm{{{negative_state}\to {positive_state}}}}}={barrier:.2f}$ kcal mol$^{{-1}}$",
        transform=ax_pmf.transAxes, ha="left", va="top", fontsize=13,
    )

    ax_fes.set(title=r"Coordination free energy", xlabel=r"coordination, $s$",
               ylabel=r"$F(s)$ (kcal mol$^{-1}$)", xlim=(0, 1))
    ax_pka.set(title=r"Apparent p$K_a$ trajectory", xlabel="t (ps; metadynamics starts at 40)",
               ylabel=r"p$K_a(t)$")
    ax_pmf.set(title="B/C proton-transfer PMF", xlabel=r"LCOD (Å)",
               ylabel=r"PMF (kcal mol$^{-1}$)", xlim=(-2, 2), ylim=(0, None))
    ax_helix.set(title="Ring-center helicity", xlabel="t (ps; metadynamics starts at 40)",
                 ylabel="Helicity (deg)")
    ax_distance.set(title="Defect proximity", xlabel="t (ps; metadynamics starts at 40)",
                    ylabel="Defect–nearest solute atom (Å)", ylim=(0, None))
    ax_hbond.set(title="Proton-transfer wire", xlabel="t (ps; metadynamics starts at 40)",
                 ylabel="Number of hydrogen bonds", ylim=(-.6, 9.6),
                 yticks=[0, 3, 6, 9])
    for ax in axes.flat:
        ax.grid(False)
        ax.legend(loc="best", fontsize=11)
    fig.suptitle("BV · focused B/C comparison · B-171 and C-99", fontsize=24)
    fig.savefig(plot, dpi=220)
    plt.close(fig)

    save_csv(output_dir / "fes.csv",
             ["label", "run", "snapshot_aligned_ps", "s", "free_energy_kcal_mol"], fes_export)
    save_csv(output_dir / "pka_timeseries.csv",
             ["label", "run", "aligned_time_ps", "pka", "kcal_mol_per_pka_unit"], pka_export)
    save_csv(output_dir / "pmf_lcod.csv", ["lcod_A", "pmf_kcal_mol"], zip(lcod, free_energy))
    save_csv(output_dir / "helicity.csv",
             ["label", "run", "aligned_time_ps", "helicity_deg", "smoothed_helicity_deg"],
             helicity_export)
    save_csv(output_dir / "defect_distance.csv",
             ["label", "run", "aligned_time_ps", "display_distance_A"], all_distances)
    save_csv(output_dir / "hydrogen_bonds.csv",
             ["label", "run", "aligned_time_ps", "hydrogen_bond_count"], all_hbonds)
    save_csv(
        output_dir / "endpoint_pka.csv",
        ["label", "run", "tdiff_aligned_ps", "tstop_ps", "target_endpoint_aligned_ps",
         "saved_endpoint_aligned_ps", "endpoint_pka"],
        endpoint_export,
    )
    save_csv(
        output_dir / "deprotonation_barriers.csv",
        ["label", "run", "fes_snapshot_aligned_ps", "protonated_minimum_s",
         "protonated_minimum_f_kcal_mol", "transition_maximum_s",
         "transition_maximum_f_kcal_mol", "deprotonated_minimum_s",
         "deprotonated_minimum_f_kcal_mol", "deprotonation_barrier_kcal_mol"],
        barrier_export,
    )
    save_csv(output_dir / "run_mapping.csv", ["label", "run", "cv_dir", "source"],
             [(d["label"], d["run"], d["cv_dir"], d["source"]) for d in data])
    (output_dir / "README.md").write_text(
        "# Focused BV B/C summary\n\n"
        "Profile: B-171 (meta-hib) and C-99 (meta-hic). The source trajectories and "
        "the original joint report remain unchanged. F(s), pKa(t), helicity, defect "
        "distance, and wire data retain the original synchronized clock: metadynamics "
        "starts at 40 ps and each trace ends at tdiff + 1.75 ps. pKa(t) is the stored "
        "summary pKa series (equivalently DeltaF divided by its run-specific conversion "
        "factor). Helicity uses the original 0.2 ps circular smoothing. Defect distance "
        "and H-bond counts are selected from the audited timestamp-aligned caches used "
        "by the original joint summary; missing defect assignments remain NaN. The LCOD "
        "panel is the existing smooth B/C umbrella PMF in three_pmf_curves.csv and is not "
        "derived from B-171 or C-99. A wire with n bridging waters contains n+1 hydrogen "
        "bonds; a missing path remains a gap. Endpoint pKa values use the final saved sample "
        "at or before tdiff + 1.75 ps. Deprotonation barriers are measured from the minimum "
        "in the protonated s=0.9–1.0 basin to the maximum between that minimum and the "
        "deprotonated s=0.0–0.1 basin; complete features are saved in "
        "deprotonation_barriers.csv.\n"
    )
    return plot


def render_original() -> None:
    colors=['blue','red','black']
    data = load_joint_data()
    out = ROOT / "reports/BV/joint_summary_b_and_c"
    out.mkdir(exist_ok=True)
    plot = ROOT / "reports/BV/joint_summary_b_and_c.png"
    # Keep superseded mixed-site data recoverable, but out of the active output.
    old = out / "superseded_mixed_site"
    for name in ["fes_mean.csv", "ensemble_statistics.csv", "fes.csv", "delta_f.csv",
                 "samples.csv", "helicity.csv", "statistics.csv", "clock_mapping.csv",
                 "reaction_markers.csv", "helicity_pre_reaction_statistics.csv",
                 "helicity_post_reaction_statistics.csv"]:
        path = out / name
        if path.exists():
            old.mkdir(exist_ok=True)
            if (old / name).exists():
                raise FileExistsError(old / name)
            path.rename(old / name)
    plt.style.use(ROOT / "plotting/lefteris.mplstyle")
    fig = plt.figure(figsize=(29, 25))
    outer = fig.add_gridspec(2, 1, left=.06, right=.98, bottom=.065, top=.95, hspace=.19)
    top = outer[0].subgridspec(1, 3, wspace=.22)
    bottom = outer[1].subgridspec(1, 3, wspace=.22)
    axes = np.empty((2, 4), dtype=object)
    for panel in range(4):
        cell = top[panel] if panel < 3 else bottom[0]
        inner = cell.subgridspec(2, 1, hspace=0)
        axes[0, panel] = fig.add_subplot(inner[0])
        axes[1, panel] = fig.add_subplot(inner[1], sharex=axes[0, panel], sharey=axes[0, panel])
    for row, (cv, label) in enumerate([("meta-hib", "B"), ("meta-hic", "C")]):
        group = [d for d in data if d["cv_dir"] == cv]
        group_out = out / label
        group_out.mkdir(exist_ok=True)
        render(group, group_out, 1.75, 10, .2, axes_override=axes[row],palette=colors)
        for line in list(axes[row,0].lines):
            if line.get_label().startswith('Mean F(s)'):
                line.remove()
            else:
                line.set_alpha(.85)
        bar = axes[row, 2]
        mean_df = float(np.mean([np.mean(d["sample_df"]) for d in group]))
        mean_pka = float(np.mean([np.mean(d["sample_df"])/d["factor"] for d in group]))
        approximate_pka = round(mean_pka * 2) / 2
        for text in bar.texts:
            if text.get_text().startswith("Across"):
                text.set_text(f"Across {len(group)} runs:\n"
                              + rf"$\langle\Delta F\rangle = {mean_df:.2f}$ kcal mol$^{{-1}}$" + "\n"
                              + rf"$\langle\mathrm{{p}}K_a\rangle\approx {approximate_pka:g}$")
        from matplotlib.container import BarContainer
        bars = [c for c in bar.containers if isinstance(c, BarContainer)]
        for container, d in zip(bars, group):
            pka = float(np.mean(d["sample_df"])/d["factor"])
            container.set_label(rf"{d['label']}: p$K_a\approx {pka:.1f}$")
        for ax in axes[row]:
            ax.set_title("")
            ax.tick_params(labelsize=10)
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(12)
            ax.legend(fontsize=8, loc="best")
            for text in ax.texts:
                text.set_fontsize(8)
            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
    limits = []
    for col in range(4):
        pair = axes[:, col]
        xmin = min(ax.get_xlim()[0] for ax in pair)
        xmax = max(ax.get_xlim()[1] for ax in pair)
        ymin = min(ax.get_ylim()[0] for ax in pair)
        ymax = max(ax.get_ylim()[1] for ax in pair)
        if col==3:
            ymin -= .25*(ymax-ymin)
        for ax in pair:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        if col == 3:
            for ax in pair:
                ax.legend(fontsize=8, loc="upper right")
        limits.append((col, xmin, xmax, ymin, ymax))
    inner = bottom[1].subgridspec(2, 1, hspace=0)
    distance_axes = [fig.add_subplot(inner[0])]
    distance_axes.append(fig.add_subplot(inner[1], sharex=distance_axes[0], sharey=distance_axes[0]))
    group_size = len([d for d in data if d["cv_dir"] == "meta-hib"])
    positions={d['label']:(tier,j) for tier,cv in enumerate(['meta-hib','meta-hic'])
               for j,d in enumerate([r for r in data if r['cv_dir']==cv])}
    for j, d in enumerate(data):
        times, distances = nearest_solute_distance(d, out)
        tier,index=positions[d['label']]
        distance_axes[tier].scatter(times, distances, color=colors[index], s=9, linewidths=0, label=d["label"])
    for j, ax in enumerate(distance_axes):
        ax.set_xlim(axes[0, 1].get_xlim())
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Defect–nearest solute atom (Å)", fontsize=12)
        ax.tick_params(labelsize=10, labelbottom=j == 1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.2)
    distance_axes[1].set_xlabel("t (ps; metadynamics starts at 40)", fontsize=11)
    # Reuse the previously recomputed eight-water H-bond paths without
    # filling missing defect assignments or recalculating trajectories.
    wire_grid = bottom[2].subgridspec(len(data), 1, hspace=0)
    wire_axes = []
    for j, d in enumerate(data):
        with (out / f"{d['label']}_coordination_wires.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        times = np.asarray([float(r["aligned_time_ps"]) for r in rows])
        wires = np.asarray([float(r["hbond_bridging_waters"]) for r in rows])
        keep = times <= d["marker"] + 1.75 + 1e-8
        ax = fig.add_subplot(wire_grid[j], sharex=wire_axes[0], sharey=wire_axes[0]) if wire_axes else fig.add_subplot(wire_grid[j])
        wire_axes.append(ax)
        ax.scatter(times[keep], wires[keep] + 1, color=colors[positions[d['label']][1]], s=9,
                   linewidths=0, label=d["label"])
        ax.set_xlim(axes[0, 1].get_xlim())
        ax.set_ylim(-.6, 9.6)
        ax.set_yticks([0, 3, 6, 9])
        ax.set_yticks(range(10), minor=True)
        ax.tick_params(labelsize=9, labelbottom=j == len(data)-1)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9, loc="upper right")
        if j in (1, group_size+1):
            ax.set_ylabel("Bridging waters", fontsize=12)
    wire_axes[group_size].spines["top"].set_linewidth(2.3)
    wire_axes[-1].set_xlabel("t (ps; metadynamics starts at 40)", fontsize=11)
    # Shared categorical positions: display both IDs at each common bar position.
    b_ids = [d["run"] for d in data if d["cv_dir"] == "meta-hib"]
    c_ids = [d["run"] for d in data if d["cv_dir"] == "meta-hic"]
    tick_labels=[(f'B-{b_ids[i]}' if i<len(b_ids) else '')+'\n'+
                 (f'C-{c_ids[i]}' if i<len(c_ids) else '') for i in range(max(len(b_ids),len(c_ids)))]
    axes[1, 2].set_xticks(range(len(tick_labels)),tick_labels)
    # Presentation-scale typography and strokes across every tier.
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=23, width=1.8, length=7)
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(27)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
        for line in ax.lines:
            line.set_linewidth(max(3.5, line.get_linewidth() * 2.8))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(max(7, line.get_markersize() * 1.4))
        for collection in ax.collections:
            if hasattr(collection, "set_sizes"):
                collection.set_sizes([42])
            elif hasattr(collection, "set_linewidth"):
                collection.set_linewidth(2.2)
        for text in ax.texts:
            text.set_fontsize(20)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(19)
            for line in legend.get_lines():
                line.set_linewidth(5)
    for ax in axes[:,3]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    # One figure-positioned ylabel centered across each complete stacked panel.
    stacks = [list(axes[:, i]) for i in range(4)] + [distance_axes, wire_axes]
    labels = [r"$F(s)$ (kcal mol$^{-1}$)", r"$\Delta F$ (kcal mol$^{-1}$)",
              r"$\langle\Delta F\rangle$ (kcal mol$^{-1}$)",
              "Ring-center helicity (deg)", "Defect–nearest solute atom (Å)",
              "Number of hydrogen bonds"]
    for stack, label in zip(stacks, labels):
        for ax in stack:
            ax.set_ylabel("")
        upper, lower = stack[0].get_position(), stack[-1].get_position()
        fig.text(upper.x0 - .030, (upper.y1 + lower.y0)/2, label,
                 rotation=90, ha="center", va="center", fontsize=27)
    fig.suptitle("BV · meta-hib (B, upper) / meta-hic (C, lower)", fontsize=32, y=.98)
    fig.savefig(plot)
    plt.close(fig)
    save_csv(out / "axis_limits.csv", ["column", "xmin", "xmax", "ymin", "ymax"], limits)
    save_csv(out / "run_mapping.csv", ["label", "run", "cv_dir", "source"],
             [(d["label"], d["run"], d["cv_dir"], d["source"]) for d in data])
    (out / "README.md").write_text(
        "# Joint B/C summary\n\n"
        "Selections and aligned tdiff values are read from each correct_summary/rerun.py; reaction markers come from its reaction_markers.csv.\n"
        "Joint-only exclusions: B runs 56,98 and C run 57. Retained: B 38,171; C 39,99,142. Separate summaries and all excluded-run source data are preserved.\n"
        "B = meta-hib; C = meta-hic. All starts are aligned to 40 ps; endpoint is tdiff+1.75 ps.\n"
        "Wide 2x3 layout: FES, Delta F, bars above; helicity time series, nearest-solute distance, H-bond wires below. B above C in each panel; wire panel has one separate trace per selected run. Colors: B38/C39 blue, B171/C99 red, C142 black.\n"
        "Distance panel: minimum periodic distance from assigned defect oxygen to any of the 78 solute atoms (including hydrogens). Uses timestamp-matched defect IDs in retained *_coordination_wires.csv. Missing distance values carry the last valid value forward; leading gaps stay NaN. Raw and display distances and a fill flag are saved separately. No defect identity is imputed.\n"
        "Water-wire panel reuses the eight-water path analysis in *_coordination_wires.csv: D-H <=1.3 A, H-A <=2.5 A, D-H-A >=135 degrees, periodic geometry. Plotted hydrogen-bond count = bridging-water count + 1: 0 means no path found within the eight-water cap, 1 is direct, up to 9 bonds. Missing assignments remain gaps. CSV retains original bridging-water counts.\n"
        "FES shows individual curves only; mean curves are hidden (mean CSVs retained). Ensemble bar annotations are averaged ONLY within the corresponding site. No B/C pooling. Bars use ten equally spaced samples per run, with population SD. Defect distances are scatter points; existing forward-fill policy is unchanged.\n"
        "Helicity time series restored with 0.2 ps circular smoothing and treact markers. Helicity histogram CSVs are retained legacy outputs, not used by this figure. Fonts enlarged throughout; helicity legends remain hidden to avoid crowding.\n"
        "Data for each site are in B/ and C/. Superseded mixed-site CSVs are archived in superseded_mixed_site/ and are not used. Separate summaries remain unchanged.\n"
    )
    print(plot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("original", "b171-c99-lcod"), default="original",
        help="report profile; default preserves the original B/C joint summary",
    )
    args = parser.parse_args()
    if args.profile == "original":
        render_original()
    else:
        print(render_focused())


if __name__ == "__main__":
    main()
