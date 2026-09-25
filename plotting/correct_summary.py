#!/usr/bin/env python3
"""Combine indexed BV acid/base reports without modifying individual summaries.

Example: python plotting/correct_summary.py --runs 39 57 99 142
Uses saved pKa/helix data and FES blocks at tdiff + the sampling window.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acid_base_BV import (load_fes_snapshot, sample_pka_window,
                          calculate_bv_torsions, read_box_lengths_from_dftb_inp)
from plot_pka_grid import PKA_FACTOR
from plot_cv_grid import first_trajectory_clock

ROOT = Path(__file__).resolve().parents[1]


def read_table(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def load_run(base: Path, entry: dict[str, str], samples: int, window: float,
             aligned_tdiff: float | None = None) -> dict:
    """Retain each report's time mapping and temperature; do not infer offsets."""
    folder = base / Path(entry["summary"]).parent
    meta = json.loads((folder / "provenance.json").read_text())
    marker = float(entry["tdiff_grid_ps"])
    raw = float(entry["tdiff_raw_ps"])
    offset = meta["clock_offset_ps"]
    if not np.isclose(raw + offset, marker, atol=1e-8, rtol=0):
        raise ValueError(f"Index/provenance clock mismatch: {folder}")
    if not np.isclose(marker, meta["grid_tdiff_ps"], atol=1e-8, rtol=0):
        raise ValueError(f"Index/provenance marker mismatch: {folder}")
    source = Path(meta["source"])
    clock = first_trajectory_clock(source / "traject")
    if clock is None:
        raise ValueError(f"Missing trajectory start time: {source}")
    raw_start = clock[1]
    display_offset = 40.0 - raw_start
    if aligned_tdiff is not None:
        raw = aligned_tdiff - display_offset
    factor = PKA_FACTOR * meta["temperature_K"]
    series = read_table(folder / "summary_pka.csv")
    # This reverses the exact existing conversion, preserving its basin/sign convention.
    df = series["pka"] * factor
    st, sv = sample_pka_window(series["time_ps"], df, raw, raw + window, samples)
    if len(sv) != samples:
        raise ValueError(f"Missing sampling window: {folder}")
    source = Path(meta["source"])
    snapshot_time, block = load_fes_snapshot(source / "fes.dat", raw + window)
    if snapshot_time > raw + window + 1e-8:
        raise ValueError(f"No FES snapshot at or before window endpoint: {folder}")
    clock = first_trajectory_clock(source / "traject")
    if clock is None:
        raise ValueError(f"Missing trajectory start time: {source}")
    raw_start = clock[1]
    display_offset = 40.0 - raw_start
    torsions = read_table(folder / "summary_bv_torsions.csv")
    helix_times = torsions["time_ps"]
    helicity = torsions["helicity_deg"]
    if raw + window > helix_times[-1] + .021:
        extra = np.arange(helix_times[-1] + .02, raw + window + 1e-8, .02)
        _, extra_helicity = calculate_bv_torsions(
            source / "traject", extra,
            ROOT / "systems/BV/solv_4.0/salt/ready.parm7", 78,
            read_box_lengths_from_dftb_inp(source / "dftb.inp"))
        if not np.all(np.isfinite(extra_helicity)):
            raise ValueError(f"Missing extended helicity frames: {source}")
        helix_times = np.concatenate([helix_times, extra])
        helicity = np.concatenate([helicity, extra_helicity])
    keep = series["time_ps"] <= raw + window + 1e-8
    hk = helix_times <= raw + window + 1e-8
    return dict(run=int(entry["run"]), marker=raw + display_offset, raw_marker=raw,
                original_marker=marker, raw_start=raw_start, display_offset=display_offset,
                offset=offset, temperature=meta["temperature_K"], factor=factor,
                snapshot_time=snapshot_time + display_offset, fes=block,
                time=series["time_ps"][keep] + display_offset, df=df[keep],
                sample_time=st + display_offset, sample_df=sv,
                helix_time=helix_times[hk] + display_offset,
                helicity=helicity[hk], source=str(source), report_folder=folder,
                cv_dir=meta["cv_dir"])


def save_csv(path: Path, header: list[str], rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def smooth_helicity(times: np.ndarray, angles: np.ndarray, width_ps: float) -> np.ndarray:
    """Centered circular moving mean; truncate the window at trajectory edges."""
    result = np.full(angles.shape, np.nan)
    for i, t in enumerate(times):
        lo, hi = np.searchsorted(times, [t-width_ps/2, t+width_ps/2], side="left")
        values = angles[lo:max(hi, i+1)]
        values = values[np.isfinite(values)]
        if values.size:
            z = np.mean(np.exp(1j*np.deg2rad(values)))
            result[i] = np.rad2deg(np.angle(z))
    return result


def render(data: list[dict], out: Path, window: float, samples: int, smooth_ps: float,
           plot_path: Path | None = None, axes_override: np.ndarray | None = None,
           palette: list[str] | None = None) -> None:
    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    if axes_override is None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")
    else:
        axes = np.asarray(axes_override)
        fig = axes.flat[0].figure
    af, at, ab, ah = axes.flat
    colors = palette if palette is not None else ["green", "black", "red", "blue"]
    if len(data) > len(colors):
        colors = colors + ["#8056A3"]
    rows = {"fes": [], "delta_f": [], "samples": [], "helicity": [], "statistics": []}
    for i, d in enumerate(data):
        c = colors[i % len(colors)]
        n = d["run"]
        label = d.get("label", str(n))
        mean, std = float(np.mean(d["sample_df"])), float(np.std(d["sample_df"]))
        pka, pstd = mean / d["factor"], std / d["factor"]
        af.plot(d["fes"][:, 0], d["fes"][:, 1], color=c,
                label=f"{label}: t={d['snapshot_time']:.2f} ps", alpha=.3, lw=1.8)
        at.plot(d["time"], d["df"], color=c,
                label=f"{label}: tdiff={d['marker']:.3f} ps", lw=1.6)
        at.scatter(d["sample_time"], d["sample_df"], color=c, edgecolor="black",
                   linewidth=.4, marker="*", s=60, zorder=5)
        at.axvline(d["marker"], color=c, ls="--", lw=1, alpha=.65)
        ab.bar(i, mean, yerr=std, capsize=5, color=c, alpha=.85,
               label=rf"{label}: p$K_a$ = {pka:.2f} $\pm$ {pstd:.2f}")
        ab.annotate(f"{mean:.2f} ± {std:.2f}", (i, mean + std),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11)
        smoothed = smooth_helicity(d["helix_time"], d["helicity"], smooth_ps)
        before_reaction = ((d["helix_time"] >= 40.0) &
                           (d["helix_time"] < d["treact"]) &
                           np.isfinite(d["helicity"]))
        values = d["helicity"][before_reaction]
        resultant = np.mean(np.exp(1j*np.deg2rad(values))) if values.size else complex(np.nan, np.nan)
        circular_mean = float(np.rad2deg(np.angle(resultant))) if abs(resultant) > 1e-12 else np.nan
        d["pre_reaction_mean"] = circular_mean
        d["pre_reaction_count"] = len(values)
        d["pre_reaction_resultant"] = float(abs(resultant))
        after_reaction = ((d["helix_time"] >= d["treact"]) &
                          (d["helix_time"] <= d["marker"] + window + 1e-8) &
                          np.isfinite(d["helicity"]))
        post_values = d["helicity"][after_reaction]
        post_resultant = np.mean(np.exp(1j*np.deg2rad(post_values))) if post_values.size else complex(np.nan, np.nan)
        post_mean = float(np.rad2deg(np.angle(post_resultant))) if abs(post_resultant) > 1e-12 else np.nan
        d["post_reaction_mean"] = post_mean
        d["post_reaction_count"] = len(post_values)
        d["post_reaction_resultant"] = float(abs(post_resultant))
        ah.plot(d["helix_time"], smoothed, color=c,
                label=f"{label}: treact={d['treact']:.3f} ps\n"
                + rf"$\langle h\rangle_{{\rm pre}}={circular_mean:.1f}^\circ$; "
                + rf"$\langle h\rangle_{{\rm post}}={post_mean:.1f}^\circ$", lw=1.5)
        ah.axvline(d["treact"], color=c, ls="--", lw=1, alpha=.65)
        rows["fes"].extend((n, d["marker"], d["snapshot_time"], s, f) for s, f in d["fes"][:, :2])
        rows["delta_f"].extend((n, t, f) for t, f in zip(d["time"], d["df"]))
        rows["samples"].extend((n, t, f, f/d["factor"]) for t, f in zip(d["sample_time"], d["sample_df"]))
        rows["helicity"].extend((n, t, h, sm) for t, h, sm in zip(d["helix_time"], d["helicity"], smoothed))
        rows["statistics"].append((n, d["marker"], d["marker"] + window, samples,
                                    d["temperature"], mean, std, pka, pstd))
    # Each FES has already been shifted to its own minimum zero. Average
    # pointwise only on identical CV grids; the band is between-run SD (ddof=0).
    grid = data[0]["fes"][:, 0]
    for d in data[1:]:
        if d["fes"].shape[0] != len(grid) or not np.allclose(d["fes"][:, 0], grid, rtol=0, atol=1e-10):
            raise ValueError("FES grids differ; cannot form a pointwise mean")
    energies = np.stack([d["fes"][:, 1] for d in data])
    fes_mean, fes_std = energies.mean(axis=0), energies.std(axis=0)
    af.plot(grid, fes_mean, color="#1F3A5F", lw=3.2,
            label=f"Mean F(s), {len(data)} runs", zorder=5)
    save_csv(out / "fes_mean.csv", ["s", "mean_free_energy_kcal_mol", "std_free_energy_kcal_mol"],
             list(zip(grid, fes_mean, fes_std)))
    af.set(xlabel=r"coordination, $s$", ylabel=r"$F(s)$ (kcal mol$^{-1}$)",
           xlim=(0, 1), title=f"FES at tdiff + {window:g} ps")
    at.set(xlabel="t (ps; metadynamics starts at 40)", ylabel=r"$\Delta F$ (kcal mol$^{-1}$)",
           title="Free-energy difference")
    ab.set(xlabel="Run ID", ylabel=r"$\langle\Delta F\rangle$ (kcal mol$^{-1}$)",
           xticks=range(len(data)), xticklabels=[d.get("label", str(d["run"])) for d in data],
           title=f"{samples} samples: tdiff to tdiff + {window:g} ps")
    ab.margins(y=.45)
    mean_df = float(np.mean([np.mean(d["sample_df"]) for d in data]))
    mean_pka = float(np.mean([np.mean(d["sample_df"]) / d["factor"] for d in data]))
    ab.text(.03, .96,
            rf"Across {len(data)} runs:" + "\n"
            + rf"$\langle\Delta F\rangle = {mean_df:.2f}$ kcal mol$^{{-1}}$" + "\n"
            + rf"$\langle\mathrm{{p}}K_a\rangle = {mean_pka:.2f}$",
            transform=ab.transAxes, ha="left", va="top", fontsize=12)
    save_csv(out / "ensemble_statistics.csv",
             ["run_count", "mean_df_kcal_mol", "mean_pka"],
             [(len(data), mean_df, mean_pka)])
    ah.set(xlabel="t (ps; metadynamics starts at 40)", ylabel="Ring-center helicity (deg)",
           title=f"Metadynamics helicity ({smooth_ps:g} ps smoothing)")
    at.set_xlim(left=40)
    ah.set_xlim(left=40)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=11, loc="best")
    if axes_override is not None:
        pass  # Caller sets the joint layout, axis limits and output filename.
    elif len({d['cv_dir'] for d in data}) > 1:
        fig.set_size_inches(19, 13)
        lower, upper = ah.get_ylim()
        ah.set_ylim(lower - .30 * (upper-lower), upper)
        ah.legend(fontsize=9, loc="lower left", ncol=2)
        af.legend(fontsize=10, loc="lower center", ncol=2)
        fig.suptitle("BV · joint summary · meta-hib (B) + meta-hic (C) · N1T48C1 · solv 4.0", fontsize=20)
    else:
        fig.suptitle(f"BV · {data[0]['cv_dir']} · N1T48C1 · solv 4.0", fontsize=20)
    if axes_override is None:
        fig.savefig(plot_path or out / "summary.png")
        plt.close(fig)
    headers = {
        "fes": ["run", "tdiff_aligned_ps", "snapshot_aligned_ps", "s", "free_energy_kcal_mol"],
        "delta_f": ["run", "aligned_time_ps", "delta_f_kcal_mol"],
        "samples": ["run", "aligned_time_ps", "delta_f_kcal_mol", "pka"],
        "helicity": ["run", "aligned_time_ps", "helicity_deg", "smoothed_helicity_deg"],
        "statistics": ["run", "tdiff_aligned_ps", "end_aligned_ps", "sample_count", "temperature_K",
                       "mean_df_kcal_mol", "std_df_kcal_mol", "mean_pka", "std_pka"],
    }
    for key in rows:
        save_csv(out / f"{key}.csv", headers[key], rows[key])
    save_csv(out / "clock_mapping.csv",
             ["run", "raw_start_ps", "original_tdiff_grid_ps", "tdiff_raw_ps", "tdiff_aligned_ps", "raw_to_aligned_offset_ps"],
             [(d["run"], d["raw_start"], d["original_marker"], d["raw_marker"], d["marker"], d["display_offset"]) for d in data])
    save_csv(out / "reaction_markers.csv", ["run", "treact_raw_ps", "treact_aligned_ps"],
             [(d["run"], d["treact"]-d["display_offset"], d["treact"]) for d in data])
    save_csv(out / "helicity_pre_reaction_statistics.csv",
             ["run", "start_aligned_ps", "stop_exclusive_aligned_ps", "sample_count",
              "circular_mean_helicity_deg", "mean_resultant_length"],
             [(d["run"], 40.0, d["treact"], d["pre_reaction_count"],
               d["pre_reaction_mean"], d["pre_reaction_resultant"]) for d in data])
    save_csv(out / "helicity_post_reaction_statistics.csv",
             ["run", "start_inclusive_aligned_ps", "stop_inclusive_aligned_ps", "sample_count",
              "circular_mean_helicity_deg", "mean_resultant_length"],
             [(d["run"], d["treact"], d["marker"] + window, d["post_reaction_count"],
               d["post_reaction_mean"], d["post_reaction_resultant"]) for d in data])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", required=True)
    parser.add_argument("--tdiff", type=float, nargs="+",
                        help="Override markers on the synchronized 40-ps-start clock, in --runs order")
    parser.add_argument("--acid-base-dir", type=Path, default=ROOT / "reports/BV/meta-hic/acid_base")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--window-ps", type=float, default=1.75, help="Post-tdiff sampling duration (ps)")
    parser.add_argument("--samples", type=int, default=10, help="Equally spaced samples, not independent windows")
    parser.add_argument("--helicity-smooth-ps", type=float, default=.2,
                        help="Centered circular smoothing width for displayed helicity (ps)")
    parser.add_argument("--reaction-markers", type=Path,
                        help="Saved summary purple markers in raw report ps")
    parser.add_argument("--reaction-s-threshold", type=float, default=.05,
                        help="If saved markers are unavailable, first sampled coordination <= this value marks reaction")
    args = parser.parse_args()
    if not np.isfinite(args.helicity_smooth_ps) or args.helicity_smooth_ps < 0:
        parser.error("Smoothing width must be finite and nonnegative")
    if args.tdiff is not None and (len(args.tdiff) != len(args.runs) or not np.all(np.isfinite(args.tdiff))):
        parser.error("Supply one finite --tdiff value per run")
    if args.window_ps <= 0 or args.samples < 2 or len(set(args.runs)) != len(args.runs):
        parser.error("Require positive window, at least two samples, and unique run IDs")
    with (args.acid_base_dir / "index.csv").open() as handle:
        entries = {int(r["run"]): r for r in csv.DictReader(handle) if r["bench_tag"] == "N1T48C1"}
    markers = args.tdiff if args.tdiff is not None else [None] * len(args.runs)
    data = [load_run(args.acid_base_dir, entries[n], args.samples, args.window_ps, t)
            for n, t in zip(args.runs, markers)]
    marker_path = args.reaction_markers
    if marker_path is None and all(d["cv_dir"] == "meta-hic" for d in data):
        marker_path = ROOT / "reports/BV/meta-hic-finalist-runs/sasa_summary_dashed_endpoints.json"
    reactions = json.loads(marker_path.read_text())["endpoints_ps"] if marker_path else {}
    for d in data:
        if str(d["run"]) in reactions:
            reaction_raw = float(reactions[str(d["run"])])
        else:
            table = read_table(d["report_folder"] / "summary.csv")
            eligible = np.flatnonzero(np.isfinite(table["coordination_s"]) &
                                     (table["coordination_s"] <= args.reaction_s_threshold) &
                                     (table["time_ps"] <= d["raw_marker"]))
            if not eligible.size:
                raise ValueError(f"No reaction threshold crossing before tdiff for run {d['run']}")
            reaction_raw = float(table["time_ps"][eligible[0]])
        d["treact"] = reaction_raw + d["display_offset"]
    out = args.out_dir or args.acid_base_dir / "correct_summary"
    out.mkdir(parents=True, exist_ok=True)
    render(data, out, args.window_ps, args.samples, args.helicity_smooth_ps)
    (out / "README.md").write_text(
        f"# Combined BV {data[0]['cv_dir']} summaries\n\n"
        f"Runs: {', '.join(map(str, args.runs))}. Markers default to ../index.csv; aligned overrides: {args.tdiff}.\n\n"
        f"FES: latest saved block at or before each tdiff+{args.window_ps:g} ps, shifted to minimum zero. Actual snapshot times are in fes.csv.\n"
        "Faint individual FES curves and their pointwise arithmetic mean are plotted, without an SD band. Each input curve is independently shifted to minimum zero. Mean/SD remain available numerically in fes_mean.csv.\n"
        "The bar-panel ensemble annotation is the equally weighted arithmetic mean of the per-run mean Delta F and per-run mean pKa, saved in ensemble_statistics.csv.\n"
        "Delta F uses the existing BV basin convention (near s=0 minus near s=1); saved pKa is converted back with 0.004576*T kcal/mol.\n"
        f"Bars: mean and population standard deviation of {args.samples} interpolated, equally spaced samples from tdiff to tdiff+{args.window_ps:g} ps.\n"
        "These correlated samples are not independent windows or uncertainty of a converged equilibrium estimate.\n"
        "All time axes use t_aligned = t_raw - first_trajectory_time + 40 ps. Physical markers and sample windows are unchanged; clock_mapping.csv records original and aligned times. Delta F dashed lines mark tdiff; stars mark samples. Helicity dashed lines mark the saved purple reaction/deprotonation markers, not tdiff.\n"
        f"Reaction-marker source: {marker_path}. When no saved marker exists, the first saved coordination s<={args.reaction_s_threshold:g} before tdiff is used; this is a CV-based reaction marker, not a bulk-diffusion test. Legacy HIC markers for 39/57/99 were reconstructed from first saved s<=0.05 and checked against the purple lines; 142 has an explicit recorded marker.\n"
        f"Helicity display: centered circular moving mean, {args.helicity_smooth_ps:g} ps wide, truncated at edges. Raw and smoothed values are saved separately; no smoothing affects free energies.\n"
        "Helicity legend averages use unsmoothed finite samples in [40 ps, treact), calculated as atan2(mean(sin h), mean(cos h)). Per-run counts and resultant lengths are saved in helicity_pre_reaction_statistics.csv.\n"
        "The second (post) circular average uses unsmoothed finite samples in [treact, tstop], where tstop=tdiff+window, not the full raw simulation endpoint. Counts and resultant lengths are saved in helicity_post_reaction_statistics.csv.\n"
        "Helicity is the existing dihedral of the four ring centers, through the individual summary endpoint.\n"
        "Individual reports and raw simulations are unchanged. Numerical data for all four panels are in the sister CSV files.\n"
    )
    print(out / "summary.png")


if __name__ == "__main__":
    main()
