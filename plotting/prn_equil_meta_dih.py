#!/usr/bin/env python3
"""Plot PRN equilibration and restart-aligned metadynamics without changing runs.

The first meta frame must be the restart frame, with inherited step number.
Absolute meta time = restart_step * equil_dt + (reported_time - first_time).
Full equilibration is retained, including any continuation beyond the branch.
Fixed H11 torsions with O1-H11 > --oh-cutoff (Angstrom) are marked separately;
this is a geometric flag, not a proton-ownership assignment. By default meta
traces stop before the first cutoff crossing, even if H11 later returns.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prn_anti_dih import (
    ROOT, NUMBER, TIME, box_from_input, read_series, run_ids,
    wrapped_plot_series, circular_mean_deg, circular_std_deg,
)


def timestep_ps(path: Path) -> float:
    match = re.search(rf"\bDELTAT\s*=\s*({NUMBER})", path.read_text(), re.I)
    if not match:
        raise ValueError(f"Missing DELTAT: {path}")
    return float(match[1].replace("D", "E").replace("d", "e")) * 1e12


def restart_origin(path: Path, eq_dt: float, meta_dt: float) -> tuple[int, float, float]:
    with path.open() as handle:
        handle.readline()
        comment = handle.readline()
    step = re.search(r"STEP NO\.\s*=\s*(\d+)", comment)
    time = TIME.search(comment)
    if not step or not time:
        raise ValueError(f"Missing restart timestamp/step: {path}")
    s = int(step[1])
    t = float(time[1].replace("D", "E")) / 1000
    if not np.isclose(t, s * meta_dt, atol=1e-5, rtol=0):
        raise ValueError(f"Unexpected restart clock: {path}")
    return s, s * eq_dt, t


def align_meta(data: np.ndarray, origin: float, reported_origin: float) -> np.ndarray:
    result = data.copy()
    result[:, 0] += origin - reported_origin
    return result


def departure_index(data: np.ndarray, cutoff: float) -> int:
    """First sampled distance above cutoff, or length if no crossing occurs."""
    hits = np.flatnonzero(data[:, 3] > cutoff)
    return int(hits[0]) if len(hits) else len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=run_ids, default=list(range(1, 21)))
    parser.add_argument("--meta-dir", default="meta-h")
    parser.add_argument("--oh-cutoff", type=float, default=1.3, help="O1-H11 distance flag in Angstrom (default 1.3)")
    parser.add_argument("--show-post-departure", action="store_true", help="Show full meta trajectories instead of truncating at first OH cutoff crossing")
    args = parser.parse_args()
    if args.oh_cutoff <= 0 or args.out.suffix != ".png":
        parser.error("Require positive OH cutoff and PNG output")
    if args.out.resolve().is_relative_to((ROOT / "systems").resolve()):
        parser.error("Output must be outside systems")
    center = 0 if "PRN-syn" in args.runs_path.resolve().parts else 180
    records = {}
    for rid in args.runs:
        base = args.runs_path / f"run-{rid}"
        eq, md = base / "equil", base / args.meta_dir
        eqbox, mbox = box_from_input(eq / "dftb.inp"), box_from_input(md / "dftb.inp")
        e = read_series(eq / "traject", [5, 3, 4, 11], eqbox)
        m = read_series(md / "traject", [5, 3, 4, 11], mbox)
        if not len(e) or not len(m):
            raise ValueError(f"No complete frames for run {rid}")
        step, origin, raw_origin = restart_origin(md / "traject", timestep_ps(eq / "dftb.inp"), timestep_ps(md / "dftb.inp"))
        if origin > e[-1, 0] + .02:
            raise ValueError(f"Restart beyond available equilibration: run {rid}")
        aligned = align_meta(m, origin, raw_origin)
        records[rid] = (e, m, aligned, origin, step)
        print(f"run-{rid}: equil {e[-1,0]:.3f} ps; meta branches at {origin:.3f} ps (step {step}); meta elapsed {m[-1,0]-raw_origin:.3f} ps", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots((len(records)+2)//3, 3, figsize=(18, 3.2*((len(records)+2)//3)), squeeze=False, sharex=True, sharey=True)
    summaries = []
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "stage", "reported_time_ps", "aligned_time_ps", "stage_elapsed_ps", "signed_dihedral_deg", "plotted_dihedral_deg", "O1_H11_distance_A", "within_oh_cutoff", "restart_equil_time_ps", "restart_step"])
        for ax, (rid, (e, m, aligned, origin, step)) in zip(axes.flat, records.items()):
            labels = []
            for stage, raw, data, color in [("equil", e, e, "tab:blue"), ("meta", m, aligned, "tab:orange")]:
                departure = float("nan")
                if stage == "meta" and not args.show_post_departure:
                    stop = departure_index(data, args.oh_cutoff)
                    if stop < len(data):
                        departure = float(data[stop, 0])
                    raw, data = raw[:stop], data[:stop]
                bound = data[:, 3] <= args.oh_cutoff
                phi = data[:, 1].copy()
                phi[~bound] = np.nan
                x, y = wrapped_plot_series(data[:, 0], phi, center)
                ax.plot(x, y, lw=.65, color=color, label=stage)
                wrapped = (data[:,1]-center+180)%360+center-180
                ax.scatter(data[~bound,0], wrapped[~bound], s=2, color="0.55", alpha=.4)
                mean = (circular_mean_deg(phi)-center+180)%360+center-180
                std = circular_std_deg(phi)
                labels.append(f"{stage}: {mean:.1f}° ± {std:.1f}°" if len(data) else f"{stage}: no pre-crossing frames")
                endpoints = [raw[0,0],raw[-1,0],data[0,0],data[-1,0]] if len(data) else [float("nan")]*4
                summaries.append([rid,stage,len(data),*endpoints,origin,step,mean,std,int((~bound).sum()),departure])
                for j, row in enumerate(data):
                    writer.writerow([rid,stage,raw[j,0],row[0],raw[j,0]-raw[0,0],row[1],wrapped[j],row[3],int(bound[j]),origin,step])
            ax.axvline(origin, color="black", ls="--", lw=.8)
            ax.text(.98,.97,"\n".join(labels),transform=ax.transAxes,ha="right",va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
            ax.set(title=f"run-{rid} | meta start {origin:.3f} ps", ylim=(center-180,center+180),yticks=np.arange(center-180,center+181,90),xlabel="Restart-aligned time (ps)",ylabel="Dihedral (°)")
            ax.grid(alpha=.2)
            ax.tick_params(labelbottom=True)
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    note = (f"Gray: O1–H11 > {args.oh_cutoff:g} Å; statistics exclude gray frames" if args.show_post_departure else f"Meta stops before first O1–H11 > {args.oh_cutoff:g} Å; circular mean ± SD uses retained frames")
    fig.suptitle(f"PRN {'syn' if center == 0 else 'anti'}: O2–CG–O1–H11 (5–3–4–11)\nBlue: full equilibration | Orange: metadynamics | Dashed: restart branch\n{note}",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.955))
    fig.savefig(args.out,dpi=160)
    plt.close(fig)
    summary = args.out.with_name(args.out.stem+"_summary.csv")
    with summary.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["run_id","stage","frames","reported_start_ps","reported_end_ps","aligned_start_ps","aligned_end_ps","restart_equil_time_ps","restart_step","bound_circular_mean_deg","bound_circular_std_deg","outside_oh_cutoff_frames","first_cutoff_crossing_aligned_ps"])
        writer.writerows(summaries)
    print(f"Saved {args.out}, aligned CSV, and {summary}")


if __name__ == "__main__":
    main()
