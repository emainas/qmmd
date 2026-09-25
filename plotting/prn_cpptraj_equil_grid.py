#!/usr/bin/env python3
"""Calculate PRN equilibration torsions with cpptraj and plot a centered grid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import parmed

from prn_anti_dih import (TIME, box_from_input, circular_mean_deg,
                          circular_std_deg, dihedral, run_ids, wrapped_plot_series)


def snapshot(source: Path, target: Path) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Freeze the input size, keep complete frames, and retain exact timestamps."""
    with source.open("rb") as f:
        blob = f.read(os.fstat(f.fileno()).st_size)
    lines = blob.splitlines(keepends=True)
    natoms = int(lines[0])
    block = natoms + 2
    frames = len(lines) // block
    if frames and not lines[frames*block-1].endswith(b"\n"):
        frames -= 1
    if not frames:
        raise ValueError(f"No complete frames: {source}")
    retained = b"".join(lines[:frames*block])
    times, audit = [], []
    for i in range(frames):
        start = i*block
        if int(lines[start]) != natoms:
            raise ValueError("Changing atom count")
        match = TIME.search(lines[start+1].decode())
        if match is None:
            raise ValueError("Missing trajectory timestamp")
        times.append(float(match[1].replace("D", "E"))/1000)
        if i in (0, frames//2, frames-1):
            audit.append(np.asarray([[float(x) for x in lines[start+1+j].split()[1:4]]
                                     for j in [5, 3, 4, 11]]))
    if np.any(np.diff(times) <= 0):
        raise ValueError("Nonmonotonic timestamps; restart requires explicit stitching")
    symbols = [lines[2+i].split()[0].decode() for i in range(11)]
    if symbols != ["C", "C", "C", "O", "O"] + ["H"]*6:
        raise ValueError("Unexpected solute atom order")
    target.write_bytes(retained)
    return np.asarray(times), audit, dict(atoms=natoms, frames=frames,
        bytes=len(retained), sha256=hashlib.sha256(retained).hexdigest(),
        ignored_tail_bytes=len(blob)-len(retained))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["PRN-syn", "PRN-anti"], default="PRN-syn")
    parser.add_argument("--runs-path", type=Path)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--runs", type=run_ids, default=run_ids("1-100"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--time-end-ps", type=float, default=40.0,
                        help="Planned equilibration end time for the shared x axis (ps).")
    parser.add_argument("--cpptraj", default="cpptraj")
    parser.add_argument("--columns", type=int, default=10)
    args = parser.parse_args()
    if args.columns < 1 or args.time_end_ps <= 0:
        parser.error("columns and time-end-ps must be positive")
    system_dir = Path("systems")/args.system/"solv_5.5"
    args.runs_path = args.runs_path or system_dir/"dftb/N1T48C1"
    args.topology = args.topology or system_dir/"salt/ready.parm7"
    args.out_dir = args.out_dir or Path("reports")/args.system/"solv_5.5/equil_dihedral"
    center = 180.0 if args.system == "PRN-anti" else 0.0
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    top = parmed.load_file(str(args.topology))
    ids = [5, 3, 4, 11]
    if [top.atoms[i-1].name for i in ids] != ["O2", "CG", "O1", "H11"]:
        raise ValueError("Topology does not match the requested dihedral")
    for a, b in zip(ids[:-1], ids[1:]):
        if top.atoms[b-1] not in top.atoms[a-1].bond_partners:
            raise ValueError("Dihedral atoms are not consecutively bonded")
    version = subprocess.check_output([args.cpptraj, "--version"], text=True).strip()
    results, records = {}, []
    with (out/"cpptraj.log").open("w") as log, (out/"cpptraj_inputs.txt").open("w") as inputs:
        for run in args.runs:
            folder = (args.runs_path/f"run-{run}"/"equil").resolve()
            box = box_from_input(folder/"dftb.inp")
            with tempfile.TemporaryDirectory(prefix=f"{args.system.lower()}-cpptraj-{run}-") as tmp:
                xyz, dat = Path(tmp)/"snapshot.xyz", Path(tmp)/"dihedral.dat"
                times, audit, info = snapshot(folder/"traject", xyz)
                if info["atoms"] != len(top.atoms):
                    raise ValueError("Trajectory/topology atom count mismatch")
                script = (f"parm {args.topology.resolve()}\ntrajin {xyz} 1 {len(times)} as xyz\n"
                          f"box x {box[0]} y {box[1]} z {box[2]} alpha 90 beta 90 gamma 90\n"
                          "fiximagedbonds :1\n"
                          f"dihedral PRN @5 @3 @4 @11 out {dat}\nrun\nquit\n")
                inputs.write(f"# run-{run}; temporary snapshot paths are recreated by this script\n{script}\n")
                proc = subprocess.run([args.cpptraj], input=script, text=True, stdout=log,
                                      stderr=subprocess.STDOUT)
                if proc.returncode:
                    raise RuntimeError(f"cpptraj failed for run {run}; see log")
                values = np.atleast_2d(np.loadtxt(dat))
                if len(values) != len(times) or not np.isfinite(values[:, 1]).all():
                    raise ValueError("cpptraj frame count or angle mismatch")
                angles = (values[:, 1] - center + 180) % 360 + center - 180
                chosen = sorted(set([0, len(times)//2, len(times)-1]))
                expected = np.asarray([dihedral(p, box) for p in audit])
                error = np.abs((angles[chosen]-expected+180)%360-180)
                if np.max(error) > .002:
                    raise ValueError(f"cpptraj/MIC audit failed: run {run}, {error}")
            results[run] = (times, angles)
            mean = (circular_mean_deg(angles)-center+180)%360+center-180
            records.append(dict(run=run, **info, start_ps=float(times[0]), end_ps=float(times[-1]),
                                circular_mean_deg=mean, circular_std_deg=circular_std_deg(angles),
                                box_A=box.tolist(), audit_max_error_deg=float(np.max(error))))
            print(f"run-{run}: {len(times)} frames, {times[-1]:.3f} ps", flush=True)
    with (out/"dihedral.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "time_ps", "dihedral_deg"])
        for run, (times, angles) in results.items():
            w.writerows((run, t, a) for t, a in zip(times, angles))
    with (out/"run_statistics.csv").open("w", newline="") as f:
        keys = ["run", "frames", "start_ps", "end_ps", "circular_mean_deg", "circular_std_deg"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(records)
    (out/"provenance.json").write_text(json.dumps(dict(cpptraj=version, atoms_one_based=ids,
        topology=str(args.topology.resolve()), source=str(args.runs_path.resolve()), runs=records,
        system=args.system, center_deg=center, planned_end_ps=args.time_end_ps,
        note="All complete equil frames, no burn-in removal. MIC bonded imaging before cpptraj torsion. Fixed original H11; does not track exchanged protons. Circular SD is angular spread, not standard error."), indent=2)+"\n")
    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    nrow = math.ceil(len(results)/args.columns)
    fig, axes = plt.subplots(nrow, args.columns, figsize=(3.3*args.columns, 2.35*nrow),
                             sharex=True, sharey=True, squeeze=False)
    for ax, record in zip(axes.flat, records):
        run = record["run"]; times, angles = results[run]
        x, y = wrapped_plot_series(times, angles, center)
        ax.axhline(center, color="0.65", lw=.5)
        ax.plot(x, y, color="#1F3A5F", lw=.6)
        ax.set_title(f"run-{run}", fontsize=10, pad=3)
        ax.text(.97, .92, f"{record['circular_mean_deg']:.1f}° ± {record['circular_std_deg']:.1f}°",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        if times[-1] < args.time_end_ps - 0.005:
            ax.text(.97, .08, f"through {times[-1]:.2f} ps", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8)
        ax.set_xlim(0, max(args.time_end_ps, max(r["end_ps"] for r in records)))
        ax.set_ylim(center-180, center+180)
        ax.set_yticks(np.arange(center-180, center+181, 90)); ax.tick_params(labelsize=8)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(results):]: ax.set_visible(False)
    fig.supxlabel("Equilibration time (ps)", fontsize=16)
    fig.supylabel(f"O2–CG–O1–H11 dihedral (degrees; centered at {center:g}°)", fontsize=16)
    fig.suptitle(f"{args.system} · {args.runs_path.parent.parent.name} · {args.runs_path.name} · {len(results)} equilibration runs\ncpptraj; circular mean ± circular SD; all complete frames available at read time", fontsize=18)
    fig.tight_layout(rect=(.015,.015,1,.955))
    png = out/f"{args.system.lower().replace('-', '_')}_equil_{len(results)}_dihedrals.png"
    fig.savefig(png, dpi=200); plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
