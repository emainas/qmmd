"""Snapshot-safe PRN screening and aligned inputs for the HIST-derived summary."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

TIME_RE = re.compile(r"AT T=\s*([\d.]+)\s*FSEC")


def snapshot_bytes(path: Path) -> bytes:
    """Never follow a growing file past its size when opened."""
    with path.open("rb") as handle:
        return handle.read(os.fstat(handle.fileno()).st_size)


def xyz_frames(path: Path) -> Iterator[tuple[float, np.ndarray]]:
    handle = io.StringIO(snapshot_bytes(path).decode())
    while line := handle.readline():
        if not line.strip():
            continue
        count = int(line)
        comment = handle.readline()
        match = TIME_RE.search(comment)
        coords = []
        for _ in range(count):
            row = handle.readline()
            if not row.endswith("\n") or len(row.split()) < 4:
                return  # Live, incomplete final frame.
            coords.append([float(x) for x in row.split()[1:4]])
        if match is None:
            raise ValueError(f"Missing trajectory timestamp: {path}")
        yield float(match[1]) / 1000, np.asarray(coords)


def mulliken_frames(path: Path, symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sum orbital charges; require every expected orbital in a complete block."""
    handle = io.StringIO(snapshot_bytes(path).decode())
    times, charges = [], []
    expected = {(i, orb) for i, element in enumerate(symbols, 1)
                for orb in (("s",) if element == "H" else ("s", "p"))}
    while line := handle.readline():
        if not line.strip():
            continue
        count, natoms = map(int, line.split())
        if natoms != len(symbols) or count != len(expected):
            raise ValueError("Mulliken basis/atom count does not match PRN trajectory")
        match = TIME_RE.search(handle.readline())
        values = np.zeros(natoms)
        seen = set()
        for _ in range(count):
            row = handle.readline()
            if not row.endswith("\n") or len(row.split()) != 4:
                return np.asarray(times), np.asarray(charges)
            atom, element, orbital, value = row.split()
            atom = int(atom)
            if (atom, orbital) not in expected or (atom, orbital) in seen:
                raise ValueError("Unexpected or duplicate Mulliken orbital")
            if symbols[atom - 1] != element:
                raise ValueError("Mulliken atom order differs from trajectory")
            seen.add((atom, orbital))
            values[atom - 1] += float(value)
        if match is None:
            raise ValueError("Missing Mulliken timestamp")
        times.append(float(match[1]) / 1000)
        charges.append(values)
    return np.asarray(times), np.asarray(charges)


def bias_samples(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    text = snapshot_bytes(path).decode()
    times, values, by_hill = [], [], {}
    for block in text.split("GAUSSIAN BIAS POTENTIAL:")[1:]:
        time = TIME_RE.search(block)
        value = re.search(r"Coordinate\s*=\s*([\d.Ee+\-]+)", block)
        width = re.search(r"Gaussian width\s*=\s*([\d.Ee+\-]+)\s*\n", block)
        if time and value and width:
            t = float(time[1]) / 1000
            times.append(t)
            values.append(float(value[1]))
            by_hill[int(block.split()[0])] = t
    return np.asarray(times), np.asarray(values), by_hill


def aligned_fes(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Match complete FES surfaces to bias timestamps by Gaussian count."""
    _, _, hill_times = bias_samples(path.parent / "biaspot")
    text = snapshot_bytes(path).decode()
    chunks = re.split(r"### FREE ENERGY SURFACE CONSISTING OF\s+(\d+) GAUSSIANS", text)
    times, blocks = [], []
    reference = None
    for i in range(1, len(chunks), 2):
        rows = []
        for line in chunks[i + 1].splitlines(keepends=True):
            parts = line.split()
            if len(parts) == 2 and line.endswith("\n"):
                rows.append([float(x) for x in parts])
        block = np.asarray(rows)
        if not rows:
            continue
        if reference is None:
            reference = block[:, 0].copy()
        if len(block) != len(reference) or not np.allclose(block[:, 0], reference):
            if i + 2 < len(chunks):
                raise ValueError("Incomplete interior FES surface")
            continue
        hill = int(chunks[i])
        if hill in hill_times:
            block[:, 1] *= 627.509474
            times.append(hill_times[hill])
            blocks.append(block)
    return np.asarray(times), blocks


def aligned_data(folder: Path, args: argparse.Namespace) -> dict:
    from acid_base_PRN import read_xyz_symbols, read_box_lengths_from_dftb_inp
    from acid_base_PRN import _minimum_image, _signed_dihedral_deg
    symbols = read_xyz_symbols(folder / "traject")
    box = read_box_lengths_from_dftb_inp(folder / "dftb.inp")
    frames = list(xyz_frames(folder / "traject"))
    if not frames:
        raise ValueError("No complete trajectory frames")
    times = np.asarray([t for t, _ in frames])
    coords = np.asarray([c for _, c in frames])
    mt, charges = mulliken_frames(folder / "mulliken", symbols)
    bt, bs, _ = bias_samples(folder / "biaspot")
    if len(mt) < 2 or len(bt) < 2:
        raise ValueError("Insufficient complete Mulliken/bias samples")
    for label, ts in (("trajectory", times), ("Mulliken", mt), ("bias", bt)):
        if np.any(np.diff(ts) <= 0):
            raise ValueError(f"Nonmonotonic {label} timestamps; restart needs explicit stitching")
    # Strict timestamp matching, not row alignment or broad nearest-neighbor matching.
    lookup = {round(t, 8): i for i, t in enumerate(mt)}
    mi = np.asarray([lookup.get(round(t, 8), -1) for t in times])
    q = np.full((len(times), len(symbols)), np.nan)
    present = mi >= 0
    q[present] = charges[mi[present]]
    oxygen_ids = np.asarray([i for i, s in enumerate(symbols, 1)
                             if i > args.solute_atoms and s == "O"])
    oq = q[:, oxygen_ids - 1]
    eligible = (oq >= args.charge_min) & (oq <= args.charge_max)
    assigned = np.any(eligible, axis=1)
    winner = np.argmax(np.where(eligible, oq, -np.inf), axis=1)
    defect = np.where(assigned, oxygen_ids[winner], np.nan)
    distance = np.full(len(times), np.nan)
    idx = np.flatnonzero(assigned)
    delta = coords[idx, defect[idx].astype(int) - 1] - coords[idx, args.acid_oxygen - 1]
    distance[idx] = np.linalg.norm(_minimum_image(delta, box), axis=1)
    coordination = np.interp(times, bt, bs, left=np.nan, right=np.nan)
    phi = np.asarray([_signed_dihedral_deg(c[np.asarray(args.dihedral) - 1], box)
                      for c in coords])
    state_distance = np.full(len(times), np.nan)
    finite = np.isfinite(distance)
    if finite.sum() >= 2:
        state_distance = np.interp(times, times[finite], distance[finite],
                                   left=np.nan, right=np.nan)
    return dict(times=times, coordination=coordination, distance=distance,
                state_distance=state_distance, defect=defect, phi=phi, box=box,
                charges=oq.T, oxygen_ids=oxygen_ids, matched=present,
                origin=float(times[0]), coords=coords, symbols=symbols,
                competitor_charge=q[:, args.competitor_oxygen - 1])


def competitor_geometry(data: dict, args: argparse.Namespace, end: float) -> np.ndarray:
    """O5 charge, defect wire, and all-H coordination on the exact frame grid."""
    from acid_base_PRN import _minimum_image, shortest_hbond_path
    endpoint = args.competitor_oxygen
    if not 1 <= endpoint <= args.solute_atoms or data["symbols"][endpoint - 1] != "O":
        raise ValueError("Competitor must be a one-based solute oxygen ID")
    hydrogen_ids = np.asarray([i for i, symbol in enumerate(data["symbols"], 1)
                               if symbol == "H"])
    rows = []
    for i, t in enumerate(data["times"]):
        if t > end:
            break
        coords = data["coords"][i]
        delta = _minimum_image(coords[hydrogen_ids - 1] - coords[endpoint - 1], data["box"])
        distances = np.linalg.norm(delta, axis=1)
        coordination = np.sum(1 / (1 + (distances / args.competitor_refdist) ** 6))
        wire = np.nan
        if np.isfinite(data["defect"][i]):
            result, _ = shortest_hbond_path(
                coords, endpoint, int(data["defect"][i]), data["oxygen_ids"], hydrogen_ids,
                covalent_cutoff=args.covalent_cutoff,
                hydrogen_acceptor_cutoff=args.hydrogen_acceptor_cutoff,
                angle_cutoff=args.angle_cutoff,
                max_bridging_waters=args.competitor_max_bridging_waters, box=data["box"])
            wire = result.bridging_water_count if result.connected else -1.
        rows.append([t, data["competitor_charge"][i], wire, coordination])
    return np.asarray(rows)


def screen(data: dict, args: argparse.Namespace) -> tuple[float, float]:
    from acid_base_PRN import detect_diffusive_start, _first_sustained_start
    t, s, d = data["times"], data["coordination"], data["distance"]
    diffuse = detect_diffusive_start(t, s, d, args.distance_min,
                                     args.deprotonated_s_max, args.returned_s_min,
                                     args.persistence_ps)
    dep = _first_sustained_start(t, (data["state_distance"] >= args.distance_min)
                                & (s <= args.deprotonated_s_max), args.persistence_ps)
    return float(t[dep]), diffuse


def save_base(path: Path, data: dict, args: argparse.Namespace, end: float) -> None:
    fields = ["time_ps", "coordination_s", "defect_oxygen_id",
              f"O{args.acid_oxygen}_Odefect_distance_A", "state_distance_interpolated_A",
              "mulliken_timestamp_matched", "dihedral_raw_deg", "dihedral_plotted_deg"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i, t in enumerate(data["times"]):
            if t > end:
                break
            writer.writerow([t, data["coordination"][i], data["defect"][i],
                             data["distance"][i], data["state_distance"][i],
                             int(data["matched"][i]), data["phi"][i],
                             ((data["bond_dihedrals"]["old_dihedral_deg"][i] - args.dihedral_center + 180) % 360
                              + args.dihedral_center - 180) if "bond_dihedrals" in data else
                             ((data["phi"][i] - args.dihedral_center + 180) % 360
                              + args.dihedral_center - 180 if t < data.get("dihedral_stop", data["deprot"]) else np.nan)])


def manual_event_times(data: dict, args: argparse.Namespace, diffuse: float) -> tuple[float, float]:
    """User-specified raw-clock diffusion; independent sustained labeled-H loss."""
    from acid_base_PRN import _first_sustained_start
    times = data["times"]
    if not np.isfinite(diffuse) or not times[0] <= diffuse <= times[-1]:
        raise ValueError("Manual diffusion time is outside the available DFTB-reported clock")
    index = _first_sustained_start(times, (data["coordination"] <= args.deprotonated_s_max)
                                   & (times < diffuse), args.persistence_ps)
    return (float(times[index]) if index is not None else float("nan")), diffuse


def main() -> None:
    from prn_anti_dih import run_ids
    import acid_base_PRN as plot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=run_ids, default=run_ids("1-20"))
    parser.add_argument("--runs-path", type=Path,
                        default=Path("systems/PRN-anti/solv_100/dftb/N1T48C1"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/PRN-anti/acid_base"))
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--diffusion-start", action="append", default=[], metavar="RUN:PS",
                        help="Manual diffusion override on the raw DFTB-reported clock; repeat per run")
    parser.add_argument("--diffusion-clock-note", default="",
                        help="Provenance for converting manually selected diffusion times")
    parser.add_argument("--acid-oxygen", type=int, default=4, help="One-based acid oxygen ID")
    parser.add_argument("--solute-atoms", type=int, default=11)
    parser.add_argument("--competitor-oxygen", type=int, default=5, help="One-based competing solute oxygen ID")
    parser.add_argument("--competitor-max-bridging-waters", type=int, default=3)
    parser.add_argument("--competitor-refdist", type=float, default=1.6,
                        help="All-H rational coordination reference distance (angstrom); exponents 6/12")
    parser.add_argument("--dihedral", nargs=4, type=int, default=[5, 3, 4, 11])
    parser.add_argument("--dihedral-center", type=float, default=180.,
                        help="Dihedral display center in degrees; use 0 for syn, 180 for anti")
    parser.add_argument("--bond-cutoff", type=float, default=1.4, help="Dihedral O-H bond assignment cutoff (angstrom)")
    parser.add_argument("--bond-ownership-margin", type=float, default=.15,
                        help="Minimum distance advantage over next nearest heavy atom (angstrom)")
    parser.add_argument("--bond-persistence-ps", type=float, default=.05,
                        help="Minimum continuous same-H bond assignment duration (ps)")
    for name, default, help_text in [
        ("charge-min", -.625, "Lower solvent-O Mulliken charge bound (e)"),
        ("charge-max", -.525, "Upper solvent-O Mulliken charge bound (e)"),
        ("distance-min", 4., "Separated acid-O to defect-O distance (angstrom)"),
        ("deprotonated-s-max", .05, "Maximum labeled-H coordination for deprotonation"),
        ("returned-s-min", .20, "Minimum labeled-H coordination for recovery"),
        ("persistence-ps", .05, "Sustained-state duration (ps)"),
        ("probability-window-ps", 1.75, "Post-diffusion analysis window (ps)"),
        ("covalent-cutoff", 1.3, "H-bond short heavy-H distance (angstrom), NOT event cutoff"),
        ("hydrogen-acceptor-cutoff", 2.5, "H-bond H-acceptor distance (angstrom)"),
        ("angle-cutoff", 135., "Minimum D-H-A angle (degrees)"),
        ("temperature", 300., "FES apparent-pKa conversion temperature (K)"),
    ]:
        parser.add_argument(f"--{name}", type=float, default=default, help=help_text)
    args = parser.parse_args()
    overrides = {}
    for entry in args.diffusion_start:
        try:
            run_text, time_text = entry.split(":")
            run_id, value = int(run_text), float(time_text)
            if run_id not in args.runs or run_id in overrides or not np.isfinite(value):
                raise ValueError
            overrides[run_id] = value
        except ValueError:
            parser.error("--diffusion-start requires a unique selected RUN:PS with finite PS")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    screening_path = args.out_dir / "screening.csv"
    summary = list(csv.DictReader(screening_path.open())) if screening_path.exists() else []
    for run in args.runs:
        source = args.runs_path / f"run-{run}" / "meta-h"
        row = dict(run=run, status="", deprotonation_elapsed_ps="", diffusion_elapsed_ps="",
                   available_elapsed_ps="", matched_mulliken_frames="", plotted="no")
        try:
            if (source / "metad-restart").exists():
                raise ValueError("Restart subdirectory present: explicit stitching required")
            with tempfile.TemporaryDirectory(prefix=f"prn-acid-{run}-") as tmp:
                snap = Path(tmp)
                for name in ("traject", "mulliken", "biaspot", "dftb.inp", "fes.dat"):
                    if name == "fes.dat" and not (source / name).exists():
                        continue
                    (snap / name).write_bytes(snapshot_bytes(source / name))
                data = aligned_data(snap, args)
                row["available_elapsed_ps"] = data["times"][-1] - data["origin"]
                row["matched_mulliken_frames"] = int(data["matched"].sum())
                manual = run in overrides
                dep, diffuse = (manual_event_times(data, args, overrides[run])
                                if manual else screen(data, args))
                data["deprot"] = dep
                data["dihedral_stop"] = dep if np.isfinite(dep) else diffuse
                row.update(status="manual diffusion override" if manual else "diffusion detected", deprotonation_elapsed_ps=dep-data["origin"],
                           diffusion_elapsed_ps=diffuse-data["origin"])
                end = diffuse + args.probability_window_ps
                if data["times"][-1] < end:
                    row["status"] = "diffusion detected; awaiting complete analysis window"
                elif not args.screen_only:
                    dest = args.out_dir / f"run-{run}"
                    dest.mkdir(exist_ok=True)
                    base, augmented = dest / "aligned.csv", dest / "summary_data.csv"
                    from prn_proton_dihedrals import bond_dihedrals
                    bond_selected = data["times"] <= end
                    bonded = bond_dihedrals(data["times"][bond_selected], data["coords"][bond_selected],
                                            data["symbols"], data["box"], args.dihedral,
                                            args.competitor_oxygen, args.bond_cutoff,
                                            args.bond_ownership_margin, args.bond_persistence_ps)
                    data["bond_dihedrals"] = bonded
                    np.savetxt(dest / "proton_dihedrals.csv", np.column_stack(list(bonded.values())),
                               delimiter=",", header=",".join(bonded), comments="")
                    save_base(base, data, args, end)
                    print(f"run-{run}: diffusion at {diffuse-data['origin']:.3f} ps; making summary", flush=True)
                    plot.augment_csv(base, augmented, snap / "traject", args.acid_oxygen,
                                     args.solute_atoms, box=data["box"],
                                     covalent_cutoff=args.covalent_cutoff,
                                     hydrogen_acceptor_cutoff=args.hydrogen_acceptor_cutoff,
                                     angle_cutoff=args.angle_cutoff)
                    selected = data["times"] <= end
                    phi = np.where(data["times"] < data["dihedral_stop"], data["phi"], np.nan)
                    plot.save_oxygen_charge_csv(dest / "oxygen_charges.csv", data["times"][selected],
                                                data["charges"][:, selected], data["oxygen_ids"])
                    fes = snap / "fes.dat"
                    competitor = competitor_geometry(data, args, end)
                    np.savetxt(dest / "competitor.csv", competitor, delimiter=",", comments="",
                               header=f"time_ps,O{args.competitor_oxygen}_charge_e,O{args.competitor_oxygen}_defect_wire_bridging_waters,O{args.competitor_oxygen}_all_H_coordination")
                    plot.plot_wire_csv(
                        augmented, dest / "summary.png", diffusive_start=diffuse,
                        probability_window_ps=args.probability_window_ps,
                        deprotonation_start=dep if np.isfinite(dep) else None, temperature=args.temperature,
                        covalent_cutoff=args.covalent_cutoff,
                        hydrogen_acceptor_cutoff=args.hydrogen_acceptor_cutoff,
                        angle_cutoff=args.angle_cutoff,
                        fes_path=fes if fes.exists() else None,
                        pka_data_out=dest / "pka.csv", fes_snapshot_out=dest / "fes_snapshot.csv",
                        oxygen_charge_times=data["times"][selected],
                        oxygen_charges=data["charges"][:, selected], oxygen_ids=data["oxygen_ids"],
                        dihedral_times=data["times"][selected], dihedrals=phi[selected],
                        dihedral_ids=args.dihedral,
                        dihedral_center=args.dihedral_center,
                        bond_dihedral_data=bonded,
                        competitor_data=competitor, competitor_id=args.competitor_oxygen,
                        competitor_max_bridging_waters=args.competitor_max_bridging_waters,
                        manual_diffusion=manual,
                        figure_title=f"{args.runs_path.parts[-4]} run-{run} | metadynamics")
                    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                    metadata.update(source=str(source.resolve()), reported_time_origin_ps=data["origin"],
                                    pka_convention="[F(min near s=0) - F(min near s=1)] / (0.004576 * T); F in kcal/mol, T in K",
                                    deprotonation_reported_ps=dep if np.isfinite(dep) else None, diffusion_reported_ps=diffuse,
                                    dihedral_display="Dynamic geometry-based ownership: old labeled O4-H11 and newly bonded O5-H; persistent same-H intervals only. No diffusion/deprotonation-time truncation.",
                                    diffusion_source="manual override" if manual else "automatic HIST criterion",
                                    deprotonation_definition="sustained labeled-H coordination <= threshold, without defect-separation requirement" if manual else "sustained separated deprotonation",
                                    caveat="HIST heuristic: interpolate missing defect distances for event detection; raw plot gaps remain NaN. Labeled-H recovery is not proof of acid reprotonation. FES pKa is an apparent estimator.")
                    (dest / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    row["plotted"] = "yes"
        except (ValueError, FileNotFoundError) as exc:
            row["status"] = str(exc)
        summary = [old for old in summary if int(old["run"]) != run]
        summary.append(row)
        summary.sort(key=lambda item: int(item["run"]))
        print(f"run-{run}: {row['status']}", flush=True)
        with (args.out_dir / "screening.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(summary)


if __name__ == "__main__":
    main()
