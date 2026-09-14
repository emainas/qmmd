"""Portable full-metadynamics PRN movie highlighting solute and assigned defect.

No connected-frame filtering, fixed water ownership, or water-wire graphics.
Raw trajectory/Mulliken files are read only; all movie files are derived outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from acid_base_PRN import _minimum_image
from prn_acid_base_analysis import aligned_data, snapshot_bytes


def display_defects(raw: np.ndarray, carry: bool) -> np.ndarray:
    """Forward-fill identities only; never backfill leading missing frames."""
    result = np.asarray(raw, dtype=float).copy()
    previous = np.nan
    for i, value in enumerate(result):
        if np.isfinite(value):
            previous = value
        elif carry:
            result[i] = previous
    return result


def image_frame(coords: np.ndarray, symbols: list[str], box: np.ndarray,
                anchor_id: int = 3, bond_cutoff: float = 1.4,
                ownership_margin: float = .15) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Center heavy atoms on C3 and image each H with its nearest heavy atom.

    Returned geometric heavy-H bonds use one-based IDs and may change each frame.
    Shared/ambiguous protons are imaged but not assigned a unique covalent bond.
    """
    heavy = np.asarray([i for i, s in enumerate(symbols) if s != "H"])
    hs = np.asarray([i for i, s in enumerate(symbols) if s == "H"])
    imaged = _minimum_image(coords - coords[anchor_id-1], box)
    vectors = _minimum_image(coords[hs, None, :] - coords[None, heavy, :], box)
    distances = np.linalg.norm(vectors, axis=2)
    order = np.argsort(distances, axis=1)
    first = order[:, 0]
    nearest = heavy[first]
    imaged[hs] = imaged[nearest] + vectors[np.arange(len(hs)), first]
    d0 = distances[np.arange(len(hs)), first]
    d1 = distances[np.arange(len(hs)), order[:, 1]]
    bound = (d0 <= bond_cutoff) & (d1-d0 >= ownership_margin)
    bonds = [(int(d)+1, int(h)+1) for d, h in zip(nearest[bound], hs[bound])]
    return imaged, bonds


VMD_SCRIPT = r'''# Full PRN solute/defect view. No wire filtering or wire graphics.
set movie_dir [file dirname [file normalize [info script]]]
set stream [open [file join $movie_dir frames.csv] r]
gets $stream header
set defect_rows {}
while {[gets $stream line] >= 0} {
    if {[string trim $line] ne ""} { lappend defect_rows [split $line ","] }
}
close $stream
mol new [file join $movie_dir trajectory.xyz] type xyz waitfor all
set defect_molid [molinfo top]
if {[molinfo $defect_molid get numframes] != [llength $defect_rows]} {
    error "Movie trajectory and frame metadata have different lengths"
}
mol delrep 0 $defect_molid
color Display Background white
display projection Orthographic
axes location Off
display depthcue on

# All atoms remain visible; no inferred static XYZ bonds are rendered.
mol representation VDW 0.12 12
mol selection all
mol color Name
mol material Transparent
mol addrep $defect_molid
# PRN skeleton and carbon-bound H atoms. H11 is shown independently below.
mol representation VDW 0.30 18
mol selection "index 0 to 9"
mol color Name
mol material Opaque
mol addrep $defect_molid
mol representation VDW 0.55 18
mol selection "index 10"
mol color ColorID 4
mol material Opaque
mol addrep $defect_molid

proc prn_defect_draw {name element op} {
    global defect_molid defect_rows
    set frame $::vmd_frame($defect_molid)
    if {$frame < 0 || $frame >= [llength $defect_rows]} { return }
    set row [lindex $defect_rows $frame]
    set selection [atomselect $defect_molid all frame $frame]
    set xyz [$selection get {x y z}]
    $selection delete
    graphics $defect_molid delete all
    graphics $defect_molid material Opaque
    # Only the four permanent PRN heavy-atom bonds are fixed.
    graphics $defect_molid color gray
    foreach pair {{1 2} {2 3} {3 4} {3 5}} {
        lassign $pair a b
        graphics $defect_molid cylinder [lindex $xyz [expr {$a-1}]] [lindex $xyz [expr {$b-1}]] radius 0.12 resolution 14
    }
    # Geometric heavy-H bonds are recalculated per frame, including transferred H.
    foreach pair [split [lindex $row 6] ";"] {
        if {$pair eq ""} { continue }
        lassign [split $pair "-"] d h
        if {$d <= 5} { set radius .09 } else { set radius .025 }
        graphics $defect_molid color gray
        graphics $defect_molid cylinder [lindex $xyz [expr {$d-1}]] [lindex $xyz [expr {$h-1}]] radius $radius resolution 10
    }
    set defect_id [lindex $row 3]
    if {$defect_id ne ""} {
        graphics $defect_molid color cyan
        graphics $defect_molid sphere [lindex $xyz [expr {$defect_id-1}]] radius .48 resolution 22
        foreach h [split [lindex $row 5] ";"] {
            if {$h eq ""} { continue }
            graphics $defect_molid sphere [lindex $xyz [expr {$h-1}]] radius .23 resolution 18
            graphics $defect_molid cylinder [lindex $xyz [expr {$defect_id-1}]] [lindex $xyz [expr {$h-1}]] radius .09 resolution 12
        }
        set status "defect O$defect_id; q=[lindex $row 4] e"
        if {[lindex $row 10] eq "carried"} {
            append status " | CARRIED identity (visual assumption)"
        }
    } else {
        set status "defect UNASSIGNED (not evidence of no excess proton)"
    }
    graphics $defect_molid color black
    graphics $defect_molid text {-7.2 -8.0 0} "meta elapsed [lindex $row 2] ps | raw t [lindex $row 1] ps" size .85 thickness 1
    graphics $defect_molid text {-7.2 -8.7 0} $status size .85 thickness 1
}
trace add variable ::vmd_frame($defect_molid) write prn_defect_draw
animate goto 0
prn_defect_draw {} {} {}
display resetview
animate speed .15
puts "Solute: large element-colored atoms. Original H11: yellow. Assigned solvent defect and its current H atoms: cyan."
puts "All frames retained, including unassigned/disconnected states. Use animation controls."
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("systems/PRN-syn/solv_100/dftb/N1T48C1/run-15/meta-h"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/PRN-syn/acid_base/run-15/vmd_solute_defect"))
    parser.add_argument("--charge-min", type=float, default=-.625, help="Defect oxygen charge minimum (e)")
    parser.add_argument("--charge-max", type=float, default=-.525, help="Defect oxygen charge maximum (e)")
    parser.add_argument("--bond-cutoff", type=float, default=1.4, help="Heavy-H covalent display cutoff (angstrom)")
    parser.add_argument("--ownership-margin", type=float, default=.15, help="Distance advantage over next nearest heavy atom (angstrom)")
    parser.add_argument("--carry-forward-defect", action="store_true",
                        help="Carry last assigned oxygen through gaps for display only; leading gaps remain unassigned")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(root / "systems"):
        parser.error("Output must be outside simulation directories")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new --out to preserve it")
    if args.charge_min > args.charge_max or args.bond_cutoff <= 0 or args.ownership_margin < 0:
        parser.error("Invalid charge or bond thresholds")
    # These atom IDs define PRN; keep input defaults explicit for the shared reader.
    args.solute_atoms, args.acid_oxygen, args.competitor_oxygen = 11, 4, 5
    args.dihedral = [5, 3, 4, 11]
    with tempfile.TemporaryDirectory(prefix="prn-defect-movie-") as tmp:
        snap = Path(tmp)
        sizes = {}
        for name in ("traject", "mulliken", "biaspot", "dftb.inp"):
            content = snapshot_bytes(args.source / name)
            sizes[name] = len(content)
            (snap / name).write_bytes(content)
        data = aligned_data(snap, args)
    args.out.mkdir(parents=True)
    fields = ["movie_frame_index", "raw_time_ps", "meta_elapsed_ps", "defect_oxygen_id",
              "defect_charge_e", "defect_hydrogen_ids", "geometric_heavy_H_bonds",
              "mulliken_timestamp_matched", "O4_defect_distance_A",
              "direct_defect_oxygen_id", "assignment_status"]
    displayed = display_defects(data["defect"], args.carry_forward_defect)
    carried_count = int(np.sum(np.isfinite(displayed) & ~np.isfinite(data["defect"])))
    assigned_count = 0
    with (args.out / "trajectory.xyz").open("w") as trajectory, (args.out / "frames.csv").open("w", newline="") as metadata:
        writer = csv.writer(metadata)
        writer.writerow(fields)
        for i, (t, coords) in enumerate(zip(data["times"], data["coords"])):
            imaged, bonds = image_frame(coords, data["symbols"], data["box"],
                                        bond_cutoff=args.bond_cutoff, ownership_margin=args.ownership_margin)
            raw_defect = data["defect"][i]
            defect = int(displayed[i]) if np.isfinite(displayed[i]) else None
            status = "direct" if np.isfinite(raw_defect) else ("carried" if defect is not None else "unassigned")
            charge = ""
            defect_hs = []
            if defect is not None:
                assigned_count += 1
                oxygen_index = int(np.flatnonzero(data["oxygen_ids"] == defect)[0])
                charge = f"{data['charges'][oxygen_index, i]:.6f}"
                defect_hs = [h for d, h in bonds if d == defect]
            elapsed = t-data["origin"]
            trajectory.write(f"{len(imaged)}\nframe={i} raw_time_ps={t:.8f} meta_elapsed_ps={elapsed:.8f}\n")
            for symbol, point in zip(data["symbols"], imaged):
                trajectory.write(f"{symbol} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f}\n")
            writer.writerow([i, f"{t:.6f}", f"{elapsed:.6f}", defect or "", charge,
                             ";".join(map(str, defect_hs)), ";".join(f"{d}-{h}" for d, h in bonds),
                             int(data["matched"][i]), data["distance"][i],
                             int(raw_defect) if np.isfinite(raw_defect) else "", status])
    (args.out / "view_defect.vmd").write_text(VMD_SCRIPT)
    provenance = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance.update(source=str(args.source.resolve()), snapshot_bytes=sizes, frames=len(data["times"]),
                      assigned_defect_frames=assigned_count, carried_defect_frames=carried_count,
                      direct_defect_frames=int(np.isfinite(data["defect"]).sum()), raw_time_start_ps=float(data["times"][0]),
                      raw_time_end_ps=float(data["times"][-1]), elapsed_end_ps=float(data["times"][-1]-data["origin"]),
                      definition="Most positive solvent-O Mulliken charge in configured interval. Exact timestamp matching. Optional identity-only forward fill for visual continuity; leading gaps remain missing. Current charges and bonds are never carried forward. Direct assignments retained separately. All complete trajectory frames retained.",
                      imaging="C3 centered; heavy atoms in nearest periodic image; every H imaged beside current nearest heavy atom. Atom identities and minimum-image geometry preserved. Periodic boundary crossings may remain visible.")
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (args.out / "README.txt").write_text(
        "Open: vmd -e view_defect.vmd\nKeep trajectory.xyz and frames.csv beside the VMD script.\n"
        "Full available metadynamics, starting at elapsed time zero; no equilibration prefix.\n"
        "All frames retained. Solute skeleton is prominent; H11 is yellow; assigned solvent defect and current bonded H atoms are cyan.\n"
        "No water-wire filtering or wire drawings.\n"
        f"Identity forward fill enabled: {args.carry_forward_defect}. Carried identities are labeled CARRIED; leading gaps stay unassigned.\n"
        "Carried identity is a visual assumption, not evidence of defect persistence. Current charges and bonds are not filled.\n"
        "Mulliken-based defect identity is a heuristic, not independent chemical confirmation.\n"
        "Coordinates are periodically imaged around C3 in this derived trajectory only.\n")
    print(f"Wrote {len(data['times'])} frames; {assigned_count} defect-assigned; elapsed 0–{provenance['elapsed_end_ps']:.3f} ps")
    print(f"Open: vmd -e {args.out / 'view_defect.vmd'}")


if __name__ == "__main__":
    main()
