from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
import numpy as np

from qmmd.cphmd_dgref import (
    ChargeSets,
    TopologyInfo,
    find_repo_root,
    read_charge_sets,
    read_topology_info,
)
from qmmd.cphmd_titr import titr_dir
from qmmd.cphmd_titr_post import (
    TitrPostConfig,
    load_titr_post_config,
    post_dir,
    run_cpptraj,
)


@dataclass(frozen=True)
class CphmdViewConfig:
    post: TitrPostConfig
    output_dir: str
    residue_name: str
    residue_id: int | None
    copy_inputs: bool
    zero_charge_tolerance: float
    focus_atoms: tuple[str, ...]
    orientation_atoms: tuple[str, str, str]
    zoom: float
    water_sphere_scale: float
    water_sphere_resolution: int
    hbond_donor_atom: str
    hbond_hydrogen_atoms: tuple[str, ...]
    hbond_water_oxygen_selection: str
    hbond_distance_cutoff: float
    hbond_angle_cutoff: float
    hbond_line_radius: float
    hbond_dash_count: int
    hbond_color_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class ViewFrame:
    frame: int
    time_ps: float
    state: int
    proton_count: int


@dataclass(frozen=True)
class DummyAtom:
    name: str
    vmd_index: int
    hidden_states: tuple[int, ...]


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _child_path(value: object, field: str, default: str) -> str:
    text = default if value is None else str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{field} must be a child path")
    return text


def load_cphmd_view_config(yaml_path: Path) -> CphmdViewConfig:
    resolved = yaml_path.resolve()
    post = load_titr_post_config(resolved)
    data = yaml.safe_load(resolved.read_text())
    raw = data.get("view", {})
    if not isinstance(raw, dict):
        raise ValueError("view must be a YAML mapping")

    output_dir = _child_path(raw.get("output_dir"), "view.output_dir", "view")
    residue_name = _nonempty_string(
        raw.get("residue_name", post.titr.system), "view.residue_name"
    )
    raw_residue_id = raw.get("residue_id")
    residue_id = None if raw_residue_id is None else int(raw_residue_id)
    if residue_id is not None and residue_id < 1:
        raise ValueError("view.residue_id must be one-based and positive")
    tolerance = float(raw.get("zero_charge_tolerance", 1.0e-8))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("view.zero_charge_tolerance must be positive and finite")
    raw_focus = raw.get("focus_atoms", [])
    if not isinstance(raw_focus, list) or not raw_focus or not all(
        isinstance(value, str) and value.strip() for value in raw_focus
    ):
        raise ValueError("view.focus_atoms must be a nonempty list of atom names")
    focus_atoms = tuple(value.strip() for value in raw_focus)
    if len(set(focus_atoms)) != len(focus_atoms):
        raise ValueError("view.focus_atoms contains duplicates")
    raw_orientation = raw.get("orientation_atoms")
    if not isinstance(raw_orientation, list) or len(raw_orientation) != 3 or not all(
        isinstance(value, str) and value.strip() for value in raw_orientation
    ):
        raise ValueError("view.orientation_atoms must contain exactly three atom names")
    orientation_values = tuple(value.strip() for value in raw_orientation)
    if len(set(orientation_values)) != 3:
        raise ValueError("view.orientation_atoms must contain three distinct atoms")
    if orientation_values[0] not in focus_atoms:
        raise ValueError("The first view.orientation_atoms entry must be a focus atom")
    orientation_atoms = (
        orientation_values[0],
        orientation_values[1],
        orientation_values[2],
    )
    zoom = float(raw.get("zoom", 4.0))
    water_sphere_scale = float(raw.get("water_sphere_scale", 0.35))
    water_sphere_resolution = int(raw.get("water_sphere_resolution", 24))
    if not math.isfinite(zoom) or zoom <= 0:
        raise ValueError("view.zoom must be positive and finite")
    if not math.isfinite(water_sphere_scale) or water_sphere_scale <= 0:
        raise ValueError("view.water_sphere_scale must be positive and finite")
    if water_sphere_resolution < 6:
        raise ValueError("view.water_sphere_resolution must be at least 6")
    raw_hbonds = raw.get("hydrogen_bonds", {})
    if not isinstance(raw_hbonds, dict):
        raise ValueError("view.hydrogen_bonds must be a YAML mapping")
    hbond_donor_atom = _nonempty_string(
        raw_hbonds.get("donor_atom"), "view.hydrogen_bonds.donor_atom"
    )
    raw_hydrogens = raw_hbonds.get("hydrogen_atoms")
    if not isinstance(raw_hydrogens, list) or not raw_hydrogens or not all(
        isinstance(value, str) and value.strip() for value in raw_hydrogens
    ):
        raise ValueError(
            "view.hydrogen_bonds.hydrogen_atoms must be a nonempty list of atom names"
        )
    hbond_hydrogen_atoms = tuple(value.strip() for value in raw_hydrogens)
    if len(set(hbond_hydrogen_atoms)) != len(hbond_hydrogen_atoms):
        raise ValueError("view.hydrogen_bonds.hydrogen_atoms contains duplicates")
    hbond_water_oxygen_selection = _nonempty_string(
        raw_hbonds.get("water_oxygen_selection", "water and name O"),
        "view.hydrogen_bonds.water_oxygen_selection",
    )
    hbond_distance_cutoff = float(raw_hbonds.get("distance_cutoff_angstrom", 2.5))
    hbond_angle_cutoff = float(raw_hbonds.get("angle_cutoff_degrees", 135.0))
    hbond_line_radius = float(raw_hbonds.get("line_radius_angstrom", 0.08))
    hbond_dash_count = int(raw_hbonds.get("dash_count", 6))
    raw_color = raw_hbonds.get("color_rgb", [0.45, 1.0, 0.45])
    if not isinstance(raw_color, list) or len(raw_color) != 3:
        raise ValueError("view.hydrogen_bonds.color_rgb must contain three values")
    hbond_color_rgb = tuple(float(value) for value in raw_color)
    if not math.isfinite(hbond_distance_cutoff) or hbond_distance_cutoff <= 0:
        raise ValueError("hydrogen-bond distance cutoff must be positive and finite")
    if not 0 < hbond_angle_cutoff <= 180:
        raise ValueError("hydrogen-bond angle cutoff must be in (0, 180] degrees")
    if not math.isfinite(hbond_line_radius) or hbond_line_radius <= 0:
        raise ValueError("hydrogen-bond line radius must be positive and finite")
    if hbond_dash_count < 2:
        raise ValueError("hydrogen-bond dash_count must be at least 2")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in hbond_color_rgb):
        raise ValueError("hydrogen-bond color_rgb values must be between 0 and 1")

    return CphmdViewConfig(
        post=post,
        output_dir=output_dir,
        residue_name=residue_name,
        residue_id=residue_id,
        copy_inputs=bool(raw.get("copy_inputs", False)),
        zero_charge_tolerance=tolerance,
        focus_atoms=focus_atoms,
        orientation_atoms=orientation_atoms,
        zoom=zoom,
        water_sphere_scale=water_sphere_scale,
        water_sphere_resolution=water_sphere_resolution,
        hbond_donor_atom=hbond_donor_atom,
        hbond_hydrogen_atoms=hbond_hydrogen_atoms,
        hbond_water_oxygen_selection=hbond_water_oxygen_selection,
        hbond_distance_cutoff=hbond_distance_cutoff,
        hbond_angle_cutoff=hbond_angle_cutoff,
        hbond_line_radius=hbond_line_radius,
        hbond_dash_count=hbond_dash_count,
        hbond_color_rgb=(hbond_color_rgb[0], hbond_color_rgb[1], hbond_color_rgb[2]),
    )


def _format_ph(ph: float) -> str:
    text = f"{ph:.6f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def _resolve_ph(requested: float, available: tuple[float, ...]) -> float:
    matches = [value for value in available if math.isclose(requested, value, abs_tol=1e-6)]
    if len(matches) != 1:
        choices = ", ".join(f"{value:g}" for value in available)
        raise ValueError(f"pH {requested:g} is not in the configured ladder: {choices}")
    return matches[0]


def read_view_frames(
    path: Path,
    ph: float,
    residue_name: str,
    residue_id: int,
    expected_frames: int,
) -> tuple[list[ViewFrame], str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing coordinate/state alignment table: {path}; run cphmd-titr-post first"
        )
    frames: list[ViewFrame] = []
    trajectories: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trajectory_frame",
            "relative_time_ps",
            "pH",
            "trajectory",
            "residue",
            "residue_id",
            "state",
            "proton_count",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Malformed coordinate/state alignment table: {path}")
        for row in reader:
            if not math.isclose(float(row["pH"]), ph, abs_tol=1e-6):
                continue
            if row["residue"] != residue_name or int(row["residue_id"]) != residue_id:
                continue
            frames.append(
                ViewFrame(
                    frame=int(row["trajectory_frame"]),
                    time_ps=float(row["relative_time_ps"]),
                    state=int(row["state"]),
                    proton_count=int(row["proton_count"]),
                )
            )
            trajectories.add(row["trajectory"])

    if not frames:
        raise ValueError(
            f"No aligned frames found for {residue_name} {residue_id} at pH {ph:g}"
        )
    expected_indices = list(range(1, expected_frames + 1))
    if [frame.frame for frame in frames] != expected_indices:
        raise ValueError(
            f"Expected exactly {expected_frames} ordered trajectory frames at pH {ph:g}"
        )
    if len(trajectories) != 1:
        raise ValueError(f"Aligned frames refer to multiple trajectories: {sorted(trajectories)}")
    return frames, trajectories.pop()


def identify_dummy_atoms(
    charges: ChargeSets,
    topology: TopologyInfo,
    tolerance: float,
) -> tuple[DummyAtom, ...]:
    if topology.atom_names != charges.atom_names:
        raise ValueError(
            "Charge-set atom ordering does not match the topology residue: "
            f"{charges.atom_names} != {topology.atom_names}"
        )
    state_charges = (charges.prot_charges, charges.deprot_charges)
    dummy_atoms: list[DummyAtom] = []
    for local_index, name in enumerate(charges.atom_names):
        is_zero = tuple(abs(values[local_index]) <= tolerance for values in state_charges)
        if all(is_zero) or not any(is_zero):
            continue
        if not name.upper().startswith("H"):
            raise ValueError(
                f"State-dependent zero-charge atom {name} is not a hydrogen"
            )
        hidden_states = tuple(state for state, hidden in enumerate(is_zero) if hidden)
        dummy_atoms.append(
            DummyAtom(
                name=name,
                vmd_index=topology.first_atom - 1 + local_index,
                hidden_states=hidden_states,
            )
        )
    if not dummy_atoms:
        raise ValueError("No state-dependent zero-charge dummy hydrogens were found")
    return tuple(dummy_atoms)


def validate_frame_states(frames: list[ViewFrame], charges: ChargeSets) -> None:
    proton_counts = (charges.proton_count_prot, charges.proton_count_deprot)
    invalid = sorted({frame.state for frame in frames if frame.state not in {0, 1}})
    if invalid:
        raise ValueError(f"The two-state viewer cannot display state indices {invalid}")
    mismatches = [
        frame.frame
        for frame in frames
        if frame.proton_count != proton_counts[frame.state]
    ]
    if mismatches:
        raise ValueError(
            "Aligned proton counts disagree with charge_sets.csv at frames: "
            + ", ".join(str(value) for value in mismatches[:10])
        )


def pin_local_frame_coordinates(
    coordinates: np.ndarray,
    focus_indices: tuple[int, ...],
    orientation_indices: tuple[int, int, int],
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("Trajectory coordinates must have shape (frames, atoms, 3)")
    site = coords[:, focus_indices, :].mean(axis=1)
    first = coords[:, orientation_indices[0], :]
    second = coords[:, orientation_indices[1], :]
    third = coords[:, orientation_indices[2], :]
    bond = second - first
    plane = third - first

    bond_norm = np.linalg.norm(bond, axis=1)
    normal = np.cross(bond, plane)
    normal_norm = np.linalg.norm(normal, axis=1)
    if np.any(bond_norm < 1.0e-8) or np.any(normal_norm < 1.0e-8):
        raise ValueError("view.orientation_atoms form a degenerate local frame")
    x_axis = bond / bond_norm[:, None]
    z_axis = normal / normal_norm[:, None]
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=1)[:, None]
    basis = np.stack((x_axis, y_axis, z_axis), axis=1)

    centered = coords - site[:, None, :]
    components = np.einsum("fij,faj->fai", basis, centered)
    pinned = np.einsum("jk,faj->fak", basis[0], components)
    pinned += site[0][None, None, :]
    return pinned


def pin_local_frame_trajectory(
    trajectory_path: Path,
    charges: ChargeSets,
    topology: TopologyInfo,
    focus_atoms: tuple[str, ...],
    orientation_atoms: tuple[str, str, str],
) -> tuple[float, float]:
    try:
        from scipy.io import netcdf_file
    except ImportError as exc:
        raise RuntimeError("cphmd-view requires scipy to pin the local site frame") from exc

    def atom_index(name: str) -> int:
        return topology.first_atom - 1 + charges.atom_names.index(name)

    focus_indices = tuple(atom_index(name) for name in focus_atoms)
    orientation_indices = tuple(atom_index(name) for name in orientation_atoms)
    with netcdf_file(trajectory_path, "a", mmap=False) as dataset:
        variable = dataset.variables.get("coordinates")
        if variable is None:
            raise ValueError(f"No coordinates variable in {trajectory_path}")
        coordinates = np.asarray(variable.data, dtype=np.float64)
        pinned = pin_local_frame_coordinates(
            coordinates,
            focus_indices,
            (
                orientation_indices[0],
                orientation_indices[1],
                orientation_indices[2],
            ),
        )
        variable[:] = pinned.astype(variable.data.dtype, copy=False)

    pinned_site = pinned[:, focus_indices, :].mean(axis=1)
    site_displacement = float(
        np.max(np.linalg.norm(pinned_site - pinned_site[0], axis=1))
    )
    bond = pinned[:, orientation_indices[1], :] - pinned[:, orientation_indices[0], :]
    bond /= np.linalg.norm(bond, axis=1)[:, None]
    angular_drift = np.degrees(
        np.arccos(np.clip(np.einsum("fi,i->f", bond, bond[0]), -1.0, 1.0))
    )
    return site_displacement, float(np.max(angular_drift))


def _tcl_word(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return "{" + escaped + "}"


def _tcl_list(values: list[object] | tuple[object, ...]) -> str:
    return "{" + " ".join(str(value) for value in values) + "}"


def render_autoimage_cpptraj_input(
    topology_path: Path,
    trajectory_path: Path,
    output_name: str,
    residue_id: int,
    focus_atoms: tuple[str, ...],
    orientation_atoms: tuple[str, str, str],
) -> str:
    focus_names = ",".join(focus_atoms)
    orientation_names = ",".join(orientation_atoms)
    return "\n".join(
        [
            f"parm {topology_path}",
            f"trajin {trajectory_path}",
            f"autoimage anchor :{residue_id}",
            f"rms first :{residue_id}@{orientation_names} mass",
            f"center :{residue_id}@{focus_names} mass",
            "box nobox",
            f"trajout {output_name} netcdf",
            "run",
            "quit",
            "",
        ]
    )


def render_vmd_script(
    *,
    topology_path: str,
    trajectory_path: str,
    ph: float,
    residue_name: str,
    residue_id: int,
    topology: TopologyInfo,
    charges: ChargeSets,
    dummy_atoms: tuple[DummyAtom, ...],
    frames: list[ViewFrame],
    focus_atoms: tuple[str, ...],
    orientation_atoms: tuple[str, str, str],
    zoom: float,
    water_sphere_scale: float,
    water_sphere_resolution: int,
    hbond_donor_atom: str,
    hbond_hydrogen_atoms: tuple[str, ...],
    hbond_water_oxygen_selection: str,
    hbond_distance_cutoff: float,
    hbond_angle_cutoff: float,
    hbond_line_radius: float,
    hbond_dash_count: int,
    hbond_color_rgb: tuple[float, float, float],
) -> str:
    ligand_indices = tuple(
        range(topology.first_atom - 1, topology.first_atom - 1 + len(topology.atom_names))
    )
    dummy_indices = tuple(atom.vmd_index for atom in dummy_atoms)
    if focus_atoms:
        unknown_focus = sorted(set(focus_atoms) - set(charges.atom_names))
        if unknown_focus:
            raise ValueError(
                "view.focus_atoms are absent from the titratable residue: "
                + ", ".join(unknown_focus)
            )
        focus_indices = tuple(
            topology.first_atom - 1 + charges.atom_names.index(name)
            for name in focus_atoms
        )
    else:
        focus_indices = tuple(
            topology.first_atom - 1 + local_index
            for local_index, name in enumerate(charges.atom_names)
            if not name.upper().startswith("H")
        )
    if not focus_indices:
        raise ValueError("No atoms are available for the VMD focus selection")
    unknown_orientation = sorted(set(orientation_atoms) - set(charges.atom_names))
    if unknown_orientation:
        raise ValueError(
            "view.orientation_atoms are absent from the titratable residue: "
            + ", ".join(unknown_orientation)
        )
    orientation_indices = tuple(
        topology.first_atom - 1 + charges.atom_names.index(name)
        for name in orientation_atoms
    )
    hbond_names = (hbond_donor_atom, *hbond_hydrogen_atoms)
    unknown_hbond = sorted(set(hbond_names) - set(charges.atom_names))
    if unknown_hbond:
        raise ValueError(
            "view.hydrogen_bonds atoms are absent from the titratable residue: "
            + ", ".join(unknown_hbond)
        )
    hbond_donor_index = topology.first_atom - 1 + charges.atom_names.index(
        hbond_donor_atom
    )
    hbond_hydrogen_indices = tuple(
        topology.first_atom - 1 + charges.atom_names.index(name)
        for name in hbond_hydrogen_atoms
    )
    hidden_by_state = {
        state: tuple(atom.vmd_index for atom in dummy_atoms if state in atom.hidden_states)
        for state in (0, 1)
    }
    labels = {
        0: "protonated",
        1: "deprotonated",
    }
    array_lines: list[str] = []
    for state in (0, 1):
        array_lines.extend(
            [
                f"set ::cphmd_hidden({state}) {_tcl_list(hidden_by_state[state])}",
                f"set ::cphmd_label({state}) {_tcl_word(labels[state])}",
            ]
        )

    return f"""# Generated by qmmd cphmd-view. VMD atom indices are zero-based.
# The coordinates are the fixed-pH pH {_format_ph(ph)} ensemble produced by cphmd-titr-post.

set script_dir [file dirname [file normalize [info script]]]
set topology [file normalize [file join $script_dir {_tcl_word(topology_path)}]]
set trajectory [file normalize [file join $script_dir {_tcl_word(trajectory_path)}]]

if {{![file exists $topology]}} {{ error "Missing topology: $topology" }}
if {{![file exists $trajectory]}} {{ error "Missing trajectory: $trajectory" }}

set ::cphmd_ph {_format_ph(ph)}
set ::cphmd_residue_name {_tcl_word(residue_name)}
set ::cphmd_residue_id {residue_id}
set ::cphmd_ligand_indices {_tcl_list(ligand_indices)}
set ::cphmd_dummy_indices {_tcl_list(dummy_indices)}
set ::cphmd_focus_indices {_tcl_list(focus_indices)}
set ::cphmd_orientation_indices {_tcl_list(orientation_indices)}
set ::cphmd_hbond_donor_index {hbond_donor_index}
set ::cphmd_hbond_hydrogen_indices {_tcl_list(hbond_hydrogen_indices)}
set ::cphmd_hbond_water_selection {_tcl_word(hbond_water_oxygen_selection)}
set ::cphmd_hbond_distance_cutoff {hbond_distance_cutoff:.6f}
set ::cphmd_hbond_angle_cutoff {hbond_angle_cutoff:.6f}
set ::cphmd_hbond_line_radius {hbond_line_radius:.6f}
set ::cphmd_hbond_dash_count {hbond_dash_count}
set ::cphmd_states {_tcl_list(tuple(frame.state for frame in frames))}
set ::cphmd_times_ps {_tcl_list(tuple(f'{frame.time_ps:.6f}' for frame in frames))}
set ::cphmd_proton_counts {_tcl_list(tuple(frame.proton_count for frame in frames))}
{os.linesep.join(array_lines)}

proc cphmd_index_selection {{indices}} {{
    if {{[llength $indices] == 0}} {{ return "none" }}
    return "index [join $indices {{ }}]"
}}

mol new $topology type parm7 waitfor all
set ::cphmd_molid [molinfo top]
mol addfile $trajectory type netcdf waitfor all molid $::cphmd_molid

set frame_count [molinfo $::cphmd_molid get numframes]
if {{$frame_count != [llength $::cphmd_states]}} {{
    error "Trajectory has $frame_count frames but the aligned CpH table has [llength $::cphmd_states]"
}}

while {{[molinfo $::cphmd_molid get numreps] > 0}} {{ mol delrep 0 $::cphmd_molid }}

color change rgb 16 0.000000 0.000000 0.000000
color change rgb 12 {hbond_color_rgb[0]:.6f} {hbond_color_rgb[1]:.6f} {hbond_color_rgb[2]:.6f}

# Protein context, if present.
mol representation NewCartoon 0.300000 10.000000 4.100000 0
mol color Structure
mol selection "protein"
mol material Opaque
mol addrep $::cphmd_molid

# Non-water, non-protein context such as cofactors and ions.
mol representation Licorice 0.150000 12.000000 12.000000
mol color Name
mol selection "not water and not protein and not ([cphmd_index_selection $::cphmd_ligand_indices])"
mol material Opaque
mol addrep $::cphmd_molid

# Smooth transparent solvent spheres; hydrogen bonds are a separate overlay.
mol representation VDW {water_sphere_scale:.6f} {water_sphere_resolution}
mol color Name
mol selection "water"
mol material Transparent
mol addrep $::cphmd_molid

# Element-colored bonds for the titratable residue. DynamicBonds prevents ghost
# bonds when a zero-charge dummy proton is hidden; color carries no state meaning.
set ::cphmd_ligand_rep [molinfo $::cphmd_molid get numreps]
mol representation DynamicBonds 1.650000 0.120000 12.000000
mol color Name
mol selection [cphmd_index_selection $::cphmd_ligand_indices]
mol material Opaque
mol addrep $::cphmd_molid

# State-dependent dummy protons are isolated so they can be colored black.
set ::cphmd_dummy_rep [molinfo $::cphmd_molid get numreps]
mol representation VDW 0.350000 24.000000
mol color ColorID 16
mol selection "none"
mol material Opaque
mol addrep $::cphmd_molid

# Atom spheres make the appearing/disappearing proton easy to see.
set ::cphmd_atom_rep [molinfo $::cphmd_molid get numreps]
mol representation VDW 0.350000 12.000000
mol color Name
mol selection [cphmd_index_selection $::cphmd_ligand_indices]
mol material Opaque
mol addrep $::cphmd_molid

# The camera center is established once on the already aligned site coordinates.
set ::cphmd_focus_selection [atomselect $::cphmd_molid \
    [cphmd_index_selection $::cphmd_focus_indices] frame 0]

# The first two atoms define the pinned bond (site -> bonded neighbor). The
# third atom defines the local molecular plane. The camera matrix is set once;
# CPPTRAJ has already put every frame into this fixed solute reference frame.
set ::cphmd_orientation_selections {{}}
foreach atom_index $::cphmd_orientation_indices {{
    lappend ::cphmd_orientation_selections \
        [atomselect $::cphmd_molid "index $atom_index" frame 0]
}}
set ::cphmd_hbond_donor_selection \
    [atomselect $::cphmd_molid "index $::cphmd_hbond_donor_index" frame 0]
array set ::cphmd_hydrogen_selection {{}}
foreach atom_index $::cphmd_hbond_hydrogen_indices {{
    set ::cphmd_hydrogen_selection($atom_index) \
        [atomselect $::cphmd_molid "index $atom_index" frame 0]
}}
set ::cphmd_water_oxygen_selection \
    [atomselect $::cphmd_molid $::cphmd_hbond_water_selection frame 0]
set ::cphmd_hbond_graphics {{}}

proc cphmd_set_side_view {{frame}} {{
    set positions {{}}
    foreach selection $::cphmd_orientation_selections {{
        $selection frame $frame
        lappend positions [lindex [$selection get {{x y z}}] 0]
    }}
    set site [lindex $positions 0]
    set bonded [lindex $positions 1]
    set plane_atom [lindex $positions 2]
    set bond_axis [vecnorm [vecsub $bonded $site]]
    set plane_vector [vecsub $plane_atom $site]
    set normal_axis [vecnorm [veccross $bond_axis $plane_vector]]
    set side_axis [vecnorm [veccross $normal_axis $bond_axis]]
    set rotation_matrix [list \
        [concat $bond_axis 0.0] \
        [concat $side_axis 0.0] \
        [concat $normal_axis 0.0] \
        {{0.0 0.0 0.0 1.0}}]
    molinfo $::cphmd_molid set rotate_matrix $rotation_matrix
}}

proc cphmd_clear_hbonds {{}} {{
    foreach graphic_id $::cphmd_hbond_graphics {{
        graphics $::cphmd_molid delete $graphic_id
    }}
    set ::cphmd_hbond_graphics {{}}
}}

proc cphmd_draw_dashed_hbond {{start end}} {{
    set delta [vecsub $end $start]
    set denominator [expr {{2.0 * $::cphmd_hbond_dash_count}}]
    graphics $::cphmd_molid materials on
    graphics $::cphmd_molid material Opaque
    graphics $::cphmd_molid color 12
    for {{set dash 0}} {{$dash < $::cphmd_hbond_dash_count}} {{incr dash}} {{
        set begin_fraction [expr {{(2.0 * $dash) / $denominator}}]
        set end_fraction [expr {{(2.0 * $dash + 1.0) / $denominator}}]
        set dash_start [vecadd $start [vecscale $begin_fraction $delta]]
        set dash_end [vecadd $start [vecscale $end_fraction $delta]]
        set graphic_id [graphics $::cphmd_molid cylinder $dash_start $dash_end \
            radius $::cphmd_hbond_line_radius resolution 16 filled yes]
        lappend ::cphmd_hbond_graphics $graphic_id
    }}
}}

proc cphmd_update_hbonds {{frame hidden}} {{
    cphmd_clear_hbonds
    $::cphmd_hbond_donor_selection frame $frame
    $::cphmd_water_oxygen_selection frame $frame
    set donor_position [lindex [$::cphmd_hbond_donor_selection get {{x y z}}] 0]
    set oxygen_positions [$::cphmd_water_oxygen_selection get {{x y z}}]
    set oxygen_indices [$::cphmd_water_oxygen_selection get index]
    set hbond_count 0
    foreach hydrogen_index $::cphmd_hbond_hydrogen_indices {{
        if {{[lsearch -exact $hidden $hydrogen_index] >= 0}} {{ continue }}
        set hydrogen_selection $::cphmd_hydrogen_selection($hydrogen_index)
        $hydrogen_selection frame $frame
        set hydrogen_position [lindex [$hydrogen_selection get {{x y z}}] 0]
        set donor_vector [vecsub $donor_position $hydrogen_position]
        set donor_length [veclength $donor_vector]
        foreach oxygen_index $oxygen_indices oxygen_position $oxygen_positions {{
            set acceptor_vector [vecsub $oxygen_position $hydrogen_position]
            set distance [veclength $acceptor_vector]
            if {{$distance > $::cphmd_hbond_distance_cutoff || $distance < 1.0e-8}} {{
                continue
            }}
            set cosine [expr {{[vecdot $donor_vector $acceptor_vector] / \
                ($donor_length * $distance)}}]
            if {{$cosine > 1.0}} {{ set cosine 1.0 }}
            if {{$cosine < -1.0}} {{ set cosine -1.0 }}
            set angle [expr {{acos($cosine) * 180.0 / acos(-1.0)}}]
            if {{$angle >= $::cphmd_hbond_angle_cutoff}} {{
                cphmd_draw_dashed_hbond $hydrogen_position $oxygen_position
                incr hbond_count
            }}
        }}
    }}
    return $hbond_count
}}

proc cphmd_update {{args}} {{
    set molid $::cphmd_molid
    set frame [molinfo $molid get frame]
    if {{$frame < 0 || $frame >= [llength $::cphmd_states]}} {{ return }}
    set state [lindex $::cphmd_states $frame]
    set hidden $::cphmd_hidden($state)
    set visible {{}}
    set visible_regular {{}}
    set visible_dummy {{}}
    foreach atom_index $::cphmd_ligand_indices {{
        if {{[lsearch -exact $hidden $atom_index] < 0}} {{
            lappend visible $atom_index
            if {{[lsearch -exact $::cphmd_dummy_indices $atom_index] >= 0}} {{
                lappend visible_dummy $atom_index
            }} else {{
                lappend visible_regular $atom_index
            }}
        }}
    }}
    set selection [cphmd_index_selection $visible]
    mol modselect $::cphmd_ligand_rep $molid $selection
    mol modselect $::cphmd_atom_rep $molid [cphmd_index_selection $visible_regular]
    mol modselect $::cphmd_dummy_rep $molid [cphmd_index_selection $visible_dummy]
    mol selupdate $::cphmd_ligand_rep $molid 1
    mol selupdate $::cphmd_atom_rep $molid 1
    mol selupdate $::cphmd_dummy_rep $molid 1
    set hbond_count [cphmd_update_hbonds $frame $hidden]

    set time_ps [lindex $::cphmd_times_ps $frame]
    set protons [lindex $::cphmd_proton_counts $frame]
    set label $::cphmd_label($state)
    set shown_frame [expr {{$frame + 1}}]
    set ::cphmd_status [format "pH %.1f | frame %d/%d | %.3f ps | state %d: %s | %d protons | %d H-bonds" \
        $::cphmd_ph $shown_frame [llength $::cphmd_states] $time_ps $state $label $protons $hbond_count]
    mol rename $molid "CpHMD pH $::cphmd_ph - $label - frame $shown_frame"
}}

# A small live dashboard follows the animation without changing coordinates.
if {{[llength [info commands toplevel]] > 0}} {{
    catch {{destroy .cphmd_view}}
    toplevel .cphmd_view
    wm title .cphmd_view "qmmd CpHMD state"
    label .cphmd_view.status -textvariable ::cphmd_status -font {{Helvetica 12 bold}} -padx 12 -pady 10
    label .cphmd_view.legend -text "black HN1 = appearing proton; light-green dashes = N-H...O(water)" -padx 12 -pady 4
    pack .cphmd_view.status .cphmd_view.legend -side top -fill x
}}

color Display Background white
display projection Orthographic
axes location Off
display resetview
animate goto 0
$::cphmd_focus_selection frame 0
set initial_focus_center [measure center $::cphmd_focus_selection weight mass]
molinfo $::cphmd_molid set center $initial_focus_center
cphmd_set_side_view 0
cphmd_update
scale by {zoom:.6f}
trace variable ::vmd_frame($::cphmd_molid) w cphmd_update

puts "qmmd CpHMD viewer loaded: {_format_ph(ph)} pH, {len(frames)} aligned frames"
puts "Titratable residue: {residue_name} {residue_id}; dummy proton(s): {', '.join(atom.name for atom in dummy_atoms)}"
puts "Use VMD's animation controls to watch protonation changes."
"""


def _write_frame_table(
    path: Path,
    frames: list[ViewFrame],
    dummy_atoms: tuple[DummyAtom, ...],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["trajectory_frame", "time_ps", "state", "state_label", "proton_count", "hidden_atoms"]
        )
        for frame in frames:
            hidden = [atom.name for atom in dummy_atoms if frame.state in atom.hidden_states]
            writer.writerow(
                [
                    frame.frame,
                    f"{frame.time_ps:.6f}",
                    frame.state,
                    "protonated" if frame.state == 0 else "deprotonated",
                    frame.proton_count,
                    ";".join(hidden),
                ]
            )


def prepare_cphmd_view(
    cfg: CphmdViewConfig,
    repo_root: Path,
    requested_ph: float,
) -> Path:
    ph = _resolve_ph(float(requested_ph), cfg.post.titr.ph_values)
    source = titr_dir(cfg.post.titr, repo_root)
    post = post_dir(cfg.post, repo_root)
    topology_path = source / cfg.post.titr.input_parm7
    if not topology_path.is_file() or topology_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty titration topology: {topology_path}")

    topology = read_topology_info(topology_path, cfg.residue_name)
    residue_id = topology.residue_number if cfg.residue_id is None else cfg.residue_id
    if residue_id != topology.residue_number:
        raise ValueError(
            f"view.residue_id={residue_id} but {cfg.residue_name} is residue "
            f"{topology.residue_number} in {topology_path.name}"
        )
    charges = read_charge_sets(cfg.post.titr.charge_sets)
    dummy_atoms = identify_dummy_atoms(charges, topology, cfg.zero_charge_tolerance)

    total_steps = int(cfg.post.titr.cntrl["nstlim"]) * int(
        cfg.post.titr.cntrl["numexchg"]
    )
    expected_frames = total_steps // int(cfg.post.titr.cntrl["ntwx"])
    frames, trajectory_name = read_view_frames(
        post / "coordinate-protonation.csv",
        ph,
        cfg.residue_name,
        residue_id,
        expected_frames,
    )
    validate_frame_states(frames, charges)
    trajectory_path = post / trajectory_name
    if not trajectory_path.is_file() or trajectory_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty fixed-pH trajectory: {trajectory_path}")

    destination = post / cfg.output_dir / f"ph-{_format_ph(ph)}"
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_reference = trajectory_path.name
    autoimage_input = render_autoimage_cpptraj_input(
        topology_path.resolve(),
        trajectory_path.resolve(),
        trajectory_reference,
        residue_id,
        cfg.focus_atoms,
        cfg.orientation_atoms,
    )
    with tempfile.TemporaryDirectory(prefix="qmmd-cphmd-view-", dir="/tmp") as tmp:
        work = Path(tmp)
        cpptraj_input = work / "autoimage.cpptraj.in"
        cpptraj_log = work / "autoimage.cpptraj.log"
        cpptraj_input.write_text(autoimage_input)
        try:
            run_cpptraj(
                cpptraj_input,
                cpptraj_log,
                cfg.post.titr.runtime.module,
                cfg.post.cpptraj_executable,
            )
        except Exception:
            if cpptraj_log.is_file():
                shutil.copy2(cpptraj_log, destination / "autoimage.cpptraj.log")
            raise
        imaged_trajectory = work / trajectory_reference
        if not imaged_trajectory.is_file() or imaged_trajectory.stat().st_size == 0:
            raise FileNotFoundError(
                f"CPPTRAJ did not create the autoimaged trajectory: {imaged_trajectory}"
            )
        site_displacement, bond_angular_drift = pin_local_frame_trajectory(
            imaged_trajectory,
            charges,
            topology,
            cfg.focus_atoms,
            cfg.orientation_atoms,
        )
        if site_displacement > 1.0e-5 or bond_angular_drift > 1.0e-3:
            raise ValueError(
                "Failed to pin the local titration frame: "
                f"site displacement={site_displacement:.6g} A, "
                f"bond angular drift={bond_angular_drift:.6g} degrees"
            )
        log_text = cpptraj_log.read_text(errors="replace")
        (destination / "autoimage.cpptraj.log").write_text(log_text)
        expected_summary = (
            f"Read {expected_frames} frames and processed {expected_frames} frames."
        )
        if expected_summary not in log_text:
            raise ValueError(
                "CPPTRAJ did not confirm the expected autoimaged frame count; "
                f"see {destination / 'autoimage.cpptraj.log'}"
            )
        shutil.copy2(imaged_trajectory, destination / trajectory_reference)
    (destination / "autoimage.cpptraj.in").write_text(autoimage_input)

    if cfg.copy_inputs:
        topology_reference = topology_path.name
        shutil.copy2(topology_path, destination / topology_reference)
    else:
        topology_reference = os.path.relpath(topology_path, destination)

    script = render_vmd_script(
        topology_path=topology_reference,
        trajectory_path=trajectory_reference,
        ph=ph,
        residue_name=cfg.residue_name,
        residue_id=residue_id,
        topology=topology,
        charges=charges,
        dummy_atoms=dummy_atoms,
        frames=frames,
        focus_atoms=cfg.focus_atoms,
        orientation_atoms=cfg.orientation_atoms,
        zoom=cfg.zoom,
        water_sphere_scale=cfg.water_sphere_scale,
        water_sphere_resolution=cfg.water_sphere_resolution,
        hbond_donor_atom=cfg.hbond_donor_atom,
        hbond_hydrogen_atoms=cfg.hbond_hydrogen_atoms,
        hbond_water_oxygen_selection=cfg.hbond_water_oxygen_selection,
        hbond_distance_cutoff=cfg.hbond_distance_cutoff,
        hbond_angle_cutoff=cfg.hbond_angle_cutoff,
        hbond_line_radius=cfg.hbond_line_radius,
        hbond_dash_count=cfg.hbond_dash_count,
        hbond_color_rgb=cfg.hbond_color_rgb,
    )
    (destination / "view.vmd").write_text(script)
    _write_frame_table(destination / "frame-states.csv", frames, dummy_atoms)
    (destination / "view-spec.yaml").write_text(
        yaml.safe_dump(
            {
                "source_config": str(cfg.post.titr.yaml_path.relative_to(repo_root)),
                "pH": ph,
                "residue": cfg.residue_name,
                "residue_id": residue_id,
                "frames": len(frames),
                "topology": topology_reference,
                "trajectory": trajectory_reference,
                "source_trajectory": str(trajectory_path.relative_to(repo_root)),
                "copy_inputs": cfg.copy_inputs,
                "periodic_imaging": f"autoimage anchor :{residue_id}",
                "solute_alignment": (
                    f"rms first :{residue_id}@{','.join(cfg.orientation_atoms)} mass"
                ),
                "site_centering": (
                    f"center :{residue_id}@{','.join(cfg.focus_atoms)} mass"
                ),
                "local_frame_validation": {
                    "maximum_site_displacement_angstrom": site_displacement,
                    "maximum_bond_angular_drift_degrees": bond_angular_drift,
                },
                "focus_atoms": list(cfg.focus_atoms),
                "orientation_atoms": list(cfg.orientation_atoms),
                "zoom": cfg.zoom,
                "water_representation": {
                    "style": "VDW",
                    "sphere_scale": cfg.water_sphere_scale,
                    "sphere_resolution": cfg.water_sphere_resolution,
                    "material": "Transparent",
                },
                "hydrogen_bonds": {
                    "donor_atom": cfg.hbond_donor_atom,
                    "hydrogen_atoms": list(cfg.hbond_hydrogen_atoms),
                    "water_oxygen_selection": cfg.hbond_water_oxygen_selection,
                    "distance_cutoff_angstrom": cfg.hbond_distance_cutoff,
                    "angle_cutoff_degrees": cfg.hbond_angle_cutoff,
                    "line_radius_angstrom": cfg.hbond_line_radius,
                    "dash_count": cfg.hbond_dash_count,
                    "color_rgb": list(cfg.hbond_color_rgb),
                },
                "dummy_atoms": [
                    {
                        "name": atom.name,
                        "amber_atom_id": atom.vmd_index + 1,
                        "vmd_atom_index": atom.vmd_index,
                        "hidden_states": list(atom.hidden_states),
                    }
                    for atom in dummy_atoms
                ],
                "state_definition": {
                    0: {
                        "label": "protonated",
                        "proton_count": charges.proton_count_prot,
                    },
                    1: {
                        "label": "deprotonated",
                        "proton_count": charges.proton_count_deprot,
                    },
                },
                "state_alignment": "coordinate-protonation.csv, matched by pH/residue/frame",
            },
            sort_keys=False,
        )
    )
    (destination / "README.txt").write_text(
        "Portable qmmd CpHMD VMD bundle\n\n"
        "From this directory run:\n\n"
        "    vmd -e view.vmd\n\n"
        "Use VMD's animation controls to play the fixed-pH trajectory. The black\n"
        "dummy proton is shown or hidden according to the\n"
        "frame-aligned CpHMD state in frame-states.csv. CPPTRAJ autoimage places\n"
        "the ligand at the periodic box center and reimages the water. CPPTRAJ\n"
        "then aligns every frame to the first-frame local site atoms and centers\n"
        "the configured site atom. The VMD camera is static in the resulting local\n"
        "side view. Water uses smooth transparent VDW spheres. Thick dashed\n"
        "light-green cylinders mark geometric N-H...O(water) hydrogen bonds.\n"
    )
    return destination


def run_cphmd_view(yaml_path: Path, ph: float) -> None:
    resolved = yaml_path.resolve()
    cfg = load_cphmd_view_config(resolved)
    destination = prepare_cphmd_view(cfg, find_repo_root(resolved), ph)
    print(f"OK: wrote CpHMD VMD viewer to {destination}")
    print(f"Run: cd {destination} && vmd -e view.vmd")
