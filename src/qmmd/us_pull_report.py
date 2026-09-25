"""Report the stitched dihedral trajectory from a sequential Amber pull."""

from __future__ import annotations

import csv
import math
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from qmmd.us_pull import (
    USPullConfig,
    find_repo_root,
    load_config,
    pull_dir,
    source_paths,
    window_centers,
)


@dataclass(frozen=True, slots=True)
class PullSample:
    time_ps: float
    window_index: int
    target_deg: float
    local_time_ps: float
    frame_in_window: int
    dihedral_raw_deg: float
    dihedral_branch_deg: float


def _cpptraj_path(path: Path) -> str:
    """Quote a path for cpptraj's input language."""
    return '"' + str(path.resolve()).replace('"', '\\"') + '"'


def render_cpptraj_input(
    topology: Path,
    trajectories: list[Path],
    atoms: tuple[int, int, int, int],
    data_path: Path,
    clock_pdb_path: Path,
    clock_residue: int,
) -> str:
    lines = [f"parm {_cpptraj_path(topology)}"]
    lines.extend(f"trajin {_cpptraj_path(trajectory)}" for trajectory in trajectories)
    # autoimage applies minimum-image reconstruction before the torsion action.
    lines.extend(
        [
            "autoimage",
            "dihedral pulled "
            + " ".join(f"@{atom}" for atom in atoms)
            + f" out {_cpptraj_path(data_path)}",
            f"strip !(:{clock_residue})",
            f"trajout {_cpptraj_path(clock_pdb_path)} pdb",
            "run",
            "quit",
            "",
        ]
    )
    return "\n".join(lines)


def amber_residue_for_atoms(
    topology: Path,
    atoms: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Return the one-based residue and first atom containing an Amber atom quartet."""
    lines = topology.read_text(errors="replace").splitlines()
    flag_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "%FLAG RESIDUE_POINTER"),
        None,
    )
    if flag_index is None:
        raise ValueError(f"Missing RESIDUE_POINTER in Amber topology: {topology}")

    pointers: list[int] = []
    for line in lines[flag_index + 2 :]:
        if line.lstrip().startswith("%FLAG"):
            break
        pointers.extend(int(value) for value in line.split())
    if not pointers:
        raise ValueError(f"Empty RESIDUE_POINTER in Amber topology: {topology}")

    residue_by_atom: dict[int, tuple[int, int]] = {}
    sorted_atoms = sorted(atoms)
    for atom in sorted_atoms:
        matches = [
            (index + 1, start)
            for index, start in enumerate(pointers)
            if start <= atom and (index + 1 == len(pointers) or atom < pointers[index + 1])
        ]
        if len(matches) != 1:
            raise ValueError(f"Could not assign Amber atom {atom} to one residue")
        residue_by_atom[atom] = matches[0]
    assignments = set(residue_by_atom.values())
    if len(assignments) != 1:
        raise ValueError("The restrained dihedral atoms must belong to one residue for the VMD clock")
    return assignments.pop()


def _vector_subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length < 1.0e-12:
        raise ValueError("Cannot orient a clock from a zero-length vector")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def read_multimodel_pdb(
    path: Path,
) -> tuple[list[str], list[str], list[list[tuple[float, float, float]]]]:
    """Read atom names, elements, and coordinates from a multi-model PDB."""
    names: list[str] = []
    elements: list[str] = []
    frames: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    current_names: list[str] = []
    current_elements: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            current = []
            current_names = []
            current_elements = []
        elif line.startswith(("ATOM  ", "HETATM")):
            current_names.append(line[12:16].strip())
            element = line[76:78].strip() or line[12:16].strip()[0]
            current_elements.append(element.capitalize())
            current.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        elif line.startswith("ENDMDL"):
            if not current:
                raise ValueError(f"Empty PDB model in {path}")
            if not frames:
                names = current_names
                elements = current_elements
            elif current_names != names or current_elements != elements:
                raise ValueError(f"PDB atom identity changed between models in {path}")
            frames.append(current)
    if not frames:
        raise ValueError(f"No PDB models found in {path}")
    return names, elements, frames


def align_clock_frame(
    coordinates: list[tuple[float, float, float]],
    atom_indices: tuple[int, int, int, int],
) -> list[tuple[float, float, float]]:
    """Rigidly place the central torsion bond on +z and the first atom toward +x."""
    first, central_a, central_b, _ = atom_indices
    origin = coordinates[central_b]
    z_axis = _normalize(_vector_subtract(coordinates[central_b], coordinates[central_a]))
    first_vector = _vector_subtract(coordinates[first], coordinates[central_a])
    projection = _dot(first_vector, z_axis)
    x_axis = _normalize(
        tuple(first_vector[index] - projection * z_axis[index] for index in range(3))  # type: ignore[arg-type]
    )
    y_axis = _cross(z_axis, x_axis)
    return [
        (
            _dot(_vector_subtract(point, origin), x_axis),
            _dot(_vector_subtract(point, origin), y_axis),
            _dot(_vector_subtract(point, origin), z_axis),
        )
        for point in coordinates
    ]


def infer_bonds(
    coordinates: list[tuple[float, float, float]], elements: list[str]
) -> list[tuple[int, int]]:
    """Infer one solute's covalent graph once; explicit MOL2 bonds prevent cross-frame bonds."""
    radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07}
    bonds: list[tuple[int, int]] = []
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            delta = _vector_subtract(coordinates[left], coordinates[right])
            distance = math.sqrt(_dot(delta, delta))
            cutoff = radii.get(elements[left], 0.77) + radii.get(elements[right], 0.77) + 0.40
            if 0.4 < distance <= cutoff:
                bonds.append((left, right))
    if not bonds:
        raise ValueError("No covalent bonds inferred for pull-clock ensemble")
    return bonds


def write_static_clock_mol2(
    pdb_path: Path,
    mol2_path: Path,
    atom_indices: tuple[int, int, int, int],
) -> int:
    """Write all aligned PDB models as bonded conformers in one static MOL2 frame."""
    names, elements, frames = read_multimodel_pdb(pdb_path)
    aligned = [align_clock_frame(frame, atom_indices) for frame in frames]
    bonds = infer_bonds(aligned[0], elements)
    atom_count = len(names)
    lines = [
        "@<TRIPOS>MOLECULE",
        "PRN_PULL_CLOCK_ENSEMBLE",
        f"{atom_count * len(aligned)} {len(bonds) * len(aligned)} {len(aligned)} 0 0",
        "SMALL",
        "NO_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for frame_index, coordinates in enumerate(aligned):
        residue = frame_index + 1
        for atom_index, (name, element, point) in enumerate(zip(names, elements, coordinates)):
            serial = frame_index * atom_count + atom_index + 1
            lines.append(
                f"{serial:7d} {name:<4s} {point[0]:12.7f} {point[1]:12.7f} "
                f"{point[2]:12.7f} {element:<3s} {residue:5d} PRN{residue:03d} 0.0"
            )
    lines.append("@<TRIPOS>BOND")
    bond_serial = 1
    for frame_index in range(len(aligned)):
        offset = frame_index * atom_count
        for left, right in bonds:
            lines.append(f"{bond_serial:7d} {offset + left + 1:7d} {offset + right + 1:7d} 1")
            bond_serial += 1
    lines.append("@<TRIPOS>SUBSTRUCTURE")
    for frame_index in range(len(aligned)):
        residue = frame_index + 1
        lines.append(
            f"{residue:7d} PRN{residue:03d} {frame_index * atom_count + 1:7d} "
            "RESIDUE 0 **** 0 ROOT"
        )
    mol2_path.write_text("\n".join(lines) + "\n")
    return len(aligned)


def render_pull_clock_vmd(cfg: USPullConfig, frame_count: int) -> str:
    """Render a VMD view of the single-frame static pull ensemble."""
    return f'''# Static PRN pull ensemble: all conformers exist in one coordinate frame.
# Launch from anywhere with: vmd -e pull-clock.vmd
set clock_dir [file dirname [file normalize [info script]]]
set clock_data [file join $clock_dir pull-clock.mol2]
if {{![file exists $clock_data]}} {{error "Missing pull clock data: $clock_data"}}
set m [mol new $clock_data type mol2 waitfor all autobonds off]
mol rename $m "{cfg.system} PULL CLOCK"
mol delrep 0 $m

# One complete reference solute, faint and thin.
mol representation Licorice 0.045 16 16
mol color ColorID 2
mol selection "index 0 to 10"
mol material Opaque
mol addrep $m

# The complete O2-CG-O1-H11 dihedral ensemble from every pull frame.
mol representation Licorice 0.025 16 16
mol color ColorID 0
mol selection "name CG O1 O2 H11"
mol material Opaque
mol addrep $m

# Emphasize the ensemble of H11 clock tips.
mol representation VDW 0.045 16
mol color ColorID 1
mol selection "name H11"
mol material Opaque
mol addrep $m

color Display Background white
display projection Orthographic
display depthcue off
axes location Off
catch {{display antialias on}}
catch {{display ambientocclusion on}}
graphics $m color gray
for {{set deg 0}} {{$deg < 360}} {{incr deg 30}} {{
    set angle [expr {{$deg*acos(-1)/180.0}}]
    graphics $m line \
        [list [expr {{1.12*cos($angle)}}] [expr {{1.12*sin($angle)}}] 0] \
        [list [expr {{1.25*cos($angle)}}] [expr {{1.25*sin($angle)}}] 0] width 2
}}
graphics $m color black
graphics $m text {{1.3 0 0}} "syn 0" size 0.7
graphics $m text {{-2.0 0 0}} "anti 180" size 0.7
display resetview
molinfo $m set rotate_matrix {{{{1 0 0 0}} {{0 1 0 0}} {{0 0 1 0}} {{0 0 0 1}}}}
molinfo $m set center_matrix {{{{1 0 0 0}} {{0 1 0 0}} {{0 0 1 0}} {{0 0 0 1}}}}
puts "{cfg.system} pull clock: {frame_count} O2-CG-O1-H11 conformers."
puts "View CG toward O1. Gray: reference PRN; blue: dihedrals; red: H11 tips."
'''


def run_cpptraj(input_path: Path, log_path: Path, amber_module: str) -> None:
    """Run cpptraj directly, or through the Amber environment module."""
    executable = shutil.which("cpptraj")
    if executable is not None:
        command = [executable, "-i", str(input_path)]
    else:
        shell_command = (
            "module purge\n"
            f"module load {shlex.quote(amber_module)}\n"
            f"exec cpptraj -i {shlex.quote(str(input_path))}"
        )
        command = ["bash", "-lc", shell_command]

    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=input_path.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def read_cpptraj_dihedrals(path: Path) -> list[float]:
    """Read the angle column from a cpptraj scalar data file."""
    angles: list[float] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed cpptraj data at {path}:{line_number}")
        angle = float(fields[1])
        if not math.isfinite(angle):
            raise ValueError(f"Non-finite dihedral at {path}:{line_number}")
        angles.append(angle)
    if not angles:
        raise ValueError(f"No dihedral samples found in {path}")
    return angles


def angle_near_target(angle_deg: float, target_deg: float) -> float:
    """Move a periodic angle onto the 360-degree branch nearest its target."""
    return target_deg + ((angle_deg - target_deg + 180.0) % 360.0) - 180.0


def stitch_samples(
    angles_deg: list[float],
    centers_deg: list[float],
    frames_per_window: int,
    sample_interval_ps: float,
) -> list[PullSample]:
    expected = len(centers_deg) * frames_per_window
    if len(angles_deg) != expected:
        raise ValueError(
            f"Expected {expected} trajectory frames "
            f"({len(centers_deg)} windows x {frames_per_window}), found {len(angles_deg)}"
        )

    samples: list[PullSample] = []
    for sample_index, angle in enumerate(angles_deg):
        window_index, frame_offset = divmod(sample_index, frames_per_window)
        target = centers_deg[window_index]
        frame_in_window = frame_offset + 1
        samples.append(
            PullSample(
                time_ps=(sample_index + 1) * sample_interval_ps,
                window_index=window_index,
                target_deg=target,
                local_time_ps=frame_in_window * sample_interval_ps,
                frame_in_window=frame_in_window,
                dihedral_raw_deg=angle,
                dihedral_branch_deg=angle_near_target(angle, target),
            )
        )
    return samples


def write_samples_csv(path: Path, samples: list[PullSample]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_ps",
                "window_index",
                "target_deg",
                "local_time_ps",
                "frame_in_window",
                "dihedral_raw_deg",
                "dihedral_branch_deg",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.time_ps:.8f}",
                    sample.window_index,
                    f"{sample.target_deg:.8f}",
                    f"{sample.local_time_ps:.8f}",
                    sample.frame_in_window,
                    f"{sample.dihedral_raw_deg:.8f}",
                    f"{sample.dihedral_branch_deg:.8f}",
                ]
            )


def plot_samples(
    path: Path,
    samples: list[PullSample],
    atoms: tuple[int, int, int, int],
) -> None:
    # Keep matplotlib optional for non-report commands and lightweight unit tests.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [sample.time_ps for sample in samples]
    measured = [sample.dihedral_branch_deg for sample in samples]
    window_targets: list[float] = []
    for sample in samples:
        if sample.window_index == len(window_targets):
            window_targets.append(sample.target_deg)
    window_duration_ps = max(sample.local_time_ps for sample in samples)
    boundaries = [index * window_duration_ps for index in range(len(window_targets) + 1)]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(times, measured, color="#1769aa", linewidth=1.35, label="Measured dihedral")
    ax.step(
        boundaries,
        [*window_targets, window_targets[-1]],
        where="post",
        color="#d1495b",
        linewidth=1.1,
        linestyle="--",
        label="Restraint target",
    )
    ax.set_xlabel("Stitched pull time (ps)")
    ax.set_ylabel("Dihedral (degrees)")
    ax.set_title("Sequential Amber pull: " + "-".join(str(atom) for atom in atoms))
    ax.set_xlim(0.0, boundaries[-1])
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def create_us_pull_report(
    cfg: USPullConfig,
    yaml_text: str,
    repo_root: Path,
) -> tuple[Path, Path, int]:
    """Validate a completed pull, calculate its torsion, and write a report."""
    output = pull_dir(cfg, repo_root)
    pull_spec = output / "pull_spec.yaml"
    if not pull_spec.is_file() or pull_spec.read_text() != yaml_text:
        raise RuntimeError(f"{pull_spec} does not exactly match the requested config")

    nstlim = int(cfg.md.cntrl["nstlim"])
    ntwx = int(cfg.md.cntrl.get("ntwx", 0))
    timestep_ps = float(cfg.md.cntrl["dt"])
    if ntwx <= 0:
        raise ValueError("md.cntrl.ntwx must be positive to report a trajectory")
    if nstlim % ntwx != 0:
        raise ValueError("md.cntrl.nstlim must be exactly divisible by ntwx")
    frames_per_window = nstlim // ntwx
    sample_interval_ps = ntwx * timestep_ps

    centers = window_centers(cfg.windows)
    trajectories: list[Path] = []
    for index in range(len(centers)):
        window = output / f"window-{index:03d}" / "pull"
        trajectory = window / "pull.nc"
        amber_output = window / "pull.out"
        if not trajectory.is_file() or trajectory.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty pull trajectory: {trajectory}")
        if not amber_output.is_file() or "Final Performance Info:" not in amber_output.read_text(
            errors="replace"
        ):
            raise RuntimeError(f"Amber pull window did not reach normal completion: {amber_output}")
        trajectories.append(trajectory)

    topology, _, _ = source_paths(cfg, repo_root)
    clock_residue, residue_first_atom = amber_residue_for_atoms(topology, cfg.restraint.atoms)
    destination = output
    cpptraj_data = destination / "pull_dihedral_cpptraj.dat"
    clock_pdb = destination / "pull-clock.pdb"
    cpptraj_input = destination / "cpptraj.in"
    cpptraj_log = destination / "cpptraj.log"
    cpptraj_input.write_text(
        render_cpptraj_input(
            topology,
            trajectories,
            cfg.restraint.atoms,
            cpptraj_data,
            clock_pdb,
            clock_residue,
        )
    )
    run_cpptraj(cpptraj_input, cpptraj_log, cfg.runtime.module)
    angles = read_cpptraj_dihedrals(cpptraj_data)
    samples = stitch_samples(angles, centers, frames_per_window, sample_interval_ps)

    csv_path = destination / "pull_dihedral.csv"
    figure_path = destination / "pull.png"
    write_samples_csv(csv_path, samples)
    plot_samples(figure_path, samples, cfg.restraint.atoms)
    clock_atom_indices = tuple(atom - residue_first_atom for atom in cfg.restraint.atoms)
    static_clock = destination / "pull-clock.mol2"
    clock_frames = write_static_clock_mol2(clock_pdb, static_clock, clock_atom_indices)
    if clock_frames != len(samples):
        raise RuntimeError(
            f"Clock has {clock_frames} conformers but dihedral data has {len(samples)} samples"
        )
    clock_pdb.unlink()
    (destination / "pull-clock.vmd").write_text(render_pull_clock_vmd(cfg, clock_frames))
    (destination / "report_spec.yaml").write_text(yaml_text)
    return figure_path, csv_path, len(samples)


def run_us_pull_report(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    yaml_text = resolved.read_text()
    cfg = load_config(resolved)
    figure, data, count = create_us_pull_report(cfg, yaml_text, find_repo_root(resolved))
    print(f"OK: wrote {count} stitched dihedral samples to {data}")
    print(f"OK: wrote US pull plot to {figure}")
