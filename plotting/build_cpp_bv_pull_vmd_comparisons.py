"""Build paired CPP/BV pull-window structures for direct VMD comparison.

For each requested window, the final available pull frame is autoimaged with
cpptraj, reduced to the solute, and BV is least-squares aligned to CPP using
all common heavy atoms.  Each output directory intentionally contains only a
two-frame XYZ file and a self-locating VMD script.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_CPPTRAJ = Path("/nas/sycamore/apps/amber/26/ambertools26/bin/cpptraj")


def read_amber_flag(path: Path, flag: str) -> list[str]:
    """Read fixed-width values from an Amber topology flag."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    marker = f"%FLAG {flag}"
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == marker)
    except StopIteration as exc:
        raise ValueError(f"Missing {marker} in {path}") from exc
    match = re.search(r"\((\d+)([A-Za-z])(\d+)", lines[start + 1])
    if not match:
        raise ValueError(f"Cannot parse format for {marker} in {path}")
    width = int(match.group(3))
    values: list[str] = []
    for line in lines[start + 2 :]:
        if line.startswith("%FLAG "):
            break
        values.extend(
            line[index : index + width].strip()
            for index in range(0, len(line), width)
            if line[index : index + width].strip()
        )
    return values


def first_residue_atom_names(topology: Path) -> list[str]:
    pointers = [int(value) for value in read_amber_flag(topology, "RESIDUE_POINTER")]
    if len(pointers) < 2 or pointers[0] != 1:
        raise ValueError(f"Cannot determine first residue size from {topology}")
    count = pointers[1] - 1
    names = read_amber_flag(topology, "ATOM_NAME")[:count]
    if len(names) != count:
        raise ValueError(f"Incomplete atom names in {topology}")
    return names


def map_common_atoms(reference: list[str], mobile: list[str]) -> tuple[np.ndarray, list[int]]:
    """Map ordered reference atoms to mobile atoms, allowing mobile insertions."""
    mapped: list[int] = []
    skipped: list[int] = []
    ref_index = mobile_index = 0
    while ref_index < len(reference) and mobile_index < len(mobile):
        if reference[ref_index] == mobile[mobile_index]:
            mapped.append(mobile_index)
            ref_index += 1
            mobile_index += 1
        else:
            skipped.append(mobile_index)
            mobile_index += 1
    skipped.extend(range(mobile_index, len(mobile)))
    if ref_index != len(reference):
        raise ValueError("CPP atom-name sequence is not an ordered subset of BV")
    return np.asarray(mapped, dtype=int), skipped


def read_xyz_frames(path: Path) -> list[tuple[list[str], np.ndarray, str]]:
    lines = path.read_text().splitlines()
    frames: list[tuple[list[str], np.ndarray, str]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        atom_count = int(lines[cursor].strip())
        if cursor + atom_count + 2 > len(lines):
            raise ValueError(f"Truncated XYZ frame in {path}")
        comment = lines[cursor + 1]
        fields = [line.split() for line in lines[cursor + 2 : cursor + 2 + atom_count]]
        if any(len(row) < 4 for row in fields):
            raise ValueError(f"Malformed XYZ coordinate in {path}")
        elements = [row[0] for row in fields]
        coordinates = np.asarray([[float(value) for value in row[1:4]] for row in fields])
        frames.append((elements, coordinates, comment))
        cursor += atom_count + 2
    if not frames:
        raise ValueError(f"No XYZ frames in {path}")
    return frames


def extract_solute_frames(
    cpptraj: Path, topology: Path, trajectory: Path, xyz_path: Path
) -> list[tuple[list[str], np.ndarray, str]]:
    input_path = xyz_path.with_suffix(".in")
    input_path.write_text(
        "\n".join(
            [
                f'parm "{topology.resolve()}"',
                f'trajin "{trajectory.resolve()}"',
                "autoimage anchor :1",
                "strip !:1",
                f'trajout "{xyz_path.resolve()}" xyz',
                "run",
                "quit",
                "",
            ]
        )
    )
    result = subprocess.run(
        [str(cpptraj), "-i", str(input_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not xyz_path.is_file():
        raise RuntimeError(
            f"cpptraj failed for {trajectory}\n{result.stdout}\n{result.stderr}"
        )
    return read_xyz_frames(xyz_path)


def align_on_heavy_atoms(
    mobile: np.ndarray, reference: np.ndarray, heavy: np.ndarray
) -> tuple[np.ndarray, float]:
    mobile_center = mobile[heavy].mean(axis=0)
    reference_center = reference[heavy].mean(axis=0)
    covariance = (mobile[heavy] - mobile_center).T @ (
        reference[heavy] - reference_center
    )
    left, _, right = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.linalg.det(left @ right)])
    rotation = left @ correction @ right
    aligned = (mobile - mobile_center) @ rotation + reference_center
    rmsd = np.sqrt(np.mean(np.sum((aligned[heavy] - reference[heavy]) ** 2, axis=1)))
    return aligned, float(rmsd)


def write_xyz_frame(handle: object, elements: list[str], xyz: np.ndarray, comment: str) -> None:
    handle.write(f"{len(elements)}\n{comment}\n")
    for element, coordinate in zip(elements, xyz, strict=True):
        handle.write(
            f"{element:<2s} {coordinate[0]: .8f} {coordinate[1]: .8f} "
            f"{coordinate[2]: .8f}\n"
        )


def vmd_script(window: int, rmsd: float, heavy_count: int) -> str:
    return f'''# CPP/BV pull comparison for window {window:03d}.
# Launch from anywhere with: vmd -e vis.vmd
set here [file dirname [file normalize [info script]]]
set coordinates [file join $here comparison.xyz]
if {{![file exists $coordinates]}} {{error "Missing coordinate file: $coordinates"}}

set cpp [mol new $coordinates type xyz first 0 last 0 step 1 waitfor all]
mol rename $cpp "CPP window {window:03d}"
mol delrep 0 $cpp
mol representation Licorice 0.13 18 18
mol color ColorID 0
mol selection all
mol material Opaque
mol addrep $cpp

set bv [mol new $coordinates type xyz first 1 last 1 step 1 waitfor all]
mol rename $bv "BV window {window:03d}"
mol delrep 0 $bv
mol representation Licorice 0.13 18 18
mol color ColorID 1
mol selection all
mol material Opaque
mol addrep $bv

color Display Background white
display projection Orthographic
display depthcue off
axes location Off
catch {{display antialias on}}
catch {{display ambientocclusion on}}
display resetview
puts "Window {window:03d}: CPP blue; BV red. Final pull frames; {heavy_count}-heavy-atom fitted RMSD = {rmsd:.3f} A."
'''


def build(args: argparse.Namespace) -> list[float]:
    cpp_names = first_residue_atom_names(args.cpp_topology)
    bv_names = first_residue_atom_names(args.bv_topology)
    bv_mapping, skipped = map_common_atoms(cpp_names, bv_names)
    skipped_names = [bv_names[index] for index in skipped]
    if len(skipped) != 1 or skipped_names != ["HC"]:
        raise ValueError(
            f"Expected BV-only HC insertion; found skipped atoms {skipped_names}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    rmsds: list[float] = []
    with tempfile.TemporaryDirectory(prefix="cpp-bv-vmd-", dir="/tmp") as temporary:
        temporary_path = Path(temporary)
        for window in range(args.first_window, args.first_window + args.window_count):
            source_label = f"window-{window:03d}"
            output_label = f"window_{window:03d}"
            cpp_trajectory = args.cpp_windows / source_label / "pull" / "pull.nc"
            bv_trajectory = args.bv_windows / source_label / "pull" / "pull.nc"
            if not cpp_trajectory.is_file() or not bv_trajectory.is_file():
                raise FileNotFoundError(
                    f"Missing pull trajectory for {source_label}: "
                    f"{cpp_trajectory} or {bv_trajectory}"
                )
            cpp_frames = extract_solute_frames(
                args.cpptraj,
                args.cpp_topology,
                cpp_trajectory,
                temporary_path / f"cpp-{window:03d}.xyz",
            )
            bv_frames = extract_solute_frames(
                args.cpptraj,
                args.bv_topology,
                bv_trajectory,
                temporary_path / f"bv-{window:03d}.xyz",
            )
            cpp_elements, cpp_xyz, _ = cpp_frames[-1]
            bv_elements, bv_xyz_full, _ = bv_frames[-1]
            if len(cpp_xyz) != len(cpp_names) or len(bv_xyz_full) != len(bv_names):
                raise ValueError(f"Solute atom count mismatch in {source_label}")
            bv_xyz = bv_xyz_full[bv_mapping]
            mapped_bv_elements = [bv_elements[index] for index in bv_mapping]
            if cpp_elements != mapped_bv_elements:
                raise ValueError(f"Mapped CPP/BV element mismatch in {source_label}")

            heavy = np.asarray(
                [element.upper() != "H" for element in cpp_elements], dtype=bool
            )
            cpp_xyz = cpp_xyz - cpp_xyz[heavy].mean(axis=0)
            bv_xyz, rmsd = align_on_heavy_atoms(bv_xyz, cpp_xyz, heavy)
            rmsds.append(rmsd)

            destination = args.output / output_label
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / "comparison.xyz").open("w") as handle:
                write_xyz_frame(
                    handle,
                    cpp_elements,
                    cpp_xyz,
                    f"CPP {source_label}; final pull frame {len(cpp_frames)}; alignment reference",
                )
                write_xyz_frame(
                    handle,
                    cpp_elements,
                    bv_xyz,
                    f"BV {source_label}; final pull frame {len(bv_frames)}; aligned on "
                    f"{int(heavy.sum())} common heavy atoms; RMSD {rmsd:.6f} A; BV-only HC omitted",
                )
            (destination / "vis.vmd").write_text(
                vmd_script(window, rmsd, int(heavy.sum()))
            )
            print(
                f"{source_label}: CPP frame {len(cpp_frames)}, BV frame {len(bv_frames)}, "
                f"heavy-atom RMSD {rmsd:.3f} A"
            )
    return rmsds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpp-windows", type=Path, default=Path("systems/CPP/solv_4.0/us-single15")
    )
    parser.add_argument(
        "--bv-windows", type=Path, default=Path("systems/BV/solv_4.0/us-single15")
    )
    parser.add_argument(
        "--cpp-topology", type=Path, default=Path("systems/CPP/solv_4.0/prep/solv.parm7")
    )
    parser.add_argument(
        "--bv-topology", type=Path, default=Path("systems/BV/solv_4.0/prep/solv.parm7")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("systems/CPP/solv_4.0/us-single15/vmd_comparison"),
    )
    parser.add_argument("--first-window", type=int, default=0)
    parser.add_argument("--window-count", type=int, default=18)
    parser.add_argument(
        "--cpptraj",
        type=Path,
        default=Path(shutil.which("cpptraj") or DEFAULT_CPPTRAJ),
    )
    args = parser.parse_args()
    if args.window_count < 1:
        parser.error("--window-count must be positive")
    if not args.cpptraj.is_file():
        parser.error(f"cpptraj executable not found: {args.cpptraj}")
    return args


def main() -> None:
    args = parse_args()
    rmsds = build(args)
    print(
        f"Wrote {len(rmsds)} overlays under {args.output}; heavy-atom RMSD range "
        f"{min(rmsds):.3f}-{max(rmsds):.3f} A"
    )


if __name__ == "__main__":
    main()
