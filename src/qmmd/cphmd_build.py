from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from qmmd.cphmd_prep import find_repo_root


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Mol2Atom:
    index: int
    name: str
    coordinates: Vector3
    atom_type: str
    residue_id: int
    residue_name: str


@dataclass(frozen=True)
class Mol2Structure:
    atoms: tuple[Mol2Atom, ...]
    bonds: frozenset[frozenset[int]]


@dataclass(frozen=True)
class DummyNames:
    syn: str
    anti: str


@dataclass(frozen=True)
class CphmdBuildConfig:
    system: str
    mol2: Path
    output_pdb: Path
    carbon: str
    oxygens: tuple[str, str]
    dummy_names: tuple[DummyNames, DummyNames]
    oh_distance_angstrom: float
    dummy_pair_angle_degrees: float


@dataclass(frozen=True)
class DummyProton:
    name: str
    parent_oxygen: str
    geometry: str
    coordinates: Vector3


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return result


def load_config(yaml_path: Path) -> CphmdBuildConfig:
    resolved_yaml = yaml_path.resolve()
    data = yaml.safe_load(resolved_yaml.read_text())
    if not isinstance(data, dict):
        raise ValueError("CpHMD build configuration must be a YAML mapping")

    system = _nonempty_string(data.get("system"), "system")
    mol2_value = _nonempty_string(data.get("mol2"), "mol2")
    mol2 = Path(mol2_value)
    if not mol2.is_absolute():
        mol2 = find_repo_root(resolved_yaml) / mol2
    mol2 = mol2.resolve()
    if not mol2.is_file():
        raise FileNotFoundError(f"Missing mol2: {mol2}")

    output_value = data.get("output_pdb", f"{mol2.stem}_cphmd.pdb")
    output_name = _nonempty_string(output_value, "output_pdb")
    output_path = Path(output_name)
    if output_path.is_absolute() or output_path.parent != Path("."):
        raise ValueError("output_pdb must be a filename written beside mol2")
    if output_path.suffix.lower() != ".pdb":
        raise ValueError("output_pdb must have a .pdb suffix")
    output_pdb = (mol2.parent / output_path).resolve()

    carboxyl = data.get("carboxyl")
    if not isinstance(carboxyl, dict):
        raise ValueError("carboxyl must be a YAML mapping")
    carbon = _nonempty_string(carboxyl.get("carbon"), "carboxyl.carbon")
    oxygen_values = carboxyl.get("oxygens")
    if not isinstance(oxygen_values, list) or len(oxygen_values) != 2:
        raise ValueError("carboxyl.oxygens must contain exactly two atom names")
    oxygens = tuple(
        _nonempty_string(value, f"carboxyl.oxygens[{index}]")
        for index, value in enumerate(oxygen_values)
    )
    if len({carbon, *oxygens}) != 3:
        raise ValueError("carboxyl carbon and oxygen atom names must be distinct")

    dummy_values = carboxyl.get("dummy_protons")
    if not isinstance(dummy_values, dict):
        raise ValueError("carboxyl.dummy_protons must be a YAML mapping")
    dummy_names: list[DummyNames] = []
    for oxygen in oxygens:
        names = dummy_values.get(oxygen)
        if not isinstance(names, dict):
            raise ValueError(
                f"carboxyl.dummy_protons.{oxygen} must define syn and anti names"
            )
        dummy_names.append(
            DummyNames(
                syn=_nonempty_string(
                    names.get("syn"), f"carboxyl.dummy_protons.{oxygen}.syn"
                ),
                anti=_nonempty_string(
                    names.get("anti"), f"carboxyl.dummy_protons.{oxygen}.anti"
                ),
            )
        )
    flat_dummy_names = [
        name
        for pair in dummy_names
        for name in (pair.syn, pair.anti)
    ]
    if len(set(flat_dummy_names)) != 4:
        raise ValueError("The four dummy proton names must be unique")
    if any(len(name) > 4 for name in flat_dummy_names):
        raise ValueError("Dummy proton names must be at most four characters for PDB")

    geometry = data.get("geometry", {})
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be a YAML mapping")
    oh_distance = _positive_float(
        geometry.get("oh_distance_angstrom", 0.9725),
        "geometry.oh_distance_angstrom",
    )
    pair_angle = _positive_float(
        geometry.get("dummy_pair_angle_degrees", 134.0),
        "geometry.dummy_pair_angle_degrees",
    )
    if pair_angle >= 180.0:
        raise ValueError("geometry.dummy_pair_angle_degrees must be below 180")

    return CphmdBuildConfig(
        system=system,
        mol2=mol2,
        output_pdb=output_pdb,
        carbon=carbon,
        oxygens=(oxygens[0], oxygens[1]),
        dummy_names=(dummy_names[0], dummy_names[1]),
        oh_distance_angstrom=oh_distance,
        dummy_pair_angle_degrees=pair_angle,
    )


def read_mol2(path: Path) -> Mol2Structure:
    atoms: list[Mol2Atom] = []
    bonds: set[frozenset[int]] = set()
    section: str | None = None

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>").strip()
            continue
        if not line.strip():
            continue
        fields = line.split()
        if section == "ATOM":
            if len(fields) < 8:
                raise ValueError(f"Malformed MOL2 atom line in {path}:{line_number}")
            try:
                atoms.append(
                    Mol2Atom(
                        index=int(fields[0]),
                        name=fields[1],
                        coordinates=(
                            float(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                        ),
                        atom_type=fields[5],
                        residue_id=int(fields[6]),
                        residue_name=fields[7],
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MOL2 atom data in {path}:{line_number}"
                ) from exc
        elif section == "BOND":
            if len(fields) < 4:
                raise ValueError(f"Malformed MOL2 bond line in {path}:{line_number}")
            try:
                atom1 = int(fields[1])
                atom2 = int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MOL2 bond data in {path}:{line_number}"
                ) from exc
            bonds.add(frozenset((atom1, atom2)))

    if not atoms:
        raise ValueError(f"No @<TRIPOS>ATOM records found in {path}")
    names = [atom.name for atom in atoms]
    if len(names) != len(set(names)):
        raise ValueError(f"MOL2 atom names must be unique: {path}")
    indices = [atom.index for atom in atoms]
    if len(indices) != len(set(indices)):
        raise ValueError(f"MOL2 atom indices must be unique: {path}")
    return Mol2Structure(atoms=tuple(atoms), bonds=frozenset(bonds))


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _scale(value: float, vector: Vector3) -> Vector3:
    return tuple(value * component for component in vector)  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3, description: str) -> Vector3:
    length = _norm(vector)
    if length < 1.0e-10:
        raise ValueError(f"Cannot define {description} from coincident/collinear atoms")
    return _scale(1.0 / length, vector)


def build_dummy_protons(
    cfg: CphmdBuildConfig,
    structure: Mol2Structure,
) -> tuple[DummyProton, ...]:
    atoms_by_name = {atom.name: atom for atom in structure.atoms}
    required = (cfg.carbon, *cfg.oxygens)
    missing = [name for name in required if name not in atoms_by_name]
    if missing:
        raise ValueError("Configured carboxyl atoms absent from MOL2: " + ", ".join(missing))
    collisions = [
        name
        for pair in cfg.dummy_names
        for name in (pair.syn, pair.anti)
        if name in atoms_by_name
    ]
    if collisions:
        raise ValueError("Dummy proton names already exist in MOL2: " + ", ".join(collisions))

    carbon_atom = atoms_by_name[cfg.carbon]
    oxygen_atoms = tuple(atoms_by_name[name] for name in cfg.oxygens)
    for oxygen in oxygen_atoms:
        if frozenset((carbon_atom.index, oxygen.index)) not in structure.bonds:
            raise ValueError(f"Configured atoms {cfg.carbon} and {oxygen.name} are not bonded")

    carbon = carbon_atom.coordinates
    oxygen_coordinates = tuple(atom.coordinates for atom in oxygen_atoms)
    normal = _normalize(
        _cross(
            _subtract(oxygen_coordinates[0], carbon),
            _subtract(oxygen_coordinates[1], carbon),
        ),
        "the carboxyl plane",
    )
    half_angle = math.radians(cfg.dummy_pair_angle_degrees / 2.0)

    generated: list[DummyProton] = []
    for index, (oxygen_atom, names) in enumerate(
        zip(oxygen_atoms, cfg.dummy_names, strict=True)
    ):
        oxygen = oxygen_atom.coordinates
        other_oxygen = oxygen_coordinates[1 - index]
        outward = _normalize(_subtract(oxygen, carbon), f"{oxygen_atom.name} outward axis")
        tangent = _normalize(
            _cross(normal, outward),
            f"{oxygen_atom.name} in-plane tangent",
        )
        radial = _scale(math.cos(half_angle), outward)
        transverse = _scale(math.sin(half_angle), tangent)
        candidates = (
            _add(oxygen, _scale(cfg.oh_distance_angstrom, _add(radial, transverse))),
            _add(oxygen, _scale(cfg.oh_distance_angstrom, _subtract(radial, transverse))),
        )
        syn, anti = sorted(
            candidates,
            key=lambda coordinates: _norm(_subtract(coordinates, other_oxygen)),
        )
        generated.extend(
            (
                DummyProton(names.syn, oxygen_atom.name, "syn", syn),
                DummyProton(names.anti, oxygen_atom.name, "anti", anti),
            )
        )
    return tuple(generated)


def _element_from_mol2(atom: Mol2Atom) -> str:
    atom_type = atom.atom_type.split(".", maxsplit=1)[0]
    letters = "".join(character for character in atom_type if character.isalpha())
    if not letters:
        letters = "".join(character for character in atom.name if character.isalpha())
    if not letters:
        return "X"
    if letters.lower().startswith("cl"):
        return "Cl"
    if letters.lower().startswith("br"):
        return "Br"
    return letters[0].upper()


def _pdb_atom_line(
    serial: int,
    name: str,
    residue_name: str,
    residue_id: int,
    coordinates: Vector3,
    element: str,
) -> str:
    if serial > 99999:
        raise ValueError("PDB output supports at most 99999 atoms")
    atom_name = f"{name:>4}" if len(name) < 4 else name[:4]
    return (
        f"HETATM{serial:5d} {atom_name} {residue_name[:3]:>3} A{residue_id:4d}    "
        f"{coordinates[0]:8.3f}{coordinates[1]:8.3f}{coordinates[2]:8.3f}"
        f"  1.00  0.00          {element:>2}"
    )


def write_pdb(
    output_path: Path,
    cfg: CphmdBuildConfig,
    structure: Mol2Structure,
    dummy_protons: tuple[DummyProton, ...],
) -> None:
    first_atom = structure.atoms[0]
    lines = [
        "REMARK 950 GENERATED BY QMMD CPHMD-BUILD",
        f"REMARK 950 SOURCE MOL2 {cfg.mol2.name}",
        "REMARK 950 FOUR DUMMY PROTONS ADDED BY CARBOXYLATE GEOMETRY ONLY",
    ]
    for serial, atom in enumerate(structure.atoms, start=1):
        lines.append(
            _pdb_atom_line(
                serial,
                atom.name,
                atom.residue_name,
                atom.residue_id,
                atom.coordinates,
                _element_from_mol2(atom),
            )
        )
    start = len(structure.atoms) + 1
    for offset, proton in enumerate(dummy_protons):
        lines.append(
            _pdb_atom_line(
                start + offset,
                proton.name,
                first_atom.residue_name,
                first_atom.residue_id,
                proton.coordinates,
                "H",
            )
        )
    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n")


def run_cphmd_build(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    structure = read_mol2(cfg.mol2)
    dummy_protons = build_dummy_protons(cfg, structure)
    write_pdb(cfg.output_pdb, cfg, structure, dummy_protons)

    print("dummy  oxygen  geometry        x         y         z")
    for proton in dummy_protons:
        x, y, z = proton.coordinates
        print(
            f"{proton.name:<6} {proton.parent_oxygen:<7} {proton.geometry:<8} "
            f"{x:9.4f} {y:9.4f} {z:9.4f}"
        )
    print(f"OK: wrote geometry-only CpHMD PDB to {cfg.output_pdb}")
