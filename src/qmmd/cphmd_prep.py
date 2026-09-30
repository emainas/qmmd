from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Mol2Atom:
    index: int
    name: str
    atom_type: str
    charge: float


@dataclass(frozen=True)
class CpHMDPrepConfig:
    system: str
    prot_mol2: Path
    proton_count_prot: int
    deprot_mol2: Path
    proton_count_deprot: int
    master: Path
    output_file: Path


@dataclass(frozen=True)
class AcidPrepConfig:
    system: str
    deprotonated_mol2: Path
    dummy_geometry_pdb: Path
    protonated_mol2: Path
    carbon: str
    oxygens: tuple[str, str]
    protonated_oxygen: str
    proton_atom: str
    dummy_names: tuple[tuple[str, str], tuple[str, str]]
    master_mol2: Path
    frcmod: Path
    charge_sets: Path


@dataclass(frozen=True)
class FullMol2Atom:
    index: int
    name: str
    coordinates: tuple[float, float, float]
    atom_type: str
    residue_id: int
    residue_name: str
    charge: float


@dataclass(frozen=True)
class Mol2Bond:
    index: int
    atom1: int
    atom2: int
    bond_type: str


@dataclass(frozen=True)
class FullMol2:
    name: str
    atoms: tuple[FullMol2Atom, ...]
    bonds: tuple[Mol2Bond, ...]


@dataclass(frozen=True)
class AcidChargeStates:
    atom_names: tuple[str, ...]
    state_names: tuple[str, ...]
    charges: tuple[tuple[float, ...], ...]
    proton_counts: tuple[int, ...]


@dataclass(frozen=True)
class ChargeMapping:
    master_atom: Mol2Atom
    deprot_atom: Mol2Atom | None

    @property
    def deprot_charge(self) -> float:
        return 0.0 if self.deprot_atom is None else self.deprot_atom.charge


def find_repo_root(start: Path) -> Path:
    resolved = start.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"Could not find repository root from {start}")


def _resolve_mol2(input_dir: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (input_dir / path).resolve()


def _load_base_config(yaml_path: Path) -> CpHMDPrepConfig:
    resolved_yaml = yaml_path.resolve()
    data = yaml.safe_load(resolved_yaml.read_text())
    if not isinstance(data, dict):
        raise ValueError("CpHMD prep configuration must be a YAML mapping")

    system = data.get("system")
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system must be a nonempty string")

    input_value = data.get("input_dir")
    if not isinstance(input_value, str) or not input_value.strip():
        raise ValueError("input_dir must be a nonempty path")
    input_dir = Path(input_value)
    repo_root: Path | None = None
    if not input_dir.is_absolute():
        repo_root = find_repo_root(resolved_yaml)
        input_dir = repo_root / input_dir
    input_dir = input_dir.resolve()

    prot_mol2 = _resolve_mol2(input_dir, data.get("prot_mol2"), "prot_mol2")
    deprot_mol2 = _resolve_mol2(input_dir, data.get("deprot_mol2"), "deprot_mol2")
    master = _resolve_mol2(input_dir, data.get("master"), "master")
    output_value = data.get("output_file")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ValueError("output_file must be a nonempty path")
    output_file = Path(output_value)
    if not output_file.is_absolute():
        if repo_root is None:
            repo_root = find_repo_root(resolved_yaml)
        output_file = repo_root / output_file
    output_file = output_file.resolve()

    proton_count_prot = data.get("proton_count_prot")
    proton_count_deprot = data.get("proton_count_deprot")
    for field, value in (
        ("proton_count_prot", proton_count_prot),
        ("proton_count_deprot", proton_count_deprot),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")

    for field, path in (
        ("prot_mol2", prot_mol2),
        ("deprot_mol2", deprot_mol2),
        ("master", master),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {field}: {path}")

    if master not in {prot_mol2, deprot_mol2}:
        raise ValueError("master must identify either prot_mol2 or deprot_mol2")

    return CpHMDPrepConfig(
        system=system.strip(),
        prot_mol2=prot_mol2,
        proton_count_prot=proton_count_prot,
        deprot_mol2=deprot_mol2,
        proton_count_deprot=proton_count_deprot,
        master=master,
        output_file=output_file,
    )


def _repo_path(repo_root: Path, value: object, field: str) -> Path:
    text = value if isinstance(value, str) else ""
    if not text.strip():
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_acid_config(yaml_path: Path, data: dict[object, object]) -> AcidPrepConfig:
    resolved = yaml_path.resolve()
    repo_root = find_repo_root(resolved)
    system = data.get("system")
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system must be a nonempty string")

    inputs = data.get("input")
    if not isinstance(inputs, dict):
        raise ValueError("input must be a YAML mapping")
    deprotonated_mol2 = _repo_path(
        repo_root, inputs.get("deprotonated_mol2"), "input.deprotonated_mol2"
    )
    dummy_geometry_pdb = _repo_path(
        repo_root, inputs.get("dummy_geometry_pdb"), "input.dummy_geometry_pdb"
    )
    protonated_mol2 = _repo_path(
        repo_root, inputs.get("protonated_mol2"), "input.protonated_mol2"
    )
    for field, path in (
        ("input.deprotonated_mol2", deprotonated_mol2),
        ("input.dummy_geometry_pdb", dummy_geometry_pdb),
        ("input.protonated_mol2", protonated_mol2),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {field}: {path}")

    carboxyl = data.get("carboxyl")
    if not isinstance(carboxyl, dict):
        raise ValueError("carboxyl must be a YAML mapping")
    carbon = carboxyl.get("carbon")
    if not isinstance(carbon, str) or not carbon.strip():
        raise ValueError("carboxyl.carbon must be a nonempty atom name")
    oxygen_values = carboxyl.get("oxygens")
    if not isinstance(oxygen_values, list) or len(oxygen_values) != 2:
        raise ValueError("carboxyl.oxygens must contain exactly two atom names")
    if not all(isinstance(value, str) and value.strip() for value in oxygen_values):
        raise ValueError("carboxyl.oxygens entries must be nonempty atom names")
    oxygens = (oxygen_values[0].strip(), oxygen_values[1].strip())
    protonated_oxygen = carboxyl.get("protonated_oxygen")
    if protonated_oxygen not in oxygens:
        raise ValueError("carboxyl.protonated_oxygen must be one of carboxyl.oxygens")
    proton_atom = carboxyl.get("proton_atom")
    if not isinstance(proton_atom, str) or not proton_atom.strip():
        raise ValueError("carboxyl.proton_atom must be a nonempty atom name")

    dummy_data = carboxyl.get("dummy_protons")
    if not isinstance(dummy_data, dict):
        raise ValueError("carboxyl.dummy_protons must be a YAML mapping")
    dummy_names: list[tuple[str, str]] = []
    for oxygen in oxygens:
        values = dummy_data.get(oxygen)
        if not isinstance(values, dict):
            raise ValueError(f"carboxyl.dummy_protons.{oxygen} must be a mapping")
        names: list[str] = []
        for geometry in ("syn", "anti"):
            value = values.get(geometry)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"carboxyl.dummy_protons.{oxygen}.{geometry} must be an atom name"
                )
            if len(value.strip()) > 4:
                raise ValueError("Dummy atom names must be at most four characters")
            names.append(value.strip())
        dummy_names.append((names[0], names[1]))
    flattened = [name for pair in dummy_names for name in pair]
    if len(set(flattened)) != 4:
        raise ValueError("The four dummy proton names must be unique")

    outputs = data.get("output")
    if not isinstance(outputs, dict):
        raise ValueError("output must be a YAML mapping")
    master_mol2 = _repo_path(repo_root, outputs.get("master_mol2"), "output.master_mol2")
    frcmod = _repo_path(repo_root, outputs.get("frcmod"), "output.frcmod")
    charge_sets = _repo_path(
        repo_root, outputs.get("charge_sets"), "output.charge_sets"
    )

    return AcidPrepConfig(
        system=system.strip(),
        deprotonated_mol2=deprotonated_mol2,
        dummy_geometry_pdb=dummy_geometry_pdb,
        protonated_mol2=protonated_mol2,
        carbon=carbon.strip(),
        oxygens=oxygens,
        protonated_oxygen=protonated_oxygen,
        proton_atom=proton_atom.strip(),
        dummy_names=(dummy_names[0], dummy_names[1]),
        master_mol2=master_mol2,
        frcmod=frcmod,
        charge_sets=charge_sets,
    )


def load_config(yaml_path: Path) -> CpHMDPrepConfig | AcidPrepConfig:
    data = yaml.safe_load(yaml_path.resolve().read_text())
    if not isinstance(data, dict):
        raise ValueError("CpHMD prep configuration must be a YAML mapping")
    chemistry = data.get("chemistry")
    if chemistry is None:
        return _load_base_config(yaml_path)
    if chemistry != "carboxylic_acid":
        raise ValueError(f"Unsupported cphmd-prep chemistry: {chemistry}")
    return _load_acid_config(yaml_path, data)


def read_mol2_atoms(path: Path) -> list[Mol2Atom]:
    atoms: list[Mol2Atom] = []
    in_atom_section = False

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
            continue
        if in_atom_section and line.startswith("@<TRIPOS>"):
            break
        if not in_atom_section or not line.strip():
            continue

        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed MOL2 atom line in {path}:{line_number}")
        try:
            atom = Mol2Atom(
                index=int(fields[0]),
                name=fields[1],
                atom_type=fields[5],
                charge=float(fields[8]),
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid MOL2 atom data in {path}:{line_number}"
            ) from exc
        atoms.append(atom)

    if not atoms:
        raise ValueError(f"No @<TRIPOS>ATOM records found in {path}")

    names = [atom.name for atom in atoms]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate atom names in {path}: {', '.join(duplicates)}")
    return atoms


def map_charges(cfg: CpHMDPrepConfig | AcidPrepConfig) -> list[ChargeMapping]:
    if not isinstance(cfg, CpHMDPrepConfig):
        raise TypeError("map_charges is the two-state base mapping helper")
    master_atoms = read_mol2_atoms(cfg.master)
    deprot_atoms = read_mol2_atoms(cfg.deprot_mol2)
    master_by_name = {atom.name: atom for atom in master_atoms}
    deprot_by_name = {atom.name: atom for atom in deprot_atoms}

    extra_deprot = sorted(set(deprot_by_name) - set(master_by_name))
    if extra_deprot:
        raise ValueError(
            "Deprotonated atoms are absent from the master: "
            + ", ".join(extra_deprot)
        )

    missing = [atom for atom in master_atoms if atom.name not in deprot_by_name]
    if len(missing) != 1:
        raise ValueError(
            "Base mapping requires exactly one master atom absent from the "
            f"deprotonated MOL2; found {len(missing)}"
        )
    dummy = missing[0]
    if not (dummy.name.upper().startswith("H") or dummy.atom_type.lower().startswith("h")):
        raise ValueError(f"The unmatched master atom is not a hydrogen: {dummy.name}")

    return [
        ChargeMapping(master_atom=atom, deprot_atom=deprot_by_name.get(atom.name))
        for atom in master_atoms
    ]


def format_mapping(
    mapping: list[ChargeMapping],
    proton_count_prot: int,
    proton_count_deprot: int,
) -> str:
    master_total = sum(item.master_atom.charge for item in mapping)
    deprot_total = sum(item.deprot_charge for item in mapping)
    rows = [
        ("atom_master", "charge_prot", "charge_deprot"),
        *(
            (
                item.master_atom.name,
                f"{item.master_atom.charge:.6f}",
                f"{item.deprot_charge:.6f}",
            )
            for item in mapping
        ),
        ("TOTAL", f"{master_total:+.6f}", f"{deprot_total:+.6f}"),
        (
            "PROTON_COUNT",
            str(proton_count_prot),
            str(proton_count_deprot),
        ),
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    return "\n".join(
        "  ".join(value.ljust(widths[column]) for column, value in enumerate(row))
        for row in rows
    )


def write_mapping_csv(
    output_file: Path,
    mapping: list[ChargeMapping],
    proton_count_prot: int,
    proton_count_deprot: int,
) -> None:
    master_total = sum(item.master_atom.charge for item in mapping)
    deprot_total = sum(item.deprot_charge for item in mapping)
    rows = [
        ["atom_master", "charge_prot", "charge_deprot"],
        *[
            [
                item.master_atom.name,
                f"{item.master_atom.charge:.6f}",
                f"{item.deprot_charge:.6f}",
            ]
            for item in mapping
        ],
        ["TOTAL", f"{master_total:+.6f}", f"{deprot_total:+.6f}"],
        ["PROTON_COUNT", str(proton_count_prot), str(proton_count_deprot)],
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def read_full_mol2(path: Path) -> FullMol2:
    lines = path.read_text().splitlines()
    try:
        molecule_index = lines.index("@<TRIPOS>MOLECULE")
    except ValueError as exc:
        raise ValueError(f"Missing @<TRIPOS>MOLECULE in {path}") from exc
    if molecule_index + 1 >= len(lines) or not lines[molecule_index + 1].strip():
        raise ValueError(f"Missing molecule name in {path}")
    molecule_name = lines[molecule_index + 1].strip()

    atoms: list[FullMol2Atom] = []
    bonds: list[Mol2Bond] = []
    section: str | None = None
    for line_number, line in enumerate(lines, start=1):
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>").strip()
            continue
        if not line.strip():
            continue
        fields = line.split()
        if section == "ATOM":
            if len(fields) < 9:
                raise ValueError(f"Malformed MOL2 atom line in {path}:{line_number}")
            try:
                atoms.append(
                    FullMol2Atom(
                        index=int(fields[0]),
                        name=fields[1],
                        coordinates=(float(fields[2]), float(fields[3]), float(fields[4])),
                        atom_type=fields[5],
                        residue_id=int(fields[6]),
                        residue_name=fields[7],
                        charge=float(fields[8]),
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
                bonds.append(
                    Mol2Bond(
                        index=int(fields[0]),
                        atom1=int(fields[1]),
                        atom2=int(fields[2]),
                        bond_type=fields[3],
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MOL2 bond data in {path}:{line_number}"
                ) from exc

    if not atoms:
        raise ValueError(f"No MOL2 atoms found in {path}")
    if len({atom.name for atom in atoms}) != len(atoms):
        raise ValueError(f"MOL2 atom names must be unique in {path}")
    if len({atom.index for atom in atoms}) != len(atoms):
        raise ValueError(f"MOL2 atom indices must be unique in {path}")
    valid_indices = {atom.index for atom in atoms}
    if any(
        bond.atom1 not in valid_indices or bond.atom2 not in valid_indices
        for bond in bonds
    ):
        raise ValueError(f"MOL2 bond references an unknown atom in {path}")
    return FullMol2(molecule_name, tuple(atoms), tuple(bonds))


def read_pdb_coordinates(path: Path) -> dict[str, tuple[float, float, float]]:
    coordinates: dict[str, tuple[float, float, float]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        if not name:
            raise ValueError(f"Missing atom name in {path}:{line_number}")
        if name in coordinates:
            raise ValueError(f"Duplicate PDB atom name {name} in {path}")
        try:
            coordinates[name] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid PDB coordinates in {path}:{line_number}") from exc
    if not coordinates:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return coordinates


def _bonded(structure: FullMol2, atom1: int, atom2: int) -> bool:
    return any(
        {bond.atom1, bond.atom2} == {atom1, atom2}
        for bond in structure.bonds
    )


def build_acid_master(
    cfg: AcidPrepConfig,
    deprotonated: FullMol2,
) -> FullMol2:
    atoms_by_name = {atom.name: atom for atom in deprotonated.atoms}
    required = (cfg.carbon, *cfg.oxygens)
    missing = [name for name in required if name not in atoms_by_name]
    if missing:
        raise ValueError("Configured carboxyl atoms absent from deprotonated MOL2: " + ", ".join(missing))
    dummy_flat = [name for pair in cfg.dummy_names for name in pair]
    collisions = sorted(set(dummy_flat) & set(atoms_by_name))
    if collisions:
        raise ValueError("Dummy names already present in deprotonated MOL2: " + ", ".join(collisions))

    carbon = atoms_by_name[cfg.carbon]
    oxygen_atoms = tuple(atoms_by_name[name] for name in cfg.oxygens)
    for oxygen in oxygen_atoms:
        if not _bonded(deprotonated, carbon.index, oxygen.index):
            raise ValueError(f"Configured atoms {cfg.carbon} and {oxygen.name} are not bonded")

    pdb_coordinates = read_pdb_coordinates(cfg.dummy_geometry_pdb)
    pdb_missing = [name for name in (*required, *dummy_flat) if name not in pdb_coordinates]
    if pdb_missing:
        raise ValueError("Atoms absent from dummy geometry PDB: " + ", ".join(pdb_missing))
    for atom in deprotonated.atoms:
        if atom.name not in pdb_coordinates:
            raise ValueError(f"Original atom {atom.name} absent from dummy geometry PDB")
        maximum_delta = max(
            abs(left - right)
            for left, right in zip(atom.coordinates, pdb_coordinates[atom.name], strict=True)
        )
        if maximum_delta > 5.1e-4:
            raise ValueError(
                f"PDB coordinate for {atom.name} differs from deprotonated MOL2"
            )

    master_atoms: list[FullMol2Atom] = []
    for atom in deprotonated.atoms:
        atom_type = "oh" if atom.name in cfg.oxygens else atom.atom_type
        master_atoms.append(
            FullMol2Atom(
                atom.index,
                atom.name,
                atom.coordinates,
                atom_type,
                atom.residue_id,
                cfg.system,
                atom.charge,
            )
        )
    residue_id = deprotonated.atoms[0].residue_id
    master_bonds = list(deprotonated.bonds)
    next_atom_index = max(atom.index for atom in deprotonated.atoms) + 1
    next_bond_index = max((bond.index for bond in deprotonated.bonds), default=0) + 1
    for oxygen, pair in zip(oxygen_atoms, cfg.dummy_names, strict=True):
        for name in pair:
            master_atoms.append(
                FullMol2Atom(
                    next_atom_index,
                    name,
                    pdb_coordinates[name],
                    "ho",
                    residue_id,
                    cfg.system,
                    0.0,
                )
            )
            master_bonds.append(
                Mol2Bond(next_bond_index, oxygen.index, next_atom_index, "1")
            )
            next_atom_index += 1
            next_bond_index += 1

    return FullMol2(cfg.system, tuple(master_atoms), tuple(master_bonds))


def build_acid_charge_states(
    cfg: AcidPrepConfig,
    deprotonated: FullMol2,
    protonated: FullMol2,
    master: FullMol2,
) -> AcidChargeStates:
    deprot_by_name = {atom.name: atom for atom in deprotonated.atoms}
    prot_by_name = {atom.name: atom for atom in protonated.atoms}
    deprot_names = set(deprot_by_name)
    prot_names = set(prot_by_name)
    if prot_names - deprot_names != {cfg.proton_atom}:
        difference = sorted(prot_names - deprot_names)
        raise ValueError(
            "Protonated MOL2 must add only carboxyl.proton_atom; found: "
            + ", ".join(difference)
        )
    if deprot_names - prot_names:
        raise ValueError(
            "Protonated MOL2 is missing atoms: "
            + ", ".join(sorted(deprot_names - prot_names))
        )
    proton = prot_by_name[cfg.proton_atom]
    protonated_oxygen = prot_by_name[cfg.protonated_oxygen]
    if not _bonded(protonated, proton.index, protonated_oxygen.index):
        raise ValueError(
            f"Source proton {cfg.proton_atom} is not bonded to {cfg.protonated_oxygen}"
        )

    state_names = ("deprot", "o1_syn", "o1_anti", "o2_syn", "o2_anti")
    dummy_flat = [name for pair in cfg.dummy_names for name in pair]
    active_dummy = (None, dummy_flat[0], dummy_flat[1], dummy_flat[2], dummy_flat[3])
    atom_names = tuple(atom.name for atom in master.atoms)
    columns: list[tuple[float, ...]] = []
    for state_index, active in enumerate(active_dummy):
        values: list[float] = []
        for name in atom_names:
            if name in deprot_by_name:
                if state_index == 0:
                    charge = deprot_by_name[name].charge
                elif state_index in {1, 2}:
                    charge = prot_by_name[name].charge
                elif name == cfg.oxygens[0]:
                    charge = prot_by_name[cfg.oxygens[1]].charge
                elif name == cfg.oxygens[1]:
                    charge = prot_by_name[cfg.oxygens[0]].charge
                else:
                    charge = prot_by_name[name].charge
            else:
                charge = proton.charge if name == active else 0.0
            values.append(charge)
        columns.append(tuple(values))
    return AcidChargeStates(
        atom_names=atom_names,
        state_names=state_names,
        charges=tuple(columns),
        proton_counts=(0, 1, 1, 1, 1),
    )


def write_full_mol2(path: Path, structure: FullMol2) -> None:
    residue_name = structure.atoms[0].residue_name
    residue_id = structure.atoms[0].residue_id
    lines = [
        "@<TRIPOS>MOLECULE",
        structure.name,
        f"{len(structure.atoms):5d} {len(structure.bonds):5d}     1     0     0",
        "SMALL",
        "USER_CHARGES",
        "",
        "",
        "@<TRIPOS>ATOM",
    ]
    for atom in structure.atoms:
        x, y, z = atom.coordinates
        lines.append(
            f"{atom.index:7d} {atom.name:<8} {x:10.4f} {y:10.4f} {z:10.4f} "
            f"{atom.atom_type:<6} {atom.residue_id:4d} {atom.residue_name:<8} "
            f"{atom.charge: .6f}"
        )
    lines.append("@<TRIPOS>BOND")
    for bond in structure.bonds:
        lines.append(
            f"{bond.index:6d} {bond.atom1:6d} {bond.atom2:6d} {bond.bond_type}"
        )
    lines.extend(
        (
            "@<TRIPOS>SUBSTRUCTURE",
            f"{residue_id:6d} {residue_name:<8} {structure.atoms[0].index:4d} "
            "TEMP              0 ****  ****    0 ROOT",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_acid_frcmod(path: Path, cfg: AcidPrepConfig, master: FullMol2) -> None:
    by_name = {atom.name: atom for atom in master.atoms}
    carbon = by_name[cfg.carbon]
    oxygen_indices = {by_name[name].index for name in cfg.oxygens}
    neighbor_indices: list[int] = []
    for bond in master.bonds:
        if bond.atom1 == carbon.index and bond.atom2 not in oxygen_indices:
            neighbor_indices.append(bond.atom2)
        elif bond.atom2 == carbon.index and bond.atom1 not in oxygen_indices:
            neighbor_indices.append(bond.atom1)
    if len(neighbor_indices) != 1:
        raise ValueError("Carboxyl carbon must have exactly one non-oxygen bonded neighbor")
    atoms_by_index = {atom.index: atom for atom in master.atoms}
    neighbor_type = atoms_by_index[neighbor_indices[0]].atom_type
    text = (
        "CpHMD dual-dummy carboxylic-acid parameters\n\n"
        "ANGLE\n"
        "ho-oh-ho    60.000   134.000\n"
        "oh-c-oh     80.000   120.000\n\n"
        "IMPROPER\n"
        "x -x -oh-ho    0       1.000     180.000       2.000\n"
        f"{neighbor_type}-oh-c -oh    0      10.500     180.000       2.000\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_acid_charge_sets(path: Path, states: AcidChargeStates) -> None:
    rows = [["atom_master", *(f"charge_{name}" for name in states.state_names)]]
    for atom_index, atom_name in enumerate(states.atom_names):
        rows.append(
            [
                atom_name,
                *(f"{column[atom_index]:.6f}" for column in states.charges),
            ]
        )
    rows.append(
        ["TOTAL", *(f"{sum(column):+.6f}" for column in states.charges)]
    )
    rows.append(["PROTON_COUNT", *(str(value) for value in states.proton_counts)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def format_acid_charge_states(states: AcidChargeStates) -> str:
    rows = [
        ["atom_master", *states.state_names],
        *[
            [
                atom_name,
                *(f"{column[atom_index]:.6f}" for column in states.charges),
            ]
            for atom_index, atom_name in enumerate(states.atom_names)
        ],
        ["TOTAL", *(f"{sum(column):+.6f}" for column in states.charges)],
        ["PROTON_COUNT", *(str(value) for value in states.proton_counts)],
    ]
    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(rows[0]))
    ]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def run_acid_prep(cfg: AcidPrepConfig) -> None:
    deprotonated = read_full_mol2(cfg.deprotonated_mol2)
    protonated = read_full_mol2(cfg.protonated_mol2)
    master = build_acid_master(cfg, deprotonated)
    states = build_acid_charge_states(cfg, deprotonated, protonated, master)
    write_full_mol2(cfg.master_mol2, master)
    write_acid_frcmod(cfg.frcmod, cfg, master)
    write_acid_charge_sets(cfg.charge_sets, states)
    print(format_acid_charge_states(states))
    print(f"OK: wrote acid master MOL2 to {cfg.master_mol2}")
    print(f"OK: wrote acid frcmod to {cfg.frcmod}")
    print(f"OK: wrote acid charge sets to {cfg.charge_sets}")


def run_cphmd_prep(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    if isinstance(cfg, AcidPrepConfig):
        run_acid_prep(cfg)
        return
    mapping = map_charges(cfg)
    print(
        format_mapping(
            mapping,
            cfg.proton_count_prot,
            cfg.proton_count_deprot,
        )
    )
    write_mapping_csv(
        cfg.output_file,
        mapping,
        cfg.proton_count_prot,
        cfg.proton_count_deprot,
    )
    print(f"OK: wrote {cfg.output_file}")
