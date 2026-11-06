#!/usr/bin/env python3

import csv
import re
import sys
import yaml
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AdditionalMol2:
    """An additional single-residue MOL2 unit combined with the primary solute."""

    unit: str
    mol2: Path


@dataclass(frozen=True)
class InterResidueBond:
    """A LEaP bond between one-based ``residue.atom`` selections."""

    atom1: str
    atom2: str


@dataclass(frozen=True)
class PackmolStructure:
    """One parameterized MOL2 component placed by Packmol."""

    name: str
    mol2: Path
    number: int


@dataclass(frozen=True)
class PackingConfig:
    """Packmol controls for positioning one or more solute components."""

    tolerance: float
    seed: int
    box: tuple[float, float, float, float, float, float]
    structures: tuple[PackmolStructure, ...]


@dataclass
class PrepConfig:
    """Class for spefifying the configuration of a solvation prep with tleap (Ambertools)."""
    name: str
    leaprc_mol: Optional[str]
    leaprc_sol: Optional[str]
    frcmods: tuple[Path, ...]
    mol2: Optional[Path]
    water_model: str
    buffer: float
    input_dir: Path
    counterion_num: int = 0
    frcmod_ion: Optional[str] = None
    counterion: Optional[str] = None
    prefix: Optional[str] = None
    additional_mol2: tuple[AdditionalMol2, ...] = ()
    inter_residue_bonds: tuple[InterResidueBond, ...] = ()
    packing: Optional[PackingConfig] = None


def _none_if_empty(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _load_frcmods(data: dict[object, object]) -> tuple[Path, ...]:
    """Load one frcmod path or an ordered list from the legacy ``frcmod`` key."""
    value = data.get("frcmod")
    if value is None or value == "":
        return ()
    values = value if isinstance(value, list) else [value]
    if not values:
        return ()
    paths = tuple(
        Path(_nonempty_string(item, f"frcmod[{index}]"))
        for index, item in enumerate(values)
    )
    basenames = [path.name for path in paths]
    if len(basenames) != len(set(basenames)):
        raise ValueError("frcmod basenames must be unique")
    return paths


def _load_additional_mol2(data: dict[object, object]) -> tuple[AdditionalMol2, ...]:
    values = data.get("additional_mol2", [])
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError("additional_mol2 must be a list")
    components: list[AdditionalMol2] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"additional_mol2[{index}] must be a mapping")
        unit = _nonempty_string(value.get("unit"), f"additional_mol2[{index}].unit")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", unit) is None:
            raise ValueError(
                f"additional_mol2[{index}].unit must be a valid LEaP identifier"
            )
        mol2 = Path(
            _nonempty_string(value.get("mol2"), f"additional_mol2[{index}].mol2")
        )
        components.append(AdditionalMol2(unit=unit, mol2=mol2))
    units = [component.unit for component in components]
    if len(units) != len(set(units)):
        raise ValueError("additional_mol2 unit names must be unique")
    if "solute" in units or "sys" in units:
        raise ValueError("additional_mol2 unit names cannot be 'solute' or 'sys'")
    return tuple(components)


def _load_inter_residue_bonds(
    data: dict[object, object],
) -> tuple[InterResidueBond, ...]:
    values = data.get("inter_residue_bonds", [])
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError("inter_residue_bonds must be a list")
    selection_pattern = re.compile(r"[1-9][0-9]*\.[A-Za-z0-9_+\-]+")
    bonds: list[InterResidueBond] = []
    for index, value in enumerate(values):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"inter_residue_bonds[{index}] must contain two residue.atom selections"
            )
        atom1 = _nonempty_string(value[0], f"inter_residue_bonds[{index}][0]")
        atom2 = _nonempty_string(value[1], f"inter_residue_bonds[{index}][1]")
        if selection_pattern.fullmatch(atom1) is None:
            raise ValueError(
                f"inter_residue_bonds[{index}][0] must use one-based residue.atom syntax"
            )
        if selection_pattern.fullmatch(atom2) is None:
            raise ValueError(
                f"inter_residue_bonds[{index}][1] must use one-based residue.atom syntax"
            )
        bonds.append(InterResidueBond(atom1=atom1, atom2=atom2))
    return tuple(bonds)


def _positive_float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _load_packing(
    data: dict[object, object], default_name: str, default_mol2: Optional[Path]
) -> Optional[PackingConfig]:
    value = data.get("packing")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("packing must be a YAML mapping")

    tolerance = _positive_float(value.get("tolerance", 2.0), "packing.tolerance")
    seed_raw = value.get("seed", 1234567)
    if isinstance(seed_raw, bool):
        raise ValueError("packing.seed must be an integer")
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("packing.seed must be an integer") from exc
    if seed <= 0:
        raise ValueError("packing.seed must be positive")

    box_raw = value.get("box")
    if not isinstance(box_raw, list) or len(box_raw) not in {3, 6}:
        raise ValueError(
            "packing.box must contain either [lx, ly, lz] or "
            "[xmin, ymin, zmin, xmax, ymax, zmax] in Angstrom"
        )
    try:
        box_values = tuple(float(item) for item in box_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("packing.box values must be numeric") from exc
    if len(box_values) == 3:
        if any(length <= 0 for length in box_values):
            raise ValueError("packing.box lengths must be positive")
        box = (0.0, 0.0, 0.0, *box_values)
    else:
        box = box_values
        if any(box[i + 3] <= box[i] for i in range(3)):
            raise ValueError("packing.box maximums must exceed minimums")

    structures_raw = value.get("structures")
    if structures_raw is None:
        if default_mol2 is None:
            raise ValueError(
                "mol2 is required when packing.structures is not provided"
            )
        copies_raw = value.get("copies")
        if copies_raw is None:
            raise ValueError(
                "packing.copies is required when packing.structures is omitted"
            )
        structures_raw = [
            {"name": default_name, "mol2": str(default_mol2), "number": copies_raw}
        ]
    elif "copies" in value:
        raise ValueError("packing.copies and packing.structures are mutually exclusive")
    if not isinstance(structures_raw, list) or not structures_raw:
        raise ValueError("packing.structures must be a nonempty list")

    structures: list[PackmolStructure] = []
    for index, item in enumerate(structures_raw):
        field = f"packing.structures[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field} must be a mapping")
        name = _nonempty_string(item.get("name"), f"{field}.name")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name) is None:
            raise ValueError(
                f"{field}.name may contain only letters, digits, '_' and '-'"
            )
        mol2 = Path(_nonempty_string(item.get("mol2"), f"{field}.mol2"))
        number_raw = item.get("number")
        if isinstance(number_raw, bool):
            raise ValueError(f"{field}.number must be a positive integer")
        try:
            number = int(number_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}.number must be a positive integer") from exc
        if number <= 0:
            raise ValueError(f"{field}.number must be a positive integer")
        structures.append(PackmolStructure(name=name, mol2=mol2, number=number))
    names = [structure.name for structure in structures]
    if len(names) != len(set(names)):
        raise ValueError("packing structure names must be unique")

    return PackingConfig(
        tolerance=tolerance,
        seed=seed,
        box=box,
        structures=tuple(structures),
    )


def load_config(yaml_path: Path) -> PrepConfig:
    
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Prep configuration must be a YAML mapping")

    mol2_raw = _none_if_empty(data.get("mol2"))
    cn_raw = _none_if_empty(data.get("counterion_num"))
    
    name = _nonempty_string(data.get("name"), "name")
    mol2 = Path(mol2_raw) if mol2_raw else None
    packing = _load_packing(data, name, mol2)
    additional_mol2 = _load_additional_mol2(data)
    inter_residue_bonds = _load_inter_residue_bonds(data)
    if packing is not None and (additional_mol2 or inter_residue_bonds):
        raise ValueError(
            "packing cannot be combined with additional_mol2 or "
            "inter_residue_bonds; pack a complete bonded MOL2 component instead"
        )

    cfg = PrepConfig(
        name=name,
        leaprc_mol=_none_if_empty(data.get("leaprc_mol")),
        leaprc_sol=_none_if_empty(data.get("leaprc_sol")),
        frcmod_ion=_none_if_empty(data.get("frcmod_ion")),
        frcmods=_load_frcmods(data),
        mol2=mol2,
        water_model=data.get("water_model"),
        buffer=float(data.get("buffer")),
        counterion=_none_if_empty(data.get("counterion")),
        counterion_num=int(cn_raw) if cn_raw is not None else 0,
        input_dir=Path(data["input_dir"]),
        prefix=data.get("prefix", "solv"),
        additional_mol2=additional_mol2,
        inter_residue_bonds=inter_residue_bonds,
        packing=packing,
    )

    return cfg


@dataclass(frozen=True)
class _Mol2PackingAtom:
    line_index: int
    name: str
    atom_type: str
    residue_id: int
    residue_name: str
    charge: float


@dataclass(frozen=True)
class _Mol2PackingTemplate:
    lines: tuple[str, ...]
    atoms: tuple[_Mol2PackingAtom, ...]
    residue_ids: tuple[int, ...]


def _prep_dir(cfg: PrepConfig) -> Path:
    base_dir = cfg.input_dir.parent
    return base_dir / f"{cfg.prefix}_{cfg.buffer:.1f}" / "prep"


def _input_path(cfg: PrepConfig, path: Path) -> Path:
    """Resolve prep inputs with the historical input_dir/basename convention."""
    return cfg.input_dir / path.name


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _read_mol2_packing_template(path: Path) -> _Mol2PackingTemplate:
    lines = path.read_text().splitlines()
    atoms: list[_Mol2PackingAtom] = []
    residue_ids: list[int] = []
    section = ""
    for line_index, line in enumerate(lines):
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>").strip().upper()
            continue
        if section != "ATOM" or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed MOL2 atom record in {path}: {line}")
        try:
            residue_id = int(fields[6])
            charge = float(fields[8])
            float(fields[2])
            float(fields[3])
            float(fields[4])
        except ValueError as exc:
            raise ValueError(f"Malformed MOL2 atom record in {path}: {line}") from exc
        atoms.append(
            _Mol2PackingAtom(
                line_index=line_index,
                name=fields[1],
                atom_type=fields[5],
                residue_id=residue_id,
                residue_name=fields[7],
                charge=charge,
            )
        )
        if residue_id not in residue_ids:
            residue_ids.append(residue_id)
    if not atoms:
        raise ValueError(f"No @<TRIPOS>ATOM records found in {path}")
    total_charge = sum(atom.charge for atom in atoms)
    if abs(total_charge - round(total_charge)) > 1.0e-4:
        raise ValueError(
            f"Packing component {path} has non-integral charge {total_charge:+.8f}"
        )
    return _Mol2PackingTemplate(
        lines=tuple(lines), atoms=tuple(atoms), residue_ids=tuple(residue_ids)
    )


def _element_from_mol2(atom: _Mol2PackingAtom) -> str:
    token = atom.atom_type.split(".", 1)[0]
    letters = "".join(character for character in token if character.isalpha())
    if not letters:
        letters = "".join(character for character in atom.name if character.isalpha())
    candidate = letters[:2].title()
    if candidate in {"Br", "Cl", "Na", "Mg", "Ca", "Fe", "Zn", "Li"}:
        return candidate
    return letters[:1].upper() or "X"


def _write_packmol_template_pdb(
    source_mol2: Path, destination: Path, template: _Mol2PackingTemplate
) -> None:
    coordinates: dict[int, tuple[float, float, float]] = {}
    for atom in template.atoms:
        fields = template.lines[atom.line_index].split()
        coordinates[atom.line_index] = tuple(float(value) for value in fields[2:5])
    pdb_lines = [f"REMARK generated from {source_mol2.name} by qmmd prep"]
    for serial, atom in enumerate(template.atoms, start=1):
        x, y, z = coordinates[atom.line_index]
        residue_name = atom.residue_name[:3].upper()
        element = _element_from_mol2(atom)
        pdb_lines.append(
            f"HETATM{serial:5d} {atom.name[:4]:<4s} {residue_name:>3s} A"
            f"{atom.residue_id % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2s}"
        )
    pdb_lines.extend(("TER", "END"))
    destination.write_text("\n".join(pdb_lines) + "\n")


def _packing_source_name(structure: PackmolStructure) -> str:
    return f"packing-source-{_safe_stem(structure.name)}.mol2"


def _packing_template_name(structure: PackmolStructure) -> str:
    return f"packing-template-{_safe_stem(structure.name)}.pdb"


def _packed_copy_name(structure: PackmolStructure, copy_index: int) -> str:
    return f"packed-{_safe_stem(structure.name)}-{copy_index:03d}.mol2"


def write_packmol_in(cfg: PrepConfig) -> Path:
    """Copy packing inputs and write the deterministic Packmol input file."""
    if cfg.packing is None:
        raise ValueError("write_packmol_in requires a packing configuration")
    prep_dir = _prep_dir(cfg)
    prep_dir.mkdir(parents=True, exist_ok=True)
    xlo, ylo, zlo, xhi, yhi, zhi = cfg.packing.box
    lines = [
        f"tolerance {cfg.packing.tolerance:g}",
        "filetype pdb",
        "output packed.pdb",
        f"seed {cfg.packing.seed}",
        "",
    ]
    for structure in cfg.packing.structures:
        source = _input_path(cfg, structure.mol2)
        if not source.is_file():
            raise FileNotFoundError(f"Missing in input_dir: {source}")
        copied = prep_dir / _packing_source_name(structure)
        shutil.copy2(source, copied)
        template = _read_mol2_packing_template(copied)
        template_pdb = prep_dir / _packing_template_name(structure)
        _write_packmol_template_pdb(copied, template_pdb, template)
        lines.extend(
            (
                f"structure {template_pdb.name}",
                f"  number {structure.number}",
                "  inside box "
                f"{xlo:g} {ylo:g} {zlo:g} {xhi:g} {yhi:g} {zhi:g}",
                "end structure",
                "",
            )
        )
    destination = prep_dir / "packmol.inp"
    destination.write_text("\n".join(lines))
    return destination


def require_packmol() -> Path:
    packmol = shutil.which("packmol")
    if not packmol:
        raise RuntimeError(
            "packmol not found in PATH. Load an AmberTools module, for example amber/26."
        )
    return Path(packmol).resolve()


def run_packmol(packmol_in: Path) -> Path:
    prep_dir = packmol_in.parent
    output_pdb = prep_dir / "packed.pdb"
    log = prep_dir / "packmol.out"
    with packmol_in.open() as stdin, log.open("w") as stdout:
        subprocess.run(
            [str(require_packmol())],
            cwd=prep_dir,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not output_pdb.is_file() or output_pdb.stat().st_size == 0:
        raise RuntimeError(f"Packmol did not create {output_pdb}")
    return output_pdb


def _read_packed_pdb(path: Path) -> list[tuple[str, float, float, float]]:
    atoms: list[tuple[str, float, float, float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            atoms.append(
                (
                    line[12:16].strip(),
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid packed PDB coordinates in {path}:{line_number}"
            ) from exc
    if not atoms:
        raise ValueError(f"No atoms found in Packmol output {path}")
    return atoms


def _write_transformed_mol2(
    template: _Mol2PackingTemplate,
    coordinates: list[tuple[str, float, float, float]],
    destination: Path,
) -> None:
    if len(coordinates) != len(template.atoms):
        raise ValueError("Packed coordinate count does not match MOL2 template")
    lines = list(template.lines)
    for atom, (packed_name, x, y, z) in zip(template.atoms, coordinates):
        if packed_name != atom.name[:4]:
            raise ValueError(
                f"Packmol atom order changed: expected {atom.name}, found {packed_name}"
            )
        fields = lines[atom.line_index].split()
        fields[2:5] = [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
        lines[atom.line_index] = " ".join(fields)
    destination.write_text("\n".join(lines) + "\n")


def materialize_packed_mol2(cfg: PrepConfig, packed_pdb: Path) -> tuple[Path, ...]:
    """Transfer Packmol coordinates into topology-preserving MOL2 copies."""
    if cfg.packing is None:
        raise ValueError("materialize_packed_mol2 requires a packing configuration")
    prep_dir = packed_pdb.parent
    packed_atoms = _read_packed_pdb(packed_pdb)
    expected = 0
    templates: dict[str, _Mol2PackingTemplate] = {}
    for structure in cfg.packing.structures:
        template = _read_mol2_packing_template(
            prep_dir / _packing_source_name(structure)
        )
        templates[structure.name] = template
        expected += structure.number * len(template.atoms)
    if len(packed_atoms) != expected:
        raise ValueError(
            f"Packmol output contains {len(packed_atoms)} atoms; expected {expected}"
        )

    outputs: list[Path] = []
    mapping_rows: list[dict[str, object]] = []
    coordinate_offset = 0
    atom_offset = 0
    residue_offset = 0
    for structure in cfg.packing.structures:
        template = templates[structure.name]
        atom_count = len(template.atoms)
        residue_count = len(template.residue_ids)
        for copy_index in range(1, structure.number + 1):
            coordinates = packed_atoms[
                coordinate_offset : coordinate_offset + atom_count
            ]
            coordinate_offset += atom_count
            output = prep_dir / _packed_copy_name(structure, copy_index)
            _write_transformed_mol2(template, coordinates, output)
            outputs.append(output)
            mapping_rows.append(
                {
                    "structure": structure.name,
                    "copy": copy_index,
                    "source_mol2": structure.mol2.as_posix(),
                    "atom_start": atom_offset + 1,
                    "atom_end": atom_offset + atom_count,
                    "residue_start": residue_offset + 1,
                    "residue_end": residue_offset + residue_count,
                }
            )
            atom_offset += atom_count
            residue_offset += residue_count
    mapping = prep_dir / "packing-map.csv"
    with mapping.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    return tuple(outputs)


def write_tleap_in(cfg: PrepConfig) -> Path:

    prep_dir = _prep_dir(cfg)
    prep_dir.mkdir(parents=True, exist_ok=True)

    if cfg.packing is not None:
        packed_loads: list[str] = []
        combined_units: list[str] = []
        for structure in cfg.packing.structures:
            for copy_index in range(1, structure.number + 1):
                filename = _packed_copy_name(structure, copy_index)
                if not (prep_dir / filename).is_file():
                    raise FileNotFoundError(
                        f"Missing packed MOL2 copy: {prep_dir / filename}; "
                        "run Packmol first"
                    )
                unit = f"pack_{_safe_stem(structure.name)}_{copy_index:03d}"
                packed_loads.append(f"{unit} = loadmol2 {filename}")
                combined_units.append(unit)
        solute_block = (
            "\n".join(packed_loads)
            + f"\nsys = combine {{{' '.join(combined_units)}}}"
        )
        bond_block = ""
    else:
        if cfg.mol2 is None:
            raise RuntimeError("mol2 is required for prep; set mol2 in the YAML")
        mol2_name = cfg.mol2.name
        mol2_src = _input_path(cfg, cfg.mol2)
        if not mol2_src.exists():
            raise FileNotFoundError(f"Missing in input_dir: {mol2_src}")
        shutil.copy2(mol2_src, prep_dir / mol2_name)
        additional_loads: list[str] = []
        combined_units = ["solute"]
        copied_names = {mol2_name}
        for component in cfg.additional_mol2:
            component_name = component.mol2.name
            if component_name in copied_names:
                raise ValueError(
                    f"MOL2 basenames must be unique in prep: {component_name}"
                )
            component_src = _input_path(cfg, component.mol2)
            if not component_src.exists():
                raise FileNotFoundError(f"Missing in input_dir: {component_src}")
            shutil.copy2(component_src, prep_dir / component_name)
            copied_names.add(component_name)
            additional_loads.append(
                f"{component.unit} = loadmol2 {component_name}"
            )
            combined_units.append(component.unit)
        load_block = "\n".join(
            [f"solute = loadmol2 {mol2_name}", *additional_loads]
        )
        if cfg.additional_mol2:
            solute_block = (
                f"{load_block}\n"
                f"sys = combine {{{' '.join(combined_units)}}}"
            )
        else:
            solute_block = f"sys = loadmol2 {mol2_name}"
        bond_block = "\n".join(
            f"bond sys.{bond.atom1} sys.{bond.atom2}"
            for bond in cfg.inter_residue_bonds
        )

    frcmod_names: list[str] = []
    for frcmod in cfg.frcmods:
        frcmod_name = frcmod.name
        frcmod_src = _input_path(cfg, frcmod)
        if not frcmod_src.exists():
            raise FileNotFoundError(f"Missing in input_dir: {frcmod_src}")
        shutil.copy2(frcmod_src, prep_dir / frcmod_name)
        frcmod_names.append(frcmod_name)

    tleap_in = prep_dir / "tleap.in"

    addions_block = ""
    if cfg.counterion_num > 0:
        if not cfg.counterion:
            raise RuntimeError("counterion_num > 0 but counterion is not set")
        addions_block = f'addions sys {cfg.counterion} {cfg.counterion_num}\n'

    text = f"""\
# Auto-generated by qmmd prep.py
# System: {cfg.name}

{f"source {cfg.leaprc_mol}" if cfg.leaprc_mol else ""}
{f"source {cfg.leaprc_sol}" if cfg.leaprc_sol else ""}

# Ions (Joung-Cheatham for TIP3P)
{f"loadamberparams {cfg.frcmod_ion}" if cfg.frcmod_ion else ""}

# Molecule params
{chr(10).join(f"loadamberparams {name}" for name in frcmod_names)}
{solute_block}

# Explicit bonds between combined residues
{bond_block}

# Solvate (buffer in Angstrom)
solvatebox sys {cfg.water_model} {cfg.buffer}

# Counterions
{addions_block.rstrip()}

saveamberparm sys {cfg.prefix}.parm7 {cfg.prefix}.rst7
quit
"""
    tleap_in.write_text(text)
    return tleap_in


def require_amber() -> Path:
    tleap = shutil.which("tleap")
    if not tleap:
        raise RuntimeError("tleap not found in PATH. Did you load AmberTools?")

    tleap_path = Path(tleap).resolve()

    result = subprocess.run([str(tleap_path), "-f", "-"], input="quit\n", text=True, capture_output=True, timeout=10,)  
    if result.returncode != 0:
        raise RuntimeError(
            f"tleap was found but failed to run:\n"
            f"  executable: {tleap_path}\n"
            f"  return code: {result.returncode}\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    
    return tleap_path

def run_tleap(cfg: PrepConfig, tleap_in: Path) -> None:
    
    prep_dir = tleap_in.parent
    out_path = prep_dir / "tleap.out"

    with out_path.open("w") as f:
        tleap = require_amber()
        subprocess.run(
            [str(tleap), "-f", tleap_in.name],
            cwd=prep_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

def run_prep(yaml_path: Path) -> None:
    yaml_path = yaml_path.resolve()
    cfg = load_config(yaml_path)
    if cfg.packing is not None:
        packmol_in = write_packmol_in(cfg)
        packed_pdb = run_packmol(packmol_in)
        materialize_packed_mol2(cfg, packed_pdb)
    tleap_in = write_tleap_in(cfg)
    run_tleap(cfg, tleap_in)

    prep_dir = tleap_in.parent

    # Freeze config for reproducibility (in the prep dir)
    (prep_dir / "spec.yaml").write_text(yaml_path.read_text())

    parm7 = prep_dir / f"{cfg.prefix}.parm7"
    rst7 = prep_dir / f"{cfg.prefix}.rst7"
    if not parm7.exists() or parm7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty output: {parm7}")
    if not rst7.exists() or rst7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty output: {rst7}")

    print(f"OK: wrote {parm7.name} and {rst7.name} in {prep_dir}")

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 prep.py prep.yaml", file=sys.stderr)
        raise SystemExit(2)

    yaml_path = Path(sys.argv[1]).resolve()
    run_prep(yaml_path)


if __name__ == "__main__":
    main()
