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


def load_config(yaml_path: Path) -> CpHMDPrepConfig:
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


def map_charges(cfg: CpHMDPrepConfig) -> list[ChargeMapping]:
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


def run_cphmd_prep(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
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
