#!/usr/bin/env python3

import os
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


def load_config(yaml_path: Path) -> PrepConfig:
    
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Prep configuration must be a YAML mapping")

    mol2_raw = _none_if_empty(data.get("mol2"))
    cn_raw = _none_if_empty(data.get("counterion_num"))
    
    cfg = PrepConfig(
        name=data["name"],
        leaprc_mol=_none_if_empty(data.get("leaprc_mol")),
        leaprc_sol=_none_if_empty(data.get("leaprc_sol")),
        frcmod_ion=_none_if_empty(data.get("frcmod_ion")),
        frcmods=_load_frcmods(data),
        mol2=Path(mol2_raw) if mol2_raw else None,
        water_model=data.get("water_model"),
        buffer=float(data.get("buffer")),
        counterion=_none_if_empty(data.get("counterion")),
        counterion_num=int(cn_raw) if cn_raw is not None else 0,
        input_dir=Path(data["input_dir"]),
        prefix=data.get("prefix", "solv"),
        additional_mol2=_load_additional_mol2(data),
        inter_residue_bonds=_load_inter_residue_bonds(data),
    )

    return cfg

def write_tleap_in(cfg: PrepConfig) -> Path:

    base_dir = cfg.input_dir.parent
    prep_dir = base_dir / f"{cfg.prefix}_{cfg.buffer:.1f}" / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mol2 is None:
        raise RuntimeError("mol2 is required for prep; set mol2 in the YAML")

    mol2_name = cfg.mol2.name
    mol2_src = cfg.input_dir / mol2_name
    
    # Existence of input checks
    if not mol2_src.exists():
        raise FileNotFoundError(f"Missing in input_dir: {mol2_src}")
    
    # Copy inputs to prep dir to run tleap in
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
        component_src = cfg.input_dir / component_name
        if not component_src.exists():
            raise FileNotFoundError(f"Missing in input_dir: {component_src}")
        shutil.copy2(component_src, prep_dir / component_name)
        copied_names.add(component_name)
        additional_loads.append(
            f"{component.unit} = loadmol2 {component_name}"
        )
        combined_units.append(component.unit)
    frcmod_names: list[str] = []
    for frcmod in cfg.frcmods:
        frcmod_name = frcmod.name
        frcmod_src = cfg.input_dir / frcmod_name
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

def run_prep(yaml_path):
    cfg = load_config(yaml_path)
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
    cfg = load_config(yaml_path)

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


if __name__ == "__main__":
    main()
