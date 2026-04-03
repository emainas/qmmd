#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import torch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


@dataclass(frozen=True)
class MDConfig:
    T: float
    dt_fs: float
    friction_per_fs: float
    equil_ps: float

    @property
    def dt(self) -> float:
        return self.dt_fs * units.fs

    @property
    def friction(self) -> float:
        return self.friction_per_fs / units.fs


@dataclass(frozen=True)
class OutputConfig:
    log_stride: int
    traj_stride: int


@dataclass(frozen=True)
class OrbModelConfig:
    model: str
    precision: str
    compile: bool
    charge: int
    spin: int
    device: str


@dataclass(frozen=True)
class OrbEquilConfig:
    system: str
    buffer: float
    prefix: str
    input_source: str
    input_name: str
    salt_dirname: str
    density_dirname: str
    md: MDConfig
    output: OutputConfig
    orb: OrbModelConfig
    cell: Optional[Tuple[float, float, float]] = None
    pbc: Optional[bool] = None


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def system_base_dir(cfg: OrbEquilConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def input_dir(cfg: OrbEquilConfig, repo_root: Path) -> Path:
    base = system_base_dir(cfg, repo_root)
    src = cfg.input_source.lower()
    if src == "density":
        return base / cfg.density_dirname
    if src == "salt":
        return base / cfg.salt_dirname
    raise RuntimeError(f"Unknown input_source: {cfg.input_source!r} (expected 'salt' or 'density')")


def steps_from_ps(ps: float, dt_fs: float) -> int:
    n = int(round(ps * 1000.0 / dt_fs))
    if n <= 0:
        raise ValueError("Requested run length yields <1 step. Check equil_ps and dt_fs.")
    return n


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def _req(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key in YAML: '{key}'")
    return d[key]


def parse_box_lengths_from_xyz_comment(comment: str) -> Optional[Tuple[float, float, float]]:
    m = re.search(
        r"Box X:\s*([0-9.+-]+)\s+0\.000\s+0\.000\s+Y:\s*0\.000\s+([0-9.+-]+)\s+0\.000\s+Z:\s*0\.000\s+0\.000\s+([0-9.+-]+)",
        comment,
    )
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def infer_cell_from_xyz(xyz_path: Path) -> Optional[Tuple[float, float, float]]:
    with xyz_path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        comment = f.readline()
    if not comment:
        return None
    return parse_box_lengths_from_xyz_comment(comment.strip())


def setup_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_config(yaml_path: Path) -> OrbEquilConfig:
    data = yaml.safe_load(yaml_path.read_text())
    md = _req(data, "md")
    output = _req(data, "output")
    orb = _get(data, "orb", {}) or {}

    cell = data.get("cell")
    if cell is not None:
        if len(cell) != 3:
            raise RuntimeError("cell must be a 3-item list, e.g. [Lx, Ly, Lz]")
        cell_tuple = (float(cell[0]), float(cell[1]), float(cell[2]))
    else:
        cell_tuple = None

    return OrbEquilConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        input_source=str(data.get("input_source", "salt")),
        input_name=str(data.get("input_name", "ready.xyz")),
        salt_dirname=data.get("salt_dirname", "salt"),
        density_dirname=data.get("density_dirname", "density"),
        md=MDConfig(
            T=float(_req(md, "T")),
            dt_fs=float(_req(md, "dt_fs")),
            friction_per_fs=float(_req(md, "friction_per_fs")),
            equil_ps=float(_req(md, "equil_ps")),
        ),
        output=OutputConfig(
            log_stride=int(_req(output, "log_stride")),
            traj_stride=int(_req(output, "traj_stride")),
        ),
        orb=OrbModelConfig(
            model=str(_get(orb, "model", "orb_v3_conservative_omol")),
            precision=str(_get(orb, "precision", "float32-high")),
            compile=bool(_get(orb, "compile", False)),
            charge=int(_get(orb, "charge", 0)),
            spin=int(_get(orb, "spin", 1)),
            device=str(_get(orb, "device", "auto")),
        ),
        cell=cell_tuple,
        pbc=data.get("pbc"),
    )


def get_orb_calculator(cfg: OrbModelConfig) -> ORBCalculator:
    if not hasattr(pretrained, cfg.model):
        raise RuntimeError(f"Unknown ORB model: {cfg.model!r}")
    model_fn = getattr(pretrained, cfg.model)
    device = setup_device(cfg.device)
    orbff = model_fn(device=device, precision=cfg.precision, compile=cfg.compile)
    return ORBCalculator(orbff, device=device)


def run_equil(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    src_dir = input_dir(cfg, repo_root)
    xyz = src_dir / cfg.input_name
    if not xyz.exists() or xyz.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty input XYZ: {xyz}")

    atoms = read(xyz)

    cell = cfg.cell
    if cell is None:
        cell = infer_cell_from_xyz(xyz)
    if cell is not None:
        atoms.set_cell(cell)
        pbc = bool(cfg.pbc) if cfg.pbc is not None else True
        atoms.set_pbc([pbc, pbc, pbc])
        atoms.wrap()
    elif cfg.pbc is not None:
        atoms.set_pbc([cfg.pbc, cfg.pbc, cfg.pbc])

    atoms.info["charge"] = cfg.orb.charge
    atoms.info["spin"] = cfg.orb.spin
    atoms.calc = get_orb_calculator(cfg.orb)

    seed = secrets.randbelow(2**31 - 1) + 1
    print(f"Using RNG seed (velocities only): {seed}")
    rng_state = np.random.get_state()
    np.random.seed(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.md.T)
    np.random.set_state(rng_state)

    n_equil = steps_from_ps(cfg.md.equil_ps, cfg.md.dt_fs)
    print(f"Equil steps: {n_equil} (equil_ps={cfg.md.equil_ps}, dt_fs={cfg.md.dt_fs})")

    out_dir = yaml_path.parent
    traj_path = out_dir / "traj.xyz"
    mdlog_path = out_dir / "md.log"

    dyn = Langevin(atoms, cfg.md.dt, temperature_K=cfg.md.T, friction=cfg.md.friction)

    with mdlog_path.open("w", encoding="utf-8") as flog:
        flog.write("# step time_ps Epot_eV Ekin_eV T_K\n")

        def log_cb() -> None:
            if dyn.nsteps % cfg.output.log_stride != 0:
                return
            tps = dyn.nsteps * cfg.md.dt_fs * 1e-3
            epot = float(atoms.get_potential_energy())
            ekin = float(atoms.get_kinetic_energy())
            temp = float(atoms.get_temperature())
            flog.write(f"{dyn.nsteps} {tps:.6f} {epot:.8f} {ekin:.8f} {temp:.3f}\n")
            flog.flush()

        def traj_cb() -> None:
            if dyn.nsteps % cfg.output.traj_stride == 0:
                write(traj_path, atoms, append=True)

        dyn.attach(log_cb, interval=1)
        dyn.attach(traj_cb, interval=1)
        dyn.run(n_equil)

    print(f"OK: wrote {mdlog_path}")
    print(f"OK: wrote {traj_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="YAML config file (spec.yaml)")
    args = ap.parse_args()
    run_equil(Path(args.yaml).resolve())


if __name__ == "__main__":
    main()
