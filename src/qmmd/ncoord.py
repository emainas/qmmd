#!/usr/bin/env python3

import os
import sys
import yaml
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NcoordConfig:
    system: str
    buffer: float
    prefix: str
    job_name: str
    gausswidth: float
    group1: int
    group2: int
    nexp: int
    mexp: int
    refdist: float
    method: str
    norm: str


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def system_base_dir(cfg: NcoordConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def dftb_dir(cfg: NcoordConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / "dftb"


def run_dir(cfg: NcoordConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.job_name


def out_dir(cfg: CpptrajPostConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / "ncoord"


def load_config(yaml_path: Path) -> NcoordConfig:
    data = yaml.safe_load(yaml_path.read_text())
    return NcoordConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        job_name=data.get("job_name", "mdequil"),
    )

def parse_group_indexes(closest_path: Path) -> int:
    """
    This function reads dftb.inp from root/<system>/<solv>/dftb/dftb.inp
    """

    if not closest_path.exists() or closest_path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty: {closest_path}")

    last = None
    for line in closest_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            last = parts

    if last is None:
        raise RuntimeError(f"No data lines found in {closest_path}")

    resid = last[1]
    if not resid.isdigit():
        raise RuntimeError(f"Expected integer resid in col2, got '{resid}' from line: {last}")

    return int(resid)


def run_ncoord(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)

    repo_root = find_repo_root(yaml_path)
    pdir = prep_dir(cfg, repo_root)
    rdir = run_dir(cfg, repo_root)
    output_dir = out_dir(cfg, repo_root)

    # Validate inferred inputs
    parm7 = pdir / f"{cfg.prefix}.parm7"
    traj = rdir / "equil-npt.nc"

    if not parm7.exists() or parm7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty inferred input parm7: {parm7}")
    if not traj.exists() or traj.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty inferred input traj: {traj}")

    # Freeze YAML in output dir (reproducibility)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spec.yaml").write_text(yaml_path.read_text())

    # Parse furthest resid from stage 1 output
    closest_path = output_dir / "closest_waters.dat"
    resid = parse_furthest_resid(closest_path)
    print(f"[INFO] Furthest water resid (closest_waters.dat col2) = {resid}")

    # Validate outputs
    ready_xyz = output_dir / "ready.xyz"
    if not ready_xyz.exists() or ready_xyz.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty output: {ready_xyz}")

    print(f"OK: outputs in {output_dir}")
    print(f"OK: wrote ready.xyz and ready.parm7 (deleted :{resid}@{cfg.delete_h})")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 ncoord.py configs/SYS/ncoord.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_ncoord(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main() 
