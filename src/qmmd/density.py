#!/usr/bin/env python3

import os
import sys
import yaml
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


@dataclass(frozen=True)
class DensityConfig:
    system: str
    buffer: float
    prefix: str
    salt_dirname: str
    mask: str
    center_mask: str
    grid: Tuple[float, float, float]
    buffer_pad: float


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def system_base_dir(cfg: DensityConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def salt_dir(cfg: DensityConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.salt_dirname


def out_dir(cfg: DensityConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / "density"


def load_config(yaml_path: Path) -> DensityConfig:
    data = yaml.safe_load(yaml_path.read_text())
    grid = data.get("grid", [0.25, 0.25, 0.25])
    if len(grid) != 3:
        raise RuntimeError("grid must be a 3-item list, e.g. [0.25, 0.25, 0.25]")
    return DensityConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        salt_dirname=data.get("salt_dirname", "salt"),
        mask=data["mask"],
        center_mask=data.get("center_mask", ":HIS"),
        grid=(float(grid[0]), float(grid[1]), float(grid[2])),
        buffer_pad=float(data.get("buffer_pad", 4.0)),
    )


def require_cpptraj() -> Path:
    amberhome = os.environ.get("AMBERHOME")
    if not amberhome:
        raise RuntimeError("AMBERHOME is not set. Did you load Amber?")

    cpptraj = shutil.which("cpptraj")
    if not cpptraj:
        raise RuntimeError("cpptraj not found in PATH. Did you load Amber?")

    cpptraj_path = Path(cpptraj).resolve()
    expected = Path(amberhome).resolve() / "bin" / "cpptraj"
    if cpptraj_path != expected:
        raise RuntimeError(
            f"cpptraj mismatch:\n"
            f"  PATH cpptraj: {cpptraj_path}\n"
            f"  AMBERHOME: {amberhome}\n"
            f"  expected: {expected}\n"
            f"Fix by re-loading the correct Amber module."
        )
    return cpptraj_path


def write_cpptraj_in(cfg: DensityConfig, parm7: Path, rst7: Path, out_path: Path) -> Path:
    gridx, gridy, gridz = cfg.grid
    text = f"""\
parm {parm7}
trajin {rst7}
volmap solute.dx {gridx} {gridy} {gridz} {cfg.mask} name SOLV centermask {cfg.center_mask} buffer {cfg.buffer_pad}
volume BoxVol out box_volume.dat
writedata solute_volume.dat SOLV[totalvol]
run
"""
    cppin = out_path / "cpptraj.in"
    cppin.write_text(text)
    return cppin


def run_cpptraj(cpptraj_in: Path) -> Path:
    cpptraj = require_cpptraj()
    out_path = cpptraj_in.with_suffix(".out")
    with out_path.open("w") as f:
        subprocess.run(
            [str(cpptraj), "-i", cpptraj_in.name],
            cwd=cpptraj_in.parent,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return out_path


def parse_volumes(cpptraj_out: Path) -> Tuple[float, float]:
    solute_vol = None
    total_vol = None
    lines = cpptraj_out.read_text().splitlines()
    for i, line in enumerate(lines):
        if "ACTION OUTPUT" in line:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if "total volume" in next_line:
                    parts = next_line.split("total volume", 1)[1].strip().split()
                    if parts:
                        solute_vol = float(parts[0])
        if "VOLUME: Avg=" in line:
            try:
                total_vol = float(line.split("VOLUME: Avg=", 1)[1].split()[0])
            except Exception:
                pass
    if solute_vol is None or total_vol is None:
        raise RuntimeError(f"Failed to parse volumes from {cpptraj_out}")
    return solute_vol, total_vol


def read_box_lengths(rst7: Path) -> Tuple[float, float, float]:
    lines = rst7.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"rst7 too short: {rst7}")
    last = lines[-1].split()
    if len(last) < 3:
        raise RuntimeError(f"Could not read box lengths from last line of {rst7}")
    return float(last[0]), float(last[1]), float(last[2])


def run_density(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    sdir = salt_dir(cfg, repo_root)
    odir = out_dir(cfg, repo_root)
    odir.mkdir(parents=True, exist_ok=True)

    parm7 = sdir / "ready.parm7"
    rst7 = sdir / "ready.rst7"
    if not parm7.exists() or parm7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty parm7: {parm7}")
    if not rst7.exists() or rst7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty rst7: {rst7}")

    (odir / "spec.yaml").write_text(yaml_path.read_text())
    cppin = write_cpptraj_in(cfg, parm7, rst7, odir)
    cppout = run_cpptraj(cppin)

    solute_vol, total_vol = parse_volumes(cppout)
    l1, l2, l3 = read_box_lengths(rst7)
    box_vol = l1 * l2 * l3

    if abs(box_vol - total_vol) > 1e-2:
        raise RuntimeError(f"Box volume mismatch: L1*L2*L3={box_vol:.4f} vs VOLUME Avg={total_vol:.4f}")

    solvent_vol = total_vol - solute_vol
    print(f"OK: solute volume = {solute_vol:.4f} Ang^3")
    print(f"OK: total volume  = {total_vol:.4f} Ang^3")
    print(f"OK: solvent volume = {solvent_vol:.4f} Ang^3")
    print(f"OK: box lengths (L1,L2,L3) = {l1:.4f}, {l2:.4f}, {l3:.4f}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 density.py configs/SYS/density/density.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_density(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
