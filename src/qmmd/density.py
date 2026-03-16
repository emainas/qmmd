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
    rho_target: float = 1.0


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
        rho_target=float(data.get("rho_target", 1.0)),
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


def write_scaled_rst7(src_rst7: Path, dst_rst7: Path, l1: float, l2: float, l3: float) -> None:
    lines = src_rst7.read_text().splitlines()
    if len(lines) < 1:
        raise RuntimeError(f"rst7 too short: {src_rst7}")
    last = lines[-1].split()
    if len(last) < 3:
        raise RuntimeError(f"Could not read box lengths from last line of {src_rst7}")
    # Preserve any extra box/angle fields beyond L1/L2/L3
    last[0] = f"{l1:.6f}"
    last[1] = f"{l2:.6f}"
    last[2] = f"{l3:.6f}"
    lines[-1] = " ".join(last)
    dst_rst7.write_text("\n".join(lines) + "\n")


def write_scaled_xyz(src_xyz: Path, dst_xyz: Path, l1: float, l2: float, l3: float) -> None:
    lines = src_xyz.read_text().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"xyz too short: {src_xyz}")
    line = lines[1]
    # Expect format: Conf 1. Box X: L1 0.000 0.000 Y: 0.000 L2 0.000 Z: 0.000 0.000 L3
    parts = line.split("Box X:")
    if len(parts) != 2:
        raise RuntimeError(f"Unexpected XYZ comment line (missing 'Box X:'): {line!r}")
    prefix = parts[0]
    suffix = parts[1]
    # Rebuild the box vector portion to avoid brittle token replacement
    box = f"Box X: {l1:.3f} 0.000 0.000 Y: 0.000 {l2:.3f} 0.000 Z: 0.000 0.000 {l3:.3f}"
    lines[1] = prefix + box
    dst_xyz.write_text("\n".join(lines) + "\n")


def count_water_residues(parm7: Path) -> int:
    text = parm7.read_text().splitlines()
    start = None
    for i, line in enumerate(text):
        if line.strip() == "%FLAG RESIDUE_LABEL":
            start = i + 2  # skip flag + format line
            break
    if start is None:
        raise RuntimeError(f"RESIDUE_LABEL section not found in {parm7}")

    labels: List[str] = []
    for line in text[start:]:
        if line.startswith("%FLAG"):
            break
        labels.extend([line[j:j + 4].strip() for j in range(0, len(line), 4)])

    labels = [x for x in labels if x]
    if not labels:
        raise RuntimeError(f"No residue labels parsed from {parm7}")

    return sum(1 for x in labels if x == "WAT")


def water_density_g_cm3(nwat: int, vol_ang3: float) -> float:
    # 1 Å^3 = 1e-24 cm^3
    na = 6.02214076e23
    mw = 18.01528  # g/mol
    mass_g = (nwat * mw) / na
    vol_cm3 = vol_ang3 * 1e-24
    return mass_g / vol_cm3


def scaled_box_lengths_for_target_rho(
    l1: float,
    l2: float,
    l3: float,
    solute_vol_ang3: float,
    nwat: int,
    rho_target: float,
) -> Tuple[float, float, float]:
    na = 6.02214076e23
    mw = 18.01528  # g/mol
    mass_g = (nwat * mw) / na
    vol_solvent_target_ang3 = (mass_g / rho_target) * 1e24
    box_vol_target = solute_vol_ang3 + vol_solvent_target_ang3
    box_vol_current = l1 * l2 * l3
    scale = (box_vol_target / box_vol_current) ** (1.0 / 3.0)
    return l1 * scale, l2 * scale, l3 * scale


def run_density(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    sdir = salt_dir(cfg, repo_root)
    odir = out_dir(cfg, repo_root)
    odir.mkdir(parents=True, exist_ok=True)

    parm7 = sdir / "ready.parm7"
    rst7 = sdir / "ready.rst7"
    xyz = sdir / "ready.xyz"
    if not parm7.exists() or parm7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty parm7: {parm7}")
    if not rst7.exists() or rst7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty rst7: {rst7}")
    if not xyz.exists() or xyz.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty xyz: {xyz}")

    (odir / "spec.yaml").write_text(yaml_path.read_text())
    cppin = write_cpptraj_in(cfg, parm7, rst7, odir)
    cppout = run_cpptraj(cppin)

    solute_vol, total_vol = parse_volumes(cppout)
    l1, l2, l3 = read_box_lengths(rst7)
    box_vol = l1 * l2 * l3
    nwat = count_water_residues(parm7)

    if abs(box_vol - total_vol) > 1e-2:
        raise RuntimeError(f"Box volume mismatch: L1*L2*L3={box_vol:.4f} vs VOLUME Avg={total_vol:.4f}")

    solvent_vol = total_vol - solute_vol
    rho = water_density_g_cm3(nwat, solvent_vol)
    l1_t, l2_t, l3_t = scaled_box_lengths_for_target_rho(
        l1, l2, l3, solute_vol, nwat, cfg.rho_target
    )
    new_box_vol = l1_t * l2_t * l3_t
    new_solvent_vol = new_box_vol - solute_vol
    new_rho = water_density_g_cm3(nwat, new_solvent_vol)
    print("OK: density summary")
    print(f"{'Quantity':<32} {'Value':>20} {'Units':>10}")
    print(f"{'-'*32} {'-'*20} {'-'*10}")
    print(f"{'Solute volume':<32} {solute_vol:>20.4f} {'Å^3':>10}")
    print(f"{'Total volume':<32} {total_vol:>20.4f} {'Å^3':>10}")
    print(f"{'Solvent volume':<32} {solvent_vol:>20.4f} {'Å^3':>10}")
    print(f"{'Water residues (WAT)':<32} {nwat:>20d} {'count':>10}")
    print(f"{'Water density (solvent)':<32} {rho:>20.4f} {'g/cm^3':>10}")
    print(f"{'Box lengths L1,L2,L3':<32} {l1:>6.2f}, {l2:>6.2f}, {l3:>6.2f} {'Å':>10}")
    print(f"{'Target rho':<32} {cfg.rho_target:>20.4f} {'g/cm^3':>10}")
    print(f"{'Target L1,L2,L3':<32} {l1_t:>6.2f}, {l2_t:>6.2f}, {l3_t:>6.2f} {'Å':>10}")

    scaled_rst7 = odir / "ready.rst7"
    write_scaled_rst7(rst7, scaled_rst7, l1_t, l2_t, l3_t)
    write_scaled_xyz(xyz, odir / "ready.xyz", l1_t, l2_t, l3_t)
    shutil.copy2(parm7, odir / "ready.parm7")
    if abs(new_rho - cfg.rho_target) > 1e-4:
        raise RuntimeError(
            f"Target density check failed: rho={new_rho:.6f} vs target={cfg.rho_target:.6f}"
        )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 density.py configs/SYS/density/density.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_density(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
