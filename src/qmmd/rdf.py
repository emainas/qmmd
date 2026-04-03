#!/usr/bin/env python3

import os
import sys
import yaml
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple


@dataclass(frozen=True)
class RDFConfig:
    system: str
    buffer: float
    prefix: str
    method_dir: str
    bench_tag: Optional[str]
    run_ids: List[int]
    cv_dir: str
    traj_name: str
    dftb_inp_name: Optional[str]
    parm_path: str
    dr: float
    r_max: float
    mask1: str
    mask2: Optional[str]
    noimage: bool
    volume: bool
    density: Optional[float]
    dataset: Optional[str]
    intrdf: Optional[str]
    rawrdf: Optional[str]


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


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


def parse_run_ids(value: Any) -> List[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        if "-" in value:
            a, b = value.split("-", 1)
            return list(range(int(a.strip()), int(b.strip()) + 1))
        return [int(value)]
    if isinstance(value, list):
        out: List[int] = []
        for item in value:
            out.extend(parse_run_ids(item))
        return out
    raise RuntimeError(f"Bad run_ids spec: {value!r}")


def load_config(yaml_path: Path) -> RDFConfig:
    data = yaml.safe_load(yaml_path.read_text())
    run_ids_val = data.get("run_ids")
    if not run_ids_val:
        raise RuntimeError("run_ids must be provided (e.g., [1,2,3])")
    run_ids = parse_run_ids(run_ids_val)

    rdf = data.get("rdf", {})
    if not rdf:
        raise RuntimeError("rdf block is required")

    return RDFConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        method_dir=str(data["method_dir"]),
        bench_tag=data.get("bench_tag"),
        run_ids=[int(x) for x in run_ids],
        cv_dir=str(data.get("cv_dir", "equil")),
        traj_name=str(data["traj_name"]),
        dftb_inp_name=data.get("dftb_inp_name"),
        parm_path=str(data["parm_path"]),
        dr=float(rdf["dr"]),
        r_max=float(rdf["r_max"]),
        mask1=str(rdf["mask1"]),
        mask2=rdf.get("mask2"),
        noimage=bool(rdf.get("noimage", False)),
        volume=bool(rdf.get("volume", False)),
        density=rdf.get("density"),
        dataset=rdf.get("dataset"),
        intrdf=rdf.get("intrdf"),
        rawrdf=rdf.get("rawrdf"),
    )


def system_base_dir(cfg: RDFConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def run_dir(cfg: RDFConfig, repo_root: Path, run_id: int) -> Path:
    base = system_base_dir(cfg, repo_root) / cfg.method_dir
    if cfg.bench_tag:
        base = base / cfg.bench_tag
    return base / f"run-{run_id}" / cfg.cv_dir


def _read_box_from_dftb_inp(dftb_inp: Path) -> Tuple[float, float, float]:
    tvs: List[Tuple[float, float, float]] = []
    lines = dftb_inp.read_text().splitlines()
    for line in lines:
        if not line.startswith("TV"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            tvs.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if len(tvs) != 3:
        raise RuntimeError(f"Expected 3 TV lines in {dftb_inp}, got {len(tvs)}")
    return tvs[0][0], tvs[1][1], tvs[2][2]


def convert_dftb_xyz_to_cpptraj(
    src: Path, dst: Path, dftb_inp: Path
) -> None:
    l1, l2, l3 = _read_box_from_dftb_inp(dftb_inp)
    comment = (
        f"Box X: {l1:.3f} 0.000 0.000 "
        f"Y: 0.000 {l2:.3f} 0.000 "
        f"Z: 0.000 0.000 {l3:.3f}"
    )
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        while True:
            line = fin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError as exc:
                raise RuntimeError(f"Invalid XYZ atom-count line: {line!r}") from exc
            _ = fin.readline()
            coords: List[Tuple[str, str, str, str]] = []
            for _ in range(n_atoms):
                atom_line = fin.readline()
                if not atom_line:
                    raise RuntimeError("Unexpected EOF while reading XYZ frame")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise RuntimeError(f"Invalid XYZ atom line: {atom_line.strip()}")
                coords.append((parts[0], parts[1], parts[2], parts[3]))
            _write_simple_frame(fout, n_atoms, comment, coords)


def resolve_parm_path(cfg: RDFConfig, repo_root: Path) -> Path:
    p = Path(cfg.parm_path)
    if p.is_absolute():
        return p
    return repo_root / p


def _trajin_line(traj: Path) -> str:
    return f"trajin {traj}"


def _sanitize_mask(mask: str) -> str:
    out = []
    prev_us = False
    for ch in mask:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s = "".join(out).strip("_")
    return s or "mask"


def _parse_lattice_from_comment(comment: str) -> Optional[Tuple[float, float, float]]:
    if "Lattice=" not in comment:
        return None
    try:
        start = comment.index('Lattice="') + len('Lattice="')
        end = comment.index('"', start)
        parts = comment[start:end].split()
        if len(parts) != 9:
            return None
        vals = [float(x) for x in parts]
        return vals[0], vals[4], vals[8]
    except Exception:
        return None


def _write_simple_frame(
    out: TextIO, n_atoms: int, comment: str, coords: List[Tuple[str, str, str, str]]
) -> None:
    out.write(f"{n_atoms}\n")
    out.write(comment + "\n")
    for sym, x, y, z in coords:
        out.write(f"{sym} {x} {y} {z}\n")


def convert_ase_xyz_to_simple_xyz(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        while True:
            line = fin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError as exc:
                raise RuntimeError(f"Invalid XYZ atom-count line: {line!r}") from exc
            comment = fin.readline()
            if not comment:
                break
            comment = comment.strip()
            lattice = _parse_lattice_from_comment(comment)
            if lattice is not None:
                l1, l2, l3 = lattice
                comment = (
                    f"Box X: {l1:.3f} 0.000 0.000 "
                    f"Y: 0.000 {l2:.3f} 0.000 "
                    f"Z: 0.000 0.000 {l3:.3f}"
                )
            coords: List[Tuple[str, str, str, str]] = []
            for _ in range(n_atoms):
                atom_line = fin.readline()
                if not atom_line:
                    raise RuntimeError("Unexpected EOF while reading XYZ frame")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise RuntimeError(f"Invalid XYZ atom line: {atom_line.strip()}")
                coords.append((parts[0], parts[1], parts[2], parts[3]))
            _write_simple_frame(fout, n_atoms, comment, coords)


def write_cpptraj_in(cfg: RDFConfig, parm: Path, traj: Path, out_dir: Path) -> Path:
    m1 = _sanitize_mask(cfg.mask1)
    m2 = _sanitize_mask(cfg.mask2) if cfg.mask2 else "all"
    out_dat = out_dir / f"rdf_{m1}_{m2}.dat"
    parts: List[str] = [
        f"parm {parm}",
        _trajin_line(traj),
    ]

    radial_parts = [
        "radial",
        f"out {out_dat.name}",
        f"{cfg.dr}",
        f"{cfg.r_max}",
        cfg.mask1,
    ]
    if cfg.mask2:
        radial_parts.append(cfg.mask2)
    if cfg.noimage:
        radial_parts.append("noimage")
    if cfg.volume:
        radial_parts.append("volume")
    if cfg.density is not None:
        radial_parts.append(f"density {cfg.density}")
    if cfg.dataset:
        radial_parts.append(cfg.dataset)
    intrdf_name = cfg.intrdf
    if intrdf_name is None:
        intrdf_name = f"rdf_{m1}_{m2}.int.dat"
    radial_parts.append(f"intrdf {intrdf_name}")
    if cfg.rawrdf:
        radial_parts.append(f"rawrdf {cfg.rawrdf}")

    parts.append(" ".join(radial_parts))
    parts.append("run")
    parts.append("")

    cppin = out_dir / "cpptraj.in"
    cppin.write_text("\n".join(parts))
    return cppin


def run_cpptraj(cpptraj_in: Path) -> None:
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


def run_rdf(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    parm = resolve_parm_path(cfg, repo_root)
    if not parm.exists():
        raise RuntimeError(f"Missing parm file: {parm}")

    for run_id in cfg.run_ids:
        rdir = run_dir(cfg, repo_root, run_id)
        traj = rdir / cfg.traj_name
        if not traj.exists() or traj.stat().st_size == 0:
            print(f"WARN: missing/empty traj: {traj} (skipping run-{run_id})")
            continue

        out_dir = rdir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.yaml").write_text(yaml_path.read_text())

        traj_for_cpp = traj
        needs_convert = traj.suffix.lower() == ".xyz"
        is_dftb_traject = cfg.method_dir.lower() == "dftb" and traj.name == "traject"
        if needs_convert or is_dftb_traject:
            converted = out_dir / "traj.cpptraj.xyz"
            if cfg.dftb_inp_name and cfg.method_dir.lower() == "dftb":
                dftb_inp = rdir / cfg.dftb_inp_name
                if not dftb_inp.exists():
                    raise RuntimeError(f"Missing dftb.inp: {dftb_inp}")
                convert_dftb_xyz_to_cpptraj(traj, converted, dftb_inp)
            else:
                convert_ase_xyz_to_simple_xyz(traj, converted)
            traj_for_cpp = converted

        cppin = write_cpptraj_in(cfg, parm, traj_for_cpp, out_dir)
        run_cpptraj(cppin)
        m1 = _sanitize_mask(cfg.mask1)
        m2 = _sanitize_mask(cfg.mask2) if cfg.mask2 else "all"
        rdf_path = out_dir / f"rdf_{m1}_{m2}.dat"
        print(f"OK: wrote {rdf_path}")
        intrdf_name = cfg.intrdf or f"rdf_{m1}_{m2}.int.dat"
        print(f"OK: wrote {out_dir / intrdf_name}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 rdf.py configs/SYS/analysis/rdf.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_rdf(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
