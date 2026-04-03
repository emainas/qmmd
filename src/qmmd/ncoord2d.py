#!/usr/bin/env python3

import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union


@dataclass(frozen=True)
class SlurmJobConfig:
    nodes: int
    ntasks: int
    cpus_per_task: int


@dataclass(frozen=True)
class SlurmConfig:
    job: SlurmJobConfig


@dataclass(frozen=True)
class GroupSpec:
    mode: str
    indices: Optional[Union[int, List[Union[int, str]]]] = None
    range: Optional[Union[str, List[int], Tuple[int, int]]] = None
    solute_end: Optional[int] = None


@dataclass(frozen=True)
class CVSpec:
    cv_title: str
    gausswidth: float
    nexp: int
    mexp: int
    refdist: float
    method: str
    norm: str
    grid_min: float
    grid_max: float
    grid_step: float
    group1: GroupSpec
    group2: GroupSpec


@dataclass(frozen=True)
class Ncoord2DConfig:
    system: str
    buffer: float
    prefix: str
    dftb_dirname: str
    replica_dirname: str
    cv_dirname: str
    run_ids: List[int]
    cv1: CVSpec
    cv2: CVSpec
    slurm: Optional[SlurmConfig] = None
    bench_tag: Optional[str] = None


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def bench_tag_from_slurm(slurm_cfg: SlurmConfig) -> str:
    job = slurm_cfg.job
    return f"N{job.nodes}T{job.ntasks}C{job.cpus_per_task}"


def system_base_dir(cfg: Ncoord2DConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def dftb_root_dir(cfg: Ncoord2DConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.dftb_dirname


def run_dir(cfg: Ncoord2DConfig, repo_root: Path, run_id: int) -> Path:
    bench = cfg.bench_tag or (bench_tag_from_slurm(cfg.slurm) if cfg.slurm else None)
    if not bench:
        raise RuntimeError("bench_tag not provided and slurm is missing; cannot infer bench folder")
    return dftb_root_dir(cfg, repo_root) / bench / f"run-{run_id}"


def load_group(data: Dict[str, Any]) -> GroupSpec:
    return GroupSpec(
        mode=str(data["mode"]),
        indices=data.get("indices"),
        range=data.get("range"),
        solute_end=data.get("solute_end"),
    )


def load_cv(data: Dict[str, Any]) -> CVSpec:
    return CVSpec(
        cv_title=str(data["cv_title"]),
        gausswidth=float(data["gausswidth"]),
        nexp=int(data["nexp"]),
        mexp=int(data["mexp"]),
        refdist=float(data["refdist"]),
        method=str(data["method"]),
        norm=str(data["norm"]),
        grid_min=float(data["grid_min"]),
        grid_max=float(data["grid_max"]),
        grid_step=float(data["grid_step"]),
        group1=load_group(data["group1"]),
        group2=load_group(data["group2"]),
    )


def parse_run_ids(value: Any) -> List[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        if "-" in value:
            a, b = parse_range(value)
            return list(range(a, b + 1))
        return [int(value)]
    if isinstance(value, list):
        out: List[int] = []
        for item in value:
            out.extend(parse_run_ids(item))
        return out
    raise RuntimeError(f"Bad run_ids spec: {value!r}")


def load_config(yaml_path: Path) -> Ncoord2DConfig:
    data = yaml.safe_load(yaml_path.read_text())

    slurm_cfg = None
    if data.get("slurm") is not None:
        slurm_cfg = SlurmConfig(job=SlurmJobConfig(**data["slurm"]["job"]))

    run_ids_val = data.get("run_ids")
    if not run_ids_val:
        raise RuntimeError("run_ids must be provided (e.g., [1,2,3])")
    run_ids = parse_run_ids(run_ids_val)

    return Ncoord2DConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        dftb_dirname=data.get("dftb_dirname", "dftb"),
        replica_dirname=data.get("replica_dirname", "equil"),
        cv_dirname=data["cv_dirname"],
        run_ids=[int(x) for x in run_ids],
        cv1=load_cv(data["cv1"]),
        cv2=load_cv(data["cv2"]),
        slurm=slurm_cfg,
        bench_tag=data.get("bench_tag"),
    )


def parse_nat_line(lines: List[str]) -> Tuple[int, int]:
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and all(p.lstrip("+-").isdigit() for p in parts):
            return i, int(parts[0])
    raise RuntimeError("Could not find natoms line in dftb.inp")


def read_symbols_from_dftb_inp(dftb_inp: Path) -> List[str]:
    lines = dftb_inp.read_text().splitlines()
    nat_idx, natoms = parse_nat_line(lines)
    start = nat_idx + 1
    end = start + natoms
    if len(lines) < end:
        raise RuntimeError(f"dftb.inp too short for natoms={natoms}: {dftb_inp}")
    symbols: List[str] = []
    for i, line in enumerate(lines[start:end], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"Bad coord line {i} in {dftb_inp}: {line!r}")
        symbols.append(parts[0])
    if len(symbols) != natoms:
        raise RuntimeError(f"Expected {natoms} coords, read {len(symbols)} in {dftb_inp}")
    return symbols


def parse_range(value: Union[str, List[int], Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str) and "-" in value:
        a, b = value.split("-", 1)
        return int(a.strip()), int(b.strip())
    raise RuntimeError(f"Bad range spec: {value!r}")


def parse_indices(value: Union[int, List[Union[int, str]]]) -> List[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        raise RuntimeError(f"Bad indices spec: {value!r}")
    out: List[int] = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and "-" in item:
            a, b = parse_range(item)
            out.extend(list(range(a, b + 1)))
        else:
            raise RuntimeError(f"Bad index item: {item!r}")
    return out


def select_group_indices(group: GroupSpec, symbols: List[str]) -> List[int]:
    mode = group.mode.lower()
    if mode == "indices":
        if group.indices is None:
            raise RuntimeError("group.indices is required for mode=indices")
        return parse_indices(group.indices)
    if mode == "range":
        if group.range is None:
            raise RuntimeError("group.range is required for mode=range")
        start, end = parse_range(group.range)
        return list(range(start, end + 1))
    if mode in ("all_water_h", "all_h"):
        if group.solute_end is None:
            raise RuntimeError("group.solute_end is required for mode=all_water_H")
        start = int(group.solute_end) + 1
        out = []
        for idx, sym in enumerate(symbols, start=1):
            if idx >= start and sym.upper() == "H":
                out.append(idx)
        if not out:
            raise RuntimeError("mode=all_water_H found no hydrogens in solvent region")
        if group.indices is None:
            return out
        extra = parse_indices(group.indices)
        merged: List[int] = []
        seen = set()
        for idx in extra + out:
            if idx in seen:
                continue
            seen.add(idx)
            merged.append(idx)
        return merged
    raise RuntimeError(f"Unknown group.mode: {group.mode!r}")


def write_metacv_block(cv: CVSpec, group1: List[int], group2: List[int]) -> str:
    header = (
        f"{cv.cv_title} {cv.gausswidth} {len(group1)} {len(group2)} "
        f"{cv.nexp} {cv.mexp} {cv.refdist} {cv.method} {cv.norm} "
        f"{cv.grid_min} {cv.grid_max} {cv.grid_step}"
    )
    return header + "\n" + " ".join(map(str, group1)) + "\n" + " ".join(map(str, group2)) + "\n"


def run_2dncoord(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)

    for run_id in cfg.run_ids:
        rdir = run_dir(cfg, repo_root, run_id)
        equil_dir = rdir / cfg.replica_dirname
        dftb_inp = equil_dir / "dftb.inp"
        if not dftb_inp.exists():
            raise RuntimeError(f"Missing dftb.inp: {dftb_inp}")

        cv_dir = rdir / cfg.cv_dirname
        if cv_dir.exists():
            print(f"SKIP: {cv_dir} already exists; not touching")
            continue
        cv_dir.mkdir(parents=True, exist_ok=True)

        symbols = read_symbols_from_dftb_inp(dftb_inp)
        g1_1 = select_group_indices(cfg.cv1.group1, symbols)
        g2_1 = select_group_indices(cfg.cv1.group2, symbols)
        g1_2 = select_group_indices(cfg.cv2.group1, symbols)
        g2_2 = select_group_indices(cfg.cv2.group2, symbols)

        metacv = cv_dir / "metacv.dat"
        text = write_metacv_block(cfg.cv1, g1_1, g2_1) + write_metacv_block(cfg.cv2, g1_2, g2_2)
        metacv.write_text(text)
        (cv_dir / "spec.yaml").write_text(yaml_path.read_text())
        print(f"OK: wrote {metacv} (run {run_id})")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 ncoord2d.py configs/SYS/ncoord/2dncoord.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_2dncoord(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
