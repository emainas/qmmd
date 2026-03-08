#!/usr/bin/env python3

import sys
import yaml
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

from qmmd.dftb import (
    ElementConfig,
    DFTBConfig,
    render_element_blocks,
    render_coords_block,
)


@dataclass(frozen=True, slots=True)
class SlurmJobConfig:
    name: str
    partition: str
    nodes: int
    ntasks: int
    mem: str
    cpus_per_task: int
    time: str
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    job: SlurmJobConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    module: str
    executable: str
    mpirun_np: int
    env: Dict[str, Any]
    strict_mode: bool = True


@dataclass(frozen=True, slots=True)
class MetaConfig:
    system: str
    buffer: float
    prefix: str
    dftb_dirname: str
    replica_dirname: str
    cv_dirname: str
    run_ids: List[int]
    restart_name: str
    dftb: DFTBConfig
    runtime: RuntimeConfig
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


def system_base_dir(cfg: MetaConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def dftb_root_dir(cfg: MetaConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.dftb_dirname


def run_dir(cfg: MetaConfig, repo_root: Path, run_id: int) -> Path:
    bench = cfg.bench_tag or (bench_tag_from_slurm(cfg.slurm) if cfg.slurm else None)
    if not bench:
        raise RuntimeError("bench_tag not provided and slurm is missing; cannot infer bench folder")
    return dftb_root_dir(cfg, repo_root) / bench / f"run-{run_id}"


def repo_params_dir(cfg: MetaConfig, repo_root: Path) -> Path:
    return repo_root / cfg.dftb.params_dir


def stage_params_dir(out_dir_path: Path) -> Path:
    return out_dir_path / "params"


def stage_skf_files(repo_params: Path, out_dir_path: Path, elements: List[str]) -> None:
    dst = stage_params_dir(out_dir_path)
    dst.mkdir(parents=True, exist_ok=True)
    for a in elements:
        for b in elements:
            src = repo_params / f"{a}-{b}.skf"
            if not src.exists():
                raise RuntimeError(f"Missing SKF file: {src}")
            link = dst / src.name
            if link.exists():
                continue
            try:
                link.symlink_to(src)
            except OSError:
                shutil.copy2(src, link)


def load_config(yaml_path: Path) -> MetaConfig:
    data = yaml.safe_load(yaml_path.read_text())

    dftb_data = data["dftb"]
    elems = [ElementConfig(**e) for e in dftb_data["elements"]]
    dftb_cfg = DFTBConfig(
        title=dftb_data["title"],
        header_lines=list(dftb_data["header_lines"]),
        charge=int(dftb_data.get("charge", 0)),
        multiplicity=int(dftb_data.get("multiplicity", 1)),
        elements=elems,
        params_dir=dftb_data.get("params_dir", "params"),
    )

    runtime_cfg = RuntimeConfig(**data["runtime"])

    slurm_cfg = None
    if data.get("slurm") is not None:
        slurm_cfg = SlurmConfig(job=SlurmJobConfig(**data["slurm"]["job"]))

    run_ids_val = data.get("run_ids")
    if not run_ids_val:
        raise RuntimeError("run_ids must be provided (e.g., [1,2,3])")
    run_ids = parse_run_ids(run_ids_val)

    return MetaConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        dftb_dirname=data.get("dftb_dirname", "dftb"),
        replica_dirname=data.get("replica_dirname", "equil"),
        cv_dirname=data["cv_dirname"],
        run_ids=run_ids,
        restart_name=data.get("restart_name", "restart"),
        dftb=dftb_cfg,
        runtime=runtime_cfg,
        slurm=slurm_cfg,
        bench_tag=data.get("bench_tag"),
    )


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


def apply_seed_to_header_lines(header_lines: List[str], seed: Optional[int]) -> List[str]:
    if seed is None:
        return header_lines
    updated: List[str] = []
    replaced = False
    for line in header_lines:
        if "RANDOMSEED=0" in line:
            updated.append(line.replace("RANDOMSEED=0", f"RANDOMSEED={seed}"))
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        return header_lines
    return updated


def parse_nat_line(lines: List[str]) -> tuple[int, int]:
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and all(p.lstrip("+-").isdigit() for p in parts):
            return i, int(parts[0])
    raise RuntimeError("Could not find natoms line in dftb.inp")


def read_dftb_inp_coords(dftb_inp: Path) -> tuple[int, List[tuple[float, float, float]], List[tuple[str, float, float, float]]]:
    lines = dftb_inp.read_text().splitlines()
    nat_idx, natoms = parse_nat_line(lines)
    start = nat_idx + 1
    end = start + natoms
    if len(lines) < end:
        raise RuntimeError(f"dftb.inp too short for natoms={natoms}: {dftb_inp}")

    coords: List[tuple[str, float, float, float]] = []
    for i, line in enumerate(lines[start:end], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"Bad coord line {i} in {dftb_inp}: {line!r}")
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as e:
            raise RuntimeError(f"Bad XYZ floats on line {i} in {dftb_inp}: {line!r}") from e
        coords.append((sym, x, y, z))

    tvs: List[tuple[float, float, float]] = []
    for line in lines[end:]:
        if not line.startswith("TV"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise RuntimeError(f"Bad TV line in {dftb_inp}: {line!r}")
        try:
            tvs.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError as e:
            raise RuntimeError(f"Bad TV floats in {dftb_inp}: {line!r}") from e
    if not tvs:
        raise RuntimeError(f"No TV vectors found in {dftb_inp}")

    return natoms, tvs, coords


def write_dftb_inp(
    cfg: MetaConfig,
    repo_root: Path,
    out_dir_path: Path,
    equil_dftb_inp: Path,
    seed: Optional[int],
) -> Path:
    if not equil_dftb_inp.exists():
        raise RuntimeError(f"Missing equil dftb.inp: {equil_dftb_inp}")

    natoms, tvs, coords = read_dftb_inp_coords(equil_dftb_inp)
    xyz_syms = list(dict.fromkeys([c[0] for c in coords]).keys())

    repo_params = repo_params_dir(cfg, repo_root)
    if not repo_params.exists():
        raise RuntimeError(f"Missing repo params dir: {repo_params}")

    ordered = [e.symbol for e in cfg.dftb.elements]
    stage_skf_files(repo_params, out_dir_path, ordered)

    header_lines = apply_seed_to_header_lines(cfg.dftb.header_lines, seed)
    header = "\n".join(header_lines).rstrip() + "\n"
    elem_block = render_element_blocks(cfg.dftb, xyz_syms)

    inp_path = out_dir_path / "dftb.inp"
    text = (
        header
        + "\n"
        + f"{cfg.dftb.title}\n"
        + "\n"
        + elem_block
        + "\n"
        + f"{natoms:5d}  {cfg.dftb.charge:d}  {cfg.dftb.multiplicity:d}\n"
        + render_coords_block(coords)
    )
    for vx, vy, vz in tvs:
        text += f"TV{vx:>18.8f}{vy:>15.8f}{vz:>15.8f}\n"
    text += "\n"
    inp_path.write_text(text)
    return inp_path


def write_run_sh(cfg: MetaConfig, out_dir_path: Path) -> Path:
    sh_path = out_dir_path / "run.sh"
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    env_lines = "\n".join([f'export {k}="{v}"' for k, v in cfg.runtime.env.items()])

    text = f"""\
#!/usr/bin/env bash
{strict}

module purge
module load {cfg.runtime.module}

{env_lines}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export OMP_STACKSIZE=1G

ulimit -s unlimited

mpirun -np {cfg.runtime.mpirun_np} "{cfg.runtime.executable}"
"""
    sh_path.write_text(text)
    sh_path.chmod(0o755)
    return sh_path


def write_slurm_sh(cfg: MetaConfig, out_dir_path: Path) -> Optional[Path]:
    if cfg.slurm is None:
        return None

    job = cfg.slurm.job
    tag = bench_tag_from_slurm(cfg.slurm)
    sh_path = out_dir_path / "slurm.sh"

    text = f"""\
#!/usr/bin/env bash
#SBATCH --job-name={job.name}-{tag}
#SBATCH --partition={job.partition}
#SBATCH --time={job.time}
#SBATCH --nodes={job.nodes}
#SBATCH --ntasks={job.ntasks}
#SBATCH --mem={job.mem}
#SBATCH --cpus-per-task={job.cpus_per_task}
#SBATCH --output={job.stdout}
#SBATCH --error={job.stderr}

bash run.sh
"""
    sh_path.write_text(text)
    sh_path.chmod(0o755)
    return sh_path


def submit_slurm(slurm_sh: Path) -> None:
    import subprocess
    subprocess.run(["sbatch", slurm_sh.name], cwd=slurm_sh.parent, check=True)


def run_meta_prep(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    yaml_text = yaml_path.read_text()

    for run_id in cfg.run_ids:
        rdir = run_dir(cfg, repo_root, run_id)
        equil_dir = rdir / cfg.replica_dirname
        cv_dir = rdir / cfg.cv_dirname

        if not cv_dir.exists():
            raise RuntimeError(f"Missing cv dir (run {run_id}): {cv_dir}")

        if (cv_dir / "dftb.inp").exists():
            print(f"SKIP: {cv_dir} already has dftb.inp; not touching")
            continue

        metacv = cv_dir / "metacv.dat"
        if not metacv.exists():
            raise RuntimeError(f"Missing metacv.dat in {cv_dir}")

        restart_src = equil_dir / cfg.restart_name
        if not restart_src.exists():
            raise RuntimeError(f"Missing restart: {restart_src}")

        seed = secrets.randbelow(2**31 - 1) + 1
        write_dftb_inp(cfg, repo_root, cv_dir, equil_dir / "dftb.inp", seed=seed)
        write_run_sh(cfg, cv_dir)
        write_slurm_sh(cfg, cv_dir)
        shutil.copy2(restart_src, cv_dir / cfg.restart_name)
        (cv_dir / "meta_spec.yaml").write_text(yaml_text)

        print(f"OK: wrote meta inputs in {cv_dir} (run {run_id})")


def run_meta_submit(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    yaml_text = yaml_path.read_text()

    if cfg.slurm is None:
        print("NOTE: slurm config not provided; no submissions made")
        return

    targets: List[Path] = []
    for run_id in cfg.run_ids:
        rdir = run_dir(cfg, repo_root, run_id)
        cv_dir = rdir / cfg.cv_dirname
        spec = cv_dir / "meta_spec.yaml"
        if not spec.exists() or spec.read_text() != yaml_text:
            print(f"SKIP: {cv_dir} meta_spec.yaml does not match config (not submitting)")
            continue
        slurm_sh = cv_dir / "slurm.sh"
        if not slurm_sh.exists():
            print(f"SKIP: missing {slurm_sh} (not submitting)")
            continue
        targets.append(cv_dir)

    if not targets:
        print("NOTE: no matching meta dirs found; nothing submitted")
        return

    print("Will submit the following meta dirs:")
    for t in targets:
        print(f"  - {t}")
    resp = input(f"Proceed to submit {len(targets)} jobs? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        print("Cancelled by user.")
        return

    for cv_dir in targets:
        slurm_sh = cv_dir / "slurm.sh"
        print(f"Submitting job via sbatch for {cv_dir}...")
        submit_slurm(slurm_sh)
        print("OK: job submitted")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 meta.py configs/SYS/meta/meta.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_meta_prep(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
