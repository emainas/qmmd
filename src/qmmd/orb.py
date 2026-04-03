#!/usr/bin/env python3

from __future__ import annotations

import sys
import yaml
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class SlurmJobConfig:
    name: str
    partition: str
    nodes: int
    ntasks: int
    cpus_per_task: int
    mem: str
    time: str
    stdout: str
    stderr: str
    gres: Optional[str] = None
    qos: Optional[str] = None
    extra: Optional[List[str]] = None


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    job: SlurmJobConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    module: str
    env: Dict[str, Any]
    conda_env: Optional[str] = None
    python: str = "python"
    strict_mode: bool = True


@dataclass(frozen=True, slots=True)
class OrbPrepConfig:
    system: str
    buffer: float
    prefix: str
    orb_dirname: str
    input_source: str
    input_name: str
    salt_dirname: str
    density_dirname: str
    runtime: RuntimeConfig
    slurm: Optional[SlurmConfig] = None
    replicas: int = 1
    append: bool = False
    replica_dirname: str = "equil"
    bench_tag: Optional[str] = None


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def system_base_dir(cfg: OrbPrepConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def orb_root_dir(cfg: OrbPrepConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.orb_dirname


def bench_tag_from_slurm(slurm_cfg: SlurmConfig) -> str:
    job = slurm_cfg.job
    return f"N{job.nodes}T{job.ntasks}C{job.cpus_per_task}"


def out_dir(cfg: OrbPrepConfig, repo_root: Path) -> Path:
    if cfg.bench_tag:
        tag = cfg.bench_tag
    elif cfg.slurm is not None:
        tag = bench_tag_from_slurm(cfg.slurm)
    else:
        tag = "N1T1C1"
    return orb_root_dir(cfg, repo_root) / tag


def load_config(yaml_path: Path) -> OrbPrepConfig:
    data = yaml.safe_load(yaml_path.read_text())

    runtime_cfg = RuntimeConfig(**data["runtime"])

    slurm_cfg = None
    if data.get("slurm") is not None:
        slurm_cfg = SlurmConfig(job=SlurmJobConfig(**data["slurm"]["job"]))

    return OrbPrepConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        orb_dirname=data.get("orb_dirname", "orb"),
        input_source=str(data.get("input_source", "salt")),
        input_name=str(data.get("input_name", "ready.xyz")),
        salt_dirname=data.get("salt_dirname", "salt"),
        density_dirname=data.get("density_dirname", "density"),
        runtime=runtime_cfg,
        slurm=slurm_cfg,
        replicas=int(data.get("replicas", 1)),
        append=bool(data.get("append", False)),
        replica_dirname=data.get("replica_dirname", "equil"),
        bench_tag=data.get("bench_tag"),
    )


def input_dir(cfg: OrbPrepConfig, repo_root: Path) -> Path:
    base = system_base_dir(cfg, repo_root)
    src = cfg.input_source.lower()
    if src == "density":
        return base / cfg.density_dirname
    if src == "salt":
        return base / cfg.salt_dirname
    raise RuntimeError(f"Unknown input_source: {cfg.input_source!r} (expected 'salt' or 'density')")


def validate_inputs(cfg: OrbPrepConfig, repo_root: Path) -> None:
    src_dir = input_dir(cfg, repo_root)
    xyz = src_dir / cfg.input_name
    if not xyz.exists() or xyz.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty required input XYZ: {xyz}")


def write_run_sh(cfg: OrbPrepConfig, out_dir_path: Path) -> Path:
    sh_path = out_dir_path / "run.sh"
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    env_lines = "\n".join([f'export {k}="{v}"' for k, v in cfg.runtime.env.items()])

    cmd = f'{cfg.runtime.python} main.py'
    if cfg.runtime.conda_env:
        cmd = f'conda run -n {cfg.runtime.conda_env} {cmd}'

    text = f"""\
#!/usr/bin/env bash
{strict}

module purge
module load {cfg.runtime.module}

{env_lines}

{cmd}
"""
    sh_path.write_text(text)
    sh_path.chmod(0o755)
    return sh_path


def write_main_py(out_dir_path: Path) -> Path:
    py_path = out_dir_path / "main.py"
    text = """\
#!/usr/bin/env python3

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
        r"Box X:\\s*([0-9.+-]+)\\s+0\\.000\\s+0\\.000\\s+Y:\\s*0\\.000\\s+([0-9.+-]+)\\s+0\\.000\\s+Z:\\s*0\\.000\\s+0\\.000\\s+([0-9.+-]+)",
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
        flog.write("# step time_ps Epot_eV Ekin_eV T_K\\n")

        def log_cb() -> None:
            if dyn.nsteps % cfg.output.log_stride != 0:
                return
            tps = dyn.nsteps * cfg.md.dt_fs * 1e-3
            epot = float(atoms.get_potential_energy())
            ekin = float(atoms.get_kinetic_energy())
            temp = float(atoms.get_temperature())
            flog.write(f"{dyn.nsteps} {tps:.6f} {epot:.8f} {ekin:.8f} {temp:.3f}\\n")
            flog.flush()

        def traj_cb() -> None:
            if dyn.nsteps % cfg.output.traj_stride == 0:
                if atoms.get_cell().volume > 0.0:
                    atoms.wrap()
                write(traj_path, atoms, append=True)

        dyn.attach(log_cb, interval=1)
        dyn.attach(traj_cb, interval=1)
        dyn.run(n_equil)

    print(f"OK: wrote {mdlog_path}")
    print(f"OK: wrote {traj_path}")


def main() -> None:
    run_equil(Path("spec.yaml").resolve())


if __name__ == "__main__":
    main()
"""
    py_path.write_text(text)
    py_path.chmod(0o755)
    return py_path


def _sbatch_line(value: str) -> str:
    if value.startswith("#SBATCH"):
        return value
    return f"#SBATCH {value}"


def write_slurm_sh(cfg: OrbPrepConfig, out_dir_path: Path) -> Optional[Path]:
    if cfg.slurm is None:
        return None

    job = cfg.slurm.job
    tag = bench_tag_from_slurm(cfg.slurm)
    sh_path = out_dir_path / "slurm.sh"

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job.name}-{tag}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --time={job.time}",
        f"#SBATCH --nodes={job.nodes}",
        f"#SBATCH --ntasks={job.ntasks}",
        f"#SBATCH --mem={job.mem}",
        f"#SBATCH --cpus-per-task={job.cpus_per_task}",
        f"#SBATCH --output={job.stdout}",
        f"#SBATCH --error={job.stderr}",
    ]
    if job.gres:
        lines.append(f"#SBATCH --gres={job.gres}")
    if job.qos:
        lines.append(f"#SBATCH --qos={job.qos}")
    if job.extra:
        for line in job.extra:
            lines.append(_sbatch_line(line))

    lines.append("")
    lines.append("bash run.sh")
    lines.append("")

    sh_path.write_text("\n".join(lines))
    sh_path.chmod(0o755)
    return sh_path


def submit_slurm(slurm_sh: Path) -> None:
    subprocess.run(["sbatch", slurm_sh.name], cwd=slurm_sh.parent, check=True)


def find_existing_run_indices(bench_dir: Path) -> List[int]:
    if not bench_dir.exists():
        return []
    indices: List[int] = []
    for p in bench_dir.iterdir():
        if p.is_dir() and p.name.startswith("run-"):
            try:
                indices.append(int(p.name.split("-", 1)[1]))
            except ValueError:
                continue
    return sorted(indices)


def run_equil_dir(bench_dir: Path, run_index: int, replica_dirname: str) -> Path:
    return bench_dir / f"run-{run_index}" / replica_dirname


def matching_run_dirs(bench_dir: Path, replica_dirname: str, yaml_text: str) -> List[Path]:
    matches: List[Path] = []
    if not bench_dir.exists():
        return matches
    for run_dir in bench_dir.iterdir():
        if not (run_dir.is_dir() and run_dir.name.startswith("run-")):
            continue
        odir = run_dir / replica_dirname
        spec = odir / "spec.yaml"
        if spec.exists() and spec.read_text() == yaml_text:
            matches.append(odir)
    return matches


def write_single_run(cfg: OrbPrepConfig, repo_root: Path, out_path: Path, yaml_text: str) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "spec.yaml").write_text(yaml_text)

    write_main_py(out_path)
    run_sh = write_run_sh(cfg, out_path)
    slurm_sh = write_slurm_sh(cfg, out_path)

    print(f"OK: wrote {run_sh.name}" + (f", {slurm_sh.name}" if slurm_sh else "") + f" in {out_path}")


def run_orb_prep(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    validate_inputs(cfg, repo_root)

    bench_dir = out_dir(cfg, repo_root)
    yaml_text = yaml_path.read_text()

    if cfg.replicas <= 1:
        odir = run_equil_dir(bench_dir, 1, cfg.replica_dirname)
        write_single_run(cfg, repo_root, odir, yaml_text)
        return

    bench_dir.mkdir(parents=True, exist_ok=True)
    existing = find_existing_run_indices(bench_dir)
    start_index = (max(existing) + 1) if (cfg.append and existing) else 1

    skipped: List[Path] = []
    for i in range(start_index, start_index + cfg.replicas):
        odir = run_equil_dir(bench_dir, i, cfg.replica_dirname)
        if odir.exists():
            print(f"SKIP: {odir} already exists; not touching or submitting")
            skipped.append(odir)
            continue
        write_single_run(cfg, repo_root, odir, yaml_text)

    if skipped:
        print(f"SKIP: {len(skipped)} existing equil dirs (not touched)")


def run_orb_submit(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    bench_dir = out_dir(cfg, repo_root)
    yaml_text = yaml_path.read_text()

    if cfg.slurm is None:
        print("NOTE: slurm config not provided; no submissions made")
        return

    if cfg.replicas <= 1:
        odir = run_equil_dir(bench_dir, 1, cfg.replica_dirname)
        spec = odir / "spec.yaml"
        if not spec.exists() or spec.read_text() != yaml_text:
            print(f"SKIP: {odir} spec.yaml does not match config (not submitting)")
            return
        slurm_sh = odir / "slurm.sh"
        if not slurm_sh.exists():
            print(f"SKIP: missing {slurm_sh} (not submitting)")
            return
        resp = input(f"About to submit 1 job: {odir}. Proceed? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Cancelled by user.")
            return
        print(f"Submitting job via sbatch for {odir}...")
        submit_slurm(slurm_sh)
        print("OK: job submitted")
        return

    targets = matching_run_dirs(bench_dir, cfg.replica_dirname, yaml_text)
    if not targets:
        print("NOTE: no matching run dirs found for this config; nothing submitted")
        return

    print("Will submit the following run dirs:")
    for odir in targets:
        print(f"  - {odir}")
    resp = input(f"Proceed to submit {len(targets)} jobs? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        print("Cancelled by user.")
        return

    for odir in targets:
        slurm_sh = odir / "slurm.sh"
        if not slurm_sh.exists():
            print(f"SKIP: missing {slurm_sh} (not submitting)")
            continue
        print(f"Submitting job via sbatch for {odir}...")
        submit_slurm(slurm_sh)
        print("OK: job submitted")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 orb.py configs/SYS/orb/orb.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_orb_prep(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
