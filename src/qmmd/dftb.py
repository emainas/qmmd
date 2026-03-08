#!/usr/bin/env python3

import sys
import yaml
import shutil
import subprocess
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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
class ElementConfig:
    symbol: str
    lmax: int
    hubbard: float


@dataclass(frozen=True, slots=True)
class DFTBConfig:
    title: str
    header_lines: List[str]
    charge: int
    multiplicity: int
    elements: List[ElementConfig]
    params_dir: str = "params"  # repo_root/params contains *.skf directly


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    system: str
    buffer: float
    prefix: str

    dftb_dirname: str
    salt_dirname: str

    dftb: DFTBConfig
    runtime: RuntimeConfig
    slurm: Optional[SlurmConfig] = None
    replicas: int = 1
    append: bool = False
    replica_dirname: str = "equil"

def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def system_base_dir(cfg: SimulationConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def salt_dir(cfg: SimulationConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.salt_dirname


def dftb_root_dir(cfg: SimulationConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.dftb_dirname


def bench_tag(job: SlurmJobConfig) -> str:
    return f"N{job.nodes}T{job.ntasks}C{job.cpus_per_task}"


def out_dir(cfg: SimulationConfig, repo_root: Path) -> Path:
    """
    dftb/
      N{nodes}T{tasks}C{cpus}/   <-- all files go here
    """
    if cfg.slurm is None:
        # local run: still create a deterministic bench folder
        # (choose 1 node / mpirun_np tasks / 1 cpu-per-task)
        nodes = 1
        ntasks = int(cfg.runtime.mpirun_np)
        cpus = 1
        tag = f"N{nodes}T{ntasks}C{cpus}"
    else:
        tag = bench_tag(cfg.slurm.job)

    return dftb_root_dir(cfg, repo_root) / tag


def repo_params_dir(cfg: SimulationConfig, repo_root: Path) -> Path:
    # repo_root/params contains *.skf directly
    return repo_root / cfg.dftb.params_dir


def stage_params_dir(out_dir_path: Path) -> Path:
    # Fortran-safe short paths: write params/*.skf relative to out_dir
    return out_dir_path / "params"


def load_config(yaml_path: Path) -> SimulationConfig:
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

    return SimulationConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        dftb_dirname=data.get("dftb_dirname", "dftb"),
        salt_dirname=data.get("salt_dirname", "salt"),
        replicas=int(data.get("replicas", 1)),
        append=bool(data.get("append", False)),
        replica_dirname=data.get("replica_dirname", "equil"),
        dftb=dftb_cfg,
        runtime=runtime_cfg,
        slurm=slurm_cfg,
    )


def parse_box_vectors_from_comment(comment: str) -> List[Tuple[float, float, float]]:
    """
    Parses:
      Conf 1. Box X: 15.612 0.000 0.000 Y: 0.000 13.439 0.000 Z: 0.000 0.000 12.950
    Returns [(x1,x2,x3),(y1,y2,y3),(z1,z2,z3)]
    """
    toks = comment.replace(":", ": ").split()

    def find_label(label: str) -> int:
        for i, t in enumerate(toks):
            if t == label:
                return i
        raise RuntimeError(f"Could not find '{label}' in XYZ comment:\n{comment}")

    ix = find_label("X:")
    iy = find_label("Y:")
    iz = find_label("Z:")

    def read3(start: int) -> Tuple[float, float, float]:
        try:
            return (float(toks[start]), float(toks[start + 1]), float(toks[start + 2]))
        except Exception as e:
            raise RuntimeError(f"Failed reading 3 floats after {toks[start-1]} in comment:\n{comment}") from e

    return [read3(ix + 1), read3(iy + 1), read3(iz + 1)]


def read_xyz_with_box(xyz_path: Path) -> Tuple[int, List[Tuple[float, float, float]], List[Tuple[str, float, float, float]]]:
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"XYZ too short: {xyz_path}")

    try:
        natoms = int(lines[0].strip())
    except ValueError as e:
        raise RuntimeError(f"Bad natoms line in {xyz_path}: {lines[0]!r}") from e

    tvs = parse_box_vectors_from_comment(lines[1].strip())

    coords: List[Tuple[str, float, float, float]] = []
    for i, line in enumerate(lines[2:], start=3):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"Bad XYZ line {i} in {xyz_path}: {line!r}")
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as e:
            raise RuntimeError(f"Bad XYZ floats on line {i} in {xyz_path}: {line!r}") from e
        coords.append((sym, x, y, z))

    if len(coords) != natoms:
        raise RuntimeError(f"XYZ natoms mismatch: header {natoms}, read {len(coords)} in {xyz_path}")

    return natoms, tvs, coords

def stage_skf_files(repo_params: Path, out_dir_path: Path, elements: List[str]) -> None:
    """
    Ensure out_dir/params contains required *.skf files (symlink preferred).
    The input will reference them as relative paths: params/A-B.skf
    """
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


def render_element_blocks(cfg: DFTBConfig, xyz_syms: List[str]) -> str:
    ordered = [e.symbol for e in cfg.elements]
    if set(ordered) != set(xyz_syms):
        raise RuntimeError(
            f"Element mismatch between YAML and ready.xyz.\n"
            f"YAML: {ordered}\n"
            f"XYZ:  {sorted(set(xyz_syms))}"
        )

    elem_map = {e.symbol: e for e in cfg.elements}

    lines: List[str] = []
    lines.append(str(len(ordered)))

    for sym in ordered:
        e = elem_map[sym]
        lines.append(f"{sym:<1s}   {e.lmax:d} {e.hubbard:.4f}")
        # RELATIVE paths (Fortran-safe)
        skfs = [f"params/{sym}-{other}.skf" for other in ordered]
        lines.append("  " + " ".join(skfs))

    return "\n".join(lines) + "\n"


def render_coords_block(coords: List[Tuple[str, float, float, float]]) -> str:
    # exactly one newline after the last coordinate line (no extra blank line)
    return "\n".join(f"{sym:<2s}{x:>15.8f}{y:>15.8f}{z:>15.8f}" for sym, x, y, z in coords) + "\n"


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


def write_dftb_inp(cfg: SimulationConfig, repo_root: Path, out_dir_path: Path, seed: Optional[int] = None) -> Path:
    sdir = salt_dir(cfg, repo_root)
    ready_xyz = sdir / "ready.xyz"
    if not ready_xyz.exists() or ready_xyz.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty ready.xyz: {ready_xyz}")

    natoms, tvs, coords = read_xyz_with_box(ready_xyz)
    xyz_syms = list(dict.fromkeys([c[0] for c in coords]).keys())  # unique symbols in appearance order

    repo_params = repo_params_dir(cfg, repo_root)
    if not repo_params.exists():
        raise RuntimeError(f"Missing repo params dir: {repo_params}")

    # Stage SKF files locally for short relative paths
    ordered = [e.symbol for e in cfg.dftb.elements]
    stage_skf_files(repo_params, out_dir_path, ordered)

    header_lines = apply_seed_to_header_lines(cfg.dftb.header_lines, seed)
    header = "\n".join(header_lines).rstrip() + "\n"
    elem_block = render_element_blocks(cfg.dftb, xyz_syms)

    inp_path = out_dir_path / "dftb.inp"

    # IMPORTANT: no extra blank line between last coordinate and first TV
    text = (
        header
        + "\n"
        + f"{cfg.dftb.title}\n"
        + "\n"
        + elem_block
        + "\n"
        + f"{natoms:5d}  {cfg.dftb.charge:d}  {cfg.dftb.multiplicity:d}\n"
        + render_coords_block(coords)  # ends with exactly one '\n'
    )

    for vx, vy, vz in tvs:
        text += f"TV{vx:>18.8f}{vy:>15.8f}{vz:>15.8f}\n"

    text += "\n"   # ← add this so that fortran input does not give EOF error
    inp_path.write_text(text)
    return inp_path


def write_run_sh(cfg: SimulationConfig, out_dir_path: Path) -> Path:
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


def write_slurm_sh(cfg: SimulationConfig, out_dir_path: Path) -> Optional[Path]:
    if cfg.slurm is None:
        return None

    job = cfg.slurm.job
    tag = bench_tag(job)
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
    subprocess.run(["sbatch", slurm_sh.name], cwd=slurm_sh.parent, check=True)


def run_local(run_sh: Path) -> None:
    subprocess.run(["bash", run_sh.name], cwd=run_sh.parent, check=True)


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


def validate_inputs(cfg: SimulationConfig, repo_root: Path) -> None:
    sdir = salt_dir(cfg, repo_root)
    ready_xyz = sdir / "ready.xyz"
    if not ready_xyz.exists() or ready_xyz.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty required input (from salt): {ready_xyz}")

    repo_params = repo_params_dir(cfg, repo_root)
    if not repo_params.exists():
        raise RuntimeError(f"Missing repo params dir: {repo_params}")


def write_single_run(cfg: SimulationConfig, repo_root: Path, out_path: Path, yaml_text: str, seed: Optional[int]) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "spec.yaml").write_text(yaml_text)

    inp = write_dftb_inp(cfg, repo_root, out_path, seed=seed)
    run_sh = write_run_sh(cfg, out_path)
    slurm_sh = write_slurm_sh(cfg, out_path)

    print(f"OK: wrote {inp.name}, {run_sh.name}" + (f", {slurm_sh.name}" if slurm_sh else "") + f" in {out_path}")
    print(f"OK: staged SKF files in {stage_params_dir(out_path)} (relative paths in dftb.inp)")


def run_dftb_prep(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    validate_inputs(cfg, repo_root)

    bench_dir = out_dir(cfg, repo_root)
    yaml_text = yaml_path.read_text()

    if cfg.replicas <= 1:
        write_single_run(cfg, repo_root, bench_dir, yaml_text, seed=None)
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
        seed = secrets.randbelow(2**31 - 1) + 1
        write_single_run(cfg, repo_root, odir, yaml_text, seed=seed)

    if skipped:
        print(f"SKIP: {len(skipped)} existing equil dirs (not touched)")


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


def run_dftb_submit(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    repo_root = find_repo_root(yaml_path)
    bench_dir = out_dir(cfg, repo_root)
    yaml_text = yaml_path.read_text()

    if cfg.slurm is None:
        print("NOTE: slurm config not provided; no submissions made")
        return

    if cfg.replicas <= 1:
        spec = bench_dir / "spec.yaml"
        if not spec.exists() or spec.read_text() != yaml_text:
            print(f"SKIP: {bench_dir} spec.yaml does not match config (not submitting)")
            return
        slurm_sh = bench_dir / "slurm.sh"
        if not slurm_sh.exists():
            print(f"SKIP: missing {slurm_sh} (not submitting)")
            return
        resp = input(f"About to submit 1 job: {bench_dir}. Proceed? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Cancelled by user.")
            return
        print(f"Submitting job via sbatch for {bench_dir}...")
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
        print("Usage: python3 dftb.py configs/SYS/dftb/dftb.yaml", file=sys.stderr)
        raise SystemExit(2)

    run_dftb(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
