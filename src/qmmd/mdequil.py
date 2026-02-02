#!/usr/bin/env python3

import sys
import yaml
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class SlurmJobConfig:
    name: str
    partition: str
    nodes: int
    time: str
    stdout: str
    stderr: str


@dataclass
class SlurmConfig:
    job: SlurmJobConfig


@dataclass
class RuntimeConfig:
    module: str
    executable: str
    env: Dict[str, Any]
    strict_mode: bool = True


@dataclass
class MDStage:
    description: str
    cntrl: Dict[str, Any]
    wt: Optional[List[Dict[str, Any]]] = None


@dataclass
class MDConfig:
    minimize: MDStage
    heat: MDStage
    equilibrate_nvt: MDStage
    equilibrate_npt: MDStage


@dataclass
class MDEquilConfig:
    system: str
    buffer: float
    prefix: str
    job_name: str
    md: MDConfig
    runtime: RuntimeConfig
    slurm: Optional[SlurmConfig] = None


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def system_base_dir(cfg: MDEquilConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def prep_dir(cfg: MDEquilConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / "prep"


def run_dir(cfg: MDEquilConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.job_name


def load_config(yaml_path: Path) -> MDEquilConfig:
    data = yaml.safe_load(yaml_path.read_text())

    md = data["md"]
    md_cfg = MDConfig(
        minimize=MDStage(**md["minimize"]),
        heat=MDStage(**md["heat"]),
        equilibrate_nvt=MDStage(**md["equilibrate_nvt"]),
        equilibrate_npt=MDStage(**md["equilibrate_npt"]),
    )

    runtime_cfg = RuntimeConfig(**data["runtime"])

    slurm_cfg = None
    if data.get("slurm") is not None:
        slurm_cfg = SlurmConfig(job=SlurmJobConfig(**data["slurm"]["job"]))

    return MDEquilConfig(
        system=data["system"],
        buffer=float(data["buffer"]),
        prefix=data.get("prefix", "solv"),
        job_name=data.get("job_name", "mdequil"),
        md=md_cfg,
        runtime=runtime_cfg,
        slurm=slurm_cfg,
    )


def render_mdin(stage: MDStage) -> str:
    lines: List[str] = []
    lines.append(stage.description)
    lines.append("&cntrl")
    for k, v in stage.cntrl.items():
        lines.append(f"  {k}={v},")
    lines.append("/")

    if stage.wt:
        for card in stage.wt:
            t = str(card["type"])
            if t.upper() == "END":
                lines.append("&wt type='END' /")
            else:
                parts = [f"type='{t}'"]
                for key in ("istep1", "istep2", "value1", "value2"):
                    if key in card:
                        parts.append(f"{key}={card[key]}")
                lines.append("&wt " + ", ".join(parts) + " /")

    return "\n".join(lines) + "\n"


def write_mdin_files(cfg: MDEquilConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "min.in").write_text(render_mdin(cfg.md.minimize))
    (out_dir / "heat.in").write_text(render_mdin(cfg.md.heat))
    (out_dir / "equil-nvt.in").write_text(render_mdin(cfg.md.equilibrate_nvt))
    (out_dir / "equil-npt.in").write_text(render_mdin(cfg.md.equilibrate_npt))



def write_run_sh(cfg: MDEquilConfig, out_dir: Path, prep_dir_path: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sh_path = out_dir / "run.sh"

    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""

    env_lines = "\n".join([f"export {k}={v}" for k, v in cfg.runtime.env.items()])

    # Use prep topology/coords as inputs
    parm7 = (prep_dir_path / f"{cfg.prefix}.parm7").resolve()
    rst7  = (prep_dir_path / f"{cfg.prefix}.rst7").resolve()

    text = f"""\
#!/usr/bin/env bash
{strict}

module purge
module load {cfg.runtime.module}

{env_lines}

echo "==> Running min/heat/equil (serial {cfg.runtime.executable})"

# Minimize
if [[ ! -f min.out ]]; then
    echo "  Minimization..."
    {cfg.runtime.executable} -O \\
      -i min.in \\
      -p "{parm7}" \\
      -c "{rst7}" \\
      -r min.rst7 \\
      -o min.out \\
      -inf min.info
else
    echo "  Skipping minimization (min.out exists)"
fi

# Heat (start from minimized)
if [[ ! -f heat.out ]]; then
    echo "  Heating..."
    {cfg.runtime.executable} -O \\
      -i heat.in \\
      -p "{parm7}" \\
      -c min.rst7 \\
      -r heat.rst7 \\
      -o heat.out \\
      -inf heat.info \\
      -x heat.nc
else
    echo "  Skipping heating (heat.out exists)"
fi

# Equilibration NVT (start from heated)
if [[ ! -f equil-nvt.out ]]; then
    echo "  Equilibration (NVT)..."
    {cfg.runtime.executable} -O \\
      -i equil-nvt.in \\
      -p "{parm7}" \\
      -c heat.rst7 \\
      -r equil-nvt.rst7 \\
      -o equil-nvt.out \\
      -inf equil-nvt.info \\
      -x equil-nvt.nc
else
    echo "  Skipping equilibration (equil-nvt.out exists)"
fi

# Equilibration NPT (start from NVT)
if [[ ! -f equil-npt.out ]]; then
    echo "  Equilibration (NPT)..."
    {cfg.runtime.executable} -O \\
      -i equil-npt.in \\
      -p "{parm7}" \\
      -c equil-nvt.rst7 \\
      -r equil-npt.rst7 \\
      -o equil-npt.out \\
      -inf equil-npt.info \\
      -x equil-npt.nc
else
    echo "  Skipping equilibration (equil-npt.out exists)"
fi
"""
    sh_path.write_text(text)
    sh_path.chmod(0o755)
    return sh_path


def write_slurm_sh(cfg: MDEquilConfig, out_dir: Path) -> Optional[Path]:
    if cfg.slurm is None:
        return None

    job = cfg.slurm.job
    sh_path = out_dir / "slurm.sh"

    text = f"""\
#!/usr/bin/env bash
#SBATCH -J {job.name}
#SBATCH -p {job.partition}
#SBATCH -N {job.nodes}
#SBATCH -t {job.time}
#SBATCH -o {job.stdout}
#SBATCH -e {job.stderr}

bash run.sh
"""
    sh_path.write_text(text)
    sh_path.chmod(0o755)
    return sh_path

def submit_slurm(slurm_sh: Path) -> None:
    subprocess.run(
        ["sbatch", slurm_sh.name],
        cwd=slurm_sh.parent,
        check=True,
    )

def run_local(run_sh: Path) -> None:
    subprocess.run(
        ["bash", run_sh.name],
        cwd=run_sh.parent,
        check=True,
    )


def run_mdequil(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)

    repo_root = find_repo_root(yaml_path)
    pdir = prep_dir(cfg, repo_root)
    output_dir = run_dir(cfg, repo_root)

    if not pdir.exists():
        raise FileNotFoundError(f"Missing prep dir: {pdir}")

    parm7 = pdir / f"{cfg.prefix}.parm7"
    rst7  = pdir / f"{cfg.prefix}.rst7"
    if not parm7.exists() or parm7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty input: {parm7}")
    if not rst7.exists() or rst7.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty input: {rst7}")

    # Write outputs next to prep (NOT inside)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_mdin_files(cfg, output_dir)
    run_sh = write_run_sh(cfg, output_dir, pdir)
    slurm_sh = write_slurm_sh(cfg, output_dir)

    (output_dir / "spec.yaml").write_text(yaml_path.read_text())

    print(f"OK: wrote md inputs + scripts in {output_dir}")
    print(f"OK: using prep inputs from {pdir}")
    
    if slurm_sh is not None:
        print("Submitting job via sbatch...")
        submit_slurm(slurm_sh)
        print("OK: job submitted")
    else:
        print("Running locally...")
        run_local(run_sh)
        print("OK: local MD run finished")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 mdequil.py configs/SYS/mdequil/mdequil.yaml", file=sys.stderr)
        raise SystemExit(2)

    yaml_path = Path(sys.argv[1]).resolve()
    run_mdequil(yaml_path)


if __name__ == "__main__":
    main()

