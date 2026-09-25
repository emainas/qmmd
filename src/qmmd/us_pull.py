"""Prepare sequential Amber restrained-MD pulling windows."""

from __future__ import annotations

import csv
import math
import shlex
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import yaml

from qmmd.mdequil import MDStage, render_dihedral_restraints, render_mdin


@dataclass(frozen=True, slots=True)
class SourceConfig:
    topology_dirname: str
    topology_name: str
    restart_dirname: str
    restart_name: str
    output_name: str


@dataclass(frozen=True, slots=True)
class WindowConfig:
    start_deg: float
    stop_deg: float
    spacing_deg: float


@dataclass(frozen=True, slots=True)
class RestraintConfig:
    atoms: tuple[int, int, int, int]
    force_constant: float


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    module: str
    executable: str
    env: dict[str, Any]
    strict_mode: bool = True


@dataclass(frozen=True, slots=True)
class SlurmJobConfig:
    name: str
    partition: str
    nodes: int
    time: str
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    job: SlurmJobConfig


@dataclass(frozen=True, slots=True)
class USPullConfig:
    system: str
    buffer: float | None
    prefix: str
    system_dir: str | None
    output_dirname: str
    source: SourceConfig
    windows: WindowConfig
    restraint: RestraintConfig
    md: MDStage
    runtime: RuntimeConfig
    slurm: SlurmConfig


def find_repo_root(start: Path) -> Path:
    path = start.resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root starting from {start}")


def _single_component(value: Any, field: str) -> str:
    text = str(value)
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"{field} must be a single directory or file name")
    return text


def window_centers(spec: WindowConfig) -> list[float]:
    start = Decimal(str(spec.start_deg))
    stop = Decimal(str(spec.stop_deg))
    spacing = Decimal(str(spec.spacing_deg))
    if not all(math.isfinite(float(value)) for value in (start, stop, spacing)):
        raise ValueError("Window start, stop, and spacing must be finite")
    if spacing <= 0:
        raise ValueError("windows.spacing_deg must be positive")

    distance = abs(stop - start)
    quotient = distance / spacing
    if quotient != quotient.to_integral_value():
        raise ValueError("Window range must be exactly divisible by spacing_deg")
    direction = Decimal(1) if stop >= start else Decimal(-1)
    count = int(quotient) + 1
    return [float(start + direction * spacing * index) for index in range(count)]


def load_config(yaml_path: Path) -> USPullConfig:
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("US pull configuration must be a YAML mapping")
    if data.get("system_dir") is None and data.get("buffer") is None:
        raise ValueError("Either system_dir or buffer is required")

    source_data = data["source"]
    source = SourceConfig(
        topology_dirname=_single_component(
            source_data.get("topology_dirname", "prep"), "source.topology_dirname"
        ),
        topology_name=_single_component(
            source_data.get("topology_name", f"{data.get('prefix', 'solv')}.parm7"),
            "source.topology_name",
        ),
        restart_dirname=_single_component(
            source_data.get("restart_dirname", "mdequil"), "source.restart_dirname"
        ),
        restart_name=_single_component(
            source_data.get("restart_name", "equil-npt.rst7"), "source.restart_name"
        ),
        output_name=_single_component(
            source_data.get("output_name", "equil-npt.out"), "source.output_name"
        ),
    )

    window_data = data["windows"]
    windows = WindowConfig(
        start_deg=float(window_data["start_deg"]),
        stop_deg=float(window_data["stop_deg"]),
        spacing_deg=float(window_data["spacing_deg"]),
    )
    centers = window_centers(windows)
    if any(center < -180.0 or center > 180.0 for center in centers):
        raise ValueError("All dihedral window centers must lie between -180 and 180 degrees")

    restraint_data = data["restraint"]
    atoms = restraint_data["atoms"]
    if (
        not isinstance(atoms, list)
        or len(atoms) != 4
        or any(type(atom) is not int or atom < 1 for atom in atoms)
        or len(set(atoms)) != 4
    ):
        raise ValueError("restraint.atoms must contain four distinct positive one-based IDs")
    force_constant = float(restraint_data["force_constant"])
    if not math.isfinite(force_constant) or force_constant <= 0:
        raise ValueError("restraint.force_constant must be positive and finite")

    md_data = data["md"]
    md = MDStage(description=str(md_data["description"]), cntrl=dict(md_data["cntrl"]))
    nstlim = int(md.cntrl.get("nstlim", 0))
    timestep_ps = float(md.cntrl.get("dt", 0.0))
    if nstlim <= 0 or not math.isfinite(timestep_ps) or timestep_ps <= 0:
        raise ValueError("md.cntrl requires positive nstlim and dt (ps)")
    if int(md.cntrl.get("irest", 0)) != 1 or int(md.cntrl.get("ntx", 0)) != 5:
        raise ValueError("Sequential pulling requires md.cntrl irest=1 and ntx=5")

    runtime = RuntimeConfig(**data["runtime"])
    job_data = data["slurm"]["job"]
    slurm = SlurmConfig(
        job=SlurmJobConfig(
            name=str(job_data["name"]),
            partition=str(job_data["partition"]),
            nodes=int(job_data.get("nodes", 2)),
            time=str(job_data["time"]),
            stdout=str(job_data["stdout"]),
            stderr=str(job_data["stderr"]),
        )
    )
    if slurm.job.partition != "small":
        raise ValueError("US pulling must use the Slurm small partition")
    if slurm.job.nodes < 2:
        raise ValueError("US pulling on the small partition requires at least 2 nodes")

    cfg = USPullConfig(
        system=str(data["system"]),
        buffer=float(data["buffer"]) if data.get("buffer") is not None else None,
        prefix=str(data.get("prefix", "solv")),
        system_dir=data.get("system_dir"),
        output_dirname=_single_component(data.get("output_dirname", "us-pull"), "output_dirname"),
        source=source,
        windows=windows,
        restraint=RestraintConfig(tuple(atoms), force_constant),
        md=md,
        runtime=runtime,
        slurm=slurm,
    )
    return cfg


def system_base_dir(cfg: USPullConfig, repo_root: Path) -> Path:
    if cfg.system_dir is not None:
        path = Path(cfg.system_dir)
        return path if path.is_absolute() else repo_root / path
    if cfg.buffer is None:
        raise ValueError("Either system_dir or buffer is required")
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def pull_dir(cfg: USPullConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / cfg.output_dirname


def source_paths(cfg: USPullConfig, repo_root: Path) -> tuple[Path, Path, Path]:
    base = system_base_dir(cfg, repo_root)
    topology = base / cfg.source.topology_dirname / cfg.source.topology_name
    restart = base / cfg.source.restart_dirname / cfg.source.restart_name
    output = base / cfg.source.restart_dirname / cfg.source.output_name
    return topology, restart, output


def _window_name(index: int) -> str:
    return f"window-{index:03d}"


def render_run_sh(
    cfg: USPullConfig,
    output_dir: Path,
    topology: Path,
    initial_restart: Path,
    centers: list[float],
) -> str:
    strict = "set -euo pipefail\n" if cfg.runtime.strict_mode else ""
    env_lines = "\n".join(
        f"export {key}={shlex.quote(str(value))}" for key, value in cfg.runtime.env.items()
    )
    lines = [
        "#!/usr/bin/env bash",
        strict.rstrip(),
        "module purge",
        f"module load {shlex.quote(cfg.runtime.module)}",
        "",
        env_lines,
        "",
        'pull_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        f"topology={shlex.quote(str(topology.resolve()))}",
        f"input_restart={shlex.quote(str(initial_restart.resolve()))}",
        "",
        f'echo "==> Sequential Amber pull: {len(centers)} windows"',
    ]
    for index, center in enumerate(centers):
        name = _window_name(index)
        center_text = f"{center:g}"
        lines.extend(
            [
                "",
                f'window="$pull_root/{name}/pull"',
                f'echo "==> {name}: target {center_text} degrees"',
                'if [[ -s "$window/pull.rst7" ]] && grep -q "Final Performance Info:" "$window/pull.out"; then',
                '    echo "  Complete; using existing restart"',
                'elif [[ -e "$window/pull.out" || -e "$window/pull.rst7" || -e "$window/pull.nc" ]]; then',
                '    echo "ERROR: partial output exists in $window; refusing to overwrite" >&2',
                "    exit 1",
                "else",
                '    (cd "$window" && ' + shlex.quote(cfg.runtime.executable) + " -O \\",
                "      -i pull.in \\",
                '      -p "$topology" ' + "\\",
                '      -c "$input_restart" ' + "\\",
                "      -r pull.rst7 " + "\\",
                "      -o pull.out " + "\\",
                "      -inf pull.info " + "\\",
                "      -x pull.nc)",
                "fi",
                'input_restart="$window/pull.rst7"',
            ]
        )
    lines.extend(["", 'echo "==> Sequential Amber pull complete"', ""])
    return "\n".join(line for line in lines if line is not None)


def render_slurm_sh(cfg: USPullConfig) -> str:
    job = cfg.slurm.job
    return f"""#!/usr/bin/env bash
#SBATCH -J {job.name}
#SBATCH -p {job.partition}
#SBATCH -N {job.nodes}
#SBATCH -t {job.time}
#SBATCH -o {job.stdout}
#SBATCH -e {job.stderr}

bash run.sh
"""


def render_windows_csv(centers: list[float]) -> str:
    stream = StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["window_index", "pull_dir", "target_deg"])
    for index, center in enumerate(centers):
        writer.writerow([index, f"{_window_name(index)}/pull", f"{center:g}"])
    return stream.getvalue()


def prepare_us_pull(cfg: USPullConfig, yaml_text: str, repo_root: Path) -> Path:
    topology, initial_restart, source_output = source_paths(cfg, repo_root)
    for label, path in (
        ("topology", topology),
        ("initial restart", initial_restart),
        ("source MD output", source_output),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty {label}: {path}")
    if "Final Performance Info:" not in source_output.read_text(errors="replace"):
        raise RuntimeError(f"Source Amber MD did not reach normal completion: {source_output}")

    centers = window_centers(cfg.windows)
    output_dir = pull_dir(cfg, repo_root)
    if output_dir.exists():
        raise FileExistsError(f"US pull directory already exists; not touching: {output_dir}")

    # Render every file before making the destination, so validation failures
    # cannot leave a partly prepared window chain.
    rendered_windows: list[tuple[str, str, str]] = []
    for index, center in enumerate(centers):
        restraint_text = render_dihedral_restraints(
            [
                {
                    "atoms": list(cfg.restraint.atoms),
                    "target_deg": center,
                    "force_constant": cfg.restraint.force_constant,
                }
            ]
        )
        input_text = render_mdin(cfg.md, "dihedral.rst")
        rendered_windows.append((_window_name(index), input_text, restraint_text))

    output_dir.mkdir(parents=True)
    for name, input_text, restraint_text in rendered_windows:
        window = output_dir / name
        window.mkdir()
        stage = window / "pull"
        stage.mkdir()
        (stage / "pull.in").write_text(input_text)
        (stage / "dihedral.rst").write_text(restraint_text)

    (output_dir / "windows.csv").write_text(render_windows_csv(centers))
    (output_dir / "pull_spec.yaml").write_text(yaml_text)
    run_sh = output_dir / "run.sh"
    run_sh.write_text(render_run_sh(cfg, output_dir, topology, initial_restart, centers))
    run_sh.chmod(0o755)
    slurm_sh = output_dir / "slurm.sh"
    slurm_sh.write_text(render_slurm_sh(cfg))
    slurm_sh.chmod(0o755)
    return output_dir


def run_us_pull_prep(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    output = prepare_us_pull(cfg, resolved.read_text(), find_repo_root(resolved))
    centers = window_centers(cfg.windows)
    duration_ps = int(cfg.md.cntrl["nstlim"]) * float(cfg.md.cntrl["dt"])
    print(
        f"OK: prepared {len(centers)} sequential Amber windows in {output} "
        f"({duration_ps:g} ps/window; {centers[0]:g} to {centers[-1]:g} degrees)"
    )


def submit_slurm(slurm_sh: Path) -> None:
    subprocess.run(["sbatch", slurm_sh.name], cwd=slurm_sh.parent, check=True)


def submit_us_pull(
    cfg: USPullConfig,
    yaml_text: str,
    repo_root: Path,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
) -> bool:
    output = pull_dir(cfg, repo_root)
    spec = output / "pull_spec.yaml"
    if not spec.is_file() or spec.read_text() != yaml_text:
        print(f"SKIP: {output} pull_spec.yaml does not match config (not submitting)")
        return False

    run_sh = output / "run.sh"
    slurm_sh = output / "slurm.sh"
    for script in (run_sh, slurm_sh):
        if not script.is_file() or script.stat().st_size == 0:
            print(f"SKIP: missing/empty {script} (not submitting)")
            return False

    print("Will submit the following US pull directory:")
    print(f"  - {output}")
    response = confirm("Proceed to submit 1 job? [y/N] ").strip().lower()
    if response not in ("y", "yes"):
        print("Cancelled by user.")
        return False

    print(f"Submitting job via sbatch for {output}...")
    (submitter or submit_slurm)(slurm_sh)
    print("OK: job submitted")
    return True


def run_us_pull_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    submit_us_pull(cfg, resolved.read_text(), find_repo_root(resolved))
