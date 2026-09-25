"""Prepare restrained DCDFTBMD equilibration inputs for umbrella windows."""

from __future__ import annotations

import math
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from qmmd.dftb import (
    DFTBConfig,
    ElementConfig,
    RuntimeConfig,
    SlurmConfig,
    SlurmJobConfig,
    parse_box_vectors_from_comment,
    render_coords_block,
    render_element_blocks,
    stage_skf_files,
)
from qmmd.us_pull import (
    USPullConfig,
    find_repo_root,
    load_config as load_pull_config,
    pull_dir,
    source_paths,
    window_centers,
)


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    amber_module: str


@dataclass(frozen=True, slots=True)
class CVConfig:
    gaussian_width_deg: float


@dataclass(frozen=True, slots=True)
class WallConfig:
    coefficient_kcal_mol_deg2: float
    exponent: int


@dataclass(frozen=True, slots=True)
class USEquilConfig:
    pull_yaml: Path
    pull: USPullConfig
    stage_dirname: str
    conversion: ConversionConfig
    cv: CVConfig
    wall: WallConfig
    dftb: DFTBConfig
    runtime: RuntimeConfig
    slurm: SlurmConfig


def _single_component(value: object, field: str) -> str:
    text = str(value)
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"{field} must be a single directory name")
    return text


def _header_value(header_lines: list[str], key: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(key)}\s*=\s*([^\s,)]+)", re.IGNORECASE)
    for line in header_lines:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def _logical_true(value: str | None) -> bool:
    return value is not None and value.upper() in {"TRUE", "T", ".TRUE."}


def _logical_false(value: str | None) -> bool:
    return value is not None and value.upper() in {"FALSE", "F", ".FALSE."}


def validate_zero_height_metawall(header_lines: list[str]) -> None:
    """Validate the documented zero-height metadynamics restraint hack."""
    metad = _header_value(header_lines, "METADYNAMICS") or _header_value(
        header_lines, "METAD"
    )
    if not _logical_true(metad):
        raise ValueError("dftb.header_lines must enable METADYNAMICS=TRUE")
    if not _logical_true(_header_value(header_lines, "METAWALL")):
        raise ValueError("dftb.header_lines must enable METAWALL=TRUE")
    if not _logical_false(_header_value(header_lines, "METAPRINTFES")):
        raise ValueError("dftb.header_lines must explicitly set METAPRINTFES=FALSE")

    height_text = _header_value(header_lines, "METAHEIGHT")
    if height_text is None or float(height_text.replace("D", "E").replace("d", "e")) != 0.0:
        raise ValueError("dftb.header_lines must set METAHEIGHT=0")

    nstep_text = _header_value(header_lines, "NSTEP")
    frequency_text = _header_value(header_lines, "METAFREQ")
    max_gauss_text = _header_value(header_lines, "METAMAXGAUSS")
    if nstep_text is None or frequency_text is None or max_gauss_text is None:
        raise ValueError("NSTEP, METAFREQ, and METAMAXGAUSS must be explicit")
    nstep = int(nstep_text)
    frequency = int(frequency_text)
    max_gauss = int(max_gauss_text)
    if nstep <= 0 or frequency <= 0:
        raise ValueError("NSTEP and METAFREQ must be positive")
    required = math.ceil(nstep / frequency)
    if max_gauss < required:
        raise ValueError(
            f"METAMAXGAUSS={max_gauss} is too small; need at least {required} "
            f"for NSTEP={nstep} and METAFREQ={frequency}"
        )


def load_config(yaml_path: Path) -> USEquilConfig:
    resolved = yaml_path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("US equilibration configuration must be a YAML mapping")

    pull_yaml_value = Path(str(data.get("pull_yaml", "pull.yaml")))
    pull_yaml = (
        pull_yaml_value
        if pull_yaml_value.is_absolute()
        else (resolved.parent / pull_yaml_value).resolve()
    )
    if not pull_yaml.is_file():
        raise RuntimeError(f"Missing pull configuration: {pull_yaml}")
    pull_cfg = load_pull_config(pull_yaml)

    dftb_data = data["dftb"]
    dftb = DFTBConfig(
        title=str(dftb_data["title"]),
        header_lines=[str(line) for line in dftb_data["header_lines"]],
        charge=int(dftb_data.get("charge", 0)),
        multiplicity=int(dftb_data.get("multiplicity", 1)),
        elements=[ElementConfig(**element) for element in dftb_data["elements"]],
        params_dir=str(dftb_data.get("params_dir", "params")),
    )
    validate_zero_height_metawall(dftb.header_lines)

    width = float(data["cv"].get("gaussian_width_deg", 0.1))
    if not math.isfinite(width) or width <= 0:
        raise ValueError("cv.gaussian_width_deg must be positive and finite")
    wall_data = data["wall"]
    coefficient = float(wall_data["coefficient_kcal_mol_deg2"])
    exponent = int(wall_data.get("exponent", 2))
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("wall.coefficient_kcal_mol_deg2 must be positive and finite")
    if exponent <= 0 or exponent % 2:
        raise ValueError("wall.exponent must be a positive even integer")

    return USEquilConfig(
        pull_yaml=pull_yaml,
        pull=pull_cfg,
        stage_dirname=_single_component(data.get("stage_dirname", "equil"), "stage_dirname"),
        conversion=ConversionConfig(**data.get("conversion", {"amber_module": "amber/26"})),
        cv=CVConfig(width),
        wall=WallConfig(coefficient, exponent),
        dftb=dftb,
        runtime=RuntimeConfig(**data["runtime"]),
        slurm=SlurmConfig(job=SlurmJobConfig(**data["slurm"]["job"])),
    )


def read_xyz(path: Path) -> tuple[int, list[tuple[float, float, float]], list[tuple[str, float, float, float]]]:
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"XYZ too short: {path}")
    natoms = int(lines[0].strip())
    vectors = parse_box_vectors_from_comment(lines[1])
    coordinates: list[tuple[str, float, float, float]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            raise RuntimeError(f"Malformed XYZ coordinate in {path}: {line!r}")
        coordinates.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
    if len(coordinates) != natoms:
        raise RuntimeError(
            f"XYZ atom-count mismatch in {path}: header {natoms}, read {len(coordinates)}"
        )
    return natoms, vectors, coordinates


def _cpptraj_path(path: Path) -> str:
    return '"' + str(path.resolve()).replace('"', '\\"') + '"'


def convert_amber_restart_to_xyz(
    topology: Path,
    restart: Path,
    xyz_path: Path,
    amber_module: str,
) -> None:
    input_path = xyz_path.with_suffix(".cpptraj.in")
    log_path = xyz_path.with_suffix(".cpptraj.log")
    input_path.write_text(
        "\n".join(
            [
                f"parm {_cpptraj_path(topology)}",
                f"trajin {_cpptraj_path(restart)}",
                "autoimage",
                f"trajout {_cpptraj_path(xyz_path)} xyz",
                "run",
                "quit",
                "",
            ]
        )
    )
    cpptraj = shutil.which("cpptraj")
    if cpptraj:
        command = [cpptraj, "-i", str(input_path)]
    else:
        shell_command = (
            "module purge\n"
            f"module load {shlex.quote(amber_module)}\n"
            f"exec cpptraj -i {shlex.quote(str(input_path))}"
        )
        command = ["bash", "-lc", shell_command]
    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=xyz_path.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    if not xyz_path.is_file() or xyz_path.stat().st_size == 0:
        raise RuntimeError(f"cpptraj did not create starting XYZ: {xyz_path}")


def render_metacv(cfg: USEquilConfig, center_deg: float) -> str:
    atoms = " ".join(str(atom) for atom in cfg.pull.restraint.atoms)
    return (
        f"BONDDIHEDRAL {cfg.cv.gaussian_width_deg:g} {atoms}\n"
        "\n"
        f"L 1 {cfg.wall.coefficient_kcal_mol_deg2:.12g} {center_deg:g} {cfg.wall.exponent}\n"
        f"U 1 {cfg.wall.coefficient_kcal_mol_deg2:.12g} {center_deg:g} {cfg.wall.exponent}\n"
    )


def render_dftb_input(
    cfg: USEquilConfig,
    natoms: int,
    vectors: list[tuple[float, float, float]],
    coordinates: list[tuple[str, float, float, float]],
    seed: int,
) -> str:
    header_lines = [
        line.replace("RANDOMSEED=0", f"RANDOMSEED={seed}")
        for line in cfg.dftb.header_lines
    ]
    symbols = list(dict.fromkeys(coordinate[0] for coordinate in coordinates))
    text = (
        "\n".join(header_lines).rstrip()
        + "\n\n"
        + cfg.dftb.title
        + "\n\n"
        + render_element_blocks(cfg.dftb, symbols)
        + "\n"
        + f"{natoms:5d}  {cfg.dftb.charge:d}  {cfg.dftb.multiplicity:d}\n"
        + render_coords_block(coordinates)
    )
    for x, y, z in vectors:
        text += f"TV{x:>18.8f}{y:>15.8f}{z:>15.8f}\n"
    return text + "\n"


def render_run_sh(cfg: USEquilConfig) -> str:
    strict = "set -euo pipefail\n" if cfg.runtime.strict_mode else ""
    environment = "\n".join(
        f'export {key}="{value}"' for key, value in cfg.runtime.env.items()
    )
    return f'''#!/usr/bin/env bash
{strict}module purge
module load {cfg.runtime.module}

{environment}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export OMP_STACKSIZE=1G
ulimit -s unlimited

mpirun -np {cfg.runtime.mpirun_np} "{cfg.runtime.executable}"
'''


def render_slurm_sh(cfg: USEquilConfig, window_index: int) -> str:
    job = cfg.slurm.job
    qos = f"#SBATCH --qos={job.qos}\n" if job.qos else ""
    tag = f"N{job.nodes}T{job.ntasks}C{job.cpus_per_task}"
    return f'''#!/usr/bin/env bash
{qos}#SBATCH --job-name={job.name}-w{window_index:03d}-{tag}
#SBATCH --partition={job.partition}
#SBATCH --time={job.time}
#SBATCH --nodes={job.nodes}
#SBATCH --ntasks={job.ntasks}
#SBATCH --mem={job.mem}
#SBATCH --cpus-per-task={job.cpus_per_task}
#SBATCH --output={job.stdout}
#SBATCH --error={job.stderr}

bash run.sh
'''


RestartConverter = Callable[[Path, Path, Path, str], None]


def prepare_us_equil(
    cfg: USEquilConfig,
    yaml_text: str,
    repo_root: Path,
    converter: RestartConverter = convert_amber_restart_to_xyz,
) -> list[Path]:
    pull_root = pull_dir(cfg.pull, repo_root)
    prepared_pull_spec = pull_root / "pull_spec.yaml"
    if (
        not prepared_pull_spec.is_file()
        or prepared_pull_spec.read_text() != cfg.pull_yaml.read_text()
    ):
        raise RuntimeError(f"{prepared_pull_spec} does not exactly match {cfg.pull_yaml}")

    topology, _, _ = source_paths(cfg.pull, repo_root)
    if not topology.is_file() or topology.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty Amber topology: {topology}")
    centers = window_centers(cfg.pull.windows)
    destinations = [
        pull_root / f"window-{index:03d}" / cfg.stage_dirname
        for index in range(len(centers))
    ]
    existing = [destination for destination in destinations if destination.exists()]
    if existing:
        raise FileExistsError(f"US equilibration directory already exists; not touching: {existing[0]}")

    restarts: list[Path] = []
    for index in range(len(centers)):
        pull_stage = pull_root / f"window-{index:03d}" / "pull"
        restart = pull_stage / "pull.rst7"
        amber_output = pull_stage / "pull.out"
        if not restart.is_file() or restart.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty pull restart: {restart}")
        if not amber_output.is_file() or "Final Performance Info:" not in amber_output.read_text(
            errors="replace"
        ):
            raise RuntimeError(f"Amber pull window did not reach normal completion: {amber_output}")
        restarts.append(restart)

    rendered: list[tuple[str, str, str]] = []
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="qmmd-us-equil-") as tmp:
        temporary = Path(tmp)
        for index, (center, restart) in enumerate(zip(centers, restarts)):
            xyz = temporary / f"window-{index:03d}.xyz"
            converter(topology, restart, xyz, cfg.conversion.amber_module)
            natoms, vectors, coordinates = read_xyz(xyz)
            if max(cfg.pull.restraint.atoms) > natoms:
                raise ValueError(
                    f"Dihedral atom ID exceeds {natoms} atoms in window-{index:03d}"
                )
            seed = secrets.randbelow(2**31 - 1) + 1
            rendered.append(
                (
                    render_dftb_input(cfg, natoms, vectors, coordinates, seed),
                    render_metacv(cfg, center),
                    xyz.read_text(),
                )
            )

    params = repo_root / cfg.dftb.params_dir
    if not params.is_dir():
        raise RuntimeError(f"Missing DFTB parameter directory: {params}")
    elements = [element.symbol for element in cfg.dftb.elements]
    for left in elements:
        for right in elements:
            parameter = params / f"{left}-{right}.skf"
            if not parameter.is_file():
                raise RuntimeError(f"Missing SKF file: {parameter}")
    for index, (destination, files) in enumerate(zip(destinations, rendered)):
        dftb_input, metacv, starting_xyz = files
        destination.mkdir()
        (destination / "dftb.inp").write_text(dftb_input)
        (destination / "metacv.dat").write_text(metacv)
        (destination / "start.xyz").write_text(starting_xyz)
        (destination / "equil_spec.yaml").write_text(yaml_text)
        run_sh = destination / "run.sh"
        run_sh.write_text(render_run_sh(cfg))
        run_sh.chmod(0o755)
        slurm_sh = destination / "slurm.sh"
        slurm_sh.write_text(render_slurm_sh(cfg, index))
        slurm_sh.chmod(0o755)
        stage_skf_files(params, destination, elements)
    return destinations


def run_us_equil_prep(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    destinations = prepare_us_equil(
        cfg,
        resolved.read_text(),
        find_repo_root(resolved),
    )
    print(
        f"OK: prepared {len(destinations)} restrained DCDFTBMD equilibration windows "
        f"under {destinations[0].parent.parent}"
    )
    print("NOTE: inputs only; no DCDFTBMD jobs were run or submitted")


def submit_slurm(slurm_sh: Path) -> None:
    """Submit one prepared window from inside its equilibration directory."""
    subprocess.run(["sbatch", slurm_sh.name], cwd=slurm_sh.parent, check=True)


def submit_us_equil(
    cfg: USEquilConfig,
    yaml_text: str,
    repo_root: Path,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
) -> bool:
    """Validate and submit the complete set of prepared equilibration windows."""
    root = pull_dir(cfg.pull, repo_root)
    centers = window_centers(cfg.pull.windows)
    targets = [
        root / f"window-{index:03d}" / cfg.stage_dirname
        for index in range(len(centers))
    ]

    problems: list[str] = []
    required_files = ("dftb.inp", "metacv.dat", "run.sh", "slurm.sh")
    for target in targets:
        spec = target / "equil_spec.yaml"
        if not spec.is_file() or spec.read_text() != yaml_text:
            problems.append(f"{spec} does not exactly match the supplied config")
            continue
        for filename in required_files:
            path = target / filename
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f"missing/empty {path}")

    if problems:
        print("SKIP: US equilibration set is incomplete or does not match config; nothing submitted")
        for problem in problems:
            print(f"  - {problem}")
        return False

    print("Will submit the following US equilibration directories:")
    for target, center in zip(targets, centers):
        print(f"  - {target} (center {center:g} degrees)")
    response = confirm(f"Proceed to submit {len(targets)} jobs? [y/N] ").strip().lower()
    if response not in ("y", "yes"):
        print("Cancelled by user.")
        return False

    submit = submitter or submit_slurm
    for target in targets:
        print(f"Submitting job via sbatch for {target}...")
        submit(target / "slurm.sh")
        print("OK: job submitted")
    return True


def run_us_equil_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    submit_us_equil(cfg, resolved.read_text(), find_repo_root(resolved))
