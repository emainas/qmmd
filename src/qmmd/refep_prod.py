"""Prepare Hamiltonian replica-exchange REFEP production inputs."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from qmmd.refep_equil import (
    RefepEquilConfig,
    load_refep_equil_config,
    refep_equil_dir,
    render_equil_mdin,
)


_AMBER_FAILURE_RE = re.compile(
    r"\b(?:nan|infinity|fatal|segmentation)\b|"
    r"shake cannot|coordinate resetting|vlimit exceeded|"
    r"bomb|terminated abnormally",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RefepProdRuntimeConfig:
    module: str
    executable: str
    mpi_launcher: str
    processes_per_replica: int
    rem_mode: int
    env: dict[str, Any]
    strict_mode: bool


@dataclass(frozen=True, slots=True)
class RefepProdSlurmConfig:
    name: str
    partition: str
    nodes: int
    ntasks: int
    tasks_per_node: int
    time: str
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RefepProdConfig:
    yaml_path: Path
    equil_yaml: Path
    equil: RefepEquilConfig
    stage_dirname: str
    description: str
    cntrl: dict[str, Any]
    runtime: RefepProdRuntimeConfig
    slurm: RefepProdSlurmConfig


@dataclass(frozen=True, slots=True)
class CompletedEquilWindow:
    index: int
    lambda_value: str
    topology: Path
    restart: Path
    topology_sha256: str
    restart_sha256: str


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _single_component(value: object, field: str) -> str:
    text = _nonempty_string(value, field)
    if Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"{field} must be a single directory name")
    return text


def _resolve_sibling_yaml(yaml_path: Path, value: object, field: str) -> Path:
    path = Path(_nonempty_string(value, field))
    resolved = path.resolve() if path.is_absolute() else (yaml_path.parent / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {field}: {resolved}")
    return resolved


def load_refep_prod_config(yaml_path: Path) -> RefepProdConfig:
    resolved = yaml_path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("REFEP production configuration must be a YAML mapping")

    equil_yaml = _resolve_sibling_yaml(resolved, data.get("equil_yaml"), "equil_yaml")
    equil = load_refep_equil_config(equil_yaml)

    mdin = _mapping(data.get("remd_mdin"), "remd_mdin")
    if len(mdin) != 1:
        raise ValueError("remd_mdin must contain exactly one MD stage")
    stage = _mapping(next(iter(mdin.values())), "remd_mdin stage")
    description = _nonempty_string(stage.get("description"), "remd_mdin.description")
    cntrl = dict(_mapping(stage.get("cntrl"), "remd_mdin.cntrl"))
    required = {
        "imin",
        "irest",
        "ntx",
        "nstlim",
        "numexchg",
        "dt",
        "temp0",
        "ntt",
        "gamma_ln",
        "ntb",
        "ntp",
        "ntc",
        "ntf",
        "cut",
        "ntpr",
        "ntwx",
        "ntwr",
        "ioutfm",
    }
    missing = sorted(required - cntrl.keys())
    if missing:
        raise ValueError("Missing REFEP production cntrl fields: " + ", ".join(missing))
    if int(cntrl["imin"]) != 0 or int(cntrl["irest"]) != 1 or int(cntrl["ntx"]) != 5:
        raise ValueError("REFEP production must restart MD with imin=0, irest=1, and ntx=5")
    if int(cntrl["ntb"]) != 1 or int(cntrl["ntp"]) != 0:
        raise ValueError("Hamiltonian replica exchange must use NVT with ntb=1 and ntp=0")
    if int(cntrl["ntc"]) != 2 or int(cntrl["ntf"]) != 2:
        raise ValueError("REFEP production requires SHAKE with ntc=2 and ntf=2")
    nstlim = int(cntrl["nstlim"])
    numexchg = int(cntrl["numexchg"])
    dt = float(cntrl["dt"])
    if nstlim <= 0 or numexchg <= 0 or dt <= 0:
        raise ValueError("nstlim, numexchg, and dt must be positive")
    if numexchg % 2:
        raise ValueError("numexchg must be even so both neighbor-pair parities are sampled equally")
    if float(cntrl["cut"]) <= 0:
        raise ValueError("remd_mdin cut must be positive")
    for key in ("ntpr", "ntwx", "ntwr"):
        if int(cntrl[key]) <= 0:
            raise ValueError(f"remd_mdin {key} must be positive")
    total_steps = nstlim * numexchg
    if total_steps % int(cntrl["ntwx"]):
        raise ValueError("ntwx must divide the total REFEP production step count")
    if int(cntrl["ntwx"]) % nstlim:
        raise ValueError("ntwx must be an integer multiple of nstlim for exchange alignment")

    runtime_data = _mapping(data.get("runtime"), "runtime")
    env = dict(_mapping(runtime_data.get("env", {}), "runtime.env"))
    runtime = RefepProdRuntimeConfig(
        module=_nonempty_string(runtime_data.get("module"), "runtime.module"),
        executable=_nonempty_string(runtime_data.get("executable"), "runtime.executable"),
        mpi_launcher=_nonempty_string(
            runtime_data.get("mpi_launcher"), "runtime.mpi_launcher"
        ),
        processes_per_replica=int(runtime_data.get("processes_per_replica", 1)),
        rem_mode=int(runtime_data.get("rem_mode", 3)),
        env=env,
        strict_mode=bool(runtime_data.get("strict_mode", True)),
    )
    if runtime.processes_per_replica <= 0:
        raise ValueError("runtime.processes_per_replica must be positive")
    if runtime.rem_mode != 3:
        raise ValueError("Hamiltonian REFEP requires runtime.rem_mode=3")

    slurm_data = _mapping(data.get("slurm"), "slurm")
    job = _mapping(slurm_data.get("job"), "slurm.job")
    slurm = RefepProdSlurmConfig(
        name=_nonempty_string(job.get("name"), "slurm.job.name"),
        partition=_nonempty_string(job.get("partition"), "slurm.job.partition"),
        nodes=int(job["nodes"]),
        ntasks=int(job["ntasks"]),
        tasks_per_node=int(job["tasks_per_node"]),
        time=_nonempty_string(job.get("time"), "slurm.job.time"),
        stdout=_nonempty_string(job.get("stdout"), "slurm.job.stdout"),
        stderr=_nonempty_string(job.get("stderr"), "slurm.job.stderr"),
    )
    if slurm.nodes <= 0 or slurm.ntasks <= 0 or slurm.tasks_per_node <= 0:
        raise ValueError("Slurm nodes, ntasks, and tasks_per_node must be positive")
    expected_tasks = equil.prep.windows * runtime.processes_per_replica
    if slurm.ntasks != expected_tasks:
        raise ValueError(
            "slurm.job.ntasks must equal lambda windows times processes_per_replica "
            f"({expected_tasks})"
        )
    if slurm.ntasks > slurm.nodes * slurm.tasks_per_node:
        raise ValueError("Slurm nodes * tasks_per_node cannot be smaller than ntasks")

    return RefepProdConfig(
        yaml_path=resolved,
        equil_yaml=equil_yaml,
        equil=equil,
        stage_dirname=_single_component(data.get("stage_dirname", "prod"), "stage_dirname"),
        description=description,
        cntrl=cntrl,
        runtime=runtime,
        slurm=slurm,
    )


def refep_prod_dir(cfg: RefepProdConfig) -> Path:
    return cfg.equil.prep.output_directory.parent / cfg.stage_dirname


def _window_name(index: int) -> str:
    return f"lambda-{index:03d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"Malformed manifest: {path}")
        return list(reader)


def validate_completed_equilibration(cfg: RefepProdConfig) -> list[CompletedEquilWindow]:
    equil_dir = refep_equil_dir(cfg.equil)
    topology_spec = cfg.equil.prep.output_directory / "refep-prep-spec.yaml"
    if (
        not topology_spec.is_file()
        or topology_spec.read_text() != cfg.equil.prep_yaml.read_text()
    ):
        raise ValueError(
            f"{topology_spec} does not exactly match the referenced prep.yaml"
        )
    spec = equil_dir / "refep-equil-spec.yaml"
    if not spec.is_file() or spec.read_text() != cfg.equil_yaml.read_text():
        raise ValueError(f"{spec} does not exactly match the referenced equil.yaml")

    topology_rows = _read_csv(
        cfg.equil.prep.output_directory / "topology-manifest.csv",
        {"window", "lambda", "topology", "sha256"},
    )
    equil_rows = _read_csv(
        equil_dir / "equil-manifest.csv",
        {"window", "lambda", "directory", "topology", "starting_restart"},
    )
    windows = cfg.equil.prep.windows
    if len(topology_rows) != windows or len(equil_rows) != windows:
        raise ValueError(f"Expected {windows} topology and equilibration manifest rows")

    completed: list[CompletedEquilWindow] = []
    final_step_re = re.compile(rf"NSTEP\s*=\s*{int(cfg.equil.cntrl['nstlim'])}\b")
    for index, (topology_row, equil_row) in enumerate(zip(topology_rows, equil_rows)):
        window = _window_name(index)
        if (
            int(topology_row["window"]) != index
            or int(equil_row["window"]) != index
            or topology_row["lambda"] != equil_row["lambda"]
            or topology_row["topology"] != f"{window}.parm7"
            or equil_row["directory"] != window
            or equil_row["topology"] != f"{window}/system.parm7"
            or equil_row["starting_restart"] != f"{window}/start.rst7"
        ):
            raise ValueError(f"Manifest mismatch for lambda window {index}")

        source_dir = equil_dir / window
        topology = source_dir / "system.parm7"
        restart = source_dir / "equil.rst7"
        expected_files = (
            topology,
            restart,
            source_dir / "equil.nc",
            source_dir / "equil.mdinfo",
            source_dir / "equil.mdout",
        )
        for path in expected_files:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty completed equilibration file: {path}")
        if _sha256(topology) != topology_row["sha256"]:
            raise ValueError(f"Equilibrated topology checksum mismatch: {topology}")
        mdin = source_dir / "equil.mdin"
        if not mdin.is_file() or mdin.read_text() != render_equil_mdin(cfg.equil, index):
            raise ValueError(f"Equilibration MDIN does not match equil.yaml: {mdin}")
        mdout = source_dir / "equil.mdout"
        mdout_text = mdout.read_text(errors="replace")
        if not final_step_re.search(mdout_text) or "Final Performance Info:" not in mdout_text:
            raise ValueError(f"Equilibration did not complete its requested steps: {mdout}")
        failure = _AMBER_FAILURE_RE.search(mdout_text)
        if failure is not None:
            raise ValueError(f"Amber failure marker {failure.group(0)!r} found in {mdout}")
        completed.append(
            CompletedEquilWindow(
                index=index,
                lambda_value=topology_row["lambda"],
                topology=topology,
                restart=restart,
                topology_sha256=topology_row["sha256"],
                restart_sha256=_sha256(restart),
            )
        )
    return completed


def _format_amber_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    return str(value)


def render_prod_mdin(cfg: RefepProdConfig) -> str:
    lines = [cfg.description, "&cntrl"]
    lines.extend(
        f"  {key}={_format_amber_value(value)}," for key, value in cfg.cntrl.items()
    )
    lines.extend(["/", ""])
    return "\n".join(lines)


def render_groupfile(cfg: RefepProdConfig) -> str:
    lines: list[str] = []
    for index in range(cfg.equil.prep.windows):
        stem = _window_name(index)
        lines.append(
            f"-O -i prod.mdin -p {stem}.parm7 -c {stem}.equil.rst7 "
            f"-o {stem}.mdout -r {stem}.rst7 -x {stem}.nc -inf {stem}.mdinfo"
        )
    return "\n".join(lines) + "\n"


def render_run_script(cfg: RefepProdConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    env_lines = "\n".join(
        f"export {key}={shlex.quote(str(value))}" for key, value in cfg.runtime.env.items()
    )
    replicas = cfg.equil.prep.windows
    ranks = replicas * cfg.runtime.processes_per_replica
    return f"""#!/usr/bin/env bash
{strict}

module purge
module load {shlex.quote(cfg.runtime.module)}

{env_lines}

mdexec="$(command -v {shlex.quote(cfg.runtime.executable)} || true)"
launcher="$(command -v {shlex.quote(cfg.runtime.mpi_launcher)} || true)"
if [[ -z "$mdexec" ]]; then
    echo "ERROR: {cfg.runtime.executable} not found after loading {cfg.runtime.module}" >&2
    exit 1
fi
if [[ -z "$launcher" ]]; then
    echo "ERROR: {cfg.runtime.mpi_launcher} not found after loading {cfg.runtime.module}" >&2
    exit 1
fi

"$launcher" -np {ranks} "$mdexec" \\
  -ng {replicas} \\
  -groupfile groupfile \\
  -rem {cfg.runtime.rem_mode} \\
  -remlog rem.log < /dev/null
"""


def render_slurm_script(cfg: RefepProdConfig) -> str:
    job = cfg.slurm
    return f"""#!/usr/bin/env bash
#SBATCH -J {job.name}
#SBATCH -p {job.partition}
#SBATCH -N {job.nodes}
#SBATCH -n {job.ntasks}
#SBATCH --ntasks-per-node={job.tasks_per_node}
#SBATCH -t {job.time}
#SBATCH -o {job.stdout}
#SBATCH -e {job.stderr}

bash run.sh
"""


def render_prod_manifest(windows: list[CompletedEquilWindow]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "window",
            "lambda",
            "topology",
            "starting_restart",
            "source_topology_sha256",
            "source_restart_sha256",
        ]
    )
    for window in windows:
        stem = _window_name(window.index)
        writer.writerow(
            [
                window.index,
                window.lambda_value,
                f"{stem}.parm7",
                f"{stem}.equil.rst7",
                window.topology_sha256,
                window.restart_sha256,
            ]
        )
    return handle.getvalue()


def render_provenance(cfg: RefepProdConfig) -> str:
    total_steps = int(cfg.cntrl["nstlim"]) * int(cfg.cntrl["numexchg"])
    return yaml.safe_dump(
        {
            "method": "Hamiltonian replica exchange REFEP",
            "amber_rem_mode": cfg.runtime.rem_mode,
            "replicas": cfg.equil.prep.windows,
            "processes_per_replica": cfg.runtime.processes_per_replica,
            "exchange_interval_steps": int(cfg.cntrl["nstlim"]),
            "exchange_attempts": int(cfg.cntrl["numexchg"]),
            "total_steps_per_replica": total_steps,
            "total_time_ps_per_replica": total_steps * float(cfg.cntrl["dt"]),
            "equilibration_config": str(cfg.equil_yaml),
        },
        sort_keys=False,
    )


def prepare_refep_prod(
    cfg: RefepProdConfig,
    output_dir: Path | None = None,
) -> Path:
    completed = validate_completed_equilibration(cfg)
    destination = output_dir.resolve() if output_dir else refep_prod_dir(cfg)
    if destination.exists():
        raise FileExistsError(
            f"REFEP production directory already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".prod-work-", dir=destination.parent) as tmp:
        work = Path(tmp)
        for window in completed:
            stem = _window_name(window.index)
            shutil.copy2(window.topology, work / f"{stem}.parm7")
            shutil.copy2(window.restart, work / f"{stem}.equil.rst7")

        (work / "prod.mdin").write_text(render_prod_mdin(cfg))
        (work / "groupfile").write_text(render_groupfile(cfg))
        (work / "lambda-manifest.csv").write_text(render_prod_manifest(completed))
        (work / "refep-prod-spec.yaml").write_text(cfg.yaml_path.read_text())
        (work / "provenance.yaml").write_text(render_provenance(cfg))
        run_script = work / "run.sh"
        run_script.write_text(render_run_script(cfg))
        run_script.chmod(0o755)
        slurm_script = work / "slurm.sh"
        slurm_script.write_text(render_slurm_script(cfg))
        slurm_script.chmod(0o755)
        Path(tmp).replace(destination)
    return destination


def run_refep_prod_prep(yaml_path: Path) -> None:
    cfg = load_refep_prod_config(yaml_path)
    destination = prepare_refep_prod(cfg)
    exchange_ps = int(cfg.cntrl["nstlim"]) * float(cfg.cntrl["dt"])
    total_ns = exchange_ps * int(cfg.cntrl["numexchg"]) / 1000.0
    frames = (
        int(cfg.cntrl["nstlim"])
        * int(cfg.cntrl["numexchg"])
        // int(cfg.cntrl["ntwx"])
    )
    print(f"OK: wrote {cfg.equil.prep.windows}-replica REFEP H-REMD inputs in {destination}")
    print(
        f"OK: exchange every {exchange_ps:g} ps for {cfg.cntrl['numexchg']} attempts; "
        f"{total_ns:g} ns and {frames} saved frames per replica"
    )
    print("NOTE: preparation only; no Amber job was run or submitted")


def submit_slurm(slurm_script: Path) -> None:
    subprocess.run(
        ["sbatch", slurm_script.name],
        cwd=slurm_script.parent,
        check=True,
    )


def _existing_prod_outputs(
    destination: Path, cfg: RefepProdConfig
) -> list[Path]:
    outputs: list[Path] = []
    for index in range(cfg.equil.prep.windows):
        stem = _window_name(index)
        for suffix in ("mdout", "rst7", "nc", "mdinfo"):
            path = destination / f"{stem}.{suffix}"
            if path.exists():
                outputs.append(path)
    for name in ("rem.log", "rem.type"):
        path = destination / name
        if path.exists():
            outputs.append(path)
    outputs.extend(destination.glob("logfile.*"))
    outputs.extend(destination.glob("slurm.*.out"))
    outputs.extend(destination.glob("slurm.*.err"))
    return sorted(set(outputs))


def submit_refep_prod(
    cfg: RefepProdConfig,
    yaml_text: str,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
    output_dir: Path | None = None,
) -> bool:
    destination = output_dir.resolve() if output_dir else refep_prod_dir(cfg)
    spec = destination / "refep-prod-spec.yaml"
    if not spec.is_file() or spec.read_text() != yaml_text:
        print(f"SKIP: {spec} does not exactly match the supplied config")
        return False

    try:
        completed = validate_completed_equilibration(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"SKIP: REFEP equilibration stage is no longer valid: {exc}")
        return False

    expected_topologies = {
        f"{_window_name(index)}.parm7" for index in range(cfg.equil.prep.windows)
    }
    expected_restarts = {
        f"{_window_name(index)}.equil.rst7"
        for index in range(cfg.equil.prep.windows)
    }
    actual_topologies = {path.name for path in destination.glob("lambda-*.parm7")}
    actual_restarts = {
        path.name for path in destination.glob("lambda-*.equil.rst7")
    }
    if actual_topologies != expected_topologies or actual_restarts != expected_restarts:
        print("SKIP: prepared lambda topology/restart set does not match the ladder")
        return False

    for window in completed:
        stem = _window_name(window.index)
        prepared_topology = destination / f"{stem}.parm7"
        prepared_restart = destination / f"{stem}.equil.rst7"
        if _sha256(prepared_topology) != window.topology_sha256:
            print(f"SKIP: {prepared_topology} differs from its validated topology")
            return False
        if _sha256(prepared_restart) != window.restart_sha256:
            print(f"SKIP: {prepared_restart} differs from its completed equilibration restart")
            return False

    expected_text = {
        "prod.mdin": render_prod_mdin(cfg),
        "groupfile": render_groupfile(cfg),
        "lambda-manifest.csv": render_prod_manifest(completed),
        "provenance.yaml": render_provenance(cfg),
        "run.sh": render_run_script(cfg),
        "slurm.sh": render_slurm_script(cfg),
    }
    for name, expected in expected_text.items():
        path = destination / name
        if not path.is_file() or path.read_text() != expected:
            print(f"SKIP: {path} is missing or does not match the supplied config")
            return False

    existing_outputs = _existing_prod_outputs(destination, cfg)
    if existing_outputs:
        print("SKIP: refusing to overwrite existing REFEP production outputs:")
        for path in existing_outputs:
            print(f"  - {path}")
        return False

    total_ps = (
        int(cfg.cntrl["nstlim"])
        * int(cfg.cntrl["numexchg"])
        * float(cfg.cntrl["dt"])
    )
    print("Will submit the following REFEP H-REMD production directory:")
    print(f"  - {destination}")
    print(f"  - {cfg.equil.prep.windows} replicas; {total_ps:g} ps per replica")
    print(
        f"  - {cfg.slurm.ntasks} MPI ranks "
        f"({cfg.runtime.processes_per_replica} per replica)"
    )
    response = confirm("Proceed to submit 1 H-REMD production job? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancelled by user.")
        return False

    slurm_script = destination / "slurm.sh"
    print(f"Submitting job via sbatch for {destination}...")
    (submitter or submit_slurm)(slurm_script)
    print("OK: job submitted")
    return True


def run_refep_prod_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_refep_prod_config(resolved)
    submit_refep_prod(cfg, resolved.read_text())
