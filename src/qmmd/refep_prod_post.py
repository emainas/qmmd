"""Prepare and submit the REFEP single-point cross-energy grid."""

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

from qmmd.refep_prod import (
    RefepProdConfig,
    load_refep_prod_config,
    refep_prod_dir,
    render_groupfile,
    render_prod_manifest,
    render_prod_mdin,
    render_provenance,
    render_run_script as render_prod_run_script,
    render_slurm_script as render_prod_slurm_script,
    validate_completed_equilibration,
)


_AMBER_FAILURE_RE = re.compile(
    r"\b(?:nan|infinity|fatal|segmentation)\b|"
    r"shake cannot|coordinate resetting|vlimit exceeded|"
    r"bomb|terminated abnormally",
    re.IGNORECASE,
)
_FRAME_RE = re.compile(r"frame\s*=\s*UNLIMITED\s*;\s*//\s*\((\d+) currently\)")
_ATOM_RE = re.compile(r"\batom\s*=\s*(\d+)\s*;")


@dataclass(frozen=True, slots=True)
class RefepPostRuntimeConfig:
    module: str
    executable: str
    launcher: str
    ncdump_executable: str
    cpptraj_executable: str
    local_workers: int
    strict_mode: bool


@dataclass(frozen=True, slots=True)
class RefepPostSlurmConfig:
    name: str
    partition: str
    nodes: int
    ntasks: int
    tasks_per_node: int
    memory: str
    time: str
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RefepProdPostConfig:
    yaml_path: Path
    prod: RefepProdConfig
    stage_dirname: str
    energy_dirname: str
    keep_mask: str | None
    stripped_dirname: str
    description: str
    cntrl: dict[str, Any]
    runtime: RefepPostRuntimeConfig
    slurm: RefepPostSlurmConfig


@dataclass(frozen=True, slots=True)
class CompletedProdWindow:
    index: int
    lambda_value: str
    topology: Path
    trajectory: Path
    restart: Path
    frame_count: int


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


def load_refep_prod_post_config(yaml_path: Path) -> RefepProdPostConfig:
    resolved = yaml_path.resolve()
    prod = load_refep_prod_config(resolved)
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("REFEP production configuration must be a YAML mapping")

    post = _mapping(data.get("single_point"), "single_point")
    raw_keep_mask = post.get("keep_mask")
    keep_mask = None
    if raw_keep_mask is not None:
        keep_mask = _nonempty_string(raw_keep_mask, "single_point.keep_mask")
        if "\n" in keep_mask or "\r" in keep_mask:
            raise ValueError("single_point.keep_mask must be a one-line Amber mask")
    mdin = _mapping(post.get("mdin"), "single_point.mdin")
    if len(mdin) != 1:
        raise ValueError("single_point.mdin must contain exactly one energy stage")
    stage = _mapping(next(iter(mdin.values())), "single_point.mdin stage")
    description = _nonempty_string(
        stage.get("description"), "single_point.mdin.description"
    )
    cntrl = dict(_mapping(stage.get("cntrl"), "single_point.mdin.cntrl"))
    required = {
        "imin",
        "maxcyc",
        "ntb",
        "ntp",
        "cut",
        "ntc",
        "ntf",
        "ioutfm",
        "ntpr",
        "ntwx",
        "ntwe",
    }
    missing = sorted(required - cntrl.keys())
    if missing:
        raise ValueError("Missing single-point cntrl fields: " + ", ".join(missing))
    if int(cntrl["imin"]) != 5 or int(cntrl["maxcyc"]) != 1:
        raise ValueError("Trajectory single-point evaluation requires imin=5 and maxcyc=1")
    if keep_mask is None:
        if int(cntrl["ntb"]) != int(prod.cntrl["ntb"]) or int(cntrl["ntp"]) != int(
            prod.cntrl["ntp"]
        ):
            raise ValueError("Single-point ntb/ntp must match REFEP production")
        if float(cntrl["cut"]) != float(prod.cntrl["cut"]):
            raise ValueError("Single-point cutoff must exactly match REFEP production")
    else:
        if int(cntrl["ntb"]) != 0 or int(cntrl["ntp"]) != 0:
            raise ValueError("Stripped implicit rescoring requires ntb=0 and ntp=0")
        if int(cntrl.get("igb", 0)) <= 0:
            raise ValueError("Stripped implicit rescoring requires a positive igb value")
        if float(cntrl["cut"]) <= 0.0:
            raise ValueError("Stripped implicit-rescoring cutoff must be positive")
    if int(cntrl["ntc"]) != int(prod.cntrl["ntc"]) or int(cntrl["ntf"]) != int(
        prod.cntrl["ntf"]
    ):
        raise ValueError("Single-point ntc/ntf must match REFEP production")
    if int(cntrl["ntpr"]) != 1 or int(cntrl["ntwx"]) != 0 or int(cntrl["ntwe"]) != 0:
        raise ValueError("Single-point evaluation requires ntpr=1, ntwx=0, and ntwe=0")

    runtime_data = _mapping(post.get("runtime"), "single_point.runtime")
    runtime = RefepPostRuntimeConfig(
        module=_nonempty_string(runtime_data.get("module"), "single_point.runtime.module"),
        executable=_nonempty_string(
            runtime_data.get("executable"), "single_point.runtime.executable"
        ),
        launcher=_nonempty_string(
            runtime_data.get("launcher"), "single_point.runtime.launcher"
        ),
        ncdump_executable=_nonempty_string(
            runtime_data.get("ncdump_executable", "ncdump"),
            "single_point.runtime.ncdump_executable",
        ),
        cpptraj_executable=_nonempty_string(
            runtime_data.get("cpptraj_executable", "cpptraj"),
            "single_point.runtime.cpptraj_executable",
        ),
        local_workers=int(runtime_data.get("local_workers", 1)),
        strict_mode=bool(runtime_data.get("strict_mode", True)),
    )
    if runtime.local_workers <= 0:
        raise ValueError("single_point.runtime.local_workers must be positive")

    slurm_data = _mapping(post.get("slurm"), "single_point.slurm")
    job = _mapping(slurm_data.get("job"), "single_point.slurm.job")
    slurm = RefepPostSlurmConfig(
        name=_nonempty_string(job.get("name"), "single_point.slurm.job.name"),
        partition=_nonempty_string(
            job.get("partition"), "single_point.slurm.job.partition"
        ),
        nodes=int(job["nodes"]),
        ntasks=int(job["ntasks"]),
        tasks_per_node=int(job["tasks_per_node"]),
        memory=_nonempty_string(job.get("memory"), "single_point.slurm.job.memory"),
        time=_nonempty_string(job.get("time"), "single_point.slurm.job.time"),
        stdout=_nonempty_string(job.get("stdout"), "single_point.slurm.job.stdout"),
        stderr=_nonempty_string(job.get("stderr"), "single_point.slurm.job.stderr"),
    )
    combinations = prod.equil.prep.windows**2
    if slurm.nodes <= 0 or slurm.ntasks <= 0 or slurm.tasks_per_node <= 0:
        raise ValueError("Single-point Slurm nodes, ntasks, and tasks_per_node must be positive")
    if slurm.ntasks != combinations:
        raise ValueError(
            "single_point.slurm.job.ntasks must equal the full energy-grid size "
            f"({combinations})"
        )
    if slurm.ntasks > slurm.nodes * slurm.tasks_per_node:
        raise ValueError("Single-point nodes * tasks_per_node cannot be smaller than ntasks")

    return RefepProdPostConfig(
        yaml_path=resolved,
        prod=prod,
        stage_dirname=_single_component(
            post.get("stage_dirname", "post"), "single_point.stage_dirname"
        ),
        energy_dirname=_single_component(
            post.get("energy_dirname", "energies"), "single_point.energy_dirname"
        ),
        keep_mask=keep_mask,
        stripped_dirname=_single_component(
            post.get("stripped_dirname", "stripped"),
            "single_point.stripped_dirname",
        ),
        description=description,
        cntrl=cntrl,
        runtime=runtime,
        slurm=slurm,
    )


def refep_prod_post_dir(cfg: RefepProdPostConfig) -> Path:
    return refep_prod_dir(cfg.prod) / cfg.stage_dirname


def _window_name(index: int) -> str:
    return f"lambda-{index:03d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _production_config_payload(text: str) -> dict[str, Any]:
    """Return only the configuration that governed the completed production run."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("REFEP production configuration must be a YAML mapping")
    payload = dict(data)
    payload.pop("single_point", None)
    payload.pop("report", None)
    return payload


def _run_ncdump(cfg: RefepProdPostConfig, trajectory: Path) -> str:
    executable = shutil.which(cfg.runtime.ncdump_executable)
    if executable:
        command = [executable, "-h", str(trajectory)]
    else:
        shell_command = "\n".join(
            [
                f"module load {shlex.quote(cfg.runtime.module)} >/dev/null 2>&1",
                f"exec {shlex.quote(cfg.runtime.ncdump_executable)} -h "
                f"{shlex.quote(str(trajectory))}",
            ]
        )
        command = ["bash", "-c", shell_command]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"ncdump failed for {trajectory}: {result.stderr.strip()}")
    return result.stdout


def _trajectory_frames(cfg: RefepProdPostConfig, trajectory: Path) -> int:
    match = _FRAME_RE.search(_run_ncdump(cfg, trajectory))
    if match is None:
        raise ValueError(f"Could not read NetCDF frame count from {trajectory}")
    return int(match.group(1))


def _trajectory_atoms(cfg: RefepProdPostConfig, trajectory: Path) -> int:
    match = _ATOM_RE.search(_run_ncdump(cfg, trajectory))
    if match is None:
        raise ValueError(f"Could not read NetCDF atom count from {trajectory}")
    return int(match.group(1))


def _parm7_atom_count(path: Path) -> int:
    lines = path.read_text(errors="replace").splitlines()
    flag = next(
        (index for index, line in enumerate(lines) if line.strip() == "%FLAG POINTERS"),
        None,
    )
    if flag is None:
        raise ValueError(f"Missing POINTERS section in {path}")
    for line in lines[flag + 1 :]:
        if line.startswith("%FORMAT"):
            continue
        if line.startswith("%FLAG"):
            break
        fields = line.split()
        if fields:
            return int(fields[0])
    raise ValueError(f"Missing NATOM value in {path}")


def _restart_atom_count(path: Path) -> int:
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2 or not lines[1].split():
        raise ValueError(f"Could not read atom count from restart {path}")
    return int(lines[1].split()[0])


def _cpptraj_token(value: str, field: str) -> str:
    if any(char in value for char in ('"', "\n", "\r")):
        raise ValueError(f"{field} cannot contain quotes or newlines")
    return f'"{value}"'


def _run_cpptraj(cfg: RefepProdPostConfig, script: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(cfg.runtime.cpptraj_executable)
    if executable:
        command = [executable]
    else:
        shell_command = "\n".join(
            [
                f"module load {shlex.quote(cfg.runtime.module)} >/dev/null 2>&1",
                f"exec {shlex.quote(cfg.runtime.cpptraj_executable)}",
            ]
        )
        command = ["bash", "-c", shell_command]
    result = subprocess.run(command, input=script, text=True, capture_output=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"cpptraj stripping failed: {message}")
    return result


def _stripped_relative_path(cfg: RefepProdPostConfig, name: str) -> str:
    return f"{cfg.stripped_dirname}/{name}"


def prepare_stripped_inputs(
    cfg: RefepProdPostConfig,
    windows: list[CompletedProdWindow],
    work: Path,
) -> None:
    if cfg.keep_mask is None:
        return
    stripped = work / cfg.stripped_dirname
    stripped.mkdir()
    strip_mask = f"!({cfg.keep_mask})"
    strip_token = _cpptraj_token(strip_mask, "derived cpptraj strip mask")
    for window in windows:
        stem = _window_name(window.index)
        topology = stripped / f"{stem}.parm7"
        trajectory = stripped / f"{stem}.nc"
        restart = stripped / f"{stem}.rst7"
        source_topology = _cpptraj_token(str(window.topology), "source topology")
        source_trajectory = _cpptraj_token(str(window.trajectory), "source trajectory")
        source_restart = _cpptraj_token(str(window.restart), "source restart")
        output_topology = _cpptraj_token(str(topology), "stripped topology")
        output_trajectory = _cpptraj_token(str(trajectory), "stripped trajectory")
        output_restart = _cpptraj_token(str(restart), "stripped restart")
        script = "\n".join(
            [
                f"parm {source_topology}",
                f"trajin {source_trajectory}",
                f"strip {strip_token}",
                f"trajout {output_trajectory} netcdf nobox novelocity "
                "notemperature noreplicadim",
                "run",
                "clear all",
                f"parm {source_topology}",
                f"trajin {source_restart}",
                f"strip {strip_token}",
                f"trajout {output_restart} restart nobox novelocity "
                "notemperature noreplicadim",
                "run",
                "clear all",
                f"parm {source_topology}",
                f"parmstrip {strip_token}",
                "parmbox nobox",
                f"parmwrite out {output_topology}",
                "run",
                "quit",
                "",
            ]
        )
        result = _run_cpptraj(cfg, script)
        (stripped / f"{stem}.cpptraj.log").write_text(result.stdout + result.stderr)
        for path in (topology, trajectory, restart):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"cpptraj did not create stripped input: {path}")
        atoms = _parm7_atom_count(topology)
        if atoms <= 0 or atoms >= _parm7_atom_count(window.topology):
            raise ValueError(
                f"keep_mask {cfg.keep_mask!r} retained an invalid atom count ({atoms})"
            )
        if _trajectory_atoms(cfg, trajectory) != atoms:
            raise ValueError(f"Stripped topology/trajectory atom mismatch for {stem}")
        if _restart_atom_count(restart) != atoms:
            raise ValueError(f"Stripped topology/restart atom mismatch for {stem}")
        if _trajectory_frames(cfg, trajectory) != window.frame_count:
            raise ValueError(f"Stripped trajectory frame-count mismatch for {stem}")


def validate_stripped_inputs(
    cfg: RefepProdPostConfig,
    windows: list[CompletedProdWindow],
    destination: Path,
) -> None:
    if cfg.keep_mask is None:
        return
    for window in windows:
        stem = _window_name(window.index)
        base = destination / cfg.stripped_dirname
        topology = base / f"{stem}.parm7"
        trajectory = base / f"{stem}.nc"
        restart = base / f"{stem}.rst7"
        log = base / f"{stem}.cpptraj.log"
        for path in (topology, trajectory, restart, log):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty stripped REFEP input: {path}")
        atoms = _parm7_atom_count(topology)
        if _trajectory_atoms(cfg, trajectory) != atoms:
            raise ValueError(f"Stripped topology/trajectory atom mismatch for {stem}")
        if _restart_atom_count(restart) != atoms:
            raise ValueError(f"Stripped topology/restart atom mismatch for {stem}")
        if _trajectory_frames(cfg, trajectory) != window.frame_count:
            raise ValueError(f"Stripped trajectory frame-count mismatch for {stem}")


def validate_completed_production(cfg: RefepProdPostConfig) -> list[CompletedProdWindow]:
    prod = cfg.prod
    destination = refep_prod_dir(prod)
    spec = destination / "refep-prod-spec.yaml"
    if not spec.is_file():
        raise FileNotFoundError(f"Missing REFEP production specification: {spec}")
    if _production_config_payload(spec.read_text()) != _production_config_payload(
        cfg.yaml_path.read_text()
    ):
        raise ValueError(
            f"{spec} does not match the production portion of the supplied prod.yaml"
        )

    equil_windows = validate_completed_equilibration(prod)
    expected_text = {
        "prod.mdin": render_prod_mdin(prod),
        "groupfile": render_groupfile(prod),
        "lambda-manifest.csv": render_prod_manifest(equil_windows),
        "provenance.yaml": render_provenance(prod),
        "run.sh": render_prod_run_script(prod),
        "slurm.sh": render_prod_slurm_script(prod),
    }
    for name, expected in expected_text.items():
        path = destination / name
        if not path.is_file() or path.read_text() != expected:
            raise ValueError(f"Prepared production input differs from prod.yaml: {path}")

    rem_log = destination / "rem.log"
    if not rem_log.is_file() or rem_log.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty replica-exchange log: {rem_log}")
    exchange_numbers = [
        int(value)
        for value in re.findall(r"^# exchange\s+(\d+)\s*$", rem_log.read_text(), re.MULTILINE)
    ]
    expected_exchanges = int(prod.cntrl["numexchg"])
    if exchange_numbers != list(range(1, expected_exchanges + 1)):
        raise ValueError("Replica-exchange log is incomplete or non-sequential")

    total_steps = int(prod.cntrl["nstlim"]) * expected_exchanges
    expected_frames = total_steps // int(prod.cntrl["ntwx"])
    final_step_re = re.compile(rf"NSTEP\s*=\s*{total_steps}\b")
    completed: list[CompletedProdWindow] = []
    for equil in equil_windows:
        stem = _window_name(equil.index)
        topology = destination / f"{stem}.parm7"
        trajectory = destination / f"{stem}.nc"
        restart = destination / f"{stem}.rst7"
        mdout = destination / f"{stem}.mdout"
        mdinfo = destination / f"{stem}.mdinfo"
        for path in (topology, trajectory, restart, mdout, mdinfo):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty REFEP production file: {path}")
        if _sha256(topology) != equil.topology_sha256:
            raise ValueError(f"Production topology checksum mismatch: {topology}")
        text = mdout.read_text(errors="replace")
        if not final_step_re.search(text) or "Final Performance Info:" not in text:
            raise ValueError(f"REFEP production did not reach {total_steps} steps: {mdout}")
        failure = _AMBER_FAILURE_RE.search(text)
        if failure is not None:
            raise ValueError(f"Amber failure marker {failure.group(0)!r} found in {mdout}")
        frames = _trajectory_frames(cfg, trajectory)
        if frames != expected_frames:
            raise ValueError(
                f"Trajectory {trajectory} contains {frames} frames; expected {expected_frames}"
            )
        completed.append(
            CompletedProdWindow(
                index=equil.index,
                lambda_value=equil.lambda_value,
                topology=topology,
                trajectory=trajectory,
                restart=restart,
                frame_count=frames,
            )
        )
    return completed


def _format_amber_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    return str(value)


def render_energy_mdin(cfg: RefepProdPostConfig) -> str:
    lines = [cfg.description, "&cntrl"]
    lines.extend(
        f"  {key}={_format_amber_value(value)}," for key, value in cfg.cntrl.items()
    )
    lines.extend(["/", ""])
    return "\n".join(lines)


def _energy_stem(sample: int, evaluation: int) -> str:
    return f"refep-j{sample:03d}-k{evaluation:03d}"


def render_task_manifest(cfg: RefepProdPostConfig, windows: list[CompletedProdWindow]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "sampling_window",
            "evaluation_window",
            "stem",
            "trajectory",
            "topology",
            "coordinate",
            "frames",
        ]
    )
    for sample in windows:
        for evaluation in windows:
            if cfg.keep_mask is None:
                trajectory = sample.trajectory.name
                topology = evaluation.topology.name
                coordinate = evaluation.restart.name
            else:
                trajectory = _stripped_relative_path(cfg, sample.trajectory.name)
                topology = _stripped_relative_path(cfg, evaluation.topology.name)
                coordinate = _stripped_relative_path(cfg, evaluation.restart.name)
            writer.writerow(
                [
                    sample.index,
                    evaluation.index,
                    _energy_stem(sample.index, evaluation.index),
                    trajectory,
                    topology,
                    coordinate,
                    sample.frame_count,
                ]
            )
    return handle.getvalue()


def render_energy_run_script(cfg: RefepProdPostConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    input_prefix = "" if cfg.keep_mask is not None else "../"
    return f"""#!/usr/bin/env bash
{strict}

script_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$script_dir"

stem="$1"
trajectory="$2"
topology="$3"
coordinate="$4"

energyexec="$(command -v {shlex.quote(cfg.runtime.executable)} || true)"
if [[ -z "$energyexec" ]]; then
    echo "ERROR: {cfg.runtime.executable} not found" >&2
    exit 1
fi

"$energyexec" -O \\
  -i energy.mdin \\
  -p "{input_prefix}$topology" \\
  -c "{input_prefix}$coordinate" \\
  -y "{input_prefix}$trajectory" \\
  -o "{cfg.energy_dirname}/$stem.mdout" \\
  -inf "{cfg.energy_dirname}/$stem.mdinfo" \\
  -r "{cfg.energy_dirname}/$stem.rst7" \\
  -x "{cfg.energy_dirname}/$stem.nc"
"""


def render_local_run_script(cfg: RefepProdPostConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    combinations = cfg.prod.equil.prep.windows**2
    return f"""#!/usr/bin/env bash
{strict}

script_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$script_dir"

workers="${{REFEP_LOCAL_WORKERS:-{cfg.runtime.local_workers}}}"
if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: REFEP_LOCAL_WORKERS must be a positive integer" >&2
    exit 1
fi

pids=()
wait_batch() {{
    local batch_status=0
    local pid
    for pid in "${{pids[@]}}"; do
        if ! wait "$pid"; then
            batch_status=1
        fi
    done
    pids=()
    return "$batch_status"
}}

status=0
launched=0
{{
    IFS= read -r _header
    while IFS=, read -r sample evaluation stem trajectory topology coordinate frames; do
        bash run-energy.sh "$stem" "$trajectory" "$topology" "$coordinate" \\
          > "{cfg.energy_dirname}/$stem.launch.log" 2>&1 &
        pids+=("$!")
        launched=$((launched + 1))
        if (( ${{#pids[@]}} >= workers )); then
            if ! wait_batch; then
                status=1
            fi
            echo "Completed $launched/{combinations} calculations"
        fi
    done
}} < task-manifest.csv

if (( ${{#pids[@]}} > 0 )); then
    if ! wait_batch; then
        status=1
    fi
fi
if (( status != 0 )); then
    echo "ERROR: one or more REFEP single-point calculations failed" >&2
    exit "$status"
fi
echo "OK: all {combinations} REFEP single-point calculations completed locally"
"""


def render_post_run_script(cfg: RefepProdPostConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    combinations = cfg.prod.equil.prep.windows**2
    return f"""#!/usr/bin/env bash
{strict}

module purge
module load {shlex.quote(cfg.runtime.module)}

launcher="$(command -v {shlex.quote(cfg.runtime.launcher)} || true)"
if [[ -z "$launcher" ]]; then
    echo "ERROR: {cfg.runtime.launcher} not found after loading {cfg.runtime.module}" >&2
    exit 1
fi

pids=()
{{
    IFS= read -r _header
    while IFS=, read -r sample evaluation stem trajectory topology coordinate frames; do
        "$launcher" --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \\
          bash run-energy.sh "$stem" "$trajectory" "$topology" "$coordinate" \\
          > "{cfg.energy_dirname}/$stem.launch.log" 2>&1 &
        pids+=("$!")
    done
}} < task-manifest.csv

status=0
for pid in "${{pids[@]}}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "ERROR: one or more REFEP single-point calculations failed" >&2
    exit "$status"
fi
echo "OK: all {combinations} REFEP single-point calculations completed"
"""


def render_post_slurm_script(cfg: RefepProdPostConfig) -> str:
    job = cfg.slurm
    return f"""#!/usr/bin/env bash
#SBATCH -J {job.name}
#SBATCH -p {job.partition}
#SBATCH -N {job.nodes}
#SBATCH -n {job.ntasks}
#SBATCH --ntasks-per-node={job.tasks_per_node}
#SBATCH --mem={job.memory}
#SBATCH -t {job.time}
#SBATCH -o {job.stdout}
#SBATCH -e {job.stderr}

bash run.sh
"""


def render_post_provenance(
    cfg: RefepProdPostConfig, windows: list[CompletedProdWindow]
) -> str:
    provenance: dict[str, Any] = {
        "method": "full REFEP cross-Hamiltonian single-point energy matrix",
        "energy_definition": "U_k(q_j)",
        "sampling_windows": len(windows),
        "evaluation_windows": len(windows),
        "calculations": len(windows) ** 2,
        "frames_per_trajectory": windows[0].frame_count,
        "total_energy_evaluations": len(windows) ** 2 * windows[0].frame_count,
        "production_config": str(cfg.yaml_path),
        "production_directory": str(refep_prod_dir(cfg.prod)),
    }
    if cfg.keep_mask is not None:
        provenance["keep_mask"] = cfg.keep_mask
        provenance["sampling_hamiltonian"] = "explicit periodic REFEP production"
        provenance["evaluation_hamiltonian"] = (
            f"stripped implicit solvent igb={cfg.cntrl.get('igb')}"
        )
        provenance["interpretation"] = (
            "GB rescoring of explicit-solvent ensembles; diagnostic rather than "
            "a rigorously sampled implicit-solvent free energy"
        )
    return yaml.safe_dump(provenance, sort_keys=False)


def prepare_refep_prod_post(
    cfg: RefepProdPostConfig,
    output_dir: Path | None = None,
) -> Path:
    windows = validate_completed_production(cfg)
    destination = output_dir.resolve() if output_dir else refep_prod_post_dir(cfg)
    if destination.exists():
        raise FileExistsError(
            f"REFEP production post directory already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".post-work-", dir=destination.parent) as tmp:
        work = Path(tmp)
        (work / cfg.energy_dirname).mkdir()
        prepare_stripped_inputs(cfg, windows, work)
        (work / "energy.mdin").write_text(render_energy_mdin(cfg))
        (work / "task-manifest.csv").write_text(render_task_manifest(cfg, windows))
        (work / "provenance.yaml").write_text(render_post_provenance(cfg, windows))
        (work / "refep-prod-post-spec.yaml").write_text(cfg.yaml_path.read_text())
        energy_run = work / "run-energy.sh"
        energy_run.write_text(render_energy_run_script(cfg))
        energy_run.chmod(0o755)
        run_script = work / "run.sh"
        run_script.write_text(render_post_run_script(cfg))
        run_script.chmod(0o755)
        local_script = work / "run-local.sh"
        local_script.write_text(render_local_run_script(cfg))
        local_script.chmod(0o755)
        slurm_script = work / "slurm.sh"
        slurm_script.write_text(render_post_slurm_script(cfg))
        slurm_script.chmod(0o755)
        Path(tmp).replace(destination)
    return destination


def run_refep_prod_post_prep(yaml_path: Path) -> None:
    cfg = load_refep_prod_post_config(yaml_path)
    destination = prepare_refep_prod_post(cfg)
    windows = cfg.prod.equil.prep.windows
    frames = (
        int(cfg.prod.cntrl["nstlim"])
        * int(cfg.prod.cntrl["numexchg"])
        // int(cfg.prod.cntrl["ntwx"])
    )
    print(f"OK: wrote REFEP single-point grid inputs in {destination}")
    print(
        f"OK: {windows}x{windows} = {windows**2} jobs; "
        f"{frames} frames each; {windows**2 * frames} energy evaluations"
    )
    print("NOTE: preparation only; no Amber job was run or submitted")


def submit_slurm(slurm_script: Path) -> None:
    subprocess.run(["sbatch", slurm_script.name], cwd=slurm_script.parent, check=True)


def _existing_energy_outputs(destination: Path, cfg: RefepProdPostConfig) -> list[Path]:
    outputs = list((destination / cfg.energy_dirname).iterdir())
    outputs.extend(destination.glob("slurm.*.out"))
    outputs.extend(destination.glob("slurm.*.err"))
    return sorted(set(outputs))


def submit_refep_prod_post(
    cfg: RefepProdPostConfig,
    yaml_text: str,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
    output_dir: Path | None = None,
) -> bool:
    destination = output_dir.resolve() if output_dir else refep_prod_post_dir(cfg)
    spec = destination / "refep-prod-post-spec.yaml"
    if not spec.is_file() or spec.read_text() != yaml_text:
        print(f"SKIP: {spec} does not exactly match the supplied config")
        return False
    try:
        windows = validate_completed_production(cfg)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"SKIP: REFEP production stage is no longer valid: {exc}")
        return False

    expected_text = {
        "energy.mdin": render_energy_mdin(cfg),
        "task-manifest.csv": render_task_manifest(cfg, windows),
        "provenance.yaml": render_post_provenance(cfg, windows),
        "run-energy.sh": render_energy_run_script(cfg),
        "run.sh": render_post_run_script(cfg),
        "slurm.sh": render_post_slurm_script(cfg),
    }
    if cfg.keep_mask is not None:
        expected_text["run-local.sh"] = render_local_run_script(cfg)
    for name, expected in expected_text.items():
        path = destination / name
        if not path.is_file() or path.read_text() != expected:
            print(f"SKIP: {path} is missing or does not match the supplied config")
            return False
    energy_dir = destination / cfg.energy_dirname
    if not energy_dir.is_dir():
        print(f"SKIP: missing energy output directory: {energy_dir}")
        return False
    try:
        validate_stripped_inputs(cfg, windows, destination)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"SKIP: stripped REFEP inputs are invalid: {exc}")
        return False
    existing = _existing_energy_outputs(destination, cfg)
    if existing:
        print("SKIP: refusing to overwrite existing REFEP single-point outputs:")
        for path in existing:
            print(f"  - {path}")
        return False

    combinations = len(windows) ** 2
    print("Will submit the following REFEP single-point energy grid:")
    print(f"  - {destination}")
    print(f"  - {len(windows)}x{len(windows)} = {combinations} calculations")
    print(f"  - {windows[0].frame_count} frames per calculation")
    response = confirm("Proceed to submit 1 single-point grid job? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancelled by user.")
        return False

    slurm_script = destination / "slurm.sh"
    print(f"Submitting job via sbatch for {destination}...")
    (submitter or submit_slurm)(slurm_script)
    print("OK: job submitted")
    return True


def run_refep_prod_post_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_refep_prod_post_config(resolved)
    submit_refep_prod_post(cfg, resolved.read_text())
