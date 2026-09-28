"""Prepare independent per-lambda NVT equilibration inputs for REFEP."""

from __future__ import annotations

import csv
import hashlib
import io
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from qmmd.refep_prep import RefepPrepConfig, load_refep_prep_config


@dataclass(frozen=True, slots=True)
class RefepEquilRuntimeConfig:
    module: str
    executable: str
    launcher: str
    strict_mode: bool


@dataclass(frozen=True, slots=True)
class RefepEquilSlurmConfig:
    name: str
    partition: str
    nodes: int
    ntasks: int
    tasks_per_node: int
    time: str
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RefepEquilConfig:
    yaml_path: Path
    prep_yaml: Path
    prep: RefepPrepConfig
    stage_dirname: str
    description: str
    cntrl: dict[str, Any]
    runtime: RefepEquilRuntimeConfig
    slurm: RefepEquilSlurmConfig


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


def load_refep_equil_config(yaml_path: Path) -> RefepEquilConfig:
    resolved = yaml_path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("REFEP equilibration configuration must be a YAML mapping")

    prep_value = Path(_nonempty_string(data.get("prep_yaml"), "prep_yaml"))
    prep_yaml = (
        prep_value.resolve()
        if prep_value.is_absolute()
        else (resolved.parent / prep_value).resolve()
    )
    if not prep_yaml.is_file():
        raise FileNotFoundError(f"Missing prep_yaml: {prep_yaml}")
    prep = load_refep_prep_config(prep_yaml)

    mdin = _mapping(data.get("equil_mdin"), "equil_mdin")
    if len(mdin) != 1:
        raise ValueError("equil_mdin must contain exactly one MD stage")
    stage = _mapping(next(iter(mdin.values())), "equil_mdin stage")
    description = _nonempty_string(stage.get("description"), "equil_mdin.description")
    cntrl = dict(_mapping(stage.get("cntrl"), "equil_mdin.cntrl"))
    required = {
        "imin",
        "irest",
        "ntx",
        "nstlim",
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
        raise ValueError("Missing REFEP equilibration cntrl fields: " + ", ".join(missing))
    if int(cntrl["imin"]) != 0 or int(cntrl["irest"]) != 1 or int(cntrl["ntx"]) != 5:
        raise ValueError("REFEP equilibration must restart MD with imin=0, irest=1, and ntx=5")
    if int(cntrl["ntb"]) != 1 or int(cntrl["ntp"]) != 0:
        raise ValueError("REFEP equilibration must use NVT with ntb=1 and ntp=0")
    if int(cntrl["nstlim"]) <= 0 or float(cntrl["dt"]) <= 0:
        raise ValueError("equil_mdin nstlim and dt must be positive")
    if int(cntrl["ntc"]) != 2 or int(cntrl["ntf"]) != 2:
        raise ValueError("REFEP equilibration requires SHAKE with ntc=2 and ntf=2")
    if float(cntrl["cut"]) <= 0:
        raise ValueError("equil_mdin cut must be positive")
    if any(int(cntrl[key]) <= 0 for key in ("ntpr", "ntwx", "ntwr")):
        raise ValueError("equil_mdin ntpr, ntwx, and ntwr must be positive")

    runtime_data = _mapping(data.get("runtime"), "runtime")
    runtime = RefepEquilRuntimeConfig(
        module=_nonempty_string(runtime_data.get("module"), "runtime.module"),
        executable=_nonempty_string(
            runtime_data.get("executable"), "runtime.executable"
        ),
        launcher=_nonempty_string(runtime_data.get("launcher"), "runtime.launcher"),
        strict_mode=bool(runtime_data.get("strict_mode", True)),
    )

    slurm_data = _mapping(data.get("slurm"), "slurm")
    job = _mapping(slurm_data.get("job"), "slurm.job")
    slurm = RefepEquilSlurmConfig(
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
    if slurm.ntasks != prep.windows:
        raise ValueError(
            "slurm.job.ntasks must equal the REFEP lambda-window count "
            f"({prep.windows})"
        )
    if slurm.ntasks > slurm.nodes * slurm.tasks_per_node:
        raise ValueError("Slurm nodes * tasks_per_node cannot be smaller than ntasks")

    return RefepEquilConfig(
        yaml_path=resolved,
        prep_yaml=prep_yaml,
        prep=prep,
        stage_dirname=_single_component(
            data.get("stage_dirname", "equil"), "stage_dirname"
        ),
        description=description,
        cntrl=cntrl,
        runtime=runtime,
        slurm=slurm,
    )


def refep_equil_dir(cfg: RefepEquilConfig) -> Path:
    return cfg.prep.output_directory.parent / cfg.stage_dirname


def _window_name(index: int) -> str:
    return f"lambda-{index:03d}"


def _format_amber_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    return str(value)


def render_equil_mdin(cfg: RefepEquilConfig, index: int) -> str:
    lam = index / (cfg.prep.windows - 1)
    lines = [
        f"{cfg.description}; lambda window {index:03d} (lambda={lam:.10f})",
        "&cntrl",
    ]
    lines.extend(
        f"  {key}={_format_amber_value(value)}," for key, value in cfg.cntrl.items()
    )
    lines.extend(["/", ""])
    return "\n".join(lines)


def render_window_run_script(cfg: RefepEquilConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    executable = shlex.quote(cfg.runtime.executable)
    return f"""#!/usr/bin/env bash
{strict}

script_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$script_dir"

mdexec="$(command -v {executable} || true)"
if [[ -z "$mdexec" ]]; then
    echo "ERROR: {cfg.runtime.executable} not found" >&2
    exit 1
fi

"$mdexec" -O \\
  -i equil.mdin \\
  -p system.parm7 \\
  -c start.rst7 \\
  -r equil.rst7 \\
  -o equil.mdout \\
  -inf equil.mdinfo \\
  -x equil.nc
"""


def render_run_script(cfg: RefepEquilConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    launcher = shlex.quote(cfg.runtime.launcher)
    windows = " ".join(_window_name(index) for index in range(cfg.prep.windows))
    return f"""#!/usr/bin/env bash
{strict}

module purge
module load {shlex.quote(cfg.runtime.module)}

launcher="$(command -v {launcher} || true)"
if [[ -z "$launcher" ]]; then
    echo "ERROR: {cfg.runtime.launcher} not found after loading {cfg.runtime.module}" >&2
    exit 1
fi

pids=()
for window in {windows}; do
    "$launcher" --exclusive --exact --ntasks=1 --cpus-per-task=1 \\
      bash "$window/run.sh" > "$window/launch.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${{pids[@]}}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "ERROR: one or more lambda equilibrations failed" >&2
    exit "$status"
fi
echo "OK: all {cfg.prep.windows} lambda equilibrations completed"
"""


def render_slurm_script(cfg: RefepEquilConfig) -> str:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_topology_manifest(topology_dir: Path, windows: int) -> list[dict[str, str]]:
    manifest_path = topology_dir / "topology-manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing REFEP topology manifest: {manifest_path}")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != windows:
        raise ValueError(
            f"REFEP topology manifest has {len(rows)} windows; expected {windows}"
        )
    expected_fields = {"window", "lambda", "topology", "sha256"}
    if not rows or not expected_fields.issubset(rows[0]):
        raise ValueError(f"Malformed REFEP topology manifest: {manifest_path}")
    for index, row in enumerate(rows):
        expected_name = f"lambda-{index:03d}.parm7"
        if int(row["window"]) != index or row["topology"] != expected_name:
            raise ValueError(
                f"Unexpected topology-manifest row {index}: {row.get('topology')}"
            )
        topology = topology_dir / expected_name
        if not topology.is_file() or topology.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty REFEP topology: {topology}")
        if _sha256(topology) != row["sha256"]:
            raise ValueError(f"REFEP topology checksum mismatch: {topology}")
    return rows


def _render_equil_manifest(rows: list[dict[str, str]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        ["window", "lambda", "directory", "topology", "starting_restart"]
    )
    for index, row in enumerate(rows):
        window = _window_name(index)
        writer.writerow(
            [index, row["lambda"], window, f"{window}/system.parm7", f"{window}/start.rst7"]
        )
    return handle.getvalue()


def _validate_topology_stage(cfg: RefepEquilConfig) -> tuple[Path, list[dict[str, str]]]:
    topology_dir = cfg.prep.output_directory
    spec = topology_dir / "refep-prep-spec.yaml"
    if not spec.is_file() or spec.read_text() != cfg.prep_yaml.read_text():
        raise ValueError(
            f"{spec} does not exactly match the referenced REFEP prep configuration"
        )
    restart = topology_dir / "common.rst7"
    if not restart.is_file() or restart.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty common REFEP restart: {restart}")
    return restart, _read_topology_manifest(topology_dir, cfg.prep.windows)


def prepare_refep_equil(
    cfg: RefepEquilConfig,
    output_dir: Path | None = None,
) -> Path:
    restart, rows = _validate_topology_stage(cfg)
    destination = output_dir.resolve() if output_dir else refep_equil_dir(cfg)
    if destination.exists():
        raise FileExistsError(
            f"REFEP equilibration directory already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".equil-work-", dir=destination.parent) as tmp:
        work = Path(tmp)
        for index, row in enumerate(rows):
            window = work / _window_name(index)
            window.mkdir()
            source_topology = cfg.prep.output_directory / row["topology"]
            shutil.copy2(source_topology, window / "system.parm7")
            shutil.copy2(restart, window / "start.rst7")
            (window / "equil.mdin").write_text(render_equil_mdin(cfg, index))
            window_run = window / "run.sh"
            window_run.write_text(render_window_run_script(cfg))
            window_run.chmod(0o755)

        (work / "equil-manifest.csv").write_text(_render_equil_manifest(rows))
        (work / "refep-equil-spec.yaml").write_text(cfg.yaml_path.read_text())
        run_script = work / "run.sh"
        run_script.write_text(render_run_script(cfg))
        run_script.chmod(0o755)
        slurm_script = work / "slurm.sh"
        slurm_script.write_text(render_slurm_script(cfg))
        slurm_script.chmod(0o755)
        Path(tmp).replace(destination)
    return destination


def run_refep_equil_prep(yaml_path: Path) -> None:
    cfg = load_refep_equil_config(yaml_path)
    destination = prepare_refep_equil(cfg)
    time_ps = int(cfg.cntrl["nstlim"]) * float(cfg.cntrl["dt"])
    print(f"OK: wrote {cfg.prep.windows} independent REFEP NVT equilibration inputs in {destination}")
    print(f"OK: each lambda window will equilibrate for {time_ps:g} ps")
    print("NOTE: preparation only; no Amber job was run or submitted")


def submit_slurm(slurm_script: Path) -> None:
    subprocess.run(
        ["sbatch", slurm_script.name],
        cwd=slurm_script.parent,
        check=True,
    )


def _existing_equil_outputs(
    destination: Path, cfg: RefepEquilConfig
) -> list[Path]:
    outputs: list[Path] = []
    for index in range(cfg.prep.windows):
        window = destination / _window_name(index)
        for name in (
            "equil.mdout",
            "equil.rst7",
            "equil.nc",
            "equil.mdinfo",
            "launch.log",
        ):
            path = window / name
            if path.exists():
                outputs.append(path)
    outputs.extend(destination.glob("slurm.*.out"))
    outputs.extend(destination.glob("slurm.*.err"))
    return sorted(set(outputs))


def submit_refep_equil(
    cfg: RefepEquilConfig,
    yaml_text: str,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
    output_dir: Path | None = None,
) -> bool:
    destination = output_dir.resolve() if output_dir else refep_equil_dir(cfg)
    spec = destination / "refep-equil-spec.yaml"
    if not spec.is_file() or spec.read_text() != yaml_text:
        print(f"SKIP: {spec} does not exactly match the supplied config")
        return False

    try:
        restart, rows = _validate_topology_stage(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"SKIP: REFEP topology stage is no longer valid: {exc}")
        return False

    expected_windows = {_window_name(index) for index in range(cfg.prep.windows)}
    actual_windows = {
        path.name
        for path in destination.glob("lambda-*")
        if path.is_dir()
    }
    if actual_windows != expected_windows:
        print("SKIP: prepared lambda-directory set does not match the topology ladder")
        return False

    expected_top_level = {
        "equil-manifest.csv": _render_equil_manifest(rows),
        "run.sh": render_run_script(cfg),
        "slurm.sh": render_slurm_script(cfg),
    }
    for name, expected in expected_top_level.items():
        path = destination / name
        if not path.is_file() or path.read_text() != expected:
            print(f"SKIP: {path} is missing or does not match the supplied config")
            return False

    restart_checksum = _sha256(restart)
    for index, row in enumerate(rows):
        window = destination / _window_name(index)
        prepared_topology = window / "system.parm7"
        prepared_restart = window / "start.rst7"
        expected_text = {
            "equil.mdin": render_equil_mdin(cfg, index),
            "run.sh": render_window_run_script(cfg),
        }
        if not prepared_topology.is_file() or _sha256(prepared_topology) != row["sha256"]:
            print(f"SKIP: {prepared_topology} differs from its validated lambda topology")
            return False
        if not prepared_restart.is_file() or _sha256(prepared_restart) != restart_checksum:
            print(f"SKIP: {prepared_restart} differs from the validated common restart")
            return False
        for name, expected in expected_text.items():
            path = window / name
            if not path.is_file() or path.read_text() != expected:
                print(f"SKIP: {path} is missing or does not match the supplied config")
                return False

    existing_outputs = _existing_equil_outputs(destination, cfg)
    if existing_outputs:
        print("SKIP: refusing to overwrite existing REFEP equilibration outputs:")
        for path in existing_outputs:
            print(f"  - {path}")
        return False

    print("Will submit the following REFEP equilibration directory:")
    print(f"  - {destination}")
    print(f"  - {cfg.prep.windows} independent lambda windows")
    response = confirm("Proceed to submit 1 equilibration job? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancelled by user.")
        return False

    slurm_script = destination / "slurm.sh"
    print(f"Submitting job via sbatch for {destination}...")
    (submitter or submit_slurm)(slurm_script)
    print("OK: job submitted")
    return True


def run_refep_equil_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_refep_equil_config(resolved)
    submit_refep_equil(cfg, resolved.read_text())
