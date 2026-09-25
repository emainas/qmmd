from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from qmmd.cphmd_dgref import (
    find_repo_root,
    read_charge_sets,
    read_topology_info,
    render_two_state_cpin,
)


_FINAL_DGREF_RE = re.compile(
    r"The value of DELTAGREF.*?DELTAGREF\s*=\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+kcal/mol"
)
_DGREF_SUCCESS = "The execution of finddgref.py ended with success."


@dataclass(frozen=True)
class TitrRuntimeConfig:
    module: str
    executable: str
    mpi_launcher: str
    processes_per_replica: int
    env: dict[str, Any]
    strict_mode: bool


@dataclass(frozen=True)
class TitrSlurmJobConfig:
    name: str
    partition: str
    nodes: int
    ntasks: int
    time: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TitrConfig:
    yaml_path: Path
    system: str
    buffer: float
    prefix: str
    job_name: str
    input_parm7: str
    input_rst7: str
    charge_sets: Path
    dgref_job_name: str
    pka_corr: tuple[float, float]
    cph_igb: int
    ph_values: tuple[float, ...]
    description: str
    cntrl: dict[str, Any]
    runtime: TitrRuntimeConfig
    slurm: TitrSlurmJobConfig


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _resolve_repo_path(value: object, field: str, repo_root: Path) -> Path:
    path = Path(_nonempty_string(value, field))
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {field}: {resolved}")
    return resolved


def load_titr_config(yaml_path: Path) -> TitrConfig:
    resolved = yaml_path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("CpHMD titration configuration must be a YAML mapping")
    repo_root = find_repo_root(resolved)

    system = _nonempty_string(data.get("system"), "system")
    prefix = _nonempty_string(data.get("prefix"), "prefix")
    job_name = _nonempty_string(data.get("job_name"), "job_name")
    input_parm7 = _nonempty_string(data.get("input_parm7"), "input_parm7")
    input_rst7 = _nonempty_string(data.get("input_rst7"), "input_rst7")
    dgref_job_name = _nonempty_string(data.get("dgref_job_name"), "dgref_job_name")
    if job_name == dgref_job_name:
        raise ValueError("job_name must differ from dgref_job_name")
    buffer = float(data["buffer"])
    if buffer <= 0:
        raise ValueError("buffer must be positive")

    cpin = _mapping(data.get("cpin"), "cpin")
    charge_sets = _resolve_repo_path(cpin.get("charge_sets"), "cpin.charge_sets", repo_root)
    pka_values = cpin.get("pka_corr")
    if not isinstance(pka_values, list) or len(pka_values) != 2:
        raise ValueError("cpin.pka_corr must contain exactly two numeric values")
    pka_corr = (float(pka_values[0]), float(pka_values[1]))
    cph_igb = int(cpin["cph_igb"])
    if cph_igb not in {1, 2, 5, 7, 8}:
        raise ValueError("cpin.cph_igb must be one of 1, 2, 5, 7, or 8")

    ladder = _mapping(data.get("ph_ladder"), "ph_ladder")
    raw_values = ladder.get("values")
    if not isinstance(raw_values, list) or len(raw_values) < 2:
        raise ValueError("ph_ladder.values must contain at least two pH values")
    ph_values = tuple(float(value) for value in raw_values)
    if any(right <= left for left, right in zip(ph_values, ph_values[1:])):
        raise ValueError("ph_ladder.values must be strictly increasing")
    if len(ph_values) % 2:
        raise ValueError("pH replica exchange requires an even number of replicas")

    mdin = _mapping(data.get("remd_mdin"), "remd_mdin")
    if len(mdin) != 1:
        raise ValueError("remd_mdin must contain exactly one MD stage")
    stage = _mapping(next(iter(mdin.values())), "remd_mdin stage")
    description = _nonempty_string(stage.get("description"), "remd_mdin.description")
    cntrl = dict(_mapping(stage.get("cntrl"), "remd_mdin.cntrl"))
    if "solvph" in cntrl:
        raise ValueError("remd_mdin.cntrl.solvph is set from ph_ladder and must be omitted")
    required = {
        "imin", "irest", "ntx", "nstlim", "numexchg", "dt", "ntb", "ntp",
        "icnstph", "ntcnstph", "ntrelax", "saltcon",
    }
    missing = sorted(required - cntrl.keys())
    if missing:
        raise ValueError("Missing RECpHMD cntrl fields: " + ", ".join(missing))
    if int(cntrl["imin"]) != 0 or int(cntrl["irest"]) != 1 or int(cntrl["ntx"]) != 5:
        raise ValueError("RECpHMD must restart MD with imin=0, irest=1, and ntx=5")
    if int(cntrl["ntb"]) != 1 or int(cntrl["ntp"]) != 0:
        raise ValueError("Explicit-solvent RECpHMD requires NVT with ntb=1 and ntp=0")
    if int(cntrl["icnstph"]) != 2:
        raise ValueError("Explicit-solvent RECpHMD requires icnstph=2")
    if int(cntrl["nstlim"]) <= 0 or int(cntrl["numexchg"]) <= 0:
        raise ValueError("nstlim and numexchg must be positive")
    if int(cntrl["ntcnstph"]) <= 0 or int(cntrl["ntrelax"]) < 0:
        raise ValueError("ntcnstph must be positive and ntrelax non-negative")
    if float(cntrl["saltcon"]) < 0:
        raise ValueError("saltcon must be non-negative")

    runtime_data = _mapping(data.get("runtime"), "runtime")
    env = _mapping(runtime_data.get("env", {}), "runtime.env")
    runtime = TitrRuntimeConfig(
        module=_nonempty_string(runtime_data.get("module"), "runtime.module"),
        executable=_nonempty_string(runtime_data.get("executable"), "runtime.executable"),
        mpi_launcher=_nonempty_string(
            runtime_data.get("mpi_launcher"), "runtime.mpi_launcher"
        ),
        processes_per_replica=int(runtime_data.get("processes_per_replica", 1)),
        env=dict(env),
        strict_mode=bool(runtime_data.get("strict_mode", True)),
    )
    if runtime.processes_per_replica <= 0:
        raise ValueError("runtime.processes_per_replica must be positive")

    slurm_data = _mapping(data.get("slurm"), "slurm")
    job = _mapping(slurm_data.get("job"), "slurm.job")
    slurm = TitrSlurmJobConfig(
        name=_nonempty_string(job.get("name"), "slurm.job.name"),
        partition=_nonempty_string(job.get("partition"), "slurm.job.partition"),
        nodes=int(job["nodes"]),
        ntasks=int(job["ntasks"]),
        time=_nonempty_string(job.get("time"), "slurm.job.time"),
        stdout=_nonempty_string(job.get("stdout"), "slurm.job.stdout"),
        stderr=_nonempty_string(job.get("stderr"), "slurm.job.stderr"),
    )
    if slurm.nodes <= 0:
        raise ValueError("slurm.job.nodes must be positive")
    expected_tasks = len(ph_values) * runtime.processes_per_replica
    if slurm.ntasks != expected_tasks:
        raise ValueError(
            "slurm.job.ntasks must equal replica count times processes_per_replica "
            f"({expected_tasks})"
        )

    return TitrConfig(
        yaml_path=resolved,
        system=system,
        buffer=buffer,
        prefix=prefix,
        job_name=job_name,
        input_parm7=input_parm7,
        input_rst7=input_rst7,
        charge_sets=charge_sets,
        dgref_job_name=dgref_job_name,
        pka_corr=pka_corr,
        cph_igb=cph_igb,
        ph_values=ph_values,
        description=description,
        cntrl=cntrl,
        runtime=runtime,
        slurm=slurm,
    )


def read_converged_dgref(log_path: Path) -> float:
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing finddgref log: {log_path}")
    text = log_path.read_text(errors="replace")
    matches = list(_FINAL_DGREF_RE.finditer(text))
    if not matches or _DGREF_SUCCESS not in text[matches[-1].end():]:
        raise ValueError(f"finddgref did not finish successfully in {log_path}")
    return float(matches[-1].group(1))


def system_base_dir(cfg: TitrConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer}"


def titr_dir(cfg: TitrConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / "cphmd" / cfg.job_name


def render_replica_mdin(cfg: TitrConfig, replica: int, ph: float) -> str:
    lines = [f"{cfg.description}; replica {replica} at pH {ph:g}", "&cntrl"]
    lines.extend(f"  {key}={value}," for key, value in cfg.cntrl.items())
    lines.append(f"  solvph={ph:g},")
    lines.append("/")
    return "\n".join(lines) + "\n"


def _replica_stem(replica: int, count: int) -> str:
    width = max(2, len(str(count)))
    return f"replica-{replica:0{width}d}"


def render_groupfile(cfg: TitrConfig) -> str:
    lines: list[str] = []
    for replica, _ph in enumerate(cfg.ph_values, start=1):
        stem = _replica_stem(replica, len(cfg.ph_values))
        lines.append(
            f"-O -i {stem}.mdin -o {stem}.mdout -p {cfg.input_parm7} "
            f"-c {cfg.input_rst7} -r {stem}.rst7 -x {stem}.nc -inf {stem}.mdinfo "
            f"-cpin {stem}.cpin -cpout {stem}.cpout -cprestrt {stem}.cprestrt"
        )
    return "\n".join(lines) + "\n"


def render_run_script(cfg: TitrConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    env_lines = "\n".join(
        f"export {key}={value}" for key, value in cfg.runtime.env.items()
    )
    replicas = len(cfg.ph_values)
    ranks = replicas * cfg.runtime.processes_per_replica
    return f"""#!/usr/bin/env bash
{strict}

module purge
module load {cfg.runtime.module}

{env_lines}

mdexec="$(command -v {cfg.runtime.executable} || true)"
launcher="$(command -v {cfg.runtime.mpi_launcher} || true)"
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
  -rem 4 \\
  -remlog rem.log < /dev/null
"""


def render_slurm_script(cfg: TitrConfig) -> str:
    job = cfg.slurm
    return f"""#!/usr/bin/env bash
#SBATCH -J {job.name}
#SBATCH -p {job.partition}
#SBATCH -N {job.nodes}
#SBATCH -n {job.ntasks}
#SBATCH -t {job.time}
#SBATCH -o {job.stdout}
#SBATCH -e {job.stderr}

bash run.sh
"""


def render_ladder_csv(cfg: TitrConfig) -> str:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["replica", "pH", "mdin"])
    for replica, ph in enumerate(cfg.ph_values, start=1):
        stem = _replica_stem(replica, len(cfg.ph_values))
        writer.writerow([replica, f"{ph:.4f}", f"{stem}.mdin"])
    return handle.getvalue()


def render_dgref_provenance(dgref: float, dgref_log: Path, repo_root: Path) -> str:
    return yaml.safe_dump(
        {
            "dgref_kcal_mol": dgref,
            "source": str(dgref_log.relative_to(repo_root)),
            "method": "final successful finddgref value",
        },
        sort_keys=False,
    )


def prepare_titration(
    cfg: TitrConfig,
    repo_root: Path,
    output_dir: Path | None = None,
) -> tuple[Path, float]:
    base = system_base_dir(cfg, repo_root)
    parm7_source = base / "prep" / cfg.input_parm7
    rst7_source = base / "mdequil" / cfg.input_rst7
    dgref_log = base / "cphmd" / cfg.dgref_job_name / "dgref.log"
    for label, path in (("parm7", parm7_source), ("rst7", rst7_source)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty {label} input: {path}")

    dgref = read_converged_dgref(dgref_log)
    charges = read_charge_sets(cfg.charge_sets)
    topology = read_topology_info(parm7_source, cfg.system)
    cpin_text = render_two_state_cpin(
        cfg.system,
        charges,
        topology,
        (f"{dgref:.6f}", "0.0"),
        cfg.pka_corr,
        cfg.cph_igb,
    )

    destination = output_dir.resolve() if output_dir else titr_dir(cfg, repo_root)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parm7_source, destination / cfg.input_parm7)
    shutil.copy2(rst7_source, destination / cfg.input_rst7)

    for replica, ph in enumerate(cfg.ph_values, start=1):
        stem = _replica_stem(replica, len(cfg.ph_values))
        (destination / f"{stem}.mdin").write_text(
            render_replica_mdin(cfg, replica, ph)
        )
        (destination / f"{stem}.cpin").write_text(cpin_text)

    (destination / "groupfile").write_text(render_groupfile(cfg))
    (destination / "ph-ladder.csv").write_text(render_ladder_csv(cfg))
    (destination / "dgref-value.yaml").write_text(
        render_dgref_provenance(dgref, dgref_log, repo_root)
    )
    run_script = destination / "run.sh"
    run_script.write_text(render_run_script(cfg))
    run_script.chmod(0o755)
    slurm_script = destination / "slurm.sh"
    slurm_script.write_text(render_slurm_script(cfg))
    slurm_script.chmod(0o755)
    (destination / "cphmd_spec.yaml").write_text(cfg.yaml_path.read_text())
    return destination, dgref


def run_cphmd_titr_prep(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_titr_config(resolved)
    destination, dgref = prepare_titration(cfg, find_repo_root(resolved))
    print(f"OK: wrote RECpHMD titration inputs in {destination}")
    print(f"OK: {len(cfg.ph_values)} replicas spanning pH {cfg.ph_values[0]:g}-{cfg.ph_values[-1]:g}")
    print(f"OK: CPIN uses converged DGref = {dgref:+.6f} kcal/mol")
    print("NOTE: preparation only; no Amber job was run or submitted")


def submit_slurm(slurm_script: Path) -> None:
    subprocess.run(
        ["sbatch", slurm_script.name],
        cwd=slurm_script.parent,
        check=True,
    )


def _existing_simulation_outputs(destination: Path, cfg: TitrConfig) -> list[Path]:
    outputs: list[Path] = []
    for replica, _ph in enumerate(cfg.ph_values, start=1):
        stem = _replica_stem(replica, len(cfg.ph_values))
        for suffix in ("mdout", "rst7", "nc", "mdinfo", "cpout", "cprestrt"):
            path = destination / f"{stem}.{suffix}"
            if path.exists():
                outputs.append(path)
    for name in ("rem.log", "rem.type"):
        path = destination / name
        if path.exists():
            outputs.append(path)
    outputs.extend(destination.glob("slurm.*.out"))
    outputs.extend(destination.glob("slurm.*.err"))
    return sorted(set(outputs))


def submit_titration(
    cfg: TitrConfig,
    yaml_text: str,
    repo_root: Path,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
    output_dir: Path | None = None,
) -> bool:
    destination = output_dir.resolve() if output_dir else titr_dir(cfg, repo_root)
    spec = destination / "cphmd_spec.yaml"
    if not spec.is_file() or spec.read_text() != yaml_text:
        print(f"SKIP: {spec} does not exactly match the supplied config")
        return False

    base = system_base_dir(cfg, repo_root)
    parm7_source = base / "prep" / cfg.input_parm7
    rst7_source = base / "mdequil" / cfg.input_rst7
    prepared_parm7 = destination / cfg.input_parm7
    prepared_rst7 = destination / cfg.input_rst7
    for label, source, prepared in (
        ("topology", parm7_source, prepared_parm7),
        ("restart", rst7_source, prepared_rst7),
    ):
        if not source.is_file() or not prepared.is_file():
            print(f"SKIP: missing {label} source or prepared copy")
            return False
        if source.read_bytes() != prepared.read_bytes():
            print(f"SKIP: prepared {label} differs from {source}")
            return False

    dgref_log = base / "cphmd" / cfg.dgref_job_name / "dgref.log"
    dgref = read_converged_dgref(dgref_log)
    charges = read_charge_sets(cfg.charge_sets)
    topology = read_topology_info(prepared_parm7, cfg.system)
    cpin_text = render_two_state_cpin(
        cfg.system,
        charges,
        topology,
        (f"{dgref:.6f}", "0.0"),
        cfg.pka_corr,
        cfg.cph_igb,
    )

    expected_text = {
        "groupfile": render_groupfile(cfg),
        "ph-ladder.csv": render_ladder_csv(cfg),
        "dgref-value.yaml": render_dgref_provenance(dgref, dgref_log, repo_root),
        "run.sh": render_run_script(cfg),
        "slurm.sh": render_slurm_script(cfg),
    }
    expected_mdin: set[str] = set()
    expected_cpin: set[str] = set()
    for replica, ph in enumerate(cfg.ph_values, start=1):
        stem = _replica_stem(replica, len(cfg.ph_values))
        mdin_name = f"{stem}.mdin"
        cpin_name = f"{stem}.cpin"
        expected_mdin.add(mdin_name)
        expected_cpin.add(cpin_name)
        expected_text[mdin_name] = render_replica_mdin(cfg, replica, ph)
        expected_text[cpin_name] = cpin_text

    actual_mdin = {path.name for path in destination.glob("replica-*.mdin")}
    actual_cpin = {path.name for path in destination.glob("replica-*.cpin")}
    if actual_mdin != expected_mdin or actual_cpin != expected_cpin:
        print("SKIP: prepared replica input set does not match the supplied pH ladder")
        return False
    for name, expected in expected_text.items():
        path = destination / name
        if not path.is_file() or path.read_text() != expected:
            print(f"SKIP: {path} is missing or does not match the supplied config")
            return False

    existing_outputs = _existing_simulation_outputs(destination, cfg)
    if existing_outputs:
        print("SKIP: refusing to overwrite existing titration outputs:")
        for path in existing_outputs:
            print(f"  - {path}")
        return False

    print("Will submit the following RECpHMD titration directory:")
    print(f"  - {destination}")
    print(f"  - {len(cfg.ph_values)} replicas")
    response = confirm("Proceed to submit 1 replica-exchange job? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancelled by user.")
        return False

    slurm_script = destination / "slurm.sh"
    print(f"Submitting job via sbatch for {destination}...")
    (submitter or submit_slurm)(slurm_script)
    print("OK: job submitted")
    return True


def run_cphmd_titr_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_titr_config(resolved)
    submit_titration(cfg, resolved.read_text(), find_repo_root(resolved))
