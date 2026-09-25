from __future__ import annotations

import csv
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


SOLVENT_IONS = {
    "WAT", "Na+", "Br-", "Cl-", "Cs+", "F-", "I-", "K+", "Li+",
    "Mg+", "Rb+", "CIO", "IB", "MG2",
}


@dataclass(frozen=True)
class ChargeSets:
    atom_names: tuple[str, ...]
    prot_charges: tuple[float, ...]
    deprot_charges: tuple[float, ...]
    proton_count_prot: int
    proton_count_deprot: int


@dataclass(frozen=True)
class TopologyInfo:
    residue_number: int
    first_atom: int
    atom_names: tuple[str, ...]
    first_solvent: int


@dataclass(frozen=True)
class RuntimeConfig:
    module: str
    executable: str
    env: dict[str, Any]
    strict_mode: bool


@dataclass(frozen=True)
class SlurmJobConfig:
    name: str
    partition: str
    nodes: int
    time: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class DgrefConfig:
    yaml_path: Path
    system: str
    buffer: float
    prefix: str
    job_name: str
    input_parm7: str
    input_rst7: str
    charge_sets: Path
    statene: tuple[str, ...]
    pka_corr: tuple[float, ...]
    cph_igb: int
    description: str
    cntrl: dict[str, Any]
    runtime: RuntimeConfig
    slurm: SlurmJobConfig
    report_discard_first: int = 0


def find_repo_root(start: Path) -> Path:
    resolved = start.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"Could not find repository root from {start}")


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _comma_values(value: object, field: str) -> list[str]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        values = [str(item).strip() for item in value]
    else:
        raise ValueError(f"{field} must be a comma-separated string or list")
    if not values or any(not item for item in values):
        raise ValueError(f"{field} contains an empty value")
    return values


def load_config(yaml_path: Path) -> DgrefConfig:
    resolved = yaml_path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("CpHMD dgref configuration must be a YAML mapping")

    system = _nonempty_string(data.get("system"), "system")
    prefix = _nonempty_string(data.get("prefix"), "prefix")
    job_name = _nonempty_string(data.get("job_name"), "job_name")
    input_parm7 = _nonempty_string(data.get("input_parm7"), "input_parm7")
    input_rst7 = _nonempty_string(data.get("input_rst7"), "input_rst7")
    buffer = float(data["buffer"])
    if buffer <= 0:
        raise ValueError("buffer must be positive")

    cpin = data.get("cpin")
    if not isinstance(cpin, dict):
        raise ValueError("cpin must be a YAML mapping")
    charge_value = _nonempty_string(cpin.get("charge_sets"), "cpin.charge_sets")
    charge_sets = Path(charge_value)
    if not charge_sets.is_absolute():
        charge_sets = find_repo_root(resolved) / charge_sets
    charge_sets = charge_sets.resolve()
    if not charge_sets.is_file():
        raise FileNotFoundError(f"Missing charge_sets: {charge_sets}")

    statene = tuple(_comma_values(cpin.get("statene"), "cpin.statene"))
    pka_values = _comma_values(cpin.get("pka_corr"), "cpin.pka_corr")
    try:
        pka_corr = tuple(float(value) for value in pka_values)
    except ValueError as exc:
        raise ValueError("cpin.pka_corr values must be numeric") from exc
    if len(statene) != len(pka_corr):
        raise ValueError("cpin.statene and cpin.pka_corr must have equal lengths")
    if sum(value.upper() == "DELTAGREF" for value in statene) != 1:
        raise ValueError("cpin.statene must contain DELTAGREF exactly once")
    cph_igb = int(cpin["cph_igb"])
    if cph_igb not in {1, 2, 5, 7, 8}:
        raise ValueError("cpin.cph_igb must be one of 1, 2, 5, 7, or 8")

    mdin = data.get("dgref_mdin")
    if not isinstance(mdin, dict) or len(mdin) != 1:
        raise ValueError("dgref_mdin must contain exactly one MD stage")
    stage = next(iter(mdin.values()))
    if not isinstance(stage, dict) or not isinstance(stage.get("cntrl"), dict):
        raise ValueError("dgref_mdin stage must contain a cntrl mapping")
    description = _nonempty_string(stage.get("description"), "dgref_mdin.description")
    cntrl = dict(stage["cntrl"])
    required_cph = {"icnstph", "ntcnstph", "solvph", "ntrelax", "saltcon"}
    missing_cph = sorted(required_cph - cntrl.keys())
    if missing_cph:
        raise ValueError("Missing CpHMD cntrl fields: " + ", ".join(missing_cph))
    if int(cntrl["icnstph"]) != 2:
        raise ValueError("dgref requires explicit-solvent icnstph=2")
    if int(cntrl.get("ntb", 0)) != 1 or int(cntrl.get("ntp", -1)) != 0:
        raise ValueError("CpHMD dgref requires constant-volume ntb=1 and ntp=0")
    if int(cntrl["ntcnstph"]) <= 0 or int(cntrl["ntrelax"]) < 0:
        raise ValueError("ntcnstph must be positive and ntrelax non-negative")
    if float(cntrl["saltcon"]) < 0:
        raise ValueError("saltcon must be non-negative")
    variable_state = next(i for i, value in enumerate(statene) if value.upper() == "DELTAGREF")
    if abs(float(cntrl["solvph"]) - pka_corr[variable_state]) > 1e-9:
        raise ValueError("solvph must equal pka_corr for the DELTAGREF state")

    runtime_data = data.get("runtime")
    if not isinstance(runtime_data, dict):
        raise ValueError("runtime must be a YAML mapping")
    env = runtime_data.get("env", {})
    if not isinstance(env, dict):
        raise ValueError("runtime.env must be a YAML mapping")
    runtime = RuntimeConfig(
        module=_nonempty_string(runtime_data.get("module"), "runtime.module"),
        executable=_nonempty_string(runtime_data.get("executable"), "runtime.executable"),
        env=dict(env),
        strict_mode=bool(runtime_data.get("strict_mode", True)),
    )

    slurm_data = data.get("slurm")
    if not isinstance(slurm_data, dict) or not isinstance(slurm_data.get("job"), dict):
        raise ValueError("slurm.job must be a YAML mapping")
    job = slurm_data["job"]
    slurm = SlurmJobConfig(
        name=_nonempty_string(job.get("name"), "slurm.job.name"),
        partition=_nonempty_string(job.get("partition"), "slurm.job.partition"),
        nodes=int(job["nodes"]),
        time=_nonempty_string(job.get("time"), "slurm.job.time"),
        stdout=_nonempty_string(job.get("stdout"), "slurm.job.stdout"),
        stderr=_nonempty_string(job.get("stderr"), "slurm.job.stderr"),
    )
    if slurm.nodes <= 0:
        raise ValueError("slurm.job.nodes must be positive")

    report_data = data.get("report", {})
    if not isinstance(report_data, dict):
        raise ValueError("report must be a YAML mapping")
    report_discard_first = int(report_data.get("discard_first", 0))
    if report_discard_first < 0:
        raise ValueError("report.discard_first must be non-negative")

    return DgrefConfig(
        yaml_path=resolved,
        system=system,
        buffer=buffer,
        prefix=prefix,
        job_name=job_name,
        input_parm7=input_parm7,
        input_rst7=input_rst7,
        charge_sets=charge_sets,
        statene=statene,
        pka_corr=pka_corr,
        cph_igb=cph_igb,
        description=description,
        cntrl=cntrl,
        runtime=runtime,
        slurm=slurm,
        report_discard_first=report_discard_first,
    )


def read_charge_sets(path: Path) -> ChargeSets:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != ["atom_master", "charge_prot", "charge_deprot"]:
        raise ValueError(f"Unexpected charge-set header in {path}")

    atom_names: list[str] = []
    prot_charges: list[float] = []
    deprot_charges: list[float] = []
    total_row: list[str] | None = None
    proton_row: list[str] | None = None
    for row in rows[1:]:
        if len(row) != 3:
            raise ValueError(f"Every charge-set row must have three columns in {path}")
        if row[0] == "TOTAL":
            total_row = row
        elif row[0] == "PROTON_COUNT":
            proton_row = row
        else:
            if total_row is not None or proton_row is not None:
                raise ValueError("Atom rows must precede TOTAL and PROTON_COUNT")
            atom_names.append(row[0])
            try:
                prot_charges.append(float(row[1]))
                deprot_charges.append(float(row[2]))
            except ValueError as exc:
                raise ValueError(f"Invalid charge row for atom {row[0]}") from exc

    if not atom_names or total_row is None or proton_row is None:
        raise ValueError("Charge sets require atom, TOTAL, and PROTON_COUNT rows")
    if len(set(atom_names)) != len(atom_names):
        raise ValueError("Charge-set atom names must be unique")
    try:
        reported_totals = (float(total_row[1]), float(total_row[2]))
        proton_counts = (int(proton_row[1]), int(proton_row[2]))
    except ValueError as exc:
        raise ValueError("Invalid TOTAL or PROTON_COUNT row") from exc
    calculated_totals = (sum(prot_charges), sum(deprot_charges))
    if any(abs(a - b) > 5e-7 for a, b in zip(reported_totals, calculated_totals)):
        raise ValueError("Reported charge totals do not match the charge arrays")
    if proton_counts[0] - proton_counts[1] != 1:
        raise ValueError("Base states must differ by exactly one proton")

    return ChargeSets(
        atom_names=tuple(atom_names),
        prot_charges=tuple(prot_charges),
        deprot_charges=tuple(deprot_charges),
        proton_count_prot=proton_counts[0],
        proton_count_deprot=proton_counts[1],
    )


def _read_prmtop_flag(lines: list[str], name: str) -> list[str]:
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == f"%FLAG {name}")
    except StopIteration as exc:
        raise ValueError(f"Missing %FLAG {name} in topology") from exc
    if start + 1 >= len(lines):
        raise ValueError(f"Missing format for %FLAG {name}")
    match = re.search(r"\(\d+[A-Za-z](\d+)", lines[start + 1])
    if match is None:
        raise ValueError(f"Unsupported topology format for %FLAG {name}")
    width = int(match.group(1))
    values: list[str] = []
    for line in lines[start + 2:]:
        if line.startswith("%FLAG"):
            break
        values.extend(
            chunk.strip()
            for offset in range(0, len(line), width)
            if (chunk := line[offset:offset + width]).strip()
        )
    return values


def read_topology_info(path: Path, residue_name: str) -> TopologyInfo:
    lines = path.read_text().splitlines()
    pointers = [int(value) for value in _read_prmtop_flag(lines, "POINTERS")]
    natom = pointers[0]
    nres = pointers[11]
    atom_names = _read_prmtop_flag(lines, "ATOM_NAME")[:natom]
    residue_labels = _read_prmtop_flag(lines, "RESIDUE_LABEL")[:nres]
    residue_pointers = [
        int(value) for value in _read_prmtop_flag(lines, "RESIDUE_POINTER")[:nres]
    ]
    matches = [i for i, label in enumerate(residue_labels) if label == residue_name]
    if len(matches) != 1:
        raise ValueError(
            f"Topology must contain exactly one {residue_name} residue; found {len(matches)}"
        )
    residue_index = matches[0]
    first_atom = residue_pointers[residue_index]
    last_atom = (
        residue_pointers[residue_index + 1] - 1
        if residue_index + 1 < nres
        else natom
    )
    first_solvent = next(
        (
            residue_pointers[i]
            for i, label in enumerate(residue_labels)
            if label in SOLVENT_IONS
        ),
        0,
    )
    if first_solvent == 0:
        raise ValueError("Explicit-solvent topology contains no recognized water or ions")
    return TopologyInfo(
        residue_number=residue_index + 1,
        first_atom=first_atom,
        atom_names=tuple(atom_names[first_atom - 1:last_atom]),
        first_solvent=first_solvent,
    )


def _wrapped_field(prefix: str, values: list[str], width: int = 80) -> list[str]:
    lines: list[str] = []
    line = prefix
    for value in values:
        word = f"{value},"
        if len(line) + len(word) > width:
            lines.append(line)
            line = f" {word}"
        else:
            line += word
    lines.append(line)
    return lines


def render_two_state_cpin(
    system: str,
    charges: ChargeSets,
    topology: TopologyInfo,
    statene: tuple[str, str],
    pka_corr: tuple[float, float],
    cph_igb: int,
) -> str:
    if topology.atom_names != charges.atom_names:
        raise ValueError(
            "Charge-set atom ordering does not match the topology residue: "
            f"{charges.atom_names} != {topology.atom_names}"
        )
    flattened = [*charges.prot_charges, *charges.deprot_charges]
    lines = [
        "&CNSTPHE_LIMITS",
        f" ntres=1, maxh=2, natchrg={len(flattened)}, ntstates=2,",
        "/",
        "&CNSTPH",
    ]
    lines.extend(_wrapped_field(" CHRGDAT=", [f"{charge:.6f}" for charge in flattened]))
    lines.extend(
        _wrapped_field(
            " PROTCNT=",
            [str(charges.proton_count_prot), str(charges.proton_count_deprot)],
        )
    )
    lines.extend(
        _wrapped_field(
            " RESNAME=",
            [f"'System: {system}'", f"'Residue: {system} {topology.residue_number}'"],
        )
    )
    lines.append(" RESSTATE=0,")
    lines.append(
        " STATEINF(0)%FIRST_ATOM="
        f"{topology.first_atom}, STATEINF(0)%FIRST_CHARGE=0, "
        "STATEINF(0)%FIRST_STATE=0,"
    )
    lines.append(
        f" STATEINF(0)%NUM_ATOMS={len(charges.atom_names)}, "
        "STATEINF(0)%NUM_STATES=2,"
    )
    lines.extend(_wrapped_field(" STATENE=", list(statene)))
    lines.extend(_wrapped_field(" PKA_CORR=", [f"{value:.4f}" for value in pka_corr]))
    lines.append(
        f" TRESCNT=1, CPHFIRST_SOL={topology.first_solvent}, "
        f"CPH_IGB={cph_igb}, CPH_INTDIEL=1.0,"
    )
    lines.append("/")
    return "\n".join(lines) + "\n"


def render_cpin(cfg: DgrefConfig, charges: ChargeSets, topology: TopologyInfo) -> str:
    if len(cfg.statene) != 2 or len(cfg.pka_corr) != 2:
        raise ValueError("Base dgref preparation currently requires exactly two states")
    return render_two_state_cpin(
        cfg.system,
        charges,
        topology,
        (cfg.statene[0], cfg.statene[1]),
        (cfg.pka_corr[0], cfg.pka_corr[1]),
        cfg.cph_igb,
    )


def render_mdin(cfg: DgrefConfig) -> str:
    lines = [cfg.description, "&cntrl"]
    lines.extend(f"  {key}={value}," for key, value in cfg.cntrl.items())
    lines.append("/")
    return "\n".join(lines) + "\n"


def render_run_script(cfg: DgrefConfig) -> str:
    strict = "set -euo pipefail" if cfg.runtime.strict_mode else ""
    env_lines = "\n".join(
        f"export {key}={value}" for key, value in cfg.runtime.env.items()
    )
    return f"""#!/usr/bin/env bash
{strict}

module purge
module load {cfg.runtime.module}

{env_lines}

mdexec="$(command -v {cfg.runtime.executable} || true)"
finddgref="$(command -v finddgref.py || true)"
if [[ -z "$mdexec" ]]; then
    echo "ERROR: {cfg.runtime.executable} not found after loading {cfg.runtime.module}" >&2
    exit 1
fi
if [[ -z "$finddgref" ]]; then
    echo "ERROR: finddgref.py not found after loading {cfg.runtime.module}" >&2
    exit 1
fi

"$finddgref" \\
  -mdexec "$mdexec" \\
  -i dgref.mdin \\
  -o dgref.out \\
  -inf mdinfo \\
  -log dgref.log \\
  -p {cfg.input_parm7} \\
  -c {cfg.input_rst7} \\
  -r dgref.rst7 \\
  -x dgref.nc \\
  -ref {cfg.input_rst7} \\
  -cpin dgref.cpin
"""


def render_slurm_script(cfg: DgrefConfig) -> str:
    job = cfg.slurm
    return f"""#!/usr/bin/env bash
#SBATCH -J {job.name}
#SBATCH -p {job.partition}
#SBATCH -N {job.nodes}
#SBATCH -t {job.time}
#SBATCH -o {job.stdout}
#SBATCH -e {job.stderr}

bash run.sh
"""


def system_base_dir(cfg: DgrefConfig, repo_root: Path) -> Path:
    return repo_root / "systems" / cfg.system / f"{cfg.prefix}_{cfg.buffer:.1f}"


def dgref_dir(cfg: DgrefConfig, repo_root: Path) -> Path:
    return system_base_dir(cfg, repo_root) / "cphmd" / cfg.job_name


def prepare_dgref(
    cfg: DgrefConfig,
    repo_root: Path,
    output_dir: Path | None = None,
) -> Path:
    base = system_base_dir(cfg, repo_root)
    parm7_source = base / "prep" / cfg.input_parm7
    rst7_source = base / "mdequil" / cfg.input_rst7
    for label, path in (("parm7", parm7_source), ("rst7", rst7_source)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty {label} input: {path}")

    charges = read_charge_sets(cfg.charge_sets)
    topology = read_topology_info(parm7_source, cfg.system)
    destination = output_dir.resolve() if output_dir else dgref_dir(cfg, repo_root)
    destination.mkdir(parents=True, exist_ok=True)

    shutil.copy2(parm7_source, destination / cfg.input_parm7)
    shutil.copy2(rst7_source, destination / cfg.input_rst7)
    (destination / "dgref.cpin").write_text(render_cpin(cfg, charges, topology))
    (destination / "dgref.mdin").write_text(render_mdin(cfg))
    run_script = destination / "run.sh"
    run_script.write_text(render_run_script(cfg))
    run_script.chmod(0o755)
    slurm_script = destination / "slurm.sh"
    slurm_script.write_text(render_slurm_script(cfg))
    slurm_script.chmod(0o755)
    (destination / "dgref_spec.yaml").write_text(cfg.yaml_path.read_text())
    return destination


def run_cphmd_dgref_prep(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    destination = prepare_dgref(cfg, find_repo_root(yaml_path))
    print(f"OK: wrote CpHMD dgref inputs in {destination}")
    print("NOTE: preparation only; no Amber job was run or submitted")


def submit_slurm(slurm_script: Path) -> None:
    subprocess.run(
        ["sbatch", slurm_script.name],
        cwd=slurm_script.parent,
        check=True,
    )


def submit_dgref(
    cfg: DgrefConfig,
    yaml_text: str,
    repo_root: Path,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
    output_dir: Path | None = None,
) -> bool:
    destination = output_dir.resolve() if output_dir else dgref_dir(cfg, repo_root)
    spec = destination / "dgref_spec.yaml"
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

    charges = read_charge_sets(cfg.charge_sets)
    topology = read_topology_info(prepared_parm7, cfg.system)
    expected_text = {
        "dgref.cpin": render_cpin(cfg, charges, topology),
        "dgref.mdin": render_mdin(cfg),
        "run.sh": render_run_script(cfg),
        "slurm.sh": render_slurm_script(cfg),
    }
    for name, expected in expected_text.items():
        path = destination / name
        if not path.is_file() or path.read_text() != expected:
            print(f"SKIP: {path} is missing or does not match the supplied config")
            return False

    print("Will submit the following CpHMD dgref directory:")
    print(f"  - {destination}")
    response = confirm("Proceed to submit 1 job? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancelled by user.")
        return False

    slurm_script = destination / "slurm.sh"
    print(f"Submitting job via sbatch for {destination}...")
    (submitter or submit_slurm)(slurm_script)
    print("OK: job submitted")
    return True


def run_cphmd_dgref_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    submit_dgref(cfg, resolved.read_text(), find_repo_root(resolved))
