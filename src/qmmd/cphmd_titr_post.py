from __future__ import annotations

import csv
import io
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from qmmd.cphmd_dgref import find_repo_root, read_charge_sets
from qmmd.cphmd_titr import TitrConfig, _replica_stem, load_titr_config, titr_dir


_STATE_LABEL_RE = re.compile(r"^(?P<residue>.+):(?P<resid>\d+)%(?P<member>\d+)$")
_FIRST_CPOUT_TIME_RE = re.compile(
    r"^Time:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*$", re.MULTILINE
)
_PROCESSED_FRAMES_RE = re.compile(
    r"Read\s+(\d+)\s+frames\s+and\s+processed\s+(\d+)\s+frames\."
)


@dataclass(frozen=True)
class TitrPostConfig:
    titr: TitrConfig
    output_dir: str
    cpptraj_executable: str
    fraction: str


@dataclass(frozen=True)
class StateColumn:
    residue: str
    residue_id: int
    member: int


CpptrajRunner = Callable[[Path, Path, str, str], None]


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def load_titr_post_config(yaml_path: Path) -> TitrPostConfig:
    resolved = yaml_path.resolve()
    titr = load_titr_config(resolved)
    data = yaml.safe_load(resolved.read_text())
    post = data.get("postprocess", {})
    if not isinstance(post, dict):
        raise ValueError("postprocess must be a YAML mapping")

    output_dir = _nonempty_string(post.get("output_dir", "post"), "postprocess.output_dir")
    output_path = Path(output_dir)
    if output_path.is_absolute() or ".." in output_path.parts or output_path == Path("."):
        raise ValueError("postprocess.output_dir must be a child path inside the titration directory")

    cpptraj_executable = _nonempty_string(
        post.get("cpptraj_executable", "cpptraj"),
        "postprocess.cpptraj_executable",
    )
    fraction = _nonempty_string(
        post.get("fraction", "protonated"), "postprocess.fraction"
    ).lower()
    if fraction not in {"protonated", "deprotonated"}:
        raise ValueError("postprocess.fraction must be protonated or deprotonated")

    ntwx = int(titr.cntrl.get("ntwx", 0))
    ntcnstph = int(titr.cntrl["ntcnstph"])
    if ntwx <= 0:
        raise ValueError("remd_mdin.cntrl.ntwx must be positive for postprocessing")
    if ntwx % ntcnstph:
        raise ValueError(
            "remd_mdin.cntrl.ntwx must be an integer multiple of ntcnstph "
            "to align coordinates and protonation states"
        )
    total_steps = int(titr.cntrl["nstlim"]) * int(titr.cntrl["numexchg"])
    if total_steps % ntcnstph or total_steps % ntwx:
        raise ValueError(
            "total RECpHMD steps must be divisible by both ntcnstph and ntwx "
            "for complete postprocessing records"
        )

    return TitrPostConfig(
        titr=titr,
        output_dir=output_dir,
        cpptraj_executable=cpptraj_executable,
        fraction=fraction,
    )


def post_dir(cfg: TitrPostConfig, repo_root: Path) -> Path:
    return titr_dir(cfg.titr, repo_root) / cfg.output_dir


def _cpptraj_replica_paths(cfg: TitrConfig, suffix: str) -> list[str]:
    return [
        f"../{_replica_stem(replica, len(cfg.ph_values))}.{suffix}"
        for replica in range(1, len(cfg.ph_values) + 1)
    ]


def render_trajectory_cpptraj_input(cfg: TitrPostConfig) -> str:
    trajectories = _cpptraj_replica_paths(cfg.titr, "nc")
    return "\n".join(
        [
            f"parm ../{cfg.titr.input_parm7}",
            f"ensemble {trajectories[0]} trajnames {','.join(trajectories[1:])}",
            "trajout fixed-ph.nc netcdf",
            "run",
            "quit",
            "",
        ]
    )


def render_protonation_cpptraj_input(cfg: TitrPostConfig) -> str:
    cpouts = _cpptraj_replica_paths(cfg.titr, "cpout")
    cpin = f"../{_replica_stem(1, len(cfg.titr.ph_values))}.cpin"
    deprot = " deprot" if cfg.fraction == "deprotonated" else ""
    fraction_data = f"fraction-{cfg.fraction}.dat"
    return "\n".join(
        [
            f"readensembledata {cpouts[0]} filenames {','.join(cpouts[1:])} "
            f"cpin {cpin} name PH",
            "sortensembledata PH",
            f"runanalysis cphstats PH[*] statsout cphstats.dat{deprot} "
            f"fracplot fracplotout {fraction_data}",
            "writedata protonation-states.dat PH[*]",
            "writedata protonation-states.nc PH[*] netcdf",
            "quit",
            "",
        ]
    )


def run_cpptraj(
    input_path: Path,
    log_path: Path,
    module: str,
    executable: str,
) -> None:
    direct = shutil.which(executable)
    if direct:
        command = [direct, "-i", input_path.name]
    else:
        shell_command = "\n".join(
            [
                f"module load {shlex.quote(module)} >/dev/null 2>&1",
                f"exec {shlex.quote(executable)} -i {shlex.quote(input_path.name)}",
            ]
        )
        command = ["bash", "-lc", shell_command]

    result = subprocess.run(
        command,
        cwd=input_path.parent,
        text=True,
        capture_output=True,
    )
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"cpptraj failed for {input_path.name}; see {log_path}"
        )


def _without_postprocess(text: str) -> object:
    data = yaml.safe_load(text)
    if isinstance(data, dict):
        data = dict(data)
        data.pop("postprocess", None)
        data.pop("titration_report", None)
    return data


def validate_completed_titration(cfg: TitrPostConfig, repo_root: Path) -> Path:
    source = titr_dir(cfg.titr, repo_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Missing titration directory: {source}")

    snapshot = source / "cphmd_spec.yaml"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing prepared configuration snapshot: {snapshot}")
    if _without_postprocess(snapshot.read_text()) != _without_postprocess(
        cfg.titr.yaml_path.read_text()
    ):
        raise ValueError(
            "Prepared titration configuration differs from the supplied YAML "
            "(excluding postprocess settings)"
        )

    required = [source / cfg.titr.input_parm7, source / "rem.log"]
    for replica in range(1, len(cfg.titr.ph_values) + 1):
        stem = _replica_stem(replica, len(cfg.titr.ph_values))
        required.extend(
            source / f"{stem}.{suffix}"
            for suffix in ("cpin", "cpout", "nc", "mdout")
        )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Missing/empty titration outputs:\n" + "\n".join(f"  - {path}" for path in missing)
        )

    incomplete: list[Path] = []
    for replica in range(1, len(cfg.titr.ph_values) + 1):
        stem = _replica_stem(replica, len(cfg.titr.ph_values))
        mdout = source / f"{stem}.mdout"
        if "5.  TIMINGS" not in mdout.read_text(errors="replace"):
            incomplete.append(mdout)
    if incomplete:
        raise ValueError(
            "Titration replicas did not finish cleanly:\n"
            + "\n".join(f"  - {path}" for path in incomplete)
        )

    exchange_numbers = [
        int(value)
        for value in re.findall(r"^# exchange\s+(\d+)\s*$", (source / "rem.log").read_text(), re.MULTILINE)
    ]
    expected_exchanges = int(cfg.titr.cntrl["numexchg"])
    if not exchange_numbers or exchange_numbers[-1] != expected_exchanges:
        raise ValueError(
            f"rem.log ends at exchange {exchange_numbers[-1] if exchange_numbers else 0}; "
            f"expected {expected_exchanges}"
        )
    return source


def parse_state_table(path: Path) -> tuple[list[StateColumn], list[list[int]]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or not lines[0].startswith("#Frame"):
        raise ValueError(f"Invalid cpptraj protonation-state table: {path}")
    labels = lines[0].split()[1:]
    columns: list[StateColumn] = []
    for label in labels:
        match = _STATE_LABEL_RE.fullmatch(label)
        if match is None:
            raise ValueError(f"Unrecognized cpptraj state label {label!r} in {path}")
        columns.append(
            StateColumn(
                residue=match.group("residue"),
                residue_id=int(match.group("resid")),
                member=int(match.group("member")),
            )
        )

    rows: list[list[int]] = []
    for expected_frame, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if len(fields) != len(columns) + 1 or int(fields[0]) != expected_frame:
            raise ValueError(f"Malformed frame {expected_frame} in {path}")
        rows.append([int(value) for value in fields[1:]])
    if not rows:
        raise ValueError(f"No protonation states found in {path}")
    return columns, rows


def parse_fraction_plot(path: Path) -> tuple[list[str], list[tuple[float, list[float]]]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or not lines[0].startswith("#pH"):
        raise ValueError(f"Invalid cpptraj fraction plot: {path}")
    labels = lines[0].split()[1:]
    rows: list[tuple[float, list[float]]] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != len(labels) + 1:
            raise ValueError(f"Malformed fraction row in {path}: {line}")
        rows.append((float(fields[0]), [float(value) for value in fields[1:]]))
    return labels, rows


def _first_cpout_time(path: Path) -> float:
    match = _FIRST_CPOUT_TIME_RE.search(path.read_text(errors="replace"))
    if match is None:
        raise ValueError(f"No CpH time record found in {path}")
    return float(match.group(1))


def _format_ph(ph: float) -> str:
    text = f"{ph:.6f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def _write_fraction_csv(
    path: Path,
    labels: list[str],
    rows: list[tuple[float, list[float]]],
    fraction: str,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pH", "residue", "residue_id", f"fraction_{fraction}"])
        parsed_labels: list[tuple[str, int]] = []
        for label in labels:
            residue, separator, resid = label.rpartition(":")
            if not separator or not resid.isdigit():
                raise ValueError(f"Unrecognized cpptraj fraction label {label!r}")
            parsed_labels.append((residue, int(resid)))
        for ph, values in rows:
            for (residue, residue_id), value in zip(parsed_labels, values):
                writer.writerow([f"{ph:.6f}", residue, residue_id, f"{value:.6f}"])


def _write_state_csvs(
    cfg: TitrPostConfig,
    work_dir: Path,
    source: Path,
    columns: list[StateColumn],
    rows: list[list[int]],
) -> tuple[int, int, float]:
    titr = cfg.titr
    ntcnstph = int(titr.cntrl["ntcnstph"])
    ntwx = int(titr.cntrl["ntwx"])
    total_steps = int(titr.cntrl["nstlim"]) * int(titr.cntrl["numexchg"])
    expected_state_records = total_steps // ntcnstph
    expected_coordinate_frames = total_steps // ntwx
    if len(rows) != expected_state_records:
        raise ValueError(
            f"cpptraj produced {len(rows)} CpH records; expected {expected_state_records}"
        )

    member_set = {column.member for column in columns}
    if member_set != set(range(len(titr.ph_values))):
        raise ValueError("Sorted CpH state members do not match the configured pH ladder")

    charges = read_charge_sets(titr.charge_sets)
    proton_counts = (charges.proton_count_prot, charges.proton_count_deprot)
    invalid_states = sorted(
        {state for row in rows for state in row if state < 0 or state >= len(proton_counts)}
    )
    if invalid_states:
        raise ValueError(f"CpH records contain undefined state indices: {invalid_states}")

    first_time = _first_cpout_time(source / f"{_replica_stem(1, len(titr.ph_values))}.cpout")
    state_dt = ntcnstph * float(titr.cntrl["dt"])
    state_csv = work_dir / "protonation-states.csv"
    with state_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record",
                "md_step",
                "cpout_time_ps",
                "pH",
                "residue",
                "residue_id",
                "state",
                "proton_count",
            ]
        )
        for record, values in enumerate(rows, start=1):
            cpout_time = first_time + (record - 1) * state_dt
            for column, state in zip(columns, values):
                writer.writerow(
                    [
                        record,
                        record * ntcnstph,
                        f"{cpout_time:.6f}",
                        f"{titr.ph_values[column.member]:.6f}",
                        column.residue,
                        column.residue_id,
                        state,
                        proton_counts[state],
                    ]
                )

    records_per_coordinate = ntwx // ntcnstph
    aligned_csv = work_dir / "coordinate-protonation.csv"
    with aligned_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trajectory_frame",
                "md_step",
                "relative_time_ps",
                "cpout_record",
                "cpout_time_ps",
                "pH",
                "trajectory",
                "residue",
                "residue_id",
                "state",
                "proton_count",
            ]
        )
        for trajectory_frame in range(1, expected_coordinate_frames + 1):
            record = trajectory_frame * records_per_coordinate
            values = rows[record - 1]
            cpout_time = first_time + (record - 1) * state_dt
            for column, state in zip(columns, values):
                ph = titr.ph_values[column.member]
                writer.writerow(
                    [
                        trajectory_frame,
                        trajectory_frame * ntwx,
                        f"{trajectory_frame * ntwx * float(titr.cntrl['dt']):.6f}",
                        record,
                        f"{cpout_time:.6f}",
                        f"{ph:.6f}",
                        f"ph-{_format_ph(ph)}.nc",
                        column.residue,
                        column.residue_id,
                        state,
                        proton_counts[state],
                    ]
                )
    return expected_state_records, expected_coordinate_frames, first_time


def _validate_trajectory_log(path: Path, expected_frames: int) -> None:
    matches = _PROCESSED_FRAMES_RE.findall(path.read_text(errors="replace"))
    if not matches:
        raise ValueError(f"Could not verify processed coordinate frames in {path}")
    read_count, processed_count = (int(value) for value in matches[-1])
    if read_count != expected_frames or processed_count != expected_frames:
        raise ValueError(
            f"cpptraj processed {processed_count}/{read_count} coordinate frames; "
            f"expected {expected_frames}"
        )


def _render_manifest_csv(cfg: TitrPostConfig, coordinate_frames: int) -> str:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["pH_member", "pH", "trajectory", "coordinate_frames", "topology"])
    for member, ph in enumerate(cfg.titr.ph_values, start=1):
        writer.writerow(
            [member, f"{ph:.6f}", f"ph-{_format_ph(ph)}.nc", coordinate_frames, f"../{cfg.titr.input_parm7}"]
        )
    return handle.getvalue()


def postprocess_titration(
    cfg: TitrPostConfig,
    repo_root: Path,
    runner: CpptrajRunner = run_cpptraj,
) -> Path:
    source = validate_completed_titration(cfg, repo_root)
    destination = post_dir(cfg, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".post-work-", dir=source) as tmp:
        work = Path(tmp)
        trajectory_input = work / "trajectory.cpptraj.in"
        trajectory_log = work / "trajectory.cpptraj.log"
        trajectory_input.write_text(render_trajectory_cpptraj_input(cfg))
        runner(
            trajectory_input,
            trajectory_log,
            cfg.titr.runtime.module,
            cfg.cpptraj_executable,
        )

        expected_coordinate_frames = (
            int(cfg.titr.cntrl["nstlim"])
            * int(cfg.titr.cntrl["numexchg"])
            // int(cfg.titr.cntrl["ntwx"])
        )
        _validate_trajectory_log(trajectory_log, expected_coordinate_frames)
        for member, ph in enumerate(cfg.titr.ph_values):
            generated = work / f"fixed-ph.nc.{member}"
            if not generated.is_file() or generated.stat().st_size == 0:
                raise FileNotFoundError(f"cpptraj did not create {generated}")
            generated.rename(work / f"ph-{_format_ph(ph)}.nc")

        protonation_input = work / "protonation.cpptraj.in"
        protonation_log = work / "protonation.cpptraj.log"
        protonation_input.write_text(render_protonation_cpptraj_input(cfg))
        runner(
            protonation_input,
            protonation_log,
            cfg.titr.runtime.module,
            cfg.cpptraj_executable,
        )

        state_data = work / "protonation-states.dat"
        state_netcdf = work / "protonation-states.nc"
        fraction_data = work / f"fraction-{cfg.fraction}.dat"
        cphstats = work / "cphstats.dat"
        for path in (state_data, state_netcdf, fraction_data, cphstats):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"cpptraj did not create {path}")

        columns, state_rows = parse_state_table(state_data)
        state_records, coordinate_frames, first_time = _write_state_csvs(
            cfg, work, source, columns, state_rows
        )
        labels, fraction_rows = parse_fraction_plot(fraction_data)
        _write_fraction_csv(
            work / f"fraction-{cfg.fraction}.csv",
            labels,
            fraction_rows,
            cfg.fraction,
        )
        (work / "trajectory-manifest.csv").write_text(
            _render_manifest_csv(cfg, coordinate_frames)
        )
        (work / "postprocess.yaml").write_text(
            yaml.safe_dump(
                {
                    "source_config": str(cfg.titr.yaml_path.relative_to(repo_root)),
                    "source_directory": str(source.relative_to(repo_root)),
                    "coordinate_sort": "cpptraj ensemble default replica-index sort (fixed pH)",
                    "protonation_sort": "cpptraj readensembledata plus sortensembledata",
                    "fraction": cfg.fraction,
                    "pH_values": list(cfg.titr.ph_values),
                    "coordinate_frames_per_pH": coordinate_frames,
                    "protonation_records_per_pH": state_records,
                    "first_cpout_time_ps": first_time,
                    "coordinate_state_alignment": (
                        "by MD step; each coordinate frame is paired to the CpH record "
                        "at the same configured step"
                    ),
                },
                sort_keys=False,
            )
        )

        destination.mkdir(parents=True, exist_ok=True)
        for path in work.iterdir():
            path.replace(destination / path.name)

    return destination


def run_cphmd_titr_post(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_titr_post_config(resolved)
    destination = postprocess_titration(cfg, find_repo_root(resolved))
    total_steps = int(cfg.titr.cntrl["nstlim"]) * int(cfg.titr.cntrl["numexchg"])
    coordinate_frames = total_steps // int(cfg.titr.cntrl["ntwx"])
    state_records = total_steps // int(cfg.titr.cntrl["ntcnstph"])
    print(f"OK: wrote fixed-pH RECpHMD analysis inputs in {destination}")
    print(
        f"OK: {len(cfg.titr.ph_values)} trajectories, {coordinate_frames} frames each; "
        f"{state_records} protonation records per pH"
    )
    print(f"OK: cphstats reports fraction {cfg.fraction}")
