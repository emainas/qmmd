"""Prepare restrained DCDFTBMD production runs for umbrella windows."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from qmmd.dftb import stage_skf_files
from qmmd.us_equil import (
    USEquilConfig,
    load_config as load_equil_config,
    read_xyz,
    render_dftb_input,
    render_metacv,
    render_run_sh,
    render_slurm_sh,
)
from qmmd.us_pull import find_repo_root, pull_dir, window_centers


@dataclass(frozen=True, slots=True)
class USProdConfig:
    equil_yaml: Path
    equil: USEquilConfig
    run: USEquilConfig
    restart_name: str
    equil_source: str = "restart"
    trajectory_name: str = "traject"
    allow_incomplete_equilibration: bool = False
    incomplete_equilibration_windows: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class LastTrajectoryFrame:
    atom_count: int
    step: int
    time_ps: float
    coordinates: tuple[tuple[str, float, float, float], ...]


def _single_filename(value: object, field: str) -> str:
    text = str(value)
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"{field} must be a single filename")
    return text


def _header_value(lines: list[str], key: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(key)}\s*=\s*([^\s,)]+)", re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def _logical_true(value: str | None) -> bool:
    return value is not None and value.upper() in {"TRUE", "T", ".TRUE."}


def _endpoint_ps(cfg: USEquilConfig, stage: str) -> float:
    nstep_text = _header_value(cfg.dftb.header_lines, "NSTEP")
    timestep_text = _header_value(cfg.dftb.header_lines, "DELTAT")
    if nstep_text is None or timestep_text is None:
        raise ValueError(f"{stage} NSTEP and DELTAT must be explicit")
    nstep = int(nstep_text)
    timestep_seconds = float(timestep_text.replace("D", "E").replace("d", "e"))
    if nstep <= 0 or not math.isfinite(timestep_seconds) or timestep_seconds <= 0:
        raise ValueError(f"{stage} NSTEP and DELTAT must be positive")
    return nstep * timestep_seconds * 1.0e12


def production_duration_ps(cfg: USProdConfig) -> float:
    """Return the duration of the newly prepared production segment."""
    duration = _endpoint_ps(cfg.run, "Production") - production_start_ps(cfg)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            "Production endpoint must be later than its configured starting time"
        )
    return duration


def production_start_ps(cfg: USProdConfig) -> float:
    """Return the time represented by production step zero."""
    if cfg.equil_source == "last_trajectory_frame":
        return 0.0
    return _endpoint_ps(cfg.equil, "Equilibration")


def dftb_terminated_normally(path: Path, tail_bytes: int = 16384) -> bool:
    """Check DCDFTBMD's final marker without loading a potentially huge output."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - tail_bytes))
        tail = stream.read().decode(errors="replace")
    return "Execution of DCDFTBMD terminated normally" in tail


_DFTB_PROGRESS_RE = re.compile(
    r"\*\*\*\s+AT\s+T=\s*([+\-0-9.EeDd]+)\s*FSEC,\s*"
    r"THIS RUN'S STEP NO\.=\s*(\d+)",
    re.IGNORECASE,
)

_TRAJECT_TIME_RE = re.compile(
    r"AT T=\s*([+\-0-9.EeDd]+)\s+FSEC,\s*THIS RUN'S STEP NO\.=\s*(\d+)",
    re.IGNORECASE,
)


def last_dftb_progress(path: Path, tail_bytes: int = 1024 * 1024) -> tuple[int, float] | None:
    """Return the last reported ``(step, time_ps)`` from a large DFTB output."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - tail_bytes))
        tail = stream.read().decode(errors="replace")
    matches = list(_DFTB_PROGRESS_RE.finditer(tail))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group(2)), float(match.group(1).replace("D", "E")) / 1000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_last_complete_trajectory_frame(path: Path) -> LastTrajectoryFrame:
    """Read the last complete DFTB XYZ frame, ignoring a growing-file tail."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty equilibration trajectory: {path}")
    last: LastTrajectoryFrame | None = None
    with path.open(errors="replace") as stream:
        while True:
            count_line = stream.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            try:
                atom_count = int(count_line)
            except ValueError as exc:
                raise RuntimeError(
                    f"Malformed DFTB trajectory atom count in {path}: "
                    f"{count_line.strip()!r}"
                ) from exc
            comment = stream.readline()
            if not comment:
                break
            match = _TRAJECT_TIME_RE.search(comment)
            if match is None:
                raise RuntimeError(
                    f"Missing DFTB time/step metadata in {path}: {comment.strip()!r}"
                )
            coordinates: list[tuple[str, float, float, float]] = []
            complete = True
            for _ in range(atom_count):
                line = stream.readline()
                if not line:
                    complete = False
                    break
                fields = line.split()
                if len(fields) < 4:
                    complete = False
                    break
                try:
                    coordinates.append(
                        (fields[0], float(fields[1]), float(fields[2]), float(fields[3]))
                    )
                except ValueError:
                    complete = False
                    break
            if not complete:
                break
            time_fs = float(match.group(1).replace("D", "E").replace("d", "e"))
            frame = LastTrajectoryFrame(
                atom_count=atom_count,
                step=int(match.group(2)),
                time_ps=time_fs / 1000.0,
                coordinates=tuple(coordinates),
            )
            if last is not None and (
                frame.step <= last.step or frame.time_ps <= last.time_ps
            ):
                raise RuntimeError(f"Non-increasing frame metadata in {path}")
            last = frame
    if last is None:
        raise RuntimeError(f"No complete coordinate frames found in {path}")
    return last


def render_last_frame_xyz(
    frame: LastTrajectoryFrame,
    vectors: list[tuple[float, float, float]],
) -> str:
    """Serialize the selected source frame with its fixed simulation cell."""
    lattice = " ".join(f"{value:.10f}" for vector in vectors for value in vector)
    lines = [
        str(frame.atom_count),
        f'Lattice="{lattice}" source_step={frame.step} source_time_ps={frame.time_ps:g}',
    ]
    lines.extend(
        f"{symbol:<2s} {x:18.10f} {y:18.10f} {z:18.10f}"
        for symbol, x, y, z in frame.coordinates
    )
    return "\n".join(lines) + "\n"


def load_config(yaml_path: Path) -> USProdConfig:
    resolved = yaml_path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("US production configuration must be a YAML mapping")

    equil_value = Path(str(data.get("equil_yaml", "equil.yaml")))
    equil_yaml = (
        equil_value
        if equil_value.is_absolute()
        else (resolved.parent / equil_value).resolve()
    )
    if not equil_yaml.is_file():
        raise RuntimeError(f"Missing equilibration configuration: {equil_yaml}")

    equil = load_equil_config(equil_yaml)
    run = load_equil_config(resolved)
    if run.pull_yaml != equil.pull_yaml:
        raise ValueError("Production and equilibration must reference the same pull_yaml")
    if run.stage_dirname == equil.stage_dirname:
        raise ValueError("Production stage_dirname must differ from equilibration stage_dirname")
    equil_source = str(data.get("equil_source", "restart"))
    if equil_source not in {"restart", "last_trajectory_frame"}:
        raise ValueError("equil_source must be 'restart' or 'last_trajectory_frame'")
    uses_restart = _logical_true(_header_value(run.dftb.header_lines, "RESTART"))
    if equil_source == "restart" and not uses_restart:
        raise ValueError("Binary-restart production must set RESTART=TRUE")
    if equil_source == "last_trajectory_frame" and uses_restart:
        raise ValueError("Trajectory-frame production must set RESTART=FALSE")
    if _logical_true(_header_value(run.dftb.header_lines, "READVELOCITY")):
        raise ValueError("US production must set READVELOCITY=FALSE")

    allow_incomplete = data.get("allow_incomplete_equilibration", False)
    if type(allow_incomplete) is not bool:
        raise ValueError("allow_incomplete_equilibration must be true or false")

    incomplete_windows_value = data.get("incomplete_equilibration_windows")
    incomplete_windows: tuple[int, ...] | None = None
    if incomplete_windows_value is not None:
        if not isinstance(incomplete_windows_value, list) or any(
            type(index) is not int for index in incomplete_windows_value
        ):
            raise ValueError("incomplete_equilibration_windows must be a list of integers")
        incomplete_windows = tuple(incomplete_windows_value)
        if len(set(incomplete_windows)) != len(incomplete_windows):
            raise ValueError("incomplete_equilibration_windows contains duplicates")
        window_count = len(window_centers(run.pull.windows))
        if any(index < 0 or index >= window_count for index in incomplete_windows):
            raise ValueError(
                f"incomplete_equilibration_windows indices must be between 0 and "
                f"{window_count - 1}"
            )
        if not allow_incomplete:
            raise ValueError(
                "incomplete_equilibration_windows requires "
                "allow_incomplete_equilibration: true"
            )
    if equil_source == "last_trajectory_frame" and (
        allow_incomplete or incomplete_windows is not None
    ):
        raise ValueError(
            "Trajectory-frame sourcing does not use incomplete-equilibration "
            "restart overrides"
        )

    cfg = USProdConfig(
        equil_yaml=equil_yaml,
        equil=equil,
        run=run,
        restart_name=_single_filename(data.get("restart_name", "restart"), "restart_name"),
        equil_source=equil_source,
        trajectory_name=_single_filename(
            data.get("trajectory_name", "traject"), "trajectory_name"
        ),
        allow_incomplete_equilibration=allow_incomplete,
        incomplete_equilibration_windows=incomplete_windows,
    )
    production_duration_ps(cfg)
    return cfg


def prepare_us_prod(
    cfg: USProdConfig,
    yaml_text: str,
    repo_root: Path,
) -> list[Path]:
    root = pull_dir(cfg.run.pull, repo_root)
    centers = window_centers(cfg.run.pull.windows)
    destinations = [
        root / f"window-{index:03d}" / cfg.run.stage_dirname
        for index in range(len(centers))
    ]
    existing = [destination for destination in destinations if destination.exists()]
    if existing:
        raise FileExistsError(f"US production directory already exists; not touching: {existing[0]}")

    source_yaml_text = cfg.equil_yaml.read_text()
    sources: list[
        tuple[
            Path | None,
            list[tuple[float, float, float]],
            list[tuple[str, float, float, float]],
            str | None,
            dict[str, object],
        ]
    ] = []
    for index in range(len(centers)):
        equil_dir = root / f"window-{index:03d}" / cfg.equil.stage_dirname
        spec = equil_dir / "equil_spec.yaml"
        if not spec.is_file() or spec.read_text() != source_yaml_text:
            raise RuntimeError(f"{spec} does not exactly match {cfg.equil_yaml}")
        output = equil_dir / "dftb.out"
        terminated_normally = dftb_terminated_normally(output)
        reference_xyz = equil_dir / "start.xyz"
        if not reference_xyz.is_file() or reference_xyz.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty equilibration reference geometry: {reference_xyz}")
        natoms, vectors, reference_coordinates = read_xyz(reference_xyz)
        provenance: dict[str, object] = {
            "window_index": index,
            "equilibration_directory": str(equil_dir.resolve()),
            "normal_termination": terminated_normally,
            "source_mode": cfg.equil_source,
        }
        if cfg.equil_source == "last_trajectory_frame":
            trajectory = equil_dir / cfg.trajectory_name
            before = trajectory.stat() if trajectory.is_file() else None
            frame = read_last_complete_trajectory_frame(trajectory)
            if frame.atom_count != natoms:
                raise RuntimeError(
                    f"Trajectory/reference atom-count mismatch in {equil_dir}: "
                    f"{frame.atom_count} != {natoms}"
                )
            if [coordinate[0] for coordinate in frame.coordinates] != [
                coordinate[0] for coordinate in reference_coordinates
            ]:
                raise RuntimeError(f"Trajectory/reference element order differs in {equil_dir}")
            frame_xyz = render_last_frame_xyz(frame, vectors)
            provenance.update(
                {
                    "trajectory_source": str(trajectory.resolve()),
                    "trajectory_bytes_at_read_start": before.st_size if before else 0,
                    "selected_frame_step": frame.step,
                    "selected_frame_time_ps": frame.time_ps,
                    "selected_frame_sha256": hashlib.sha256(
                        frame_xyz.encode()
                    ).hexdigest(),
                    "velocities_preserved": False,
                }
            )
            sources.append(
                (None, vectors, list(frame.coordinates), frame_xyz, provenance)
            )
        else:
            incomplete_override = cfg.allow_incomplete_equilibration and (
                cfg.incomplete_equilibration_windows is None
                or index in cfg.incomplete_equilibration_windows
            )
            if not terminated_normally and not incomplete_override:
                detail = ""
                if cfg.allow_incomplete_equilibration:
                    detail = (
                        f"; window {index} is not in "
                        "incomplete_equilibration_windows"
                    )
                raise RuntimeError(
                    f"Equilibration did not terminate normally: {output}{detail}"
                )
            restart = equil_dir / cfg.restart_name
            if not restart.is_file() or restart.stat().st_size == 0:
                raise RuntimeError(f"Missing/empty equilibration restart: {restart}")
            checkpoint = equil_dir / f"{cfg.restart_name}_chk"
            if not terminated_normally and checkpoint.is_file():
                if checkpoint.stat().st_size != restart.stat().st_size:
                    raise RuntimeError(
                        f"Incomplete equilibration restart/checkpoint sizes differ: "
                        f"{restart}, {checkpoint}"
                    )
                if restart.stat().st_mtime_ns < checkpoint.stat().st_mtime_ns:
                    raise RuntimeError(
                        "Incomplete equilibration restart is older than its "
                        f"checkpoint: {restart}"
                    )
            progress = last_dftb_progress(output)
            provenance.update(
                {
                    "incomplete_equilibration_override": not terminated_normally
                    and incomplete_override,
                    "restart_source": str(restart.resolve()),
                    "restart_bytes": restart.stat().st_size,
                    "restart_sha256": _sha256(restart),
                    "velocities_preserved": True,
                }
            )
            if progress is not None:
                provenance["last_dftb_output_step"] = progress[0]
                provenance["last_dftb_output_time_ps"] = progress[1]
            sources.append(
                (restart, vectors, reference_coordinates, None, provenance)
            )

    params = repo_root / cfg.run.dftb.params_dir
    if not params.is_dir():
        raise RuntimeError(f"Missing DFTB parameter directory: {params}")
    elements = [element.symbol for element in cfg.run.dftb.elements]
    for left in elements:
        for right in elements:
            parameter = params / f"{left}-{right}.skf"
            if not parameter.is_file():
                raise RuntimeError(f"Missing SKF file: {parameter}")

    rendered: list[tuple[str, str]] = []
    for center, (_, vectors, coordinates, _, provenance) in zip(centers, sources):
        natoms = len(coordinates)
        if max(cfg.run.pull.restraint.atoms) > natoms:
            raise ValueError(
                f"Dihedral atom ID exceeds {natoms} atoms in window "
                f"{provenance['window_index']}"
            )
        seed = secrets.randbelow(2**31 - 1) + 1
        rendered.append(
            (
                render_dftb_input(cfg.run, natoms, vectors, coordinates, seed),
                render_metacv(cfg.run, center),
            )
        )

    for index, (destination, source, files) in enumerate(
        zip(destinations, sources, rendered)
    ):
        restart, _, _, frame_xyz, provenance = source
        dftb_input, metacv = files
        destination.mkdir()
        (destination / "dftb.inp").write_text(dftb_input)
        (destination / "metacv.dat").write_text(metacv)
        if restart is not None:
            shutil.copy2(restart, destination / cfg.restart_name)
        if frame_xyz is not None:
            (destination / "equil_last_frame.xyz").write_text(frame_xyz)
        (destination / "equil_source.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        (destination / "prod_spec.yaml").write_text(yaml_text)
        run_sh = destination / "run.sh"
        run_sh.write_text(render_run_sh(cfg.run))
        run_sh.chmod(0o755)
        slurm_sh = destination / "slurm.sh"
        slurm_sh.write_text(render_slurm_sh(cfg.run, index))
        slurm_sh.chmod(0o755)
        stage_skf_files(params, destination, elements)
        if provenance.get("incomplete_equilibration_override", False):
            progress_text = "unknown time"
            if "last_dftb_output_time_ps" in provenance:
                progress_text = f"{provenance['last_dftb_output_time_ps']:g} ps"
            print(
                f"WARNING: window-{index:03d} uses a valid non-normal equilibration "
                f"restart; dftb.out reached {progress_text}"
            )
        if cfg.equil_source == "last_trajectory_frame":
            print(
                f"SOURCE: window-{index:03d} trajectory frame at "
                f"{provenance['selected_frame_time_ps']:g} ps "
                f"(step {provenance['selected_frame_step']})"
            )
    return destinations


def run_us_prod_prep(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    destinations = prepare_us_prod(cfg, resolved.read_text(), find_repo_root(resolved))
    print(
        f"OK: prepared {len(destinations)} restrained DCDFTBMD production windows "
        f"under {destinations[0].parent.parent} ({production_duration_ps(cfg):g} ps/window)"
    )
    print("NOTE: inputs only; no DCDFTBMD jobs were run or submitted")


def submit_slurm(slurm_sh: Path) -> None:
    """Submit one prepared production window from its own directory."""
    subprocess.run(["sbatch", slurm_sh.name], cwd=slurm_sh.parent, check=True)


def submit_us_prod(
    cfg: USProdConfig,
    yaml_text: str,
    repo_root: Path,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
) -> bool:
    """Validate and submit the complete set of prepared production windows."""
    root = pull_dir(cfg.run.pull, repo_root)
    centers = window_centers(cfg.run.pull.windows)
    targets = [
        root / f"window-{index:03d}" / cfg.run.stage_dirname
        for index in range(len(centers))
    ]

    problems: list[str] = []
    required_files = [
        "dftb.inp",
        "metacv.dat",
        "run.sh",
        "slurm.sh",
        "equil_source.json",
    ]
    if cfg.equil_source == "restart":
        required_files.append(cfg.restart_name)
    else:
        required_files.append("equil_last_frame.xyz")
    for target in targets:
        spec = target / "prod_spec.yaml"
        if not spec.is_file() or spec.read_text() != yaml_text:
            problems.append(f"{spec} does not exactly match the supplied config")
            continue
        for filename in required_files:
            path = target / filename
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f"missing/empty {path}")
    if problems:
        print("SKIP: US production set is incomplete or does not match config; nothing submitted")
        for problem in problems:
            print(f"  - {problem}")
        return False

    print("Will submit the following US production directories:")
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


def run_us_prod_submit(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    submit_us_prod(cfg, resolved.read_text(), find_repo_root(resolved))
