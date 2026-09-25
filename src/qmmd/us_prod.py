"""Prepare restrained DCDFTBMD production runs for umbrella windows."""

from __future__ import annotations

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
    """Return new production time beyond the cumulative equilibration restart."""
    duration = _endpoint_ps(cfg.run, "Production") - production_start_ps(cfg)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            "Production endpoint must be later than the equilibration restart endpoint"
        )
    return duration


def production_start_ps(cfg: USProdConfig) -> float:
    """Return the cumulative simulation time stored in the equilibration restart."""
    return _endpoint_ps(cfg.equil, "Equilibration")


def dftb_terminated_normally(path: Path, tail_bytes: int = 16384) -> bool:
    """Check DCDFTBMD's final marker without loading a potentially huge output."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - tail_bytes))
        tail = stream.read().decode(errors="replace")
    return "Execution of DCDFTBMD terminated normally" in tail


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
    if not _logical_true(_header_value(run.dftb.header_lines, "RESTART")):
        raise ValueError("Production dftb.header_lines must set RESTART=TRUE")
    if _logical_true(_header_value(run.dftb.header_lines, "READVELOCITY")):
        raise ValueError("Binary restart production must set READVELOCITY=FALSE")

    cfg = USProdConfig(
        equil_yaml=equil_yaml,
        equil=equil,
        run=run,
        restart_name=_single_filename(data.get("restart_name", "restart"), "restart_name"),
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
    sources: list[tuple[Path, Path]] = []
    for index in range(len(centers)):
        equil_dir = root / f"window-{index:03d}" / cfg.equil.stage_dirname
        spec = equil_dir / "equil_spec.yaml"
        if not spec.is_file() or spec.read_text() != source_yaml_text:
            raise RuntimeError(f"{spec} does not exactly match {cfg.equil_yaml}")
        output = equil_dir / "dftb.out"
        if not dftb_terminated_normally(output):
            raise RuntimeError(f"Equilibration did not terminate normally: {output}")
        restart = equil_dir / cfg.restart_name
        if not restart.is_file() or restart.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty equilibration restart: {restart}")
        reference_xyz = equil_dir / "start.xyz"
        if not reference_xyz.is_file() or reference_xyz.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty equilibration reference geometry: {reference_xyz}")
        sources.append((restart, reference_xyz))

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
    for center, (_, reference_xyz) in zip(centers, sources):
        natoms, vectors, coordinates = read_xyz(reference_xyz)
        if max(cfg.run.pull.restraint.atoms) > natoms:
            raise ValueError(f"Dihedral atom ID exceeds {natoms} atoms in {reference_xyz}")
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
        restart, _ = source
        dftb_input, metacv = files
        destination.mkdir()
        (destination / "dftb.inp").write_text(dftb_input)
        (destination / "metacv.dat").write_text(metacv)
        shutil.copy2(restart, destination / cfg.restart_name)
        (destination / "prod_spec.yaml").write_text(yaml_text)
        run_sh = destination / "run.sh"
        run_sh.write_text(render_run_sh(cfg.run))
        run_sh.chmod(0o755)
        slurm_sh = destination / "slurm.sh"
        slurm_sh.write_text(render_slurm_sh(cfg.run, index))
        slurm_sh.chmod(0o755)
        stage_skf_files(params, destination, elements)
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
    required_files = (
        "dftb.inp",
        "metacv.dat",
        cfg.restart_name,
        "run.sh",
        "slurm.sh",
    )
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
