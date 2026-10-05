"""Report completed independent REFEP lambda-window equilibrations."""

from __future__ import annotations

import csv
import math
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qmmd.cphmd_dgref import find_repo_root
from qmmd.refep_equil import (
    RefepEquilConfig,
    equil_simulation_payload,
    load_refep_equil_config,
    refep_equil_dir,
)


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_THERMO_RE = re.compile(
    rf"NSTEP\s*=\s*(\d+)\s+TIME\(PS\)\s*=\s*({_NUMBER})\s+"
    rf"TEMP\(K\)\s*=\s*({_NUMBER})\s+PRESS\s*=\s*({_NUMBER})[^\n]*\n"
    rf"\s*Etot\s*=\s*({_NUMBER})\s+EKtot\s*=\s*({_NUMBER})\s+"
    rf"EPtot\s*=\s*({_NUMBER})"
)
_AMBER_FAILURE_RE = re.compile(
    r"\b(?:nan|infinity|fatal|segmentation)\b|"
    r"shake cannot|coordinate resetting|vlimit exceeded|"
    r"bomb|terminated abnormally",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RefepDihedralConfig:
    label: str
    atom_ids: tuple[int, int, int, int]
    target_deg: float


@dataclass(frozen=True, slots=True)
class RefepEquilReportConfig:
    equil: RefepEquilConfig
    output_dir: str
    style: Path
    cpptraj_executable: str
    dihedral: RefepDihedralConfig | None


@dataclass(frozen=True, slots=True)
class ThermoSample:
    window: int
    lambda_value: float
    step: int
    time_ps: float
    temperature_k: float
    pressure_bar: float
    total_energy_kcal_mol: float
    kinetic_energy_kcal_mol: float
    potential_energy_kcal_mol: float


@dataclass(frozen=True, slots=True)
class DihedralSample:
    window: int
    lambda_value: float
    frame: int
    time_ps: float
    raw_deg: float
    branch_deg: float


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _child_name(value: object, field: str, default: str) -> str:
    text = default if value is None else _nonempty_string(value, field)
    path = Path(text)
    if path.name != text or text in {".", ".."}:
        raise ValueError(f"{field} must be a single child-directory name")
    return text


def load_refep_equil_report_config(yaml_path: Path) -> RefepEquilReportConfig:
    resolved = yaml_path.resolve()
    equil = load_refep_equil_config(resolved)
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("REFEP equilibration configuration must be a YAML mapping")

    report_raw = data.get("report", {})
    report = _mapping(report_raw, "report")
    output_dir = _child_name(report.get("output_dir"), "report.output_dir", "report")
    cpptraj_executable = _nonempty_string(
        report.get("cpptraj_executable", "cpptraj"),
        "report.cpptraj_executable",
    )
    style = Path(str(report.get("style", "plotting/prl.mplstyle")))
    if not style.is_absolute():
        style = find_repo_root(resolved) / style
    if not style.is_file():
        raise FileNotFoundError(f"Missing matplotlib style: {style}")

    dihedral_raw = data.get("dihedral")
    dihedral: RefepDihedralConfig | None = None
    if dihedral_raw is not None:
        values = _mapping(dihedral_raw, "dihedral")
        raw_ids = values.get("atom_ids")
        if not isinstance(raw_ids, list) or len(raw_ids) != 4:
            raise ValueError("dihedral.atom_ids must contain four one-based atom IDs")
        atom_ids = tuple(int(value) for value in raw_ids)
        if any(value < 1 for value in atom_ids) or len(set(atom_ids)) != 4:
            raise ValueError("dihedral.atom_ids must be four distinct positive IDs")
        target_deg = float(values["target_deg"])
        if not -180.0 <= target_deg <= 180.0:
            raise ValueError("dihedral.target_deg must be between -180 and 180 degrees")
        dihedral = RefepDihedralConfig(
            label=_nonempty_string(values.get("label"), "dihedral.label"),
            atom_ids=atom_ids,  # type: ignore[arg-type]
            target_deg=target_deg,
        )

    return RefepEquilReportConfig(
        equil=equil,
        output_dir=output_dir,
        style=style,
        cpptraj_executable=cpptraj_executable,
        dihedral=dihedral,
    )


def report_dir(cfg: RefepEquilReportConfig) -> Path:
    return refep_equil_dir(cfg.equil) / cfg.output_dir


def parse_thermo_samples(
    text: str,
    window: int,
    lambda_value: float,
    dt_ps: float,
) -> list[ThermoSample]:
    by_step: dict[int, ThermoSample] = {}
    for match in _THERMO_RE.finditer(text):
        step = int(match.group(1))
        if step <= 0 or step in by_step:
            continue
        by_step[step] = ThermoSample(
            window=window,
            lambda_value=lambda_value,
            step=step,
            time_ps=step * dt_ps,
            temperature_k=float(match.group(3)),
            pressure_bar=float(match.group(4)),
            total_energy_kcal_mol=float(match.group(5)),
            kinetic_energy_kcal_mol=float(match.group(6)),
            potential_energy_kcal_mol=float(match.group(7)),
        )
    if not by_step:
        raise ValueError("No Amber thermodynamic samples found")
    return [by_step[step] for step in sorted(by_step)]


def angle_near_target(angle_deg: float, target_deg: float) -> float:
    return angle_deg + 360.0 * math.floor((target_deg - angle_deg + 180.0) / 360.0)


def render_dihedral_cpptraj_input(
    topology: Path,
    trajectory: Path,
    output: Path,
    dihedral: RefepDihedralConfig,
) -> str:
    masks = " ".join(f"@{atom_id}" for atom_id in dihedral.atom_ids)
    return "\n".join(
        [
            f"parm {topology}",
            f"trajin {trajectory}",
            f"dihedral REFEP_DIH {masks} out {output}",
            "run",
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
        raise RuntimeError(f"cpptraj failed; see {log_path}")


def read_cpptraj_dihedral(path: Path) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed dihedral row at {path}:{line_number}")
        frame = int(fields[0])
        angle = float(fields[1])
        if not math.isfinite(angle):
            raise ValueError(f"Non-finite dihedral at {path}:{line_number}")
        values.append((frame, angle))
    if not values:
        raise ValueError(f"No dihedral samples found in {path}")
    return values


def _validate_and_collect_thermo(
    cfg: RefepEquilReportConfig,
) -> tuple[Path, list[ThermoSample]]:
    source = refep_equil_dir(cfg.equil)
    spec = source / "refep-equil-spec.yaml"
    if (
        not spec.is_file()
        or equil_simulation_payload(spec.read_text())
        != equil_simulation_payload(cfg.equil.yaml_path.read_text())
    ):
        raise ValueError("Prepared REFEP equilibration does not match the supplied YAML")

    dt = float(cfg.equil.cntrl["dt"])
    final_step = int(cfg.equil.cntrl["nstlim"])
    samples: list[ThermoSample] = []
    for index in range(cfg.equil.prep.windows):
        window = source / f"lambda-{index:03d}"
        required = [
            window / "system.parm7",
            window / "equil.nc",
            window / "equil.rst7",
            window / "equil.mdout",
        ]
        missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(
                "Missing/empty equilibration outputs:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        mdout_text = (window / "equil.mdout").read_text(errors="replace")
        if "Final Performance Info:" not in mdout_text:
            raise ValueError(f"Equilibration did not finish cleanly: {window / 'equil.mdout'}")
        failure = _AMBER_FAILURE_RE.search(mdout_text)
        if failure is not None:
            raise ValueError(
                f"Amber failure marker {failure.group(0)!r} found in {window / 'equil.mdout'}"
            )
        window_samples = parse_thermo_samples(
            mdout_text,
            index,
            index / (cfg.equil.prep.windows - 1),
            dt,
        )
        if window_samples[-1].step != final_step:
            raise ValueError(
                f"Equilibration ends at step {window_samples[-1].step}; "
                f"expected {final_step}: {window / 'equil.mdout'}"
            )
        samples.extend(window_samples)
    return source, samples


def _collect_dihedrals(
    cfg: RefepEquilReportConfig,
    source: Path,
    work: Path,
) -> list[DihedralSample]:
    if cfg.dihedral is None:
        return []
    expected_frames = int(cfg.equil.cntrl["nstlim"]) // int(cfg.equil.cntrl["ntwx"])
    frame_dt = int(cfg.equil.cntrl["ntwx"]) * float(cfg.equil.cntrl["dt"])
    samples: list[DihedralSample] = []
    combined_inputs: list[str] = []
    combined_logs: list[str] = []
    for index in range(cfg.equil.prep.windows):
        window = source / f"lambda-{index:03d}"
        input_path = work / f"dihedral-{index:03d}.cpptraj.in"
        output_path = work / f"dihedral-{index:03d}.dat"
        log_path = work / f"dihedral-{index:03d}.cpptraj.log"
        input_text = render_dihedral_cpptraj_input(
            window / "system.parm7",
            window / "equil.nc",
            output_path,
            cfg.dihedral,
        )
        input_path.write_text(input_text)
        run_cpptraj(
            input_path,
            log_path,
            cfg.equil.runtime.module,
            cfg.cpptraj_executable,
        )
        values = read_cpptraj_dihedral(output_path)
        if len(values) != expected_frames:
            raise ValueError(
                f"Window {index:03d} has {len(values)} dihedral frames; "
                f"expected {expected_frames}"
            )
        lambda_value = index / (cfg.equil.prep.windows - 1)
        for frame, angle in values:
            samples.append(
                DihedralSample(
                    window=index,
                    lambda_value=lambda_value,
                    frame=frame,
                    time_ps=frame * frame_dt,
                    raw_deg=angle,
                    branch_deg=angle_near_target(angle, cfg.dihedral.target_deg),
                )
            )
        combined_inputs.extend([f"# lambda-{index:03d}", input_text])
        combined_logs.extend([f"===== lambda-{index:03d} =====", log_path.read_text()])
    (work / "dihedral-cpptraj.in").write_text("\n".join(combined_inputs))
    (work / "dihedral-cpptraj.log").write_text("\n".join(combined_logs))
    return samples


def _write_thermo_csv(path: Path, samples: list[ThermoSample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "lambda",
                "step",
                "time_ps",
                "temperature_k",
                "pressure_bar",
                "total_energy_kcal_mol",
                "kinetic_energy_kcal_mol",
                "potential_energy_kcal_mol",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.window,
                    f"{sample.lambda_value:.10f}",
                    sample.step,
                    f"{sample.time_ps:.6f}",
                    f"{sample.temperature_k:.6f}",
                    f"{sample.pressure_bar:.6f}",
                    f"{sample.total_energy_kcal_mol:.6f}",
                    f"{sample.kinetic_energy_kcal_mol:.6f}",
                    f"{sample.potential_energy_kcal_mol:.6f}",
                ]
            )


def write_dihedral_csv(path: Path, samples: list[DihedralSample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["window", "lambda", "frame", "time_ps", "dihedral_raw_deg", "dihedral_branch_deg"]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.window,
                    f"{sample.lambda_value:.10f}",
                    sample.frame,
                    f"{sample.time_ps:.6f}",
                    f"{sample.raw_deg:.8f}",
                    f"{sample.branch_deg:.8f}",
                ]
            )


def _plot_thermo(
    cfg: RefepEquilReportConfig,
    path: Path,
    samples: list[ThermoSample],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    windows = cfg.equil.prep.windows
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, windows))
    fig, (temperature_ax, energy_ax) = plt.subplots(
        2, 1, figsize=(10.0, 7.8), sharex=True, constrained_layout=True
    )
    for index in range(windows):
        trace = [sample for sample in samples if sample.window == index]
        time = [sample.time_ps for sample in trace]
        temperature_ax.plot(
            time,
            [sample.temperature_k for sample in trace],
            color=colors[index],
            lw=0.9,
            alpha=0.78,
        )
        energy_ax.plot(
            time,
            [sample.potential_energy_kcal_mol for sample in trace],
            color=colors[index],
            lw=0.9,
            alpha=0.78,
        )
    temperature_ax.axhline(
        float(cfg.equil.cntrl["temp0"]), color="#C43C39", ls="--", lw=1.2
    )
    temperature_ax.set(ylabel="Temperature (K)", title="Temperature by lambda window")
    energy_ax.set(
        xlabel="Equilibration time (ps)",
        ylabel="Potential energy (kcal/mol)",
        title="Potential energy by lambda window",
    )
    scalar = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.0, 1.0))
    fig.colorbar(scalar, ax=[temperature_ax, energy_ax], label=r"$\lambda$")
    fig.suptitle(
        f"{cfg.equil.prep.system} REFEP independent-window equilibration",
        fontsize=15,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_dihedral_timeseries(
    path: Path,
    style: Path,
    system: str,
    stage: str,
    dihedral: RefepDihedralConfig,
    windows: int,
    samples: list[DihedralSample],
    *,
    scatter: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    columns = 4
    rows = math.ceil(windows / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.0, 2.45 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, windows))
    target = dihedral.target_deg
    for index, ax in enumerate(axes_array[:windows]):
        trace = [sample for sample in samples if sample.window == index]
        time = [sample.time_ps for sample in trace]
        angle = [sample.branch_deg for sample in trace]
        if scatter:
            ax.scatter(
                time,
                angle,
                color=colors[index],
                s=5.0,
                alpha=0.70,
                edgecolors="none",
            )
        else:
            ax.plot(time, angle, color=colors[index], lw=1.0)
        ax.axhline(target, color="#C43C39", ls="--", lw=0.9)
        ax.set_title(f"λ={index / (windows - 1):.3f}")
        ax.set_ylim(target - 190.0, target + 190.0)
        ax.set_yticks([target - 180.0, target, target + 180.0])
    for ax in axes_array[windows:]:
        ax.set_visible(False)
    for ax in axes_array[-columns:]:
        ax.set_xlabel("Time (ps)")
    for ax in axes_array[::columns]:
        ax.set_ylabel("Dihedral (degrees)")
    atom_text = "–".join(str(value) for value in dihedral.atom_ids)
    fig.suptitle(
        f"{system} {dihedral.label} {stage} time series "
        f"(atoms {atom_text}; target {target:g}°)",
        fontsize=14,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def create_refep_equil_report(
    cfg: RefepEquilReportConfig,
    output_dir: Path | None = None,
) -> tuple[Path, int, int]:
    source, thermo = _validate_and_collect_thermo(cfg)
    destination = output_dir.resolve() if output_dir else report_dir(cfg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".equil-report-", dir=destination.parent) as tmp:
        work = Path(tmp)
        dihedrals = _collect_dihedrals(cfg, source, work)
        _write_thermo_csv(work / "equil-thermo.csv", thermo)
        _plot_thermo(cfg, work / "equil-report.png", thermo)
        if cfg.dihedral is not None:
            write_dihedral_csv(work / "equil-dihedral.csv", dihedrals)
            plot_dihedral_timeseries(
                work / "equil-dihedral.png",
                cfg.style,
                cfg.equil.prep.system,
                "equilibration",
                cfg.dihedral,
                cfg.equil.prep.windows,
                dihedrals,
            )
        summary = {
            "system": cfg.equil.prep.system,
            "windows": cfg.equil.prep.windows,
            "equilibration_ps_per_window": (
                int(cfg.equil.cntrl["nstlim"]) * float(cfg.equil.cntrl["dt"])
            ),
            "thermodynamic_samples": len(thermo),
            "dihedral": None,
        }
        if cfg.dihedral is not None:
            summary["dihedral"] = {
                "label": cfg.dihedral.label,
                "atom_ids_one_based": list(cfg.dihedral.atom_ids),
                "target_deg": cfg.dihedral.target_deg,
                "samples": len(dihedrals),
            }
        (work / "report-summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
        destination.mkdir(parents=True, exist_ok=True)
        for path in work.iterdir():
            path.replace(destination / path.name)
    return destination, len(thermo), len(dihedrals)


def run_refep_equil_report(yaml_path: Path) -> None:
    cfg = load_refep_equil_report_config(yaml_path)
    destination, thermo_count, dihedral_count = create_refep_equil_report(cfg)
    print(f"OK: wrote REFEP equilibration report to {destination}")
    print(f"OK: collected {thermo_count} thermodynamic samples")
    if cfg.dihedral is not None:
        print(
            f"OK: collected {dihedral_count} {cfg.dihedral.label} dihedral samples "
            f"from {cfg.equil.prep.windows} windows"
        )
