"""Prepare a common hydroxide substitution for Amber umbrella-pull windows."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from qmmd.us_pull import (
    USPullConfig,
    find_repo_root,
    load_config as load_pull_config,
    pull_dir,
    source_paths,
    window_centers,
)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    amber_module: str = "amber/26"


@dataclass(frozen=True, slots=True)
class USSaltConfig:
    pull_yaml: Path
    pull: USPullConfig
    output_dirname: str
    counterion: str
    solute_mask: str
    nclosest: int
    delete_h: str
    minimum_solute_distance_angstrom: float
    runtime: RuntimeConfig


@dataclass(frozen=True, slots=True)
class WaterDistance:
    residue: int
    first_atom: int
    minimum: float
    mean: float
    maximum: float


CpptrajRunner = Callable[[Path, Path, str], None]


def _single_component(value: object, field: str) -> str:
    text = str(value)
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"{field} must be a single directory name")
    return text


def load_config(yaml_path: Path) -> USSaltConfig:
    resolved = yaml_path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("US salt configuration must be a YAML mapping")

    pull_value = Path(str(data.get("pull_yaml", "pull.yaml")))
    pull_yaml = pull_value if pull_value.is_absolute() else (resolved.parent / pull_value).resolve()
    if not pull_yaml.is_file():
        raise RuntimeError(f"Missing pull configuration: {pull_yaml}")

    counterion = str(data.get("counterion", "Cl-")).strip()
    solute_mask = str(data.get("solute_mask", ":1")).strip()
    delete_h = str(data.get("delete_h", "H1")).strip()
    if not counterion or not solute_mask:
        raise ValueError("counterion and solute_mask must not be empty")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9'*+\-]*", delete_h):
        raise ValueError("delete_h must be one Amber atom name")

    nclosest = int(data.get("nclosest", 999999))
    if nclosest < 1:
        raise ValueError("nclosest must be positive")
    minimum_distance = float(data.get("minimum_solute_distance_angstrom", 0.0))
    if not math.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("minimum_solute_distance_angstrom must be finite and non-negative")

    return USSaltConfig(
        pull_yaml=pull_yaml,
        pull=load_pull_config(pull_yaml),
        output_dirname=_single_component(data.get("output_dirname", "salt"), "output_dirname"),
        counterion=counterion,
        solute_mask=solute_mask,
        nclosest=nclosest,
        delete_h=delete_h,
        minimum_solute_distance_angstrom=minimum_distance,
        runtime=RuntimeConfig(**data.get("runtime", {})),
    )


def _cpptraj_path(path: Path) -> str:
    return '"' + str(path.resolve()).replace('"', '\\"') + '"'


def run_cpptraj(input_path: Path, log_path: Path, amber_module: str) -> None:
    cpptraj = shutil.which("cpptraj")
    if cpptraj:
        command = [cpptraj, "-i", input_path.name]
    else:
        shell = (
            "module purge\n"
            f"module load {shlex.quote(amber_module)}\n"
            f"exec cpptraj -i {shlex.quote(input_path.name)}"
        )
        command = ["bash", "-lc", shell]
    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=input_path.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def render_selection_input(
    topology: Path,
    restarts: list[Path],
    counterion: str,
    solute_mask: str,
    nclosest: int,
    output: Path,
) -> str:
    lines = [f"parm {_cpptraj_path(topology)}"]
    lines.extend(f"trajin {_cpptraj_path(restart)}" for restart in restarts)
    lines.extend(
        [
            "autoimage",
            f"strip :{counterion}",
            f"closest {nclosest} {solute_mask} first closestout {_cpptraj_path(output)}",
            "run",
            "quit",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_water_distances(path: Path, expected_frames: int) -> list[WaterDistance]:
    by_residue: dict[int, list[tuple[int, float, int]]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            frame = int(fields[0])
            residue = int(fields[1])
            distance = float(fields[2])
            first_atom = int(fields[3])
        except ValueError:
            continue
        if frame < 1 or residue < 1 or first_atom < 1 or not math.isfinite(distance):
            raise RuntimeError(f"Invalid closest-water row: {line}")
        by_residue.setdefault(residue, []).append((frame, distance, first_atom))

    if not by_residue:
        raise RuntimeError(f"No closest-water data rows found in {path}")

    summaries: list[WaterDistance] = []
    expected = set(range(1, expected_frames + 1))
    for residue, samples in by_residue.items():
        frames = {frame for frame, _, _ in samples}
        if frames != expected or len(samples) != expected_frames:
            continue
        first_atoms = {first_atom for _, _, first_atom in samples}
        if len(first_atoms) != 1:
            raise RuntimeError(f"Water residue {residue} changed first-atom identity across frames")
        distances = [distance for _, distance, _ in samples]
        summaries.append(
            WaterDistance(
                residue=residue,
                first_atom=first_atoms.pop(),
                minimum=min(distances),
                mean=sum(distances) / len(distances),
                maximum=max(distances),
            )
        )
    if not summaries:
        raise RuntimeError(
            f"No water residue had one distance sample in every one of {expected_frames} frames"
        )
    return sorted(summaries, key=lambda item: item.residue)


def select_common_water(summaries: list[WaterDistance]) -> WaterDistance:
    """Choose the water maximizing its minimum solute distance over all windows."""
    if not summaries:
        raise ValueError("At least one water-distance summary is required")
    return max(summaries, key=lambda item: (item.minimum, item.mean, -item.residue))


def _write_distance_summary(
    path: Path, summaries: list[WaterDistance], selected: WaterDistance
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "residue_after_counterion_removal",
                "first_atom_after_counterion_removal",
                "minimum_distance_A",
                "mean_distance_A",
                "maximum_distance_A",
                "selected",
            ]
        )
        for item in summaries:
            writer.writerow(
                [
                    item.residue,
                    item.first_atom,
                    f"{item.minimum:.6f}",
                    f"{item.mean:.6f}",
                    f"{item.maximum:.6f}",
                    item.residue == selected.residue,
                ]
            )


def _require_output(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"cpptraj did not create required output: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_us_salt(
    cfg: USSaltConfig,
    yaml_text: str,
    repo_root: Path,
    runner: CpptrajRunner = run_cpptraj,
) -> tuple[Path, list[Path], WaterDistance]:
    root = pull_dir(cfg.pull, repo_root)
    pull_spec = root / "pull_spec.yaml"
    if not pull_spec.is_file() or pull_spec.read_text() != cfg.pull_yaml.read_text():
        raise RuntimeError(f"{pull_spec} does not exactly match {cfg.pull_yaml}")

    topology, _, _ = source_paths(cfg.pull, repo_root)
    _require_output(topology)
    centers = window_centers(cfg.pull.windows)
    restarts: list[Path] = []
    for index in range(len(centers)):
        pull_stage = root / f"window-{index:03d}" / "pull"
        restart = pull_stage / "pull.rst7"
        output = pull_stage / "pull.out"
        _require_output(restart)
        if not output.is_file() or "Final Performance Info:" not in output.read_text(
            errors="replace"
        ):
            raise RuntimeError(f"Amber pull window did not reach normal completion: {output}")
        restarts.append(restart)

    common_destination = root / cfg.output_dirname
    destinations = [
        root / f"window-{index:03d}" / cfg.output_dirname for index in range(len(centers))
    ]
    existing = [path for path in [common_destination, *destinations] if path.exists()]
    if existing:
        raise FileExistsError(f"US salt output already exists; not touching: {existing[0]}")

    with tempfile.TemporaryDirectory(dir=root, prefix=".us-salt-prep-") as tmp:
        staging = Path(tmp)
        common = staging / "common"
        windows = staging / "windows"
        common.mkdir()
        windows.mkdir()

        distances = common / "common_water_distances.dat"
        selection_input = common / "select_common_water.in"
        selection_log = common / "select_common_water.out"
        selection_input.write_text(
            render_selection_input(
                topology,
                restarts,
                cfg.counterion,
                cfg.solute_mask,
                cfg.nclosest,
                distances,
            )
        )
        runner(selection_input, selection_log, cfg.runtime.amber_module)
        _require_output(distances)
        summaries = summarize_water_distances(distances, len(restarts))
        selected = select_common_water(summaries)
        if selected.minimum < cfg.minimum_solute_distance_angstrom:
            raise RuntimeError(
                f"Best common water residue {selected.residue} approaches the solute to "
                f"{selected.minimum:.3f} A, below configured minimum "
                f"{cfg.minimum_solute_distance_angstrom:.3f} A"
            )
        _write_distance_summary(common / "water_distance_summary.csv", summaries, selected)

        no_ion_topology = common / "no_ion.parm7"
        topology_1 = common / "topology_remove_counterion.in"
        topology_1.write_text(
            f"parm {_cpptraj_path(topology)}\n"
            f"parmstrip :{cfg.counterion}\n"
            f"parmwrite out {_cpptraj_path(no_ion_topology)}\n"
            "run\nquit\n"
        )
        runner(topology_1, common / "topology_remove_counterion.out", cfg.runtime.amber_module)
        _require_output(no_ion_topology)

        ready_topology = common / "ready.parm7"
        topology_2 = common / "topology_make_hydroxide.in"
        topology_2.write_text(
            f"parm {_cpptraj_path(no_ion_topology)}\n"
            f"parmstrip :{selected.residue}@{cfg.delete_h}\n"
            f"parmwrite out {_cpptraj_path(ready_topology)}\n"
            "run\nquit\n"
        )
        runner(topology_2, common / "topology_make_hydroxide.out", cfg.runtime.amber_module)
        _require_output(ready_topology)

        selection_record = {
            "strategy": "maximize the minimum solute-water distance over all pull windows",
            "residue_after_counterion_removal": selected.residue,
            "first_atom_after_counterion_removal": selected.first_atom,
            "deleted_atom_name": cfg.delete_h,
            "minimum_distance_angstrom": selected.minimum,
            "mean_distance_angstrom": selected.mean,
            "maximum_distance_angstrom": selected.maximum,
            "window_count": len(restarts),
        }
        (common / "selected_water.yaml").write_text(
            yaml.safe_dump(selection_record, sort_keys=False)
        )
        (common / "salt_spec.yaml").write_text(yaml_text)

        topology_hash = _sha256(ready_topology)
        for index, restart in enumerate(restarts):
            stage = windows / f"window-{index:03d}" / cfg.output_dirname
            stage.mkdir(parents=True)
            no_ion_restart = stage / "no_ion.rst7"
            first_input = stage / "cpptraj_1.in"
            first_input.write_text(
                f"parm {_cpptraj_path(topology)}\n"
                f"trajin {_cpptraj_path(restart)}\n"
                "autoimage\n"
                f"strip :{cfg.counterion}\n"
                f"trajout {_cpptraj_path(no_ion_restart)} restart\n"
                "run\nquit\n"
            )
            runner(first_input, stage / "cpptraj_1.out", cfg.runtime.amber_module)
            _require_output(no_ion_restart)

            ready_restart = stage / "ready.rst7"
            ready_xyz = stage / "ready.xyz"
            second_input = stage / "cpptraj_2.in"
            second_input.write_text(
                f"parm {_cpptraj_path(no_ion_topology)}\n"
                f"trajin {_cpptraj_path(no_ion_restart)}\n"
                f"strip :{selected.residue}@{cfg.delete_h}\n"
                f"trajout {_cpptraj_path(ready_restart)} restart\n"
                f"trajout {_cpptraj_path(ready_xyz)} xyz\n"
                "run\nquit\n"
            )
            runner(second_input, stage / "cpptraj_2.out", cfg.runtime.amber_module)
            _require_output(ready_restart)
            _require_output(ready_xyz)
            shutil.copy2(ready_topology, stage / "ready.parm7")
            if _sha256(stage / "ready.parm7") != topology_hash:
                raise RuntimeError(f"Topology copy checksum mismatch in window-{index:03d}")
            (stage / "salt_spec.yaml").write_text(yaml_text)
            (stage / "selected_water.yaml").write_text(
                yaml.safe_dump(selection_record, sort_keys=False)
            )

        common.replace(common_destination)
        committed: list[Path] = []
        for index, destination in enumerate(destinations):
            source = windows / f"window-{index:03d}" / cfg.output_dirname
            source.replace(destination)
            committed.append(destination)

    return common_destination, committed, selected


def run_us_salt_prep(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    common, windows, selected = prepare_us_salt(
        cfg, resolved.read_text(), find_repo_root(resolved)
    )
    print(
        f"OK: salted {len(windows)} umbrella windows using common water residue "
        f"{selected.residue} (minimum solute distance {selected.minimum:.3f} A)"
    )
    print(f"OK: common provenance and topology in {common}")
    print("NOTE: preparation only; no DCDFTBMD jobs were run or submitted")
