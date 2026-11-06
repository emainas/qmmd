#!/usr/bin/env python3
"""Append sequential Amber pull windows and DFTB umbrella equilibrations."""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from qmmd.mdequil import MDStage, render_dihedral_restraints, render_mdin
from qmmd.us_equil import (
    load_config as load_equil_config,
    prepare_us_equil,
    submit_us_equil,
)
from qmmd.us_pull import (
    find_repo_root,
    load_config as load_pull_config,
    pull_dir,
    source_paths,
)
from qmmd.us_salt import (
    render_selection_input,
    run_cpptraj,
    summarize_water_distances,
)


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    center_deg: float


@dataclass(frozen=True, slots=True)
class ExtensionConfig:
    path: Path
    text: str
    base_pull_yaml: Path
    base_equil_yaml: Path
    source_window_index: int
    windows: tuple[Window, ...]
    pull_duration_ps: float
    pull_timestep_fs: float
    pull_job: dict[str, object]


def canonical_dihedral_deg(center_deg: float) -> float:
    if not math.isfinite(center_deg):
        raise ValueError("Window centers must be finite")
    if -180.0 <= center_deg <= 180.0:
        return center_deg
    return (center_deg + 180.0) % 360.0 - 180.0


def _resolve_config(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_extension(path: Path) -> ExtensionConfig:
    resolved = path.resolve()
    text = resolved.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Torsion extension YAML must be a mapping")
    windows_data = data.get("windows")
    if not isinstance(windows_data, list) or not windows_data:
        raise ValueError("windows must be a non-empty list")
    windows: list[Window] = []
    for position, item in enumerate(windows_data):
        if not isinstance(item, dict):
            raise ValueError(f"windows[{position}] must be a mapping")
        index = item.get("window_index")
        center = float(item["center_deg"])
        if type(index) is not int or index < 0 or not math.isfinite(center):
            raise ValueError(f"Invalid windows[{position}]")
        windows.append(Window(index, center))
    if len({window.index for window in windows}) != len(windows):
        raise ValueError("Window indices must be unique")
    if len({window.center_deg for window in windows}) != len(windows):
        raise ValueError("Window centers must be unique")
    pull_data = data.get("pull")
    if not isinstance(pull_data, dict):
        raise ValueError("pull must be a mapping")
    job = pull_data.get("slurm")
    if not isinstance(job, dict):
        raise ValueError("pull.slurm must be a mapping")
    if str(job.get("partition")) != "small" or int(job.get("nodes", 0)) < 2:
        raise ValueError("Extension pulling requires the small partition and at least 2 nodes")
    return ExtensionConfig(
        path=resolved,
        text=text,
        base_pull_yaml=_resolve_config(resolved.parent, data["base_pull_yaml"]),
        base_equil_yaml=_resolve_config(resolved.parent, data["base_equil_yaml"]),
        source_window_index=int(data["source_window_index"]),
        windows=tuple(windows),
        pull_duration_ps=float(pull_data.get("duration_ps", 1.0)),
        pull_timestep_fs=float(pull_data.get("timestep_fs", 1.0)),
        pull_job=dict(job),
    )


def _require_complete_pull(stage: Path) -> None:
    restart = stage / "pull.rst7"
    output = stage / "pull.out"
    if not restart.is_file() or restart.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty pull restart: {restart}")
    if not output.is_file() or "Final Performance Info:" not in output.read_text(errors="replace"):
        raise RuntimeError(f"Pull did not terminate normally: {output}")


def _extension_names(cfg: ExtensionConfig) -> tuple[str, str, str]:
    tag = cfg.path.stem.replace("-dih", "").replace("_", "-")
    return f"run-{tag}.sh", f"slurm-{tag}.sh", f"{tag}_spec.yaml"


def prepare_pull(cfg: ExtensionConfig, repo_root: Path) -> tuple[Path, Path]:
    pull_cfg = load_pull_config(cfg.base_pull_yaml)
    root = pull_dir(pull_cfg, repo_root)
    prepared_spec = root / "pull_spec.yaml"
    if not prepared_spec.is_file() or prepared_spec.read_text() != cfg.base_pull_yaml.read_text():
        raise RuntimeError(f"{prepared_spec} does not match {cfg.base_pull_yaml}")
    source = root / f"window-{cfg.source_window_index:03d}" / "pull"
    _require_complete_pull(source)
    topology, _, _ = source_paths(pull_cfg, repo_root)
    if not topology.is_file() or topology.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty topology: {topology}")
    destinations = [root / f"window-{window.index:03d}" for window in cfg.windows]
    existing = [destination for destination in destinations if destination.exists()]
    if existing:
        raise FileExistsError(f"Extension window already exists; not touching: {existing[0]}")

    timestep_ps = cfg.pull_timestep_fs / 1000.0
    steps = round(cfg.pull_duration_ps / timestep_ps)
    if steps <= 0 or not math.isclose(steps * timestep_ps, cfg.pull_duration_ps):
        raise ValueError("Pull duration must be an integer number of timesteps")
    cntrl = dict(pull_cfg.md.cntrl)
    cntrl.update({"nstlim": steps, "dt": timestep_ps, "irest": 1, "ntx": 5})
    stage_config = MDStage(
        description=f"{pull_cfg.system} single15 right-extension pulling window",
        cntrl=cntrl,
    )
    rendered: list[tuple[Window, str, str]] = []
    for window in cfg.windows:
        target = canonical_dihedral_deg(window.center_deg)
        restraint = render_dihedral_restraints(
            [{
                "atoms": list(pull_cfg.restraint.atoms),
                "target_deg": target,
                "force_constant": pull_cfg.restraint.force_constant,
            }]
        )
        rendered.append((window, render_mdin(stage_config, "dihedral.rst"), restraint))

    run_name, slurm_name, spec_name = _extension_names(cfg)
    for window, mdin, restraint in rendered:
        stage = root / f"window-{window.index:03d}" / "pull"
        stage.mkdir(parents=True)
        (stage / "pull.in").write_text(mdin)
        (stage / "dihedral.rst").write_text(restraint)
        (stage / "extension_spec.yaml").write_text(cfg.text)

    env = "\n".join(
        f"export {key}={shlex.quote(str(value))}" for key, value in pull_cfg.runtime.env.items()
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail" if pull_cfg.runtime.strict_mode else "",
        "module purge",
        f"module load {shlex.quote(pull_cfg.runtime.module)}",
        "",
        env,
        "",
        'pull_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        f"topology={shlex.quote(str(topology.resolve()))}",
        f'input_restart="$pull_root/window-{cfg.source_window_index:03d}/pull/pull.rst7"',
        "",
    ]
    for window in cfg.windows:
        canonical = canonical_dihedral_deg(window.center_deg)
        lines.extend([
            f'window="$pull_root/window-{window.index:03d}/pull"',
            f'echo "==> window-{window.index:03d}: unwrapped {window.center_deg:g} degrees; restraint {canonical:g} degrees"',
            'if [[ -s "$window/pull.rst7" ]] && grep -q "Final Performance Info:" "$window/pull.out"; then',
            '    echo "  Complete; using existing restart"',
            'elif [[ -e "$window/pull.out" || -e "$window/pull.rst7" || -e "$window/pull.nc" ]]; then',
            '    echo "ERROR: partial output exists in $window; refusing to overwrite" >&2',
            "    exit 1",
            "else",
            '    (cd "$window" && ' + shlex.quote(pull_cfg.runtime.executable) + " -O \\",
            "      -i pull.in \\",
            "      -p \"$topology\" \\",
            "      -c \"$input_restart\" \\",
            "      -r pull.rst7 \\",
            "      -o pull.out \\",
            "      -inf pull.info \\",
            "      -x pull.nc)",
            "fi",
            'input_restart="$window/pull.rst7"',
            "",
        ])
    lines.append('echo "==> Right-extension pull complete"')
    run_path = root / run_name
    run_path.write_text("\n".join(line for line in lines if line is not None) + "\n")
    run_path.chmod(0o755)

    job = cfg.pull_job
    slurm = f'''#!/usr/bin/env bash
#SBATCH --job-name={job["name"]}
#SBATCH --partition={job["partition"]}
#SBATCH --nodes={int(job["nodes"])}
#SBATCH --time={job["time"]}
#SBATCH --output={job["stdout"]}
#SBATCH --error={job["stderr"]}

bash {run_name}
'''
    slurm_path = root / slurm_name
    slurm_path.write_text(slurm)
    slurm_path.chmod(0o755)
    (root / spec_name).write_text(cfg.text)
    return run_path, slurm_path


def submit_pull(cfg: ExtensionConfig, repo_root: Path) -> None:
    pull_cfg = load_pull_config(cfg.base_pull_yaml)
    root = pull_dir(pull_cfg, repo_root)
    _, slurm_name, spec_name = _extension_names(cfg)
    spec = root / spec_name
    if not spec.is_file() or spec.read_text() != cfg.text:
        raise RuntimeError(f"Extension specification mismatch: {spec}")
    for window in cfg.windows:
        stage = root / f"window-{window.index:03d}" / "pull"
        for filename in ("pull.in", "dihedral.rst", "extension_spec.yaml"):
            if not (stage / filename).is_file():
                raise RuntimeError(f"Missing prepared pull input: {stage / filename}")
    subprocess.run(["sbatch", slurm_name], cwd=root, check=True)


def prepare_existing_salt(cfg: ExtensionConfig, repo_root: Path) -> None:
    equil_cfg = load_equil_config(cfg.base_equil_yaml)
    if equil_cfg.conversion.source_stage != "salt" or equil_cfg.conversion.salt_yaml is None:
        raise ValueError("The base equilibration does not use a salted source")
    root = pull_dir(equil_cfg.pull, repo_root)
    common = root / "salt"
    selected_data = yaml.safe_load((common / "selected_water.yaml").read_text())
    selected_residue = int(selected_data["residue_after_counterion_removal"])
    delete_h = str(selected_data["deleted_atom_name"])
    salt_data = yaml.safe_load(equil_cfg.conversion.salt_yaml.read_text())
    topology, _, _ = source_paths(equil_cfg.pull, repo_root)
    restarts: list[Path] = []
    for window in cfg.windows:
        stage = root / f"window-{window.index:03d}" / "pull"
        _require_complete_pull(stage)
        restarts.append(stage / "pull.rst7")
        if (stage.parent / "salt").exists():
            raise FileExistsError(f"Salt directory already exists: {stage.parent / 'salt'}")

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="qmmd-us-salt-extension-") as tmp:
        temporary = Path(tmp)
        distances = temporary / "water_distances.dat"
        selection = temporary / "selection.in"
        selection.write_text(render_selection_input(
            topology,
            restarts,
            str(salt_data.get("counterion", "Cl-")),
            str(salt_data.get("solute_mask", ":1")),
            int(salt_data.get("nclosest", 999999)),
            distances,
        ))
        run_cpptraj(selection, temporary / "selection.out", equil_cfg.conversion.amber_module)
        summaries = summarize_water_distances(distances, len(restarts))
        matching = [item for item in summaries if item.residue == selected_residue]
        if len(matching) != 1:
            raise RuntimeError(f"Previously selected water residue {selected_residue} was not found")
        minimum_allowed = float(salt_data.get("minimum_solute_distance_angstrom", 0.0))
        if matching[0].minimum < minimum_allowed:
            raise RuntimeError(
                f"Selected water approaches the solute to {matching[0].minimum:.3f} A; "
                f"minimum is {minimum_allowed:.3f} A"
            )

    no_ion_topology = common / "no_ion.parm7"
    salt_spec = equil_cfg.conversion.salt_yaml.read_text()
    for window, restart in zip(cfg.windows, restarts, strict=True):
        stage = root / f"window-{window.index:03d}" / "salt"
        stage.mkdir()
        first_input = stage / "cpptraj_1.in"
        first_input.write_text(
            f'parm "{topology.resolve()}"\ntrajin "{restart.resolve()}"\nautoimage\n'
            f'strip :{salt_data.get("counterion", "Cl-")}\n'
            f'trajout "{(stage / "no_ion.rst7").resolve()}" restart\nrun\nquit\n'
        )
        run_cpptraj(first_input, stage / "cpptraj_1.out", equil_cfg.conversion.amber_module)
        second_input = stage / "cpptraj_2.in"
        second_input.write_text(
            f'parm "{no_ion_topology.resolve()}"\n'
            f'trajin "{(stage / "no_ion.rst7").resolve()}"\n'
            f'strip :{selected_residue}@{delete_h}\n'
            f'trajout "{(stage / "ready.rst7").resolve()}" restart\n'
            f'trajout "{(stage / "ready.xyz").resolve()}" xyz\nrun\nquit\n'
        )
        run_cpptraj(second_input, stage / "cpptraj_2.out", equil_cfg.conversion.amber_module)
        ready = stage / "ready.xyz"
        if not ready.is_file() or ready.stat().st_size == 0:
            raise RuntimeError(f"Salt conversion failed: {ready}")
        (stage / "salt_spec.yaml").write_text(salt_spec)
        (stage / "extension_water_check.yaml").write_text(yaml.safe_dump({
            "residue_after_counterion_removal": selected_residue,
            "minimum_distance_angstrom": matching[0].minimum,
            "required_minimum_distance_angstrom": minimum_allowed,
        }, sort_keys=False))


def extension_equil_config(cfg: ExtensionConfig):
    base = load_equil_config(cfg.base_equil_yaml)
    targets = tuple((window.index, window.center_deg) for window in cfg.windows)
    job = replace(
        base.slurm.job,
        name=f"{base.slurm.job.name}-right",
        time="3-00:00:00",
    )
    return replace(base, window_targets=targets, slurm=replace(base.slurm, job=job))


def prepare_equil(cfg: ExtensionConfig, repo_root: Path) -> list[Path]:
    return prepare_us_equil(extension_equil_config(cfg), cfg.text, repo_root)


def submit_equil(cfg: ExtensionConfig, repo_root: Path) -> bool:
    return submit_us_equil(
        extension_equil_config(cfg),
        cfg.text,
        repo_root,
        confirm=lambda _prompt: "yes",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("pull-prep", "pull-submit", "salt-prep", "equil-prep", "equil-submit"),
    )
    parser.add_argument("yaml", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_extension(args.yaml)
    repo_root = find_repo_root(cfg.path)
    if args.action == "pull-prep":
        run_path, slurm_path = prepare_pull(cfg, repo_root)
        print(f"OK: prepared {len(cfg.windows)} pull windows via {run_path} and {slurm_path}")
    elif args.action == "pull-submit":
        submit_pull(cfg, repo_root)
    elif args.action == "salt-prep":
        prepare_existing_salt(cfg, repo_root)
        print(f"OK: salted {len(cfg.windows)} extension windows with the established BV mapping")
    elif args.action == "equil-prep":
        outputs = prepare_equil(cfg, repo_root)
        print(f"OK: prepared {len(outputs)} DFTB equilibration windows")
    else:
        if not submit_equil(cfg, repo_root):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
