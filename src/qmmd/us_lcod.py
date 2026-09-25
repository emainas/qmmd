"""Hand-seeded LCOD umbrella sampling from an existing DFTB trajectory."""

from __future__ import annotations

import csv
import math
import re
import secrets
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import yaml

from qmmd.dftb import (
    DFTBConfig,
    ElementConfig,
    RuntimeConfig,
    SlurmConfig,
    SlurmJobConfig,
    stage_skf_files,
)
from qmmd.us_equil import (
    read_xyz,
    render_dftb_input,
    render_run_sh,
    render_slurm_sh,
    validate_zero_height_metawall,
)
from qmmd.us_pull import find_repo_root


TIME_RE = re.compile(r"AT\s+T=\s*([-+0-9.EeDd]+)\s+FSEC", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LCODCV:
    atoms: tuple[int, int, int, int]
    gaussian_width_angstrom: float
    label: str = ""


@dataclass(frozen=True, slots=True)
class LCODWindows:
    start_angstrom: float
    stop_angstrom: float
    spacing_angstrom: float


@dataclass(frozen=True, slots=True)
class LCODPullConfig:
    system: str
    buffer: float
    prefix: str
    output_dirname: str
    trajectory: Path
    dftb_input: Path
    windows: LCODWindows
    cv: LCODCV


@dataclass(frozen=True, slots=True)
class WallConfig:
    coefficient_kcal_mol_angstrom2: float
    exponent: int


@dataclass(frozen=True, slots=True)
class LCODEquilConfig:
    pull_yaml: Path
    pull: LCODPullConfig
    stage_dirname: str
    cv: LCODCV
    wall: WallConfig
    dftb: DFTBConfig
    runtime: RuntimeConfig
    slurm: SlurmConfig


def _single_component(value: object, field: str) -> str:
    text = str(value)
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"{field} must be a single directory name")
    return text


def lcod_centers(cfg: LCODWindows) -> list[float]:
    span = cfg.stop_angstrom - cfg.start_angstrom
    if cfg.spacing_angstrom <= 0 or span < 0:
        raise ValueError("LCOD windows require stop >= start and positive spacing")
    intervals = round(span / cfg.spacing_angstrom)
    if not math.isclose(
        intervals * cfg.spacing_angstrom, span, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("LCOD range must be exactly divisible by spacing")
    return [cfg.start_angstrom + index * cfg.spacing_angstrom for index in range(intervals + 1)]


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def load_pull_config(path: Path) -> LCODPullConfig:
    resolved = path.resolve()
    root = find_repo_root(resolved)
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("LCOD pull configuration must be a YAML mapping")
    cv_data = data["cv"]
    if cv_data.get("type") != "BONDDISTANCEDIFFERENCE":
        raise ValueError("cv.type must be BONDDISTANCEDIFFERENCE")
    atoms = tuple(int(atom) for atom in cv_data["atoms"])
    if len(atoms) != 4 or min(atoms) < 1:
        raise ValueError("cv.atoms must contain four positive one-based IDs")
    width = float(cv_data["gaussian_width_angstrom"])
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Gaussian width must be positive and finite")
    windows = LCODWindows(**data["windows"])
    lcod_centers(windows)
    source = data["source"]
    cfg = LCODPullConfig(
        system=str(data["system"]),
        buffer=float(data["buffer"]),
        prefix=str(data.get("prefix", "solv")),
        output_dirname=_single_component(data.get("output_dirname", "us-lcod"), "output_dirname"),
        trajectory=_resolve(root, source["trajectory"]),
        dftb_input=_resolve(root, source["dftb_input"]),
        windows=windows,
        cv=LCODCV(atoms, width, str(cv_data.get("label", ""))),
    )
    for source_path in (cfg.trajectory, cfg.dftb_input):
        if not source_path.is_file() or source_path.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty LCOD source: {source_path}")
    return cfg


def pull_root(cfg: LCODPullConfig, repo_root: Path) -> Path:
    return (
        repo_root
        / "systems"
        / cfg.system
        / f"{cfg.prefix}_{cfg.buffer:.1f}"
        / cfg.output_dirname
    )


def read_box_vectors(path: Path) -> np.ndarray:
    vectors = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("TV"):
            fields = line.split()
            if len(fields) == 4:
                vectors.append([float(value) for value in fields[1:]])
    if len(vectors) != 3:
        raise RuntimeError(f"Expected three TV vectors in {path}")
    box = np.asarray(vectors, dtype=float)
    if abs(np.linalg.det(box)) < 1.0e-10:
        raise ValueError(f"Singular periodic box in {path}")
    return box


def minimum_image(vector: np.ndarray, box: np.ndarray) -> np.ndarray:
    fractional = vector @ np.linalg.inv(box)
    fractional -= np.rint(fractional)
    return fractional @ box


def iter_trajectory(path: Path) -> Iterator[tuple[float, list[str], np.ndarray]]:
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                return
            if not line.strip():
                continue
            natoms = int(line)
            comment = stream.readline()
            match = TIME_RE.search(comment)
            if match is None:
                raise ValueError(f"Missing DFTB time in {path}")
            symbols: list[str] = []
            coordinates = np.empty((natoms, 3), dtype=float)
            for index in range(natoms):
                atom_line = stream.readline()
                if not atom_line:
                    return  # Ignore a partial frame at a growing-file tail.
                fields = atom_line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed trajectory atom line: {atom_line!r}")
                symbols.append(fields[0])
                coordinates[index] = [float(value) for value in fields[1:4]]
            time_ps = float(match.group(1).replace("D", "E").replace("d", "e")) / 1000.0
            yield time_ps, symbols, coordinates


def calculate_lcod(coordinates: np.ndarray, atoms: tuple[int, int, int, int], box: np.ndarray) -> float:
    a, b, c, d = (atom - 1 for atom in atoms)
    first = np.linalg.norm(minimum_image(coordinates[a] - coordinates[b], box))
    second = np.linalg.norm(minimum_image(coordinates[c] - coordinates[d], box))
    return float(first - second)


def render_xyz(symbols: list[str], coordinates: np.ndarray, box: np.ndarray) -> str:
    lattice = " ".join(f"{value:.10f}" for value in box.ravel())
    lines = [str(len(symbols)), f'Lattice="{lattice}"']
    lines.extend(
        f"{symbol:<2s} {point[0]:18.10f} {point[1]:18.10f} {point[2]:18.10f}"
        for symbol, point in zip(symbols, coordinates)
    )
    return "\n".join(lines) + "\n"


def prepare_lcod_pull(cfg: LCODPullConfig, yaml_text: str, repo_root: Path) -> Path:
    output = pull_root(cfg, repo_root)
    if output.exists():
        raise FileExistsError(f"LCOD umbrella root already exists; not touching: {output}")
    centers = lcod_centers(cfg.windows)
    box = read_box_vectors(cfg.dftb_input)
    best: list[tuple[float, float, str] | None] = [None] * len(centers)
    atom_names: list[str] | None = None
    frame_count = 0
    for time_ps, symbols, coordinates in iter_trajectory(cfg.trajectory):
        frame_count += 1
        if max(cfg.cv.atoms) > len(symbols):
            raise ValueError("LCOD atom ID exceeds source trajectory atom count")
        selected_names = [symbols[atom - 1] for atom in cfg.cv.atoms]
        if atom_names is None:
            atom_names = selected_names
        elif atom_names != selected_names:
            raise ValueError("LCOD source atom identities changed between frames")
        value = calculate_lcod(coordinates, cfg.cv.atoms, box)
        xyz = None
        for index, center in enumerate(centers):
            error = abs(value - center)
            if best[index] is None or error < best[index][0]:
                if xyz is None:
                    xyz = render_xyz(symbols, coordinates, box)
                best[index] = (error, time_ps, xyz)
    if frame_count == 0 or any(item is None for item in best):
        raise RuntimeError(f"No complete source frames found in {cfg.trajectory}")
    output.mkdir(parents=True)
    rows = []
    for index, (center, item) in enumerate(zip(centers, best)):
        assert item is not None
        _, time_ps, xyz = item
        _, _, coords = read_xyz_text(xyz)
        actual = calculate_lcod(np.asarray([[x, y, z] for _, x, y, z in coords]), cfg.cv.atoms, box)
        stage = output / f"window-{index:03d}" / "pull"
        stage.mkdir(parents=True)
        (stage / "start.xyz").write_text(xyz)
        rows.append((index, center, actual, actual - center, time_ps))
    with (output / "windows.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_index", "target_lcod_A", "selected_lcod_A", "error_A", "source_time_ps"])
        writer.writerows(rows)
    (output / "pull_spec.yaml").write_text(yaml_text)
    return output


def read_xyz_text(text: str):
    """Parse rendered XYZ text through the package reader without persistent files."""
    lines = text.splitlines()
    natoms = int(lines[0])
    lattice = [float(value) for value in lines[1].split('Lattice="', 1)[1].rstrip('"').split()]
    vectors = [tuple(lattice[index:index + 3]) for index in range(0, 9, 3)]
    coords = [(f[0], float(f[1]), float(f[2]), float(f[3])) for f in (line.split() for line in lines[2:])]
    if len(coords) != natoms:
        raise ValueError("Rendered XYZ atom-count mismatch")
    return natoms, vectors, coords


def create_lcod_pull_report(cfg: LCODPullConfig, yaml_text: str, repo_root: Path) -> tuple[Path, Path]:
    root = pull_root(cfg, repo_root)
    if not (root / "pull_spec.yaml").is_file() or (root / "pull_spec.yaml").read_text() != yaml_text:
        raise RuntimeError("Prepared LCOD pull snapshot does not match config")
    rows = list(csv.DictReader((root / "windows.csv").open()))
    if len(rows) != len(lcod_centers(cfg.windows)):
        raise RuntimeError("LCOD window table is incomplete")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = np.array([float(row["target_lcod_A"]) for row in rows])
    actual = np.array([float(row["selected_lcod_A"]) for row in rows])
    times = np.array([float(row["source_time_ps"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(targets, targets, "--", color="0.4", label="Target")
    axes[0].plot(targets, actual, "o-", color="#1769aa", ms=3.5, label="Selected source frame")
    axes[0].set_ylabel("Selected LCOD (Å)")
    axes[0].legend(frameon=False)
    axes[1].plot(targets, times, "o-", color="#d1495b", ms=3.5)
    axes[1].set(xlabel="Umbrella target LCOD (Å)", ylabel="Source trajectory time (ps)")
    for axis in axes:
        axis.grid(alpha=0.22)
    label = cfg.cv.label or (
        f"r({cfg.cv.atoms[0]},{cfg.cv.atoms[1]}) − "
        f"r({cfg.cv.atoms[2]},{cfg.cv.atoms[3]})"
    )
    fig.suptitle(f"{cfg.system} handmade LCOD umbrella seeds: {label}")
    figure = root / "pull.png"
    fig.savefig(figure, dpi=300)
    plt.close(fig)
    return figure, root / "windows.csv"


def load_equil_config(path: Path) -> LCODEquilConfig:
    resolved = path.resolve()
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("LCOD equilibration configuration must be a YAML mapping")
    pull_value = Path(str(data.get("pull_yaml", "pull.yaml")))
    pull_yaml = pull_value if pull_value.is_absolute() else (resolved.parent / pull_value).resolve()
    pull = load_pull_config(pull_yaml)
    cv_data = data["cv"]
    atoms = tuple(int(atom) for atom in cv_data["atoms"])
    if cv_data.get("type") != "BONDDISTANCEDIFFERENCE" or atoms != pull.cv.atoms:
        raise ValueError("Equilibration LCOD type/atoms must match pull.yaml")
    cv = LCODCV(atoms, float(cv_data["gaussian_width_angstrom"]))
    wall_data = data["wall"]
    wall = WallConfig(float(wall_data["coefficient_kcal_mol_angstrom2"]), int(wall_data.get("exponent", 2)))
    if not math.isfinite(wall.coefficient_kcal_mol_angstrom2) or wall.coefficient_kcal_mol_angstrom2 <= 0:
        raise ValueError("LCOD wall coefficient must be positive and finite")
    if wall.exponent <= 0 or wall.exponent % 2:
        raise ValueError("LCOD wall exponent must be a positive even integer")
    dftb_data = data["dftb"]
    dftb = DFTBConfig(
        title=str(dftb_data["title"]),
        header_lines=[str(line) for line in dftb_data["header_lines"]],
        charge=int(dftb_data.get("charge", 0)),
        multiplicity=int(dftb_data.get("multiplicity", 1)),
        elements=[ElementConfig(**element) for element in dftb_data["elements"]],
        params_dir=str(dftb_data.get("params_dir", "params")),
    )
    validate_zero_height_metawall(dftb.header_lines)
    return LCODEquilConfig(
        pull_yaml=pull_yaml,
        pull=pull,
        stage_dirname=_single_component(data.get("stage_dirname", "equil"), "stage_dirname"),
        cv=cv,
        wall=wall,
        dftb=dftb,
        runtime=RuntimeConfig(**data["runtime"]),
        slurm=SlurmConfig(job=SlurmJobConfig(**data["slurm"]["job"])),
    )


def render_lcod_metacv(cfg: LCODEquilConfig, center: float) -> str:
    atoms = " ".join(str(atom) for atom in cfg.cv.atoms)
    coefficient = cfg.wall.coefficient_kcal_mol_angstrom2
    return (
        f"BONDDISTANCEDIFFERENCE {cfg.cv.gaussian_width_angstrom:g} {atoms}\n\n"
        f"L 1 {coefficient:g} {center:g} {cfg.wall.exponent}\n"
        f"U 1 {coefficient:g} {center:g} {cfg.wall.exponent}\n"
    )


def prepare_lcod_equil(cfg: LCODEquilConfig, yaml_text: str, repo_root: Path) -> list[Path]:
    root = pull_root(cfg.pull, repo_root)
    if not (root / "pull_spec.yaml").is_file() or (root / "pull_spec.yaml").read_text() != cfg.pull_yaml.read_text():
        raise RuntimeError("Prepared LCOD pull snapshot does not match pull.yaml")
    centers = lcod_centers(cfg.pull.windows)
    destinations = [root / f"window-{index:03d}" / cfg.stage_dirname for index in range(len(centers))]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"LCOD equilibration directory already exists: {existing[0]}")
    params = repo_root / cfg.dftb.params_dir
    elements = [element.symbol for element in cfg.dftb.elements]
    for left in elements:
        for right in elements:
            if not (params / f"{left}-{right}.skf").is_file():
                raise RuntimeError(f"Missing SKF file: {params / f'{left}-{right}.skf'}")
    rendered = []
    for index, center in enumerate(centers):
        source = root / f"window-{index:03d}" / "pull" / "start.xyz"
        natoms, vectors, coordinates = read_xyz(source)
        seed = secrets.randbelow(2**31 - 1) + 1
        rendered.append((source, render_dftb_input(cfg, natoms, vectors, coordinates, seed), render_lcod_metacv(cfg, center)))
    for index, (destination, files) in enumerate(zip(destinations, rendered)):
        source, dftb_input, metacv = files
        destination.mkdir()
        (destination / "dftb.inp").write_text(dftb_input)
        (destination / "metacv.dat").write_text(metacv)
        shutil.copy2(source, destination / "start.xyz")
        (destination / "equil_spec.yaml").write_text(yaml_text)
        run_sh = destination / "run.sh"; run_sh.write_text(render_run_sh(cfg)); run_sh.chmod(0o755)
        slurm_sh = destination / "slurm.sh"; slurm_sh.write_text(render_slurm_sh(cfg, index)); slurm_sh.chmod(0o755)
        stage_skf_files(params, destination, elements)
    return destinations


def submit_slurm(script: Path) -> None:
    subprocess.run(["sbatch", script.name], cwd=script.parent, check=True)


def submit_lcod_equil(
    cfg: LCODEquilConfig,
    yaml_text: str,
    repo_root: Path,
    confirm: Callable[[str], str] = input,
    submitter: Callable[[Path], None] | None = None,
) -> bool:
    root = pull_root(cfg.pull, repo_root)
    centers = lcod_centers(cfg.pull.windows)
    targets = [root / f"window-{index:03d}" / cfg.stage_dirname for index in range(len(centers))]
    problems = []
    for target in targets:
        spec = target / "equil_spec.yaml"
        if not spec.is_file() or spec.read_text() != yaml_text:
            problems.append(f"{spec} does not match config")
            continue
        for name in ("dftb.inp", "metacv.dat", "start.xyz", "run.sh", "slurm.sh"):
            path = target / name
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f"missing/empty {path}")
    if problems:
        print("SKIP: LCOD equilibration set failed preflight; nothing submitted")
        for problem in problems: print(f"  - {problem}")
        return False
    print(f"Will submit the following {cfg.pull.system} LCOD equilibration directories:")
    for target, center in zip(targets, centers): print(f"  - {target} (center {center:g} A)")
    if confirm(f"Proceed to submit {len(targets)} jobs? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled by user."); return False
    submit = submitter or submit_slurm
    for target in targets:
        print(f"Submitting job via sbatch for {target}...")
        submit(target / "slurm.sh")
        print("OK: job submitted")
    return True


def run_pull_prep(path: Path) -> None:
    resolved = path.resolve(); cfg = load_pull_config(resolved)
    root = prepare_lcod_pull(cfg, resolved.read_text(), find_repo_root(resolved))
    print(f"OK: prepared {len(lcod_centers(cfg.windows))} handmade LCOD windows under {root}")


def run_pull_report(path: Path) -> None:
    resolved = path.resolve(); cfg = load_pull_config(resolved)
    figure, data = create_lcod_pull_report(cfg, resolved.read_text(), find_repo_root(resolved))
    print(f"OK: wrote LCOD pull-selection data to {data}")
    print(f"OK: wrote LCOD pull-selection plot to {figure}")


def run_equil_prep(path: Path) -> None:
    resolved = path.resolve(); cfg = load_equil_config(resolved)
    outputs = prepare_lcod_equil(cfg, resolved.read_text(), find_repo_root(resolved))
    print(f"OK: prepared {len(outputs)} {cfg.pull.system} LCOD equilibration windows under {outputs[0].parent.parent}")
    print("NOTE: inputs only; no DCDFTBMD jobs were run or submitted")


def run_equil_submit(path: Path) -> None:
    resolved = path.resolve(); cfg = load_equil_config(resolved)
    submit_lcod_equil(cfg, resolved.read_text(), find_repo_root(resolved))
