from __future__ import annotations

import csv
import hashlib
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from qmmd.cphmd_dgref import find_repo_root
from qmmd.cphmd_prep import Mol2Atom, read_mol2_atoms


AMBER_CHARGE_SCALE = 18.2223
_FORMAT_RE = re.compile(
    r"%FORMAT\(\s*(?P<count>\d+)(?P<kind>[aAiIeEfF])(?P<width>\d+)"
    r"(?:\.\d+)?\s*\)"
)


@dataclass(frozen=True)
class RefepPrepConfig:
    yaml_path: Path
    system: str
    buffer: float
    prefix: str
    input_parm7: Path
    input_rst7: Path
    deprot_mol2: Path
    residue_id: int
    residue_name: str
    dummy_atoms: tuple[str, ...]
    expected_charge_lambda0: int
    expected_charge_lambda1: int
    charge_tolerance: float
    windows: int
    lambda0_label: str
    lambda1_label: str
    output_directory: Path
    plot_name: str
    style: Path
    amber_module: str
    parmed_executable: str


@dataclass(frozen=True)
class AmberTopology:
    path: Path
    sections: dict[str, tuple[str, tuple[str, ...]]]
    atom_names: tuple[str, ...]
    residue_labels: tuple[str, ...]
    residue_pointers: tuple[int, ...]
    charges: np.ndarray

    @property
    def atom_count(self) -> int:
        return len(self.atom_names)


@dataclass(frozen=True)
class LigandChargeMap:
    atom_indices: tuple[int, ...]
    atom_names: tuple[str, ...]
    lambda0_charges: np.ndarray
    lambda1_charges: np.ndarray
    dummy_atoms: tuple[str, ...]


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _resolve_path(root: Path, value: object, field: str) -> Path:
    text = _required_string(value, field)
    path = Path(text)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_refep_prep_config(yaml_path: Path) -> RefepPrepConfig:
    resolved = yaml_path.resolve()
    root = find_repo_root(resolved)
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError("REFEP prep configuration must be a YAML mapping")

    ligand = data.get("ligand")
    interpolation = data.get("interpolation")
    output = data.get("output")
    runtime = data.get("runtime")
    for field, value in (
        ("ligand", ligand),
        ("interpolation", interpolation),
        ("output", output),
        ("runtime", runtime),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a YAML mapping")

    input_parm7 = _resolve_path(root, data.get("input_parm7"), "input_parm7")
    input_rst7 = _resolve_path(root, data.get("input_rst7"), "input_rst7")
    deprot_mol2 = _resolve_path(root, data.get("deprot_mol2"), "deprot_mol2")
    style = _resolve_path(root, output.get("style"), "output.style")
    output_directory = _resolve_path(
        root, output.get("directory"), "output.directory"
    )
    for field, path in (
        ("input_parm7", input_parm7),
        ("input_rst7", input_rst7),
        ("deprot_mol2", deprot_mol2),
        ("output.style", style),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {field}: {path}")
    try:
        output_directory.relative_to(root)
    except ValueError as exc:
        raise ValueError("output.directory must be inside the qmmd repository") from exc

    residue_id = int(ligand.get("residue_id", 0))
    if residue_id < 1:
        raise ValueError("ligand.residue_id must be one-based and positive")
    raw_dummy = ligand.get("dummy_atoms", [])
    if not isinstance(raw_dummy, list) or not all(
        isinstance(value, str) and value.strip() for value in raw_dummy
    ):
        raise ValueError("ligand.dummy_atoms must be a list of atom names")
    dummy_atoms = tuple(value.strip() for value in raw_dummy)
    if len(set(dummy_atoms)) != len(dummy_atoms):
        raise ValueError("ligand.dummy_atoms contains duplicates")

    charge_tolerance = float(ligand.get("charge_tolerance", 0.01))
    if charge_tolerance <= 0:
        raise ValueError("ligand.charge_tolerance must be positive")
    windows = int(interpolation.get("windows", 0))
    if windows < 3:
        raise ValueError("interpolation.windows must be at least 3")
    plot_name = _required_string(output.get("plot"), "output.plot")
    if Path(plot_name).name != plot_name or not plot_name.lower().endswith(".png"):
        raise ValueError("output.plot must be a PNG filename without directory components")

    return RefepPrepConfig(
        yaml_path=resolved,
        system=_required_string(data.get("system"), "system"),
        buffer=float(data.get("buffer")),
        prefix=_required_string(data.get("prefix"), "prefix"),
        input_parm7=input_parm7,
        input_rst7=input_rst7,
        deprot_mol2=deprot_mol2,
        residue_id=residue_id,
        residue_name=_required_string(ligand.get("residue_name"), "ligand.residue_name"),
        dummy_atoms=dummy_atoms,
        expected_charge_lambda0=int(ligand.get("expected_charge_lambda0")),
        expected_charge_lambda1=int(ligand.get("expected_charge_lambda1")),
        charge_tolerance=charge_tolerance,
        windows=windows,
        lambda0_label=_required_string(
            interpolation.get("lambda0_label"), "interpolation.lambda0_label"
        ),
        lambda1_label=_required_string(
            interpolation.get("lambda1_label"), "interpolation.lambda1_label"
        ),
        output_directory=output_directory,
        plot_name=plot_name,
        style=style,
        amber_module=_required_string(runtime.get("module"), "runtime.module"),
        parmed_executable=_required_string(
            runtime.get("parmed_executable"), "runtime.parmed_executable"
        ),
    )


def _parse_section_values(format_line: str, lines: list[str]) -> tuple[str, ...]:
    match = _FORMAT_RE.fullmatch(format_line.strip())
    if match is None:
        raise ValueError(f"Unsupported Amber topology format: {format_line}")
    kind = match.group("kind").lower()
    width = int(match.group("width"))
    if kind == "a":
        return tuple(
            line[start : start + width].strip()
            for line in lines
            for start in range(0, len(line), width)
            if line[start : start + width].strip()
        )
    return tuple(value for line in lines for value in line.split())


def read_amber_topology(path: Path) -> AmberTopology:
    lines = path.read_text().splitlines()
    sections: dict[str, tuple[str, tuple[str, ...]]] = {}
    index = 0
    while index < len(lines):
        if not lines[index].startswith("%FLAG "):
            index += 1
            continue
        name = lines[index][6:].strip()
        if index + 1 >= len(lines) or not lines[index + 1].startswith("%FORMAT"):
            raise ValueError(f"Missing format for %FLAG {name} in {path}")
        format_line = lines[index + 1]
        index += 2
        data_lines: list[str] = []
        while index < len(lines) and not lines[index].startswith("%FLAG "):
            data_lines.append(lines[index])
            index += 1
        sections[name] = (format_line, _parse_section_values(format_line, data_lines))

    required = {"POINTERS", "ATOM_NAME", "RESIDUE_LABEL", "RESIDUE_POINTER", "CHARGE"}
    missing = sorted(required - sections.keys())
    if missing:
        raise ValueError(f"Missing Amber topology sections in {path}: {', '.join(missing)}")
    atom_count = int(sections["POINTERS"][1][0])
    atom_names = sections["ATOM_NAME"][1]
    if len(atom_names) != atom_count:
        raise ValueError(
            f"ATOM_NAME has {len(atom_names)} entries but POINTERS specifies {atom_count}"
        )
    raw_charges = np.asarray([float(value.replace("D", "E")) for value in sections["CHARGE"][1]])
    if len(raw_charges) != atom_count:
        raise ValueError(
            f"CHARGE has {len(raw_charges)} entries but POINTERS specifies {atom_count}"
        )
    return AmberTopology(
        path=path,
        sections=sections,
        atom_names=atom_names,
        residue_labels=sections["RESIDUE_LABEL"][1],
        residue_pointers=tuple(int(value) for value in sections["RESIDUE_POINTER"][1]),
        charges=raw_charges / AMBER_CHARGE_SCALE,
    )


def residue_atom_indices(topology: AmberTopology, residue_id: int) -> tuple[int, ...]:
    if residue_id > len(topology.residue_pointers):
        raise ValueError(
            f"Topology has {len(topology.residue_pointers)} residues; requested {residue_id}"
        )
    start = topology.residue_pointers[residue_id - 1] - 1
    stop = (
        topology.residue_pointers[residue_id] - 1
        if residue_id < len(topology.residue_pointers)
        else topology.atom_count
    )
    return tuple(range(start, stop))


def build_ligand_charge_map(
    cfg: RefepPrepConfig,
    topology: AmberTopology,
    deprot_atoms: list[Mol2Atom],
) -> LigandChargeMap:
    residue_index = cfg.residue_id - 1
    actual_label = topology.residue_labels[residue_index]
    if actual_label != cfg.residue_name:
        raise ValueError(
            f"Topology residue {cfg.residue_id} is {actual_label!r}, expected {cfg.residue_name!r}"
        )
    atom_indices = residue_atom_indices(topology, cfg.residue_id)
    atom_names = tuple(topology.atom_names[index] for index in atom_indices)
    if len(set(atom_names)) != len(atom_names):
        raise ValueError(
            f"Residue {cfg.residue_id} contains duplicate atom names; name mapping is ambiguous"
        )
    deprot_by_name = {atom.name: atom for atom in deprot_atoms}
    extra = sorted(set(deprot_by_name) - set(atom_names))
    missing = tuple(name for name in atom_names if name not in deprot_by_name)
    if extra:
        raise ValueError(
            "Deprotonated MOL2 atoms absent from the topology residue: " + ", ".join(extra)
        )
    if set(missing) != set(cfg.dummy_atoms):
        raise ValueError(
            f"Mapped missing atoms {list(missing)} do not match ligand.dummy_atoms "
            f"{list(cfg.dummy_atoms)}"
        )
    invalid_dummy = [name for name in missing if not name.upper().startswith("H")]
    if invalid_dummy:
        raise ValueError("Only hydrogen atoms may be zero-charge dummies: " + ", ".join(invalid_dummy))

    lambda0 = topology.charges[np.asarray(atom_indices)]
    lambda1 = np.asarray(
        [0.0 if name in missing else deprot_by_name[name].charge for name in atom_names],
        dtype=float,
    )
    for label, charge, expected in (
        (cfg.lambda0_label, float(lambda0.sum()), cfg.expected_charge_lambda0),
        (cfg.lambda1_label, float(lambda1.sum()), cfg.expected_charge_lambda1),
    ):
        if abs(charge - expected) > cfg.charge_tolerance:
            raise ValueError(
                f"{label} ligand charge {charge:+.6f} differs from expected "
                f"{expected:+d} by more than {cfg.charge_tolerance:g} e"
            )
    return LigandChargeMap(
        atom_indices=atom_indices,
        atom_names=atom_names,
        lambda0_charges=lambda0,
        lambda1_charges=lambda1,
        dummy_atoms=missing,
    )


def render_endpoint_parmed_input(
    cfg: RefepPrepConfig, mapping: LigandChargeMap, output_name: str
) -> str:
    lines = [f"parm {cfg.input_parm7}"]
    for index, charge in zip(mapping.atom_indices, mapping.lambda1_charges):
        lines.append(f"change charge @{index + 1} {charge:.10f} quiet")
    lines.extend([f"outparm {output_name}", "quit", ""])
    return "\n".join(lines)


def render_interpolation_parmed_input(cfg: RefepPrepConfig) -> str:
    final_index = cfg.windows - 1
    return "\n".join(
        [
            f"parm lambda-{final_index:03d}.parm7",
            "parm lambda-000.parm7",
            f"interpolate {cfg.windows - 2} parm2 lambda-{final_index:03d}.parm7 "
            "eleconly prefix interpolated startnum 1",
            "quit",
            "",
        ]
    )


def run_parmed(
    input_path: Path,
    log_path: Path,
    cfg: RefepPrepConfig,
    parm7: Path | None = None,
    rst7: Path | None = None,
) -> None:
    if (parm7 is None) != (rst7 is None):
        raise ValueError("parm7 and rst7 must be supplied together")
    extra_args = [] if parm7 is None else ["-p", str(parm7), "-c", str(rst7)]
    direct = shutil.which(cfg.parmed_executable)
    if direct:
        command = [direct, "-n", "-s", *extra_args, "-i", input_path.name]
    else:
        extra_text = " ".join(shlex.quote(value) for value in extra_args)
        shell_command = "\n".join(
            [
                f"module load {shlex.quote(cfg.amber_module)} >/dev/null 2>&1",
                f"exec {shlex.quote(cfg.parmed_executable)} -n -s {extra_text} "
                f"-i {shlex.quote(input_path.name)}",
            ]
        )
        # The module function is exported on the cluster; a non-login shell avoids
        # unrelated site-profile side effects in the scientific provenance log.
        command = ["bash", "-c", shell_command]
    result = subprocess.run(
        command,
        cwd=input_path.parent,
        text=True,
        capture_output=True,
    )
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"ParmEd failed for {input_path.name}; see {log_path}")


def _assert_same_noncharge_topology(reference: AmberTopology, candidate: AmberTopology) -> None:
    if reference.sections.keys() != candidate.sections.keys():
        raise ValueError(f"Topology sections differ between {reference.path} and {candidate.path}")
    for name in reference.sections:
        if name == "CHARGE":
            continue
        if reference.sections[name][1] != candidate.sections[name][1]:
            raise ValueError(
                f"Non-charge topology section {name} differs in {candidate.path.name}"
            )


def _validate_endpoint(
    cfg: RefepPrepConfig,
    source: AmberTopology,
    endpoint: AmberTopology,
    mapping: LigandChargeMap,
) -> None:
    _assert_same_noncharge_topology(source, endpoint)
    ligand_indices = np.asarray(mapping.atom_indices)
    if not np.allclose(
        endpoint.charges[ligand_indices], mapping.lambda1_charges, atol=5.0e-7, rtol=0.0
    ):
        raise ValueError("Deprotonated endpoint charges do not match the MOL2 charge mapping")
    environment = np.ones(source.atom_count, dtype=bool)
    environment[ligand_indices] = False
    if not np.allclose(
        endpoint.charges[environment], source.charges[environment], atol=1.0e-10, rtol=0.0
    ):
        raise ValueError("Charges outside the selected ligand residue changed")
    expected_delta = cfg.expected_charge_lambda1 - cfg.expected_charge_lambda0
    observed_delta = float(endpoint.charges.sum() - source.charges.sum())
    if abs(observed_delta - expected_delta) > cfg.charge_tolerance:
        raise ValueError(
            f"Full-system endpoint charge change is {observed_delta:+.6f} e; "
            f"expected {expected_delta:+d} e"
        )


def _validate_interpolation(
    cfg: RefepPrepConfig,
    source: AmberTopology,
    endpoint: AmberTopology,
    topologies: list[AmberTopology],
) -> None:
    if len(topologies) != cfg.windows:
        raise ValueError(f"Expected {cfg.windows} lambda topologies; found {len(topologies)}")
    for index, topology in enumerate(topologies):
        _assert_same_noncharge_topology(source, topology)
        lam = index / (cfg.windows - 1)
        expected = source.charges + lam * (endpoint.charges - source.charges)
        if not np.allclose(topology.charges, expected, atol=5.0e-7, rtol=0.0):
            maximum = float(np.max(np.abs(topology.charges - expected)))
            raise ValueError(
                f"lambda-{index:03d}.parm7 is not linearly interpolated; "
                f"maximum charge error {maximum:.3g} e"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_charge_tables(
    cfg: RefepPrepConfig,
    work: Path,
    mapping: LigandChargeMap,
    topologies: list[AmberTopology],
) -> None:
    with (work / "endpoint-charge-map.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["atom_index", "atom_name", "charge_lambda0", "charge_lambda1", "delta_charge", "dummy"]
        )
        for index, name, q0, q1 in zip(
            mapping.atom_indices,
            mapping.atom_names,
            mapping.lambda0_charges,
            mapping.lambda1_charges,
        ):
            writer.writerow(
                [index + 1, name, f"{q0:.8f}", f"{q1:.8f}", f"{q1-q0:.8f}", name in mapping.dummy_atoms]
            )

    ligand_indices = np.asarray(mapping.atom_indices)
    with (work / "charge-interpolation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "lambda",
                *mapping.atom_names,
                "ligand_total_charge",
                "system_total_charge",
            ]
        )
        for index, topology in enumerate(topologies):
            lam = index / (cfg.windows - 1)
            charges = topology.charges[ligand_indices]
            writer.writerow(
                [
                    index,
                    f"{lam:.10f}",
                    *[f"{value:.8f}" for value in charges],
                    f"{charges.sum():+.8f}",
                    f"{topology.charges.sum():+.8f}",
                ]
            )

    with (work / "topology-manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["window", "lambda", "topology", "ligand_charge", "system_charge", "sha256"]
        )
        for index, topology in enumerate(topologies):
            charges = topology.charges[ligand_indices]
            writer.writerow(
                [
                    index,
                    f"{index / (cfg.windows - 1):.10f}",
                    topology.path.name,
                    f"{charges.sum():+.8f}",
                    f"{topology.charges.sum():+.8f}",
                    _sha256(topology.path),
                ]
            )


def plot_charge_interpolation(
    cfg: RefepPrepConfig,
    path: Path,
    mapping: LigandChargeMap,
    topologies: list[AmberTopology],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    lambdas = np.linspace(0.0, 1.0, cfg.windows)
    ligand_indices = np.asarray(mapping.atom_indices)
    charges = np.asarray([topology.charges[ligand_indices] for topology in topologies])

    fig, (atom_ax, total_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.5),
        height_ratios=[2.25, 1.0],
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(mapping.atom_names)))
    for column, (name, color) in enumerate(zip(mapping.atom_names, colors)):
        dummy = name in mapping.dummy_atoms
        atom_ax.plot(
            lambdas,
            charges[:, column],
            marker="o",
            markersize=3.2,
            linewidth=1.8 if dummy else 1.0,
            color=color,
            label=f"{name} (dummy)" if dummy else name,
        )
    atom_ax.axhline(0.0, color="0.65", linewidth=0.7)
    atom_ax.set_ylabel("Atomic charge (e)")
    atom_ax.set_title(f"{cfg.system} single-topology electrostatic interpolation")
    atom_ax.legend(ncol=3, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    ligand_totals = charges.sum(axis=1)
    total_ax.plot(
        lambdas,
        ligand_totals,
        marker="o",
        color="black",
    )
    total_ax.axhline(0.0, color="0.65", linewidth=0.7)
    total_ax.set_xlabel(r"$\lambda$")
    total_ax.set_ylabel("Ligand charge (e)")
    total_ax.set_xlim(0.0, 1.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def prepare_refep_topologies(cfg: RefepPrepConfig) -> Path:
    source = read_amber_topology(cfg.input_parm7)
    mapping = build_ligand_charge_map(cfg, source, read_mol2_atoms(cfg.deprot_mol2))
    destination = cfg.output_directory
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".topology-work-", dir=destination.parent) as tmp:
        work = Path(tmp)
        restart_input = work / "validate-common-restart.parmed.in"
        restart_log = work / "validate-common-restart.parmed.log"
        restart_input.write_text("quit\n")
        run_parmed(
            restart_input,
            restart_log,
            cfg,
            parm7=cfg.input_parm7,
            rst7=cfg.input_rst7,
        )

        endpoint_index = cfg.windows - 1
        endpoint_name = f"lambda-{endpoint_index:03d}.parm7"
        endpoint_input = work / "make-deprotonated-endpoint.parmed.in"
        endpoint_log = work / "make-deprotonated-endpoint.parmed.log"
        endpoint_input.write_text(render_endpoint_parmed_input(cfg, mapping, endpoint_name))
        run_parmed(endpoint_input, endpoint_log, cfg)
        endpoint_path = work / endpoint_name
        if not endpoint_path.is_file() or endpoint_path.stat().st_size == 0:
            raise FileNotFoundError(f"ParmEd did not create {endpoint_path}")
        shutil.copy2(cfg.input_parm7, work / "lambda-000.parm7")
        shutil.copy2(cfg.input_rst7, work / "common.rst7")

        endpoint = read_amber_topology(endpoint_path)
        _validate_endpoint(cfg, source, endpoint, mapping)

        interpolation_input = work / "interpolate.parmed.in"
        interpolation_log = work / "interpolate.parmed.log"
        interpolation_input.write_text(render_interpolation_parmed_input(cfg))
        run_parmed(interpolation_input, interpolation_log, cfg)
        for index in range(1, endpoint_index):
            generated = work / f"interpolated.{index}"
            if not generated.is_file() or generated.stat().st_size == 0:
                raise FileNotFoundError(f"ParmEd did not create {generated}")
            generated.rename(work / f"lambda-{index:03d}.parm7")

        topologies = [
            read_amber_topology(work / f"lambda-{index:03d}.parm7")
            for index in range(cfg.windows)
        ]
        _validate_interpolation(cfg, source, endpoint, topologies)
        write_charge_tables(cfg, work, mapping, topologies)
        plot_charge_interpolation(cfg, work / cfg.plot_name, mapping, topologies)
        (work / "refep-prep-spec.yaml").write_text(cfg.yaml_path.read_text())
        (work / "provenance.yaml").write_text(
            yaml.safe_dump(
                {
                    "method": "single-topology electrostatic interpolation",
                    "source_parm7": str(cfg.input_parm7),
                    "source_rst7": str(cfg.input_rst7),
                    "deprotonated_charge_source": str(cfg.deprot_mol2),
                    "residue_id": cfg.residue_id,
                    "residue_name": cfg.residue_name,
                    "dummy_atoms": list(mapping.dummy_atoms),
                    "windows": cfg.windows,
                    "lambda_formula": "window / (windows - 1)",
                    "charge_formula": "q(lambda) = (1-lambda)*q0 + lambda*q1",
                    "interpolated_terms": "charges only (ParmEd eleconly)",
                    "common_restart_sha256": _sha256(work / "common.rst7"),
                },
                sort_keys=False,
            )
        )

        destination.mkdir(parents=True, exist_ok=True)
        for path in work.iterdir():
            path.replace(destination / path.name)
    return destination


def run_refep_prep(yaml_path: Path) -> None:
    cfg = load_refep_prep_config(yaml_path.resolve())
    destination = prepare_refep_topologies(cfg)
    print(f"OK: wrote {cfg.windows} single-topology REFEP parameter files in {destination}")
    print(
        f"OK: lambda 0 = {cfg.lambda0_label}; lambda 1 = {cfg.lambda1_label}; "
        f"dummy atom(s) = {', '.join(cfg.dummy_atoms)}"
    )
    print(f"OK: wrote charge interpolation plot {destination / cfg.plot_name}")
