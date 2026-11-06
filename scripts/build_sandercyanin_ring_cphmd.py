#!/usr/bin/env python3
"""Transplant TPS+2xPRX into equilibrated Sandercyanin and build a hybrid CPIN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import parmed as pmd
import yaml
from parmed.amber import AmberParm
from parmed.amber import titratable_residues as standard
from parmed.tools import actions


SOLVENT_NAMES = {
    "WAT", "Na+", "Br-", "Cl-", "Cs+", "F-", "I-", "K+", "Li+",
    "Mg+", "Rb+", "CIO", "IB", "MG2",
}
TPS_STATES = ("ppp", "app", "bpp", "cpp", "dpp")
TPS_STATENE = (81.91213251, 0.0, 52.90237631, 44.85032306, 41.58796450)
TPS_PROTCNT = (4, 3, 3, 3, 3)
DEFAULT_TPS_MICROSTATE_PKAS = (4.9, 4.5, 4.5, 4.9)
DEFAULT_PH_VALUES = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adjacency(atoms: list[pmd.Atom]) -> list[set[int]]:
    index = {atom: i for i, atom in enumerate(atoms)}
    graph = [set() for _ in atoms]
    for i, atom in enumerate(atoms):
        for partner in atom.bond_partners:
            if partner in index:
                graph[i].add(index[partner])
    return graph


def heavy_graph(atoms: list[pmd.Atom]) -> tuple[list[pmd.Atom], list[set[int]]]:
    heavy = [atom for atom in atoms if atom.atomic_number != 1]
    return heavy, adjacency(heavy)


def refined_colors(
    old_atoms: list[pmd.Atom], old_graph: list[set[int]],
    new_atoms: list[pmd.Atom], new_graph: list[set[int]],
) -> tuple[list[int], list[int]]:
    old_colors: list[object] = [
        (atom.atomic_number, len(old_graph[i])) for i, atom in enumerate(old_atoms)
    ]
    new_colors: list[object] = [
        (atom.atomic_number, len(new_graph[i])) for i, atom in enumerate(new_atoms)
    ]
    for _ in range(len(old_atoms)):
        old_sig = [
            repr((old_colors[i], tuple(sorted((old_colors[j] for j in old_graph[i]), key=repr))))
            for i in range(len(old_atoms))
        ]
        new_sig = [
            repr((new_colors[i], tuple(sorted((new_colors[j] for j in new_graph[i]), key=repr))))
            for i in range(len(new_atoms))
        ]
        if Counter(old_sig) != Counter(new_sig):
            raise ValueError("Legacy BLA and TPS+2xPRX heavy-atom graphs are not isomorphic")
        identifiers = {value: i for i, value in enumerate(sorted(set(old_sig + new_sig)))}
        next_old = [identifiers[value] for value in old_sig]
        next_new = [identifiers[value] for value in new_sig]
        if next_old == old_colors and next_new == new_colors:
            break
        old_colors, new_colors = next_old, next_new
    return [int(value) for value in old_colors], [int(value) for value in new_colors]


def heavy_isomorphisms(
    old_atoms: list[pmd.Atom], old_graph: list[set[int]],
    new_atoms: list[pmd.Atom], new_graph: list[set[int]],
) -> list[dict[int, int]]:
    old_colors, new_colors = refined_colors(old_atoms, old_graph, new_atoms, new_graph)
    candidates = {
        i: [j for j, color in enumerate(new_colors) if color == old_colors[i]]
        for i in range(len(old_atoms))
    }
    order = sorted(range(len(old_atoms)), key=lambda i: (len(candidates[i]), -len(old_graph[i])))
    answers: list[dict[int, int]] = []

    def search(position: int, mapping: dict[int, int], used: set[int]) -> None:
        if len(answers) >= 64:
            return
        if position == len(order):
            answers.append(dict(mapping))
            return
        old_index = order[position]
        for new_index in candidates[old_index]:
            if new_index in used:
                continue
            if any(
                ((other in old_graph[old_index]) != (mapped in new_graph[new_index]))
                for other, mapped in mapping.items()
            ):
                continue
            mapping[old_index] = new_index
            used.add(new_index)
            search(position + 1, mapping, used)
            used.remove(new_index)
            del mapping[old_index]

    search(0, {}, set())
    if not answers:
        raise ValueError("Could not map the legacy BLA heavy-atom graph onto TPS+2xPRX")
    return answers


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    fitted = source @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return rotation, translation, rmsd


def coordinate(atom: pmd.Atom) -> np.ndarray:
    return np.asarray([atom.xx, atom.xy, atom.xz], dtype=float)


def box_vectors(box: np.ndarray) -> np.ndarray:
    lengths = np.asarray(box[:3], dtype=float)
    alpha, beta, gamma = np.deg2rad(np.asarray(box[3:6], dtype=float))
    vector_a = np.asarray([lengths[0], 0.0, 0.0])
    vector_b = np.asarray([lengths[1] * np.cos(gamma), lengths[1] * np.sin(gamma), 0.0])
    cx = lengths[2] * np.cos(beta)
    cy = lengths[2] * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(lengths[2] ** 2 - cx ** 2 - cy ** 2, 0.0))
    return np.asarray([vector_a, vector_b, [cx, cy, cz]])


def periodic_minimum_distance(
    first: np.ndarray, second: np.ndarray, vectors: np.ndarray
) -> float:
    inverse = np.linalg.inv(vectors)
    best = np.inf
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    for point in first:
        delta = second - point
        fractional = delta @ inverse
        fractional -= np.rint(fractional)
        candidates = (fractional[:, None, :] + shifts[None, :, :]) @ vectors
        best = min(best, float(np.linalg.norm(candidates, axis=2).min()))
    return best


def nearby_titratable_residues(
    structure: pmd.Structure, cutoff: float
) -> tuple[list[int], list[dict[str, object]]]:
    if structure.box is None:
        raise ValueError("The equilibrated monomer restart has no periodic box")
    vectors = box_vectors(np.asarray(structure.box, dtype=float))
    bla = structure.residues[169]
    bla_heavy = np.asarray([coordinate(atom) for atom in bla.atoms if atom.atomic_number != 1])
    allowed = {
        name for name in standard.titratable_residues
        if getattr(standard, name).typ == "ph"
    }
    records: list[dict[str, object]] = []
    selected: list[int] = []
    for residue_number, residue in enumerate(structure.residues[:169], start=1):
        if residue.name not in allowed:
            continue
        residue_heavy = np.asarray(
            [coordinate(atom) for atom in residue.atoms if atom.atomic_number != 1]
        )
        distance = periodic_minimum_distance(bla_heavy, residue_heavy, vectors)
        included = distance <= cutoff
        records.append(
            {
                "residue_number": residue_number,
                "residue_name": residue.name,
                "minimum_heavy_atom_distance_angstrom": distance,
                "included": included,
            }
        )
        if included:
            selected.append(residue_number)
    return selected, sorted(records, key=lambda record: float(record["minimum_heavy_atom_distance_angstrom"]))


def choose_heavy_mapping(
    old_atoms: list[pmd.Atom], new_atoms: list[pmd.Atom]
) -> tuple[dict[pmd.Atom, pmd.Atom], np.ndarray, np.ndarray, float]:
    old_heavy, old_graph = heavy_graph(old_atoms)
    new_heavy, new_graph = heavy_graph(new_atoms)
    if len(old_heavy) != len(new_heavy):
        raise ValueError("Legacy and replacement BLA heavy-atom counts differ")
    best: tuple[float, dict[int, int], np.ndarray, np.ndarray] | None = None
    for mapping in heavy_isomorphisms(old_heavy, old_graph, new_heavy, new_graph):
        source = np.asarray([coordinate(new_heavy[mapping[i]]) for i in range(len(old_heavy))])
        target = np.asarray([coordinate(atom) for atom in old_heavy])
        rotation, translation, rmsd = kabsch(source, target)
        if best is None or rmsd < best[0]:
            best = (rmsd, mapping, rotation, translation)
    assert best is not None
    rmsd, mapping, rotation, translation = best
    atom_mapping = {old_heavy[i]: new_heavy[j] for i, j in mapping.items()}
    return atom_mapping, rotation, translation, rmsd


def complete_hydrogen_mapping(
    old_atoms: list[pmd.Atom], new_atoms: list[pmd.Atom],
    mapping: dict[pmd.Atom, pmd.Atom], rotation: np.ndarray, translation: np.ndarray,
) -> tuple[dict[pmd.Atom, pmd.Atom], pmd.Atom]:
    old_h_by_parent: dict[pmd.Atom, list[pmd.Atom]] = {}
    new_h_by_parent: dict[pmd.Atom, list[pmd.Atom]] = {}
    for atom in old_atoms:
        if atom.atomic_number == 1:
            parent = next(partner for partner in atom.bond_partners if partner.atomic_number != 1)
            old_h_by_parent.setdefault(parent, []).append(atom)
    for atom in new_atoms:
        if atom.atomic_number == 1:
            parent = next(partner for partner in atom.bond_partners if partner.atomic_number != 1)
            new_h_by_parent.setdefault(parent, []).append(atom)

    mapped_new = set(mapping.values())
    for old_parent, old_hydrogens in old_h_by_parent.items():
        new_parent = mapping[old_parent]
        candidates = [atom for atom in new_h_by_parent.get(new_parent, []) if atom not in mapped_new]
        if len(candidates) < len(old_hydrogens):
            raise ValueError(f"Too few replacement hydrogens attached to {new_parent.name}")
        best: tuple[float, tuple[pmd.Atom, ...]] | None = None
        for selected in itertools.permutations(candidates, len(old_hydrogens)):
            score = sum(
                float(np.sum(((coordinate(new_h) @ rotation + translation) - coordinate(old_h)) ** 2))
                for old_h, new_h in zip(old_hydrogens, selected)
            )
            if best is None or score < best[0]:
                best = (score, selected)
        assert best is not None
        for old_h, new_h in zip(old_hydrogens, best[1]):
            mapping[old_h] = new_h
            mapped_new.add(new_h)

    unmatched = [atom for atom in new_atoms if atom not in mapped_new]
    if len(unmatched) != 1 or unmatched[0].atomic_number != 1:
        raise ValueError("Expected exactly one new ring hydrogen after BLA mapping")
    return mapping, unmatched[0]


def augment_legacy_bla(
    source_parm7: Path, source_rst7: Path, template_parm7: Path, template_rst7: Path,
    charge_sets: Path,
) -> tuple[
    pmd.Structure, list[dict[str, object]], dict[str, object],
    list[str], dict[str, list[float]],
]:
    old = pmd.load_file(str(source_parm7), str(source_rst7))
    template = pmd.load_file(str(template_parm7), str(template_rst7))
    if old.residues[169].name != "BLA":
        raise ValueError("Legacy topology residue 170 is not BLA")
    if [residue.name for residue in template.residues[:3]] != ["TPS", "PRX", "PRX"]:
        raise ValueError("Replacement topology must begin with TPS, PRX, PRX")

    old_bla_atoms = list(old.residues[169].atoms)
    template_bla = template[":1-3"].copy(pmd.Structure)
    template_atoms = list(template_bla.atoms)
    mapping, rotation, translation, heavy_rmsd = choose_heavy_mapping(old_bla_atoms, template_atoms)
    mapping, added_hydrogen = complete_hydrogen_mapping(
        old_bla_atoms, template_atoms, mapping, rotation, translation
    )

    old_velocities = np.asarray(old.velocities, dtype=float)
    old_atom_index = {atom: atom.idx for atom in old.atoms}
    mapped_old_by_new = {new_atom: old_atom for old_atom, new_atom in mapping.items()}
    legacy_bla = old[":170"].copy(pmd.Structure)
    legacy_atoms = list(legacy_bla.atoms)
    old_local_index = {atom: index for index, atom in enumerate(old_bla_atoms)}
    new_to_augmented = {
        new_atom: legacy_atoms[old_local_index[old_atom]]
        for old_atom, new_atom in mapping.items()
    }

    # The old tail-only topology lacks one of the four ring N-H sites. Add that
    # single atom at the end of BLA and copy only its local bonded terms.
    added_atom = pmd.Atom(
        atomic_number=1,
        name="HXR",
        type=added_hydrogen.type,
        charge=added_hydrogen.charge,
        mass=added_hydrogen.mass,
        solvent_radius=added_hydrogen.solvent_radius,
        screen=added_hydrogen.screen,
        rmin=added_hydrogen.rmin,
        epsilon=added_hydrogen.epsilon,
        rmin14=added_hydrogen.rmin_14,
        epsilon14=added_hydrogen.epsilon_14,
    )
    added_atom.atom_type = deepcopy(added_hydrogen.atom_type)
    legacy_residue = legacy_bla.residues[0]
    legacy_bla.add_atom(
        added_atom, legacy_residue.name, legacy_residue.number,
        legacy_residue.chain, legacy_residue.insertion_code, legacy_residue.segid,
    )
    new_to_augmented[added_hydrogen] = added_atom

    def mapped(atom: pmd.Atom) -> pmd.Atom:
        try:
            return new_to_augmented[atom]
        except KeyError as exc:
            raise ValueError(f"Unmapped atom {atom.name} in added-H force term") from exc

    for bond in added_hydrogen.bonds:
        interaction_type = deepcopy(bond.type)
        legacy_bla.bond_types.append(interaction_type)
        legacy_bla.bonds.append(
            pmd.Bond(mapped(bond.atom1), mapped(bond.atom2), interaction_type, order=bond.order)
        )
    for angle in added_hydrogen.angles:
        interaction_type = deepcopy(angle.type)
        legacy_bla.angle_types.append(interaction_type)
        legacy_bla.angles.append(
            pmd.Angle(mapped(angle.atom1), mapped(angle.atom2), mapped(angle.atom3), interaction_type)
        )
    for dihedral in added_hydrogen.dihedrals:
        interaction_type = deepcopy(dihedral.type)
        legacy_bla.dihedral_types.append(interaction_type)
        legacy_bla.dihedrals.append(
            pmd.Dihedral(
                mapped(dihedral.atom1), mapped(dihedral.atom2),
                mapped(dihedral.atom3), mapped(dihedral.atom4),
                improper=dihedral.improper, ignore_end=dihedral.ignore_end,
                type=interaction_type,
            )
        )

    parent = next(
        partner for partner in added_hydrogen.bond_partners if partner.atomic_number != 1
    )
    old_parent = mapped_old_by_new[parent]
    transformed_parent = coordinate(parent) @ rotation + translation
    transformed_h = coordinate(added_hydrogen) @ rotation + translation
    added_coordinate = coordinate(old_parent) + (transformed_h - transformed_parent)
    legacy_bla.coordinates = np.vstack(
        [np.asarray([coordinate(atom) for atom in old_bla_atoms]), added_coordinate]
    )
    legacy_bla.velocities = np.vstack(
        [old_velocities[[old_atom_index[atom] for atom in old_bla_atoms]], old_velocities[old_atom_index[old_parent]]]
    )

    tps_atom_names, tps_charges = read_tps_charges(charge_sets)
    tps_charge_lookup = {
        state: dict(zip(tps_atom_names, tps_charges[state])) for state in TPS_STATES
    }
    state_charges: dict[str, list[float]] = {}
    topology_atom_names = [atom.name for atom in legacy_bla.atoms]
    for state in TPS_STATES:
        charges: list[float] = []
        for old_atom in old_bla_atoms:
            source_atom = mapping[old_atom]
            if source_atom.residue.name == "TPS":
                charges.append(tps_charge_lookup[state][source_atom.name])
            else:
                charges.append(float(source_atom.charge))
        if added_hydrogen.residue.name != "TPS":
            raise ValueError("The added ring proton did not map to TPS")
        charges.append(tps_charge_lookup[state][added_hydrogen.name])
        expected = -1.0 if state == "ppp" else -2.0
        if abs(sum(charges) - expected) > 1.0e-7:
            raise ValueError(
                f"Mapped whole-BLA state {state} has charge {sum(charges):+.8f}; "
                f"expected {expected:+.1f}"
            )
        state_charges[state] = charges
    for atom, charge in zip(legacy_bla.atoms, state_charges["ppp"]):
        atom.charge = charge

    # The new PPP + two deprotonated PRX master is one charge unit less negative
    # than legacy BLA. Remove the farther of the two equilibrated sodium ions.
    sodium_residues = [residue for residue in old.residues if residue.name == "Na+"]
    if len(sodium_residues) != 2:
        raise ValueError(f"Expected two legacy sodium ions, found {len(sodium_residues)}")
    bla_xyz = np.asarray([coordinate(atom) for atom in old_bla_atoms])
    vectors = box_vectors(np.asarray(old.box, dtype=float))
    sodium_distance = {
        residue: periodic_minimum_distance(
            bla_xyz, np.asarray([coordinate(residue.atoms[0])]), vectors
        )
        for residue in sodium_residues
    }
    removed_sodium = max(sodium_residues, key=sodium_distance.get)

    protein = old[":1-169"].copy(pmd.Structure)
    if removed_sodium.idx + 1 != 171:
        raise ValueError(
            "The farther sodium is not legacy residue 171; refusing an unexpected splice"
        )
    suffix_mask = f":172-{len(old.residues)}"
    suffix = old[suffix_mask].copy(pmd.Structure)
    combined = protein + legacy_bla + suffix
    combined.box = np.asarray(old.box, dtype=float)

    mapping_rows = []
    for old_atom, new_atom in sorted(mapping.items(), key=lambda item: item[0].idx):
        mapping_rows.append(
            {
                "legacy_atom_id": old_atom.idx + 1,
                "legacy_atom_name": old_atom.name,
                "charge_source_residue": new_atom.residue.name,
                "charge_source_atom_name": new_atom.name,
                "new_bla_atom_name": old_atom.name,
                "new_bla_atom_id": old_local_index[old_atom] + 1,
            }
        )
    mapping_rows.append(
        {
            "legacy_atom_id": "",
            "legacy_atom_name": "",
            "charge_source_residue": added_hydrogen.residue.name,
            "charge_source_atom_name": added_hydrogen.name,
            "new_bla_atom_name": added_atom.name,
            "new_bla_atom_id": len(legacy_bla.atoms),
        }
    )
    details = {
        "legacy_bla_atoms": len(old_bla_atoms),
        "replacement_bla_atoms": len(legacy_bla.atoms),
        "heavy_atom_mapping_rmsd_angstrom": heavy_rmsd,
        "added_ring_hydrogen": added_atom.name,
        "added_ring_hydrogen_charge_source": added_hydrogen.name,
        "removed_sodium_legacy_residue": removed_sodium.idx + 1,
        "removed_sodium_distance_to_bla_angstrom": sodium_distance[removed_sodium],
    }
    return combined, mapping_rows, details, topology_atom_names, state_charges


def read_tps_charges(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    atom_names = [row["atom_master"] for row in rows if row["atom_master"] not in {"TOTAL", "PROTON_COUNT", "H2C", "H3B"}]
    charges = {
        state: [float(row[f"charge_{state}"]) for row in rows if row["atom_master"] in atom_names]
        for state in TPS_STATES
    }
    return atom_names, charges


def tps_pka_corrections(
    microscopic_pkas: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float]:
    """Convert PPP-to-tautomer pKas to Amber's five-state correction gauge."""
    reference = max(microscopic_pkas)
    return (reference, *(reference - value for value in microscopic_pkas))


def custom_bla(
    atom_names: list[str],
    charges: dict[str, list[float]],
    pka_corr: tuple[float, float, float, float, float],
):
    residue = standard.TitratableResidue("BLA", atom_names, typ="ph", pka=4.5)
    for state, energy, protcnt, correction in zip(
        TPS_STATES, TPS_STATENE, TPS_PROTCNT, pka_corr
    ):
        refene = standard._ReferenceEnergy(igb2=energy)
        refene.solvent_energies(igb2=energy)
        residue.add_state(
            charges=charges[state], refene=refene, refene_old=refene,
            protcnt=protcnt, pka_corr=correction,
        )
    residue.check()
    return residue


def build_cpin(
    parm7: Path, destination: Path, nearby_residues: list[int],
    bla_atom_names: list[str], bla_charges: dict[str, list[float]],
    pka_corr: tuple[float, float, float, float, float],
) -> list[dict[str, object]]:
    parm = AmberParm(str(parm7))
    bla = custom_bla(bla_atom_names, bla_charges, pka_corr)
    sites = (*nearby_residues, 170)
    first_solvent = next(
        residue.atoms[0].idx + 1 for residue in parm.residues if residue.name in SOLVENT_NAMES
    )
    residue_list = standard.TitratableResidueList(
        system_name="Sandercyanin monomer: BLA rings + nearby residues",
        solvated=True,
        first_solvent=first_solvent,
    )
    records: list[dict[str, object]] = []
    for residue_number in sorted(sites):
        topology_residue = parm.residues[residue_number - 1]
        definition = bla if residue_number == 170 else getattr(standard, topology_residue.name)
        topology_names = [atom.name for atom in topology_residue.atoms]
        if topology_names != definition.atom_list:
            raise ValueError(
                f"Residue {residue_number} {topology_residue.name} atom order does not "
                "match its CpHMD definition"
            )
        residue_list.add_residue(definition, residue_number, topology_residue.atoms[0].idx + 1)
        records.append(
            {
                "residue_number": residue_number,
                "residue_name": topology_residue.name,
                "atom_count": len(topology_residue.atoms),
                "state_count": len(definition.states),
            }
        )
    with destination.open("w") as handle:
        residue_list.write_cpin(handle, igb=2, intdiel=1.0, oldfmt=False, typ="ph")
    return records


def render_mdin(ph: float, numexchg: int) -> str:
    return f"""Sandercyanin monomer BLA-ring plus nearby-residue RECpHMD; pH {ph:.1f}
&cntrl
 iwrap=1, imin=0, irest=1, ntx=5,
 nstlim=1000, numexchg={numexchg}, dt=0.002,
 tempi=300.0, temp0=300.0, ntt=3, gamma_ln=1.0, ig=-1,
 ntb=1, ntp=0, ntc=2, ntf=2, cut=8.0,
 ntpr=500, ntwx=500, ntwr=1000, ioutfm=1,
 icnstph=2, solvph={ph:.1f}, ntcnstph=100, ntrelax=100, saltcon=0.1,
/
"""


def write_remd_inputs(
    destination: Path, parm7_name: str, rst7_name: str, cpin_text: str,
    ph_values: tuple[float, ...], production_ns: float, partition: str,
    nodes: int, ranks_per_replica: int, walltime: str, job_name: str,
) -> None:
    numexchg_float = production_ns * 500.0
    numexchg = int(round(numexchg_float))
    if abs(numexchg - numexchg_float) > 1.0e-8 or numexchg <= 0 or numexchg % 2:
        raise ValueError("production_ns must yield a positive even number of 2-ps exchanges")
    group_lines = []
    for index, ph in enumerate(ph_values, start=1):
        stem = f"replica-{index:02d}"
        (destination / f"{stem}.mdin").write_text(render_mdin(ph, numexchg))
        (destination / f"{stem}.cpin").write_text(cpin_text)
        group_lines.append(
            f"-O -i {stem}.mdin -p {parm7_name} -c {rst7_name} -cpin {stem}.cpin "
            f"-o {stem}.mdout -r {stem}.rst7 -x {stem}.nc -cpout {stem}.cpout "
            f"-cprestrt {stem}.cprestrt -inf {stem}.mdinfo"
        )
    (destination / "groupfile").write_text("\n".join(group_lines) + "\n")
    (destination / "ph-ladder.csv").write_text(
        "replica,pH\n" + "".join(f"{i},{ph:.1f}\n" for i, ph in enumerate(ph_values, start=1))
    )
    total_ranks = ranks_per_replica * len(ph_values)
    run = destination / "run.sh"
    run.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmodule purge\nmodule load amber/26\n"
        f"export TEMP=300\nmpirun -np {total_ranks} pmemd.MPI -ng {len(ph_values)} "
        "-groupfile groupfile -rem 4 -remlog rem.log\n"
    )
    run.chmod(0o755)
    slurm = destination / "slurm.sh"
    slurm.write_text(
        f"#!/usr/bin/env bash\n#SBATCH -J {job_name}\n"
        f"#SBATCH -p {partition}\n#SBATCH -N {nodes}\n"
        f"#SBATCH --ntasks={total_ranks}\n"
        f"#SBATCH --ntasks-per-node={(total_ranks + nodes - 1) // nodes}\n"
        f"#SBATCH -t {walltime}\n"
        "#SBATCH -o slurm.%j.out\n#SBATCH -e slurm.%j.err\n\nbash run.sh\n"
    )
    slurm.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parm7", type=Path, required=True)
    parser.add_argument("--source-rst7", type=Path, required=True)
    parser.add_argument("--template-parm7", type=Path, required=True)
    parser.add_argument("--template-rst7", type=Path, required=True)
    parser.add_argument("--charge-sets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nearby-cutoff", type=float, default=5.0)
    parser.add_argument("--production-ns", type=float, default=1.0)
    parser.add_argument("--ph-values", type=float, nargs="+", default=DEFAULT_PH_VALUES)
    parser.add_argument(
        "--tps-microstate-pkas",
        type=float,
        nargs=4,
        metavar=("APP", "BPP", "CPP", "DPP"),
        default=DEFAULT_TPS_MICROSTATE_PKAS,
        help="PPP-to-APP/BPP/CPP/DPP microscopic pKas",
    )
    parser.add_argument("--partition", default="small")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--ranks-per-replica", type=int, default=2)
    parser.add_argument("--walltime", default="48:00:00")
    parser.add_argument("--job-name", default="sc-mon-ring-cphmd-p49")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    source = pmd.load_file(str(args.source_parm7), str(args.source_rst7))
    nearby_residues, distance_records = nearby_titratable_residues(
        source, args.nearby_cutoff
    )
    combined, mapping_rows, details, bla_atom_names, bla_charges = augment_legacy_bla(
        args.source_parm7, args.source_rst7, args.template_parm7, args.template_rst7,
        args.charge_sets,
    )
    args.output_dir.mkdir(parents=True)
    parm7 = args.output_dir / "mon.rings-nearby.parm7"
    rst7 = args.output_dir / "mon.rings-nearby.rst7"
    combined.save(str(parm7), overwrite=True)
    combined.save(str(rst7), format="rst7", overwrite=True)
    finalized = AmberParm(str(parm7))
    actions.changeRadii(finalized, "mbondi2").execute()
    actions.change(
        finalized, "RADII",
        "(:AS4,GL4@OD=,OE=)|(:BLA@O1A,O2A,O1D,O2D)", 1.3,
    ).execute()
    finalized.write_parm(str(parm7))

    with (args.output_dir / "bla-atom-mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    cpin = args.output_dir / "mon.rings-nearby.cpin"
    with (args.output_dir / "nearby-titratable-residues.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distance_records[0]))
        writer.writeheader()
        writer.writerows(distance_records)
    microscopic_pkas = tuple(float(value) for value in args.tps_microstate_pkas)
    if len(microscopic_pkas) != 4 or not all(np.isfinite(microscopic_pkas)):
        raise ValueError("--tps-microstate-pkas requires four finite values")
    pka_corr = tps_pka_corrections(microscopic_pkas)  # type: ignore[arg-type]
    site_records = build_cpin(
        parm7, cpin, nearby_residues, bla_atom_names, bla_charges, pka_corr
    )
    ph_values = tuple(float(value) for value in args.ph_values)
    write_remd_inputs(
        args.output_dir, parm7.name, rst7.name, cpin.read_text(),
        ph_values, args.production_ns, args.partition, args.nodes,
        args.ranks_per_replica, args.walltime, args.job_name,
    )

    provenance = {
        "source_parm7": str(args.source_parm7.resolve()),
        "source_rst7": str(args.source_rst7.resolve()),
        "source_parm7_sha256": sha256(args.source_parm7),
        "source_rst7_sha256": sha256(args.source_rst7),
        "replacement_bla_template_parm7": str(args.template_parm7.resolve()),
        "replacement_bla_template_rst7": str(args.template_rst7.resolve()),
        "tps_charge_sets": str(args.charge_sets.resolve()),
        "tps_states": list(TPS_STATES),
        "tps_statene_kcal_mol": list(TPS_STATENE),
        "tps_microstate_pkas": dict(zip(TPS_STATES[1:], microscopic_pkas)),
        "tps_pka_corr": list(pka_corr),
        "nearby_residue_selection": (
            f"periodic heavy-atom distance <= {args.nearby_cutoff:g} A from legacy BLA"
        ),
        "titrating_residues": site_records,
        "production_ns_per_replica": args.production_ns,
        "ph_values": list(ph_values),
        "slurm": {
            "partition": args.partition,
            "nodes": args.nodes,
            "ranks_per_replica": args.ranks_per_replica,
            "total_ranks": args.ranks_per_replica * len(ph_values),
            "walltime": args.walltime,
        },
        **details,
        "output_parm7_sha256": sha256(parm7),
        "output_rst7_sha256": sha256(rst7),
        "output_cpin_sha256": sha256(cpin),
    }
    (args.output_dir / "provenance.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))
    print(f"OK: prepared Sandercyanin ring CpHMD inputs in {args.output_dir}")


if __name__ == "__main__":
    main()
