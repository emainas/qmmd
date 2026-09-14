"""Target-atom Shrake–Rupley SASA with periodic geometry (angstrom units)."""
from __future__ import annotations

import numpy as np

RADII = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52}


def sphere_points(count: int) -> np.ndarray:
    if count < 100:
        raise ValueError('Use at least 100 surface points')
    i = np.arange(count)
    z = 1 - 2 * (i + .5) / count
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z*z)
    return np.column_stack([r*np.cos(phi), r*np.sin(phi), z])


def mic(x: np.ndarray, box: np.ndarray) -> np.ndarray:
    return x - np.round(x / box) * box


def atom_sasa(offsets: np.ndarray, blocker_radii: np.ndarray, target_radius: float,
              probe: float, points: np.ndarray) -> float:
    """Offsets are already minimum-imaged relative to the target atom."""
    radius = target_radius + probe
    expanded = blocker_radii + probe
    near = np.linalg.norm(offsets, axis=1) < radius + expanded
    accessible = np.ones(len(points), dtype=bool)
    surface = points * radius
    for center, other_radius in zip(offsets[near], expanded[near]):
        accessible &= np.sum((surface - center)**2, axis=1) >= other_radius**2
    return float(4*np.pi*radius**2 * accessible.mean())


def nc_sasa(coords: np.ndarray, elements: np.ndarray, box: np.ndarray, target: int,
            solute_atoms: int, probe: float, points: np.ndarray, bond_cutoff: float,
            radii: dict[str, float] = RADII) -> tuple[float, float, int]:
    """Target is zero-based internally; dynamically include solute-bound H.

    H belongs to its nearest heavy atom only when within bond_cutoff. Waters
    and solvent-bound H do not block SASA. Return all-atom, heavy-only, NC-H count.
    """
    offsets = mic(coords - coords[target], box)
    heavy = np.flatnonzero(elements != 'H')
    solute_heavy = heavy[(heavy < solute_atoms) & (heavy != target)]
    # Only H capable of blocking the target surface need ownership assignment.
    reach = radii[elements[target]] + radii['H'] + 2*probe
    hydrogens = np.flatnonzero((elements == 'H') & (np.linalg.norm(offsets, axis=1) < reach))
    if len(hydrogens):
        distances = np.linalg.norm(mic(coords[hydrogens, None] - coords[heavy], box), axis=2)
        nearest = distances.argmin(axis=1)
        owners = heavy[nearest]
        bonded = distances[np.arange(len(hydrogens)), nearest] <= bond_cutoff
        solute_h = hydrogens[bonded & (owners < solute_atoms)]
        nh = int(np.count_nonzero(bonded & (owners == target)))
    else:
        solute_h, nh = np.array([], dtype=int), 0
    blockers = np.concatenate([solute_heavy, solute_h])
    full = atom_sasa(offsets[blockers], np.array([radii[e] for e in elements[blockers]]),
                     radii[elements[target]], probe, points)
    heavy_sasa = atom_sasa(offsets[solute_heavy], np.array([radii[e] for e in elements[solute_heavy]]),
                          radii[elements[target]], probe, points)
    return full, heavy_sasa, nh
