"""Geometry-based, dynamic proton ownership for PRN dihedral overlays."""
from __future__ import annotations

import numpy as np


def persistent_ids(times: np.ndarray, ids: np.ndarray, duration_ps: float) -> np.ndarray:
    """Keep same-ID intervals lasting long enough; never bridge missing frames.

    Accepted intervals include their first frame (retrospective confirmation).
    NaN means unassigned, ambiguous, or too short, not chemically unprotonated.
    """
    output = np.full(ids.shape, np.nan)
    steps = np.diff(times)
    if np.any(steps <= 0) or duration_ps < 0:
        raise ValueError("Times must increase and persistence must be nonnegative")
    max_gap = 1.5 * np.median(steps) if len(steps) else np.inf
    start = 0
    while start < len(ids):
        if not np.isfinite(ids[start]):
            start += 1
            continue
        end = start + 1
        while end < len(ids) and ids[end] == ids[start] and times[end] - times[end-1] <= max_gap:
            end += 1
        if times[end-1] - times[start] + 1.e-10 >= duration_ps:
            output[start:end] = ids[start]
        start = end
    return output


def bond_dihedrals(
    times: np.ndarray, coords: np.ndarray, symbols: list[str], box: np.ndarray,
    old_quartet: list[int], new_oxygen_id: int, cutoff_A: float = 1.4,
    ownership_margin_A: float = .15, persistence_ps: float = .05,
) -> dict[str, np.ndarray]:
    """Track old O-H and a dynamically acquired H on the neighboring oxygen.

    Each H belongs to its uniquely closest heavy atom if distance <= cutoff
    and the second closest heavy atom is at least margin farther away. An O5
    frame with multiple assigned H atoms is ambiguous and omitted. No permanent
    water ownership, charge-window criterion, or diffusion-time truncation is used.
    """
    from acid_base_PRN import _minimum_image, _signed_dihedral_deg
    if cutoff_A <= 0 or ownership_margin_A < 0 or persistence_ps < 0:
        raise ValueError("Invalid bond-detection thresholds")
    reference, carbon, old_oxygen, old_h = old_quartet
    if reference != new_oxygen_id:
        raise ValueError("Old dihedral must start with the neighboring oxygen")
    for atom, symbol in ((old_oxygen, "O"), (new_oxygen_id, "O"), (carbon, "C"), (old_h, "H")):
        if not 1 <= atom <= len(symbols) or symbols[atom-1] != symbol:
            raise ValueError("PRN dihedral atom IDs do not match O-C-O-H elements")
    hydrogens = np.asarray([i for i, symbol in enumerate(symbols, 1) if symbol == "H"])
    heavy = np.asarray([i for i, symbol in enumerate(symbols, 1) if symbol != "H"])
    old_h_index = int(np.flatnonzero(hydrogens == old_h)[0])
    n = len(times)
    old_candidates, new_candidates = np.full(n, np.nan), np.full(n, np.nan)
    old_distance, new_distance = np.full(n, np.nan), np.full(n, np.nan)
    new_counts = np.zeros(n, dtype=int)
    old_owner = np.full(n, np.nan)
    site_candidates = {o: np.full(n, np.nan) for o in (old_oxygen, new_oxygen_id)}
    site_distances = {o: np.full(n, np.nan) for o in site_candidates}
    for i, frame in enumerate(coords):
        vectors = _minimum_image(frame[hydrogens-1, None, :] - frame[None, heavy-1, :], box)
        distances = np.linalg.norm(vectors, axis=2)
        order = np.argsort(distances, axis=1)[:, :2]
        first = distances[np.arange(len(hydrogens)), order[:, 0]]
        second = distances[np.arange(len(hydrogens)), order[:, 1]]
        owned = (first <= cutoff_A) & (second-first >= ownership_margin_A)
        owners = np.where(owned, heavy[order[:, 0]], -1)
        for oxygen in site_candidates:
            acquired = np.flatnonzero((owners == oxygen) & (hydrogens != old_h))
            # Multiple new H atoms are ambiguous; do not choose arbitrarily.
            if len(acquired) == 1:
                k = acquired[0]
                site_candidates[oxygen][i] = hydrogens[k]
                site_distances[oxygen][i] = first[k]
        old_owner[i] = owners[old_h_index] if owners[old_h_index] > 0 else np.nan
        old_distance[i] = np.linalg.norm(_minimum_image(frame[old_h-1] - frame[old_oxygen-1], box))
        if owners[old_h_index] == old_oxygen:
            old_candidates[i] = old_h
        matches = np.flatnonzero(owners == new_oxygen_id)
        new_counts[i] = len(matches)
        if len(matches) == 1:
            h_index = matches[0]
            new_candidates[i] = hydrogens[h_index]
            new_distance[i] = first[h_index]
    old_ids = persistent_ids(times, old_candidates, persistence_ps)
    new_ids = persistent_ids(times, new_candidates, persistence_ps)
    old_phi, new_phi = np.full(n, np.nan), np.full(n, np.nan)
    for i, frame in enumerate(coords):
        if np.isfinite(old_ids[i]):
            old_phi[i] = _signed_dihedral_deg(frame[np.asarray(old_quartet)-1], box)
        if np.isfinite(new_ids[i]):
            quartet = np.asarray([old_oxygen, carbon, new_oxygen_id, int(new_ids[i])])
            new_phi[i] = _signed_dihedral_deg(frame[quartet-1], box)
    result = dict(time_ps=times, old_hydrogen_id=old_ids, old_H_nearest_owner_id=old_owner,
                old_bond_distance_A=old_distance, old_dihedral_deg=old_phi,
                new_candidate_hydrogen_id=new_candidates, new_assigned_H_count=new_counts,
                new_bond_distance_A=new_distance, new_hydrogen_id=new_ids,
                new_dihedral_deg=new_phi)
    for oxygen, candidates in site_candidates.items():
        ids = persistent_ids(times, candidates, persistence_ps)
        result[f"O{oxygen}_acquired_hydrogen_id"] = ids
        result[f"O{oxygen}_acquired_bond_distance_A"] = np.where(np.isfinite(ids), site_distances[oxygen], np.nan)
    result["acquired_bond_cutoff_A"] = np.full(n, cutoff_A)
    return result
