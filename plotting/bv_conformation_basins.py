"""Periodic angular-density basins and snapshot-safe dihedral extraction."""
from __future__ import annotations

import hashlib
import io
import itertools
import os
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + 180.0) % 360.0 - 180.0


def dihedrals(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Signed torsions for (...,4,3) coordinates, using minimum-image bonds."""
    bonds = np.diff(points, axis=-2)
    bonds -= np.round(bonds / box) * box
    b0, b1, b2 = (bonds[..., i, :] for i in range(3))
    n0, n1 = np.cross(b0, b1), np.cross(b1, b2)
    lengths = [np.linalg.norm(v, axis=-1, keepdims=True) for v in (n0, n1, b1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        u0, u1, axis = [v / n for v, n in zip((n0, n1, b1), lengths)]
    return np.degrees(np.arctan2(np.sum(np.cross(u0, u1) * axis, axis=-1),
                                 np.sum(u0 * u1, axis=-1)))


def read_torsions(path: Path, quartets: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a size-bounded XYZ snapshot; quartets are one-based atom IDs."""
    with path.open("rb") as f:
        raw = f.read(os.fstat(f.fileno()).st_size)
    provenance = {"source": str(path.resolve()), "snapshot_bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest(), "incomplete_tail_ignored": False}
    stream = io.BytesIO(raw)
    times, selected = [], []
    wanted = sorted(set(quartets.ravel()))
    expected_count = None
    while line := stream.readline():
        if not line.strip():
            continue
        count = int(line)
        if expected_count is None:
            expected_count = count
        if count != expected_count or max(wanted) > count:
            raise ValueError("Trajectory atom count changed or quartet is out of bounds")
        comment = stream.readline()
        rows = list(itertools.islice(stream, count))
        if len(rows) != count or not rows[-1].endswith(b"\n") or len(rows[-1].split()) != 4:
            provenance["incomplete_tail_ignored"] = True
            break
        match = re.search(rb"AT T=\s*([\d.]+)\s*FSEC", comment)
        if match is None:
            raise ValueError("Missing XYZ timestamp")
        coords = {i: np.asarray(rows[i - 1].split()[1:], dtype=float) for i in wanted}
        selected.append([[coords[i] for i in q] for q in quartets])
        times.append(float(match[1]) / 1000.0)
    t = np.asarray(times)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Need at least two strictly increasing timestamps; do not silently stitch restarts")
    phi = dihedrals(np.asarray(selected), box)
    if not np.all(np.isfinite(phi)):
        raise ValueError("Degenerate or non-finite torsion geometry")
    return t, phi, provenance


def circular_mean_std(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.mean(np.exp(1j * np.radians(angles)), axis=0)
    mean = wrap_degrees(np.degrees(np.angle(z)))
    std = np.degrees(np.sqrt(-2 * np.log(np.clip(np.abs(z), 1e-15, 1.0))))
    return mean, std


def density_basins(angles: np.ndarray, bin_deg: float = 10.0,
                   bandwidth_deg: float = 15.0, merge_deg: float = 30.0,
                   min_population: float = 0.05) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """26-neighbor uphill catchments of a wrapped Gaussian-smoothed histogram.

    Nearby peak catchments merge into the higher-population representative.
    Groups below min_population remain label 0, not reassigned to major basins.
    Populations are fractions of observed frames, not equilibrium probabilities.
    """
    if not 0 < bin_deg <= 60 or not np.isclose(360 / bin_deg, round(360 / bin_deg)):
        raise ValueError("bin_deg must divide 360 and lie in (0,60]")
    if bandwidth_deg <= 0 or merge_deg < 0 or not 0 < min_population <= 1:
        raise ValueError("Invalid bandwidth, merge radius or minimum population")
    a = wrap_degrees(np.asarray(angles, dtype=float))
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("angles must be a nonempty finite (N,3) array")
    n = int(round(360 / bin_deg))
    edges = np.linspace(-180, 180, n + 1)
    hist, _ = np.histogramdd(a, bins=(edges, edges, edges))
    density = gaussian_filter(hist / len(a), bandwidth_deg / bin_deg, mode="wrap")
    grid_ids = np.arange(n ** 3).reshape((n, n, n))
    parent = grid_ids.copy()
    best = density.copy()
    for shift in itertools.product((-1, 0, 1), repeat=3):
        if shift == (0, 0, 0):
            continue
        neighbor = np.roll(density, shift, axis=(0, 1, 2))
        improve = neighbor > best
        parent[improve] = np.roll(grid_ids, shift, axis=(0, 1, 2))[improve]
        best[improve] = neighbor[improve]
    roots = parent.ravel()
    for _ in range(n ** 3):
        updated = roots[roots]
        if np.array_equal(updated, roots):
            break
        roots = updated
    else:
        raise RuntimeError("Density ascent did not converge")
    cells = np.minimum(((a + 180) / bin_deg).astype(int), n - 1)
    flat = np.ravel_multi_index(cells.T, (n, n, n))
    frame_roots = roots[flat]
    unique, counts = np.unique(frame_roots, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    groups: list[dict] = []
    for root, count in zip(unique[order], counts[order]):
        peak = -180 + (np.asarray(np.unravel_index(root, (n, n, n))) + .5) * bin_deg
        nearby = [(np.linalg.norm(wrap_degrees(peak - group["mode_deg"])), i)
                  for i, group in enumerate(groups)]
        nearby = [(d, i) for d, i in nearby if d <= merge_deg]
        if nearby:
            group = groups[min(nearby)[1]]
            group["roots"].append(root)
            group["count"] += int(count)
        else:
            groups.append(dict(mode_deg=peak, roots=[root], count=int(count)))
    groups.sort(key=lambda g: -g["count"])
    labels = np.zeros(len(a), dtype=int)
    basins = []
    for group in groups:
        population = group["count"] / len(a)
        if population < min_population:
            continue
        mask = np.isin(frame_roots, group["roots"])
        label = len(basins) + 1
        labels[mask] = label
        mean, std = circular_mean_std(a[mask])
        basins.append(dict(label=label, count=int(mask.sum()), population=population,
                           mode_deg=group["mode_deg"].tolist(), mean_deg=mean.tolist(),
                           circular_std_deg=std.tolist()))
    return labels, basins, density.ravel()[flat]
