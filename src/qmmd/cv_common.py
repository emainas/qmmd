#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np


BIASPOT_COORD_RE = re.compile(r"Coordinate\s*=")
BIASPOT_COORD_VAL_RE = re.compile(r"Coordinate\s*=\s*([+-]?[0-9.]+)")
STEP_RE = re.compile(r"STEP NO\.\s*=\s*([0-9]+)")


@dataclass(frozen=True)
class CVRun:
    run_id: int
    traj: Path
    biaspot: Path


def discover_runs(runs_path: Path, cv_dir: str, traj_name: str, biaspot_name: str) -> List[CVRun]:
    out: List[CVRun] = []
    for p in sorted(runs_path.iterdir()):
        if not (p.is_dir() and p.name.startswith("run-")):
            continue
        try:
            run_id = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        run_cv = p / cv_dir
        traj = run_cv / traj_name
        biaspot = run_cv / biaspot_name
        if traj.exists() and biaspot.exists():
            out.append(CVRun(run_id, traj, biaspot))
    return out


def parse_run_ids(value) -> List[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        if "-" in value:
            a, b = value.split("-", 1)
            return list(range(int(a.strip()), int(b.strip()) + 1))
        return [int(value)]
    if isinstance(value, list):
        out: List[int] = []
        for item in value:
            out.extend(parse_run_ids(item))
        return out
    raise RuntimeError(f"Bad run_ids spec: {value!r}")


def parse_biaspot_count(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if BIASPOT_COORD_RE.search(line):
                n += 1
    return n


def parse_biaspot_values(path: Path) -> List[float]:
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = BIASPOT_COORD_VAL_RE.search(line)
            if m:
                vals.append(float(m.group(1)))
    return vals


def parse_first_step_biaspot(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                return int(m.group(1))
    raise RuntimeError(f"No STEP NO found in {path}")


def parse_biaspot_stride(path: Path) -> int:
    steps: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                if len(steps) == 2:
                    break
    if len(steps) < 2:
        raise RuntimeError(f"Need at least 2 STEP NO entries in {path} to infer stride")
    if steps[1] <= steps[0]:
        raise RuntimeError(f"Non-increasing STEP NO in biaspot: {steps[0]} -> {steps[1]}")
    return steps[1] - steps[0]


def parse_first_step_traj(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
        if not line:
            raise RuntimeError(f"Empty trajectory: {path}")
        comment = f.readline()
        if not comment:
            raise RuntimeError(f"Missing comment line in trajectory: {path}")
        m = STEP_RE.search(comment)
        if not m:
            raise RuntimeError(f"No STEP NO found in trajectory comment: {path}")
        return int(m.group(1))


def parse_traj_stride(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
        if not line:
            raise RuntimeError(f"Empty trajectory: {path}")
        comment1 = f.readline()
        if not comment1:
            raise RuntimeError(f"Missing comment line in trajectory: {path}")
        m1 = STEP_RE.search(comment1)
        if not m1:
            raise RuntimeError(f"No STEP NO found in trajectory comment: {path}")
        n_atoms = int(line.strip())
        for _ in range(n_atoms):
            if not f.readline():
                raise RuntimeError(f"Unexpected EOF in trajectory: {path}")
        line2 = f.readline()
        if not line2:
            raise RuntimeError(f"Missing second frame in trajectory: {path}")
        comment2 = f.readline()
        if not comment2:
            raise RuntimeError(f"Missing second comment line in trajectory: {path}")
        m2 = STEP_RE.search(comment2)
        if not m2:
            raise RuntimeError(f"No STEP NO found in trajectory comment: {path}")
        step1 = int(m1.group(1))
        step2 = int(m2.group(1))
        if step2 <= step1:
            raise RuntimeError(f"Non-increasing STEP NO in trajectory: {step1} -> {step2}")
        return step2 - step1


def system_base_dir(system: str, prefix: str, buffer: float, repo_root: Path) -> Path:
    return repo_root / "systems" / system / f"{prefix}_{buffer:.1f}"


def iter_xyz_coords(path: Path) -> Iterator[np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom-count line: {line}") from exc
            comment = f.readline()
            if not comment:
                break
            coords = np.empty((n_atoms, 3), dtype=float)
            for i in range(n_atoms):
                atom_line = f.readline()
                if not atom_line:
                    raise ValueError("Unexpected EOF while reading XYZ frame")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid XYZ atom line: {atom_line.strip()}")
                coords[i, 0] = float(parts[1])
                coords[i, 1] = float(parts[2])
                coords[i, 2] = float(parts[3])
            yield coords


def count_xyz_frames(path: Path) -> int:
    n_frames = 0
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError:
                raise ValueError(f"Invalid XYZ atom-count line: {line}")
            _ = f.readline()  # comment
            for _ in range(n_atoms):
                if not f.readline():
                    return n_frames
            n_frames += 1
    return n_frames


def iter_xyz_stride(path: Path, stride: int, start: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    for frame_idx, coords in enumerate(iter_xyz_coords(path)):
        if frame_idx < start:
            continue
        if (frame_idx - start) % stride == 0:
            yield frame_idx, coords


def iter_xyz_window(
    path: Path,
    start: int,
    stop: int | None,
    stride: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if start < 0:
        raise ValueError("start must be >= 0")
    for frame_idx, coords in enumerate(iter_xyz_coords(path)):
        if frame_idx < start:
            continue
        if stop is not None and frame_idx > stop:
            break
        if (frame_idx - start) % stride == 0:
            yield frame_idx, coords


def one_based_to_zero(indices: List[int]) -> np.ndarray:
    return np.array(indices, dtype=int) - 1


def validate_indices(indices: np.ndarray, n_atoms: int, name: str) -> None:
    if indices.size == 0:
        raise ValueError(f"{name} is empty")
    if np.any(indices < 0) or np.any(indices >= n_atoms):
        bad = indices[(indices < 0) | (indices >= n_atoms)]
        raise IndexError(f"{name} has out-of-range indices for n_atoms={n_atoms}: {bad.tolist()}")


def select_frames(
    traj: Path, target_indices: List[int]
) -> Iterator[Tuple[int, np.ndarray]]:
    target = sorted(target_indices)
    if not target:
        return
    ptr = 0
    target_set = set(target)
    for frame_idx, coords in enumerate(iter_xyz_coords(traj)):
        if frame_idx in target_set:
            yield frame_idx, coords
            ptr += 1
            if ptr >= len(target):
                break
