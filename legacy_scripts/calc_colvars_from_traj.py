#!/usr/bin/env python3
"""Compute custom colvars from an XYZ trajectory using a YAML configuration.

Implemented colvars:
1) Rational coordination number between two atom groups:
   n_ij = sum_{i in g1} sum_{j in g2} (1-(r_ij/r0)^p)/(1-(r_ij/r0)^(p+q))

2) Distance between two atom groups:
   s = | r_g1 - r_g2 |

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

try:
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    raise SystemExit("tqdm is required. Install it with: pip install tqdm") from exc


EPS = 1.0e-12


@dataclass
class RationalCNConfig:
    enabled: bool
    group1_indices: list[int]
    group2_indices: list[int]
    r0: float
    p: float
    q: float


@dataclass
class DistConfig:
    enabled: bool
    group1_indices: list[int]
    group2_indices: list[int]
    beta: float


@dataclass
class IOConfig:
    trajectory: Path
    output_root: Path
    stride: int
    one_based_indices: bool
    write_combined_csv: bool


@dataclass
class ColvarConfig:
    io: IOConfig
    rational_cn: RationalCNConfig
    soft_min: SoftMinConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute custom colvars from XYZ trajectory")
    parser.add_argument("--config", type=Path, default=Path("configs/colvar_calc.yaml"), help="YAML config path")
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional trajectory override")
    parser.add_argument("--stride", type=int, default=None, help="Optional stride override")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional output_root override")
    parser.add_argument("--molecule", default="TEA", help="Molecule name for default output path")
    return parser.parse_args()


def _as_int_list(values: list[int]) -> list[int]:
    return [int(v) for v in values]


def load_config(path: Path) -> ColvarConfig:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    io_cfg = cfg.get("io", {})
    c_cfg = cfg.get("colvars", {})
    rc_cfg = c_cfg.get("rational_coordination", {})
    sm_cfg = c_cfg.get("soft_min_distance", {})

    output_root = io_cfg.get("output_root")
    if output_root is None:
        # backward-compat fallback
        old_csv = io_cfg.get("output_csv", "reports/manual_colvars/colvars_manual.csv")
        output_root = str(Path(old_csv).parent)

    io = IOConfig(
        trajectory=Path(io_cfg["trajectory"]),
        output_root=Path(output_root),
        stride=max(1, int(io_cfg.get("stride", 1))),
        one_based_indices=bool(io_cfg.get("one_based_indices", True)),
        write_combined_csv=bool(io_cfg.get("write_combined_csv", True)),
    )

    rational_cn = RationalCNConfig(
        enabled=bool(rc_cfg.get("enabled", True)),
        group1_indices=_as_int_list(rc_cfg["group1_indices"]),
        group2_indices=_as_int_list(rc_cfg["group2_indices"]),
        r0=float(rc_cfg["r0"]),
        p=float(rc_cfg["p"]),
        q=float(rc_cfg["q"]),
    )

    soft_min = SoftMinConfig(
        enabled=bool(sm_cfg.get("enabled", True)),
        group1_indices=_as_int_list(sm_cfg["group1_indices"]),
        group2_indices=_as_int_list(sm_cfg["group2_indices"]),
        beta=float(sm_cfg["beta"]),
    )

    return ColvarConfig(io=io, rational_cn=rational_cn, soft_min=soft_min)


def maybe_to_zero_based(indices: list[int], one_based: bool) -> np.ndarray:
    arr = np.array(indices, dtype=int)
    if one_based:
        arr = arr - 1
    return arr


def validate_indices(indices: np.ndarray, n_atoms: int, name: str) -> None:
    if indices.size == 0:
        raise ValueError(f"{name} is empty")
    if np.any(indices < 0) or np.any(indices >= n_atoms):
        bad = indices[(indices < 0) | (indices >= n_atoms)]
        raise IndexError(f"{name} has out-of-range indices for n_atoms={n_atoms}: {bad.tolist()}")


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
            n_atoms = int(line)
            _ = f.readline()  # comment
            for _ in range(n_atoms):
                if not f.readline():
                    return n_frames
            n_frames += 1
    return n_frames


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


def pairwise_distances(coords: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    c1 = coords[g1]
    c2 = coords[g2]
    d = c1[:, None, :] - c2[None, :, :]
    return np.linalg.norm(d, axis=2)


def rational_coordination_number(dists: np.ndarray, r0: float, p: float, q: float) -> float:
    x = dists / max(r0, EPS)
    num = 1.0
    den = 1.0 + np.power(x, p)
    limit_at_one = p / max(p + q, EPS)
    term = np.where(np.abs(den) < 1.0e-14, limit_at_one, num / den)
    return float(np.sum(term))


def soft_min_distance(dists: np.ndarray, beta: float) -> float:
    if np.any(dists <= EPS):
        return 0.0
    a = beta / dists
    a_max = float(np.max(a))
    lse = a_max + float(np.log(np.sum(np.exp(a - a_max))))
    return float(beta / max(lse, EPS))


def compute_colvars(config: ColvarConfig) -> list[dict[str, float]]:
    traj = config.io.trajectory
    stride = max(1, config.io.stride)

    total_frames = count_xyz_frames(traj)

    rows: list[dict[str, float]] = []
    g1_cn: np.ndarray | None = None
    g2_cn: np.ndarray | None = None
    g1_sm: np.ndarray | None = None
    g2_sm: np.ndarray | None = None

    for frame_idx, coords in tqdm(
        enumerate(iter_xyz_coords(traj)),
        total=total_frames,
        desc=f"Frames ({traj.name})",
        unit="frame",
    ):
        if frame_idx % stride != 0:
            continue

        if g1_cn is None:
            n_atoms = coords.shape[0]
            if config.rational_cn.enabled:
                g1_cn = maybe_to_zero_based(config.rational_cn.group1_indices, config.io.one_based_indices)
                g2_cn = maybe_to_zero_based(config.rational_cn.group2_indices, config.io.one_based_indices)
                validate_indices(g1_cn, n_atoms, "rational_coordination.group1_indices")
                validate_indices(g2_cn, n_atoms, "rational_coordination.group2_indices")

            if config.soft_min.enabled:
                g1_sm = maybe_to_zero_based(config.soft_min.group1_indices, config.io.one_based_indices)
                g2_sm = maybe_to_zero_based(config.soft_min.group2_indices, config.io.one_based_indices)
                validate_indices(g1_sm, n_atoms, "soft_min_distance.group1_indices")
                validate_indices(g2_sm, n_atoms, "soft_min_distance.group2_indices")

        row: dict[str, float] = {"frame": float(frame_idx)}

        if config.rational_cn.enabled and g1_cn is not None and g2_cn is not None:
            d_cn = pairwise_distances(coords, g1_cn, g2_cn)
            row["rational_cn"] = rational_coordination_number(
                d_cn, config.rational_cn.r0, config.rational_cn.p, config.rational_cn.q
            )

        if config.soft_min.enabled and g1_sm is not None and g2_sm is not None:
            d_sm = pairwise_distances(coords, g1_sm, g2_sm)
            row["soft_min_distance"] = soft_min_distance(d_sm, config.soft_min.beta)

        rows.append(row)

    return rows


def mirrored_output_dir(trajectory: Path, output_root: Path, project_root: Path) -> Path:
    t = trajectory.resolve()
    out = output_root.resolve()
    root = project_root.resolve()
    try:
        rel = t.relative_to(root)
        # Preferred mirror: output/.../manual_colvars/<colvar_type>/run-X/<method>/
        # for input like data/molecules/<molecule>/colvars/<colvar_type>/run-X/<method>/traj.xyz
        if len(rel.parts) >= 7 and rel.parts[0] == "data" and rel.parts[1] == "molecules" and rel.parts[3] == "colvars":
            tail = Path(*rel.parts[4:-1])
            return out / tail
        # Backward compatibility: data/colvars/<colvar_type>/run-X/<method>/traj.xyz
        if len(rel.parts) >= 5 and rel.parts[0] == "data" and rel.parts[1] == "colvars":
            tail = Path(*rel.parts[2:-1])
            return out / tail
        return out / rel.parent
    except ValueError:
        return out / t.stem


def write_dat(path: Path, rows: list[dict[str, float]], key: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if key in row:
                f.write(f"{int(row['frame']):12d} {row[key]:20.10f}\n")


def write_combined_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["frame"]
    if rows and "rational_cn" in rows[0]:
        fieldnames.append("rational_cn")
    if rows and "soft_min_distance" in rows[0]:
        fieldnames.append("soft_min_distance")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent

    if args.trajectory is not None:
        cfg.io.trajectory = args.trajectory
    if args.stride is not None:
        cfg.io.stride = max(1, int(args.stride))
    if args.output_root is not None:
        cfg.io.output_root = args.output_root
    elif cfg.io.output_root == Path("output/manual_colvars"):
        cfg.io.output_root = project_root / "output" / "molecules" / args.molecule / "manual_colvars"

    rows = compute_colvars(cfg)
    if not rows:
        raise SystemExit("No frames were processed. Check trajectory and stride.")

    out_dir = mirrored_output_dir(cfg.io.trajectory, cfg.io.output_root, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    colvar1 = out_dir / "colvar1.dat"
    colvar2 = out_dir / "colvar2.dat"
    write_dat(colvar1, rows, "rational_cn")
    write_dat(colvar2, rows, "soft_min_distance")

    if cfg.io.write_combined_csv:
        write_combined_csv(out_dir / "colvars_manual.csv", rows)

    print(f"Trajectory: {cfg.io.trajectory}")
    print(f"Stride: {cfg.io.stride}")
    print(f"Frames processed: {len(rows)}")
    print(f"Output dir: {out_dir}")
    print(f"Wrote: {colvar1}")
    print(f"Wrote: {colvar2}")


if __name__ == "__main__":
    main()
