#!/usr/bin/env python3

from __future__ import annotations

import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from qmmd.cv_common import (
    CVRun,
    discover_runs,
    parse_biaspot_count,
    parse_biaspot_values,
    one_based_to_zero,
    validate_indices,
    select_frames,
    parse_run_ids,
    system_base_dir,
    iter_xyz_window,
    parse_first_step_biaspot,
    parse_first_step_traj,
    parse_traj_stride,
    parse_biaspot_stride,
    count_xyz_frames,
)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

@dataclass(frozen=True)
class CVCfg:
    runs: List[CVRun]
    group1: List[int]
    group2: List[int]
    r0: float
    p: float
    q: float
    meta_start: Optional[int]
    meta_stop: Optional[int]
    meta_stride: Optional[int]
    system: str
    buffer: float
    prefix: str
    dftb_dirname: str
    bench_tag: str
    cv_dir: str
    traj_name: str
    biaspot_name: str
    equil_dirname: Optional[str]
    equil_traj_name: Optional[str]
    equil_start: Optional[int]
    equil_stop: Optional[int]
    equil_stride: Optional[int]
    validate_tol: float


def load_config(path: Path) -> CVCfg:
    data = yaml.safe_load(path.read_text())
    system = data["system"]
    buffer = float(data["buffer"])
    prefix = data.get("prefix", "solv")
    dftb_dirname = data.get("dftb_dirname", "dftb")
    bench_tag = data["bench_tag"]
    prod_cfg = data.get("prod", {})
    cv_dir = prod_cfg["cv_dir"]
    traj_name = prod_cfg.get("traj_name", "traject")
    biaspot_name = prod_cfg.get("biaspot_name", "biaspot")
    equil_cfg = data.get("equil", {})
    equil_dirname = equil_cfg.get("dirname")
    equil_traj_name = equil_cfg.get("traj_name")
    equil_start = equil_cfg.get("start")
    equil_stop = equil_cfg.get("stop")
    equil_stride = equil_cfg.get("stride")
    meta_start = prod_cfg.get("start")
    meta_stop = prod_cfg.get("stop")
    meta_stride = prod_cfg.get("stride")
    validate_tol = float(data.get("validate_tol", 1e-5))

    repo_root = Path(__file__).resolve().parent.parent.parent
    runs_path = system_base_dir(system, prefix, buffer, repo_root) / dftb_dirname / bench_tag
    runs = discover_runs(runs_path, cv_dir, traj_name, biaspot_name)
    if "run_ids" in data:
        wanted = set(parse_run_ids(data["run_ids"]))
        runs = [r for r in runs if r.run_id in wanted]
    return CVCfg(
        runs=runs,
        group1=[int(x) for x in data["group1"]],
        group2=[int(x) for x in data["group2"]],
        r0=float(data["r0"]),
        p=float(data["p"]),
        q=float(data["q"]),
        meta_start=meta_start,
        meta_stop=int(meta_stop) if meta_stop is not None else None,
        meta_stride=meta_stride,
        system=system,
        buffer=buffer,
        prefix=prefix,
        dftb_dirname=dftb_dirname,
        bench_tag=bench_tag,
        cv_dir=cv_dir,
        traj_name=traj_name,
        biaspot_name=biaspot_name,
        equil_dirname=equil_dirname,
        equil_traj_name=equil_traj_name,
        equil_start=int(equil_start) if equil_start is not None else None,
        equil_stop=int(equil_stop) if equil_stop is not None else None,
        equil_stride=int(equil_stride) if equil_stride is not None else None,
        validate_tol=validate_tol,
    )


def rational_coordination(dists: np.ndarray, r0: float, p: float, q: float) -> float:
    x = dists / r0
    num = 1.0 - np.power(x, p)
    den = 1.0 - np.power(x, q)
    close = np.isclose(den, 0.0)
    out = np.empty_like(x, dtype=float)
    out[~close] = num[~close] / den[~close]
    if np.any(close):
        out[close] = p / q
    return float(np.sum(out))


def compute_run(cfg: CVCfg, run: CVRun, validate: bool = False) -> Path:
    user_start = cfg.meta_start is not None
    user_stride = cfg.meta_stride is not None
    n_bias = parse_biaspot_count(run.biaspot)
    if n_bias == 0:
        raise RuntimeError(f"No Coordinate entries in {run.biaspot}")
    step0_traj = parse_first_step_traj(run.traj)
    step0_bias = parse_first_step_biaspot(run.biaspot)
    stride_traj = parse_traj_stride(run.traj)

    if cfg.meta_stride is None:
        stride_bias = parse_biaspot_stride(run.biaspot)
        if stride_bias % stride_traj != 0:
            raise RuntimeError(
                f"Auto-stride failed: bias_stride {stride_bias} not divisible by traj_stride {stride_traj}"
            )
        meta_stride = stride_bias // stride_traj
    else:
        meta_stride = int(cfg.meta_stride)

    if cfg.meta_start is None:
        if (step0_bias - step0_traj) % stride_traj != 0:
            raise RuntimeError(
                f"Auto-start failed: (bias_step0 - traj_step0) not divisible by stride_traj "
                f"({step0_bias} - {step0_traj}) % {stride_traj} != 0"
            )
        meta_start = (step0_bias - step0_traj) // stride_traj
    else:
        meta_start = int(cfg.meta_start)

    if cfg.meta_stop is None:
        if not user_start and not user_stride:
            meta_stop = meta_start + (n_bias - 1) * meta_stride
        else:
            meta_stop = None
    else:
        meta_stop = cfg.meta_stop

    out_dir = run.traj.parent / "manual-cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coord.dat"

    g1 = one_based_to_zero(cfg.group1)
    g2 = one_based_to_zero(cfg.group2)

    equil_vals: List[float] = []
    equil_dist: List[float] = []
    if cfg.equil_dirname and cfg.equil_traj_name:
        equil_traj = run.traj.parent.parent / cfg.equil_dirname / cfg.equil_traj_name
        if not equil_traj.exists():
            raise RuntimeError(f"Missing equil traj: {equil_traj}")
        stride_eq = cfg.equil_stride if cfg.equil_stride is not None else meta_stride
        start_eq = cfg.equil_start or 0
        stop_eq = cfg.equil_stop
        total_frames = count_xyz_frames(equil_traj)
        if stop_eq is None:
            stop_eq_calc = total_frames - 1
        else:
            stop_eq_calc = min(stop_eq, total_frames - 1)
        total_iters = max(0, ((stop_eq_calc - start_eq) // stride_eq) + 1) if stop_eq_calc >= start_eq else 0
        iterator = iter_xyz_window(equil_traj, start_eq, stop_eq, stride_eq)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                desc=f"run-{run.run_id} equil",
                unit="frame",
                total=total_iters,
                file=sys.stdout,
                dynamic_ncols=True,
            )
        for _, coords in iterator:
            if len(equil_vals) == 0:
                n_atoms = coords.shape[0]
                validate_indices(g1, n_atoms, "group1")
                validate_indices(g2, n_atoms, "group2")
            c1 = coords[g1]
            c2 = coords[g2]
            d = c1[:, None, :] - c2[None, :, :]
            dists = np.linalg.norm(d, axis=2)
            equil_vals.append(rational_coordination(dists, cfg.r0, cfg.p, cfg.q))
            equil_dist.append(float(np.mean(dists)))

    prod_vals: List[float] = []
    prod_dist: List[float] = []
    total_frames = count_xyz_frames(run.traj)
    if meta_stop is None:
        meta_stop_calc = total_frames - 1
    else:
        meta_stop_calc = min(meta_stop, total_frames - 1)
    total_iters = max(0, ((meta_stop_calc - meta_start) // meta_stride) + 1) if meta_stop_calc >= meta_start else 0
    iterator = iter_xyz_window(run.traj, meta_start, meta_stop, meta_stride)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            desc=f"run-{run.run_id} prod",
            unit="frame",
            total=total_iters,
            file=sys.stdout,
            dynamic_ncols=True,
        )
    if total_iters == 0:
        print(
            f"[WARN] run-{run.run_id} prod has zero frames: start={meta_start} stop={meta_stop} "
            f"stride={meta_stride} total_frames={total_frames}"
        )
    for _, coords in iterator:
        if len(equil_vals) == 0 and len(prod_vals) == 0:
            n_atoms = coords.shape[0]
            validate_indices(g1, n_atoms, "group1")
            validate_indices(g2, n_atoms, "group2")
        c1 = coords[g1]
        c2 = coords[g2]
        d = c1[:, None, :] - c2[None, :, :]
        dists = np.linalg.norm(d, axis=2)
        prod_vals.append(rational_coordination(dists, cfg.r0, cfg.p, cfg.q))
        prod_dist.append(float(np.mean(dists)))

    rows: List[float] = equil_vals + prod_vals
    dist_rows: List[float] = equil_dist + prod_dist

    with out_path.open("w", encoding="utf-8") as f:
        for i, val in enumerate(rows, start=1):
            f.write(f"{i:8d} {val:20.10f}\n")
    prod_path = out_dir / "coord_prod.dat"
    with prod_path.open("w", encoding="utf-8") as f:
        for i, val in enumerate(prod_vals, start=1):
            f.write(f"{i:8d} {val:20.10f}\n")
    dist_path = out_dir / "dist.dat"
    with dist_path.open("w", encoding="utf-8") as f:
        for i, val in enumerate(dist_rows, start=1):
            f.write(f"{i:8d} {val:20.10f}\n")
    dist_prod_path = out_dir / "dist_prod.dat"
    with dist_prod_path.open("w", encoding="utf-8") as f:
        for i, val in enumerate(prod_dist, start=1):
            f.write(f"{i:8d} {val:20.10f}\n")

    if validate:
        bias_vals = parse_biaspot_values(run.biaspot)
        if len(prod_vals) != len(bias_vals):
            print(
                f"[WARN] validation length mismatch for run-{run.run_id}: "
                f"computed {len(prod_vals)} vs biaspot {len(bias_vals)}; "
                f"comparing first {min(len(prod_vals), len(bias_vals))} entries"
            )
        n = min(len(prod_vals), len(bias_vals))
        diffs = np.abs(np.array(prod_vals[:n]) - np.array(bias_vals[:n]))
        if np.any(diffs > cfg.validate_tol):
            idx = int(np.argmax(diffs))
            print(
                f"[WARN] validation failed for run-{run.run_id} at index {idx}: "
                f"computed={prod_vals[idx]:.8f} biaspot={bias_vals[idx]:.8f}"
            )
        else:
            print(f"OK: validation passed for run-{run.run_id}")

    print(f"OK: run-{run.run_id} -> {out_path}")
    print(f"OK: run-{run.run_id} -> {prod_path}")
    print(f"OK: run-{run.run_id} -> {dist_path}")
    print(f"OK: run-{run.run_id} -> {dist_prod_path}")
    return out_path


def run_cv_coord(yaml_path: Path, validate: bool = False) -> None:
    cfg = load_config(yaml_path)
    for run in cfg.runs:
        print(f"==> run-{run.run_id}")
        compute_run(cfg, run, validate=validate)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 cv_coord.py configs/SYS/cv/cv_coord.yaml", file=sys.stderr)
        raise SystemExit(2)
    run_cv_coord(Path(sys.argv[1]).resolve(), validate=False)


if __name__ == "__main__":
    main()
