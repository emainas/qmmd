#!/usr/bin/env python3
"""
equil_report.py

2x2 report:
  - Upper left: mdequil NPT temperature (left y) + density (right y)
  - Upper right: histogram of NPT Ewald error estimate
  - Lower left: DFTB equil temperature (left y) + pressure (right y)
  - Lower right: RDF curves (mask 11 and 14)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


TIME_RE = re.compile(r"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root (pyproject.toml) starting from {start}")


def parse_npt_temp(out_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    temps: List[float] = []
    for line in out_path.read_text().splitlines():
        if "NSTEP" not in line or "TIME(PS)" not in line or "TEMP(K)" not in line:
            continue
        parts = line.split()
        try:
            t_idx = parts.index("TIME(PS)") + 2
            temp_idx = parts.index("TEMP(K)") + 2
            times.append(float(parts[t_idx]))
            temps.append(float(parts[temp_idx]))
        except Exception:
            continue
    return np.array(times), np.array(temps)


def parse_npt_density(out_path: Path) -> np.ndarray:
    dens: List[float] = []
    for line in out_path.read_text().splitlines():
        if "Density" not in line:
            continue
        parts = line.split()
        try:
            dens.append(float(parts[-1]))
        except Exception:
            continue
    return np.array(dens)


def parse_ewald_errors(out_path: Path) -> np.ndarray:
    vals: List[float] = []
    for line in out_path.read_text().splitlines():
        if "Ewald error estimate:" not in line:
            continue
        parts = line.split()
        try:
            vals.append(float(parts[-1]))
        except Exception:
            continue
    return np.array(vals)


def parse_dftb_temp_pressure(out_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    temps: List[float] = []
    press: List[float] = []
    current_t = None
    for line in out_path.read_text().splitlines():
        m = TIME_RE.search(line)
        if m:
            current_t = float(m.group(1)) / 1000.0
            continue
        if "TEMPERATURE" in line:
            parts = line.split()
            try:
                val = float(parts[-2])
                if current_t is not None:
                    times.append(current_t)
                    temps.append(val)
            except Exception:
                continue
        if "PRESSURE" in line and "Pa" in line:
            parts = line.split()
            try:
                val = float(parts[-2])
                if current_t is not None:
                    press.append(val)
            except Exception:
                continue
    t = np.array(times)
    temp = np.array(temps)
    p = np.array(press)
    n = min(len(t), len(p))
    return t[:n], temp[:n], p[:n]


def read_rdf(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Bad RDF format: {path}")
    return data[:, 0], data[:, 1]


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x))


def main() -> None:
    p = argparse.ArgumentParser(description="Equilibration report (2x2 panels).")
    p.add_argument("--system", required=True)
    p.add_argument("--buffer", required=True, type=float)
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--bench-tag", required=True)
    p.add_argument("--amber-equil-dir", default="mdequil", help="Amber equil base dir (contains equil-npt.out)")
    p.add_argument("--dftb-dir", default="dftb")
    p.add_argument("--dftb-equil-dir", default="equil-nvt-postAmberNPT")
    p.add_argument("--rdf-dir", default=None, help="Defaults to dftb-equil-dir")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    style = script_dir / "prl.mplstyle"
    if style.exists():
        plt.style.use(style)

    repo_root = find_repo_root(Path.cwd())
    solv_tag = f"solv_{args.buffer:.1f}"

    # Paths
    npt_out = repo_root / "systems" / args.system / solv_tag / args.amber_equil_dir / "equil-npt.out"
    if not npt_out.exists():
        raise SystemExit(f"Missing NPT out: {npt_out}")

    dftb_out = (
        repo_root
        / "systems"
        / args.system
        / solv_tag
        / args.dftb_dir
        / args.bench_tag
        / f"run-{args.run_id}"
        / args.dftb_equil_dir
        / "dftb.out"
    )
    if not dftb_out.exists():
        raise SystemExit(f"Missing DFTB out: {dftb_out}")

    rdf_dir = args.rdf_dir or args.dftb_equil_dir
    rdf_base = (
        repo_root
        / "systems"
        / args.system
        / solv_tag
        / args.dftb_dir
        / args.bench_tag
        / f"run-{args.run_id}"
        / rdf_dir
        / "analysis"
    )
    rdf11 = rdf_base / "rdf_11_WAT_O.dat"
    rdf14 = rdf_base / "rdf_14_WAT_O.dat"
    if not rdf11.exists() or not rdf14.exists():
        raise SystemExit(f"Missing RDF files in {rdf_base}")

    # Load data
    npt_t, npt_temp = parse_npt_temp(npt_out)
    npt_rho = parse_npt_density(npt_out)
    if npt_rho.size >= 2:
        npt_rho = npt_rho[:-2]
    n = min(len(npt_t), len(npt_rho))
    npt_t = npt_t[:n]
    npt_temp = npt_temp[:n]
    npt_rho = npt_rho[:n]

    ewald = parse_ewald_errors(npt_out)
    dftb_t, dftb_temp, dftb_press = parse_dftb_temp_pressure(dftb_out)
    r11, g11 = read_rdf(rdf11)
    r14, g14 = read_rdf(rdf14)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=220)

    # Upper left: NPT temp + density
    ax = axes[0, 0]
    npt_temp_mu, npt_temp_sd = mean_std(npt_temp)
    npt_rho_mu, npt_rho_sd = mean_std(npt_rho)

    temp_line = ax.plot(
        npt_t,
        npt_temp,
        color="#1f77b4",
        lw=1.5,
        label=f"T = {npt_temp_mu:.2f} ± {npt_temp_sd:.2f} K",
    )[0]
    ax.set_title("NPT Temperature / Density")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    dens_line = ax2.plot(
        npt_t,
        npt_rho,
        color="#d62728",
        lw=1.5,
        label=f"ρ = {npt_rho_mu:.4f} ± {npt_rho_sd:.4f}",
    )[0]
    ax2.set_ylabel("Density (g/cm$^3$)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.legend(handles=[temp_line], frameon=False, fontsize=8, loc="upper left")
    ax2.legend(handles=[dens_line], frameon=False, fontsize=8, loc="upper right")

    # Upper right: Ewald error histogram
    ax = axes[0, 1]
    ewald_mu, ewald_sd = mean_std(ewald)
    ax.hist(ewald, bins=40, color="#1f77b4", alpha=0.75, density=True, label="Ewald error")
    if ewald.size > 0:
        xs = np.linspace(float(np.min(ewald)), float(np.max(ewald)), 300)
        ys = (1.0 / (ewald_sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - ewald_mu) / ewald_sd) ** 2)
        ax.plot(xs, ys, color="#ff7f0e", lw=1.5, label=f"μ={ewald_mu:.3e}, σ={ewald_sd:.3e}")
    ax.set_title("NPT Ewald Error Distribution")
    ax.set_xlabel("Ewald error estimate")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False, fontsize=8)

    # Lower left: DFTB equil temp + pressure
    ax = axes[1, 0]
    dftb_temp_mu, dftb_temp_sd = mean_std(dftb_temp)
    dftb_press_mu, dftb_press_sd = mean_std(dftb_press)

    temp2_line = ax.plot(
        dftb_t,
        dftb_temp,
        color="#1f77b4",
        lw=1.5,
        label=f"T = {dftb_temp_mu:.2f} ± {dftb_temp_sd:.2f} K",
    )[0]
    ax.set_title("DFTB Equil Temperature / Pressure")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    press_line = ax2.plot(
        dftb_t,
        dftb_press,
        color="#2ca02c",
        lw=1.5,
        label=f"P = {dftb_press_mu:.2e} ± {dftb_press_sd:.2e} Pa",
    )[0]
    ax2.set_ylabel("Pressure (Pa)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.legend(handles=[temp2_line], frameon=False, fontsize=8, loc="upper left")
    ax2.legend(handles=[press_line], frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(axes[0, 0].get_ylim())

    # Lower right: RDFs
    ax = axes[1, 1]
    ax.plot(r11, g11, label="rdf 11", lw=1.6, color="#1f77b4")
    ax.plot(r14, g14, label="rdf 14", lw=1.6, color="#d62728")
    ax.set_title("RDFs (11 vs 14)")
    ax.set_xlabel("r ($\\AA$)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0.0, 7.5)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()

    out = args.out
    if out is None:
        out = (
            repo_root
            / "reports"
            / f"{args.system}_{solv_tag}_run{args.run_id}_equil_report.png"
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
