"""Report live dihedral traces from restrained DCDFTBMD equilibration windows."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from qmmd.us_equil import USEquilConfig, load_config
from qmmd.us_pull import find_repo_root, pull_dir, window_centers
from qmmd.us_pull_report import angle_near_target


_TIME_RE = re.compile(
    r"AT\s+T=\s*([-+0-9.EeDd]+)\s+FSEC.*STEP\s+NO\.=\s*(\d+)",
    re.IGNORECASE,
)
_COORDINATE_RE = re.compile(
    r"Coordinate\s*=\s*([-+0-9.EeDd]+)\s+Degree", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class EquilSample:
    window_index: int
    target_deg: float
    time_ps: float
    step: int
    dihedral_raw_deg: float
    dihedral_branch_deg: float


def read_biaspot_samples(
    path: Path, window_index: int, target_deg: float
) -> list[EquilSample]:
    """Read complete CV records from a possibly growing DCDFTBMD biaspot file."""
    samples: list[EquilSample] = []
    current_time_fs: float | None = None
    current_step: int | None = None
    # Reading once gives a finite snapshot while DCDFTBMD may continue appending.
    for line in path.read_text(errors="replace").splitlines():
        time_match = _TIME_RE.search(line)
        if time_match:
            current_time_fs = float(time_match.group(1).replace("D", "E").replace("d", "e"))
            current_step = int(time_match.group(2))
            continue
        coordinate_match = _COORDINATE_RE.search(line)
        if not coordinate_match or current_time_fs is None or current_step is None:
            continue
        angle = float(coordinate_match.group(1).replace("D", "E").replace("d", "e"))
        if not math.isfinite(angle) or not math.isfinite(current_time_fs):
            raise ValueError(f"Non-finite CV record in {path}")
        samples.append(
            EquilSample(
                window_index=window_index,
                target_deg=target_deg,
                time_ps=current_time_fs / 1000.0,
                step=current_step,
                dihedral_raw_deg=angle,
                dihedral_branch_deg=angle_near_target(angle, target_deg),
            )
        )
        current_time_fs = None
        current_step = None
    return samples


def collect_equil_samples(
    cfg: USEquilConfig, yaml_text: str, repo_root: Path
) -> tuple[list[EquilSample], list[int]]:
    """Collect currently available samples and report windows without usable data."""
    root = pull_dir(cfg.pull, repo_root)
    samples: list[EquilSample] = []
    missing: list[int] = []
    for index, center in enumerate(window_centers(cfg.pull.windows)):
        stage = root / f"window-{index:03d}" / cfg.stage_dirname
        spec = stage / "equil_spec.yaml"
        if not spec.is_file() or spec.read_text() != yaml_text:
            missing.append(index)
            continue
        biaspot = stage / "biaspot"
        if not biaspot.is_file() or biaspot.stat().st_size == 0:
            missing.append(index)
            continue
        window_samples = read_biaspot_samples(biaspot, index, center)
        if window_samples:
            samples.extend(window_samples)
        else:
            missing.append(index)
    if not samples:
        raise RuntimeError(f"No complete dihedral samples found under {root}")
    return samples, missing


def write_samples_csv(path: Path, samples: list[EquilSample]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "window_index",
                "target_deg",
                "time_ps",
                "step",
                "dihedral_raw_deg",
                "dihedral_branch_deg",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.window_index,
                    f"{sample.target_deg:.8f}",
                    f"{sample.time_ps:.8f}",
                    sample.step,
                    f"{sample.dihedral_raw_deg:.8f}",
                    f"{sample.dihedral_branch_deg:.8f}",
                ]
            )


def plot_samples(
    path: Path,
    samples: list[EquilSample],
    atoms: tuple[int, int, int, int],
    total_windows: int,
    *,
    stage_label: str = "Equilibration",
) -> None:
    # Plotting stays optional for preparation/submission commands.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[int, list[EquilSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.window_index, []).append(sample)

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    colors = plt.get_cmap("turbo")
    for color_index, (window_index, trace) in enumerate(sorted(grouped.items())):
        color = colors(color_index / max(1, len(grouped) - 1))
        target = trace[0].target_deg
        ax.plot(
            [sample.time_ps for sample in trace],
            [sample.dihedral_branch_deg for sample in trace],
            color=color,
            linewidth=1.25,
            marker=".",
            markersize=2.5,
            label=f"w{window_index:03d}: {target:g}°",
        )
        ax.hlines(
            target,
            0.0,
            trace[-1].time_ps,
            color=color,
            linewidth=0.75,
            linestyle="--",
            alpha=0.65,
        )

    ax.set_xlabel(f"{stage_label} time within each window (ps)")
    ax.set_ylabel("Dihedral (degrees; branch nearest target)")
    ax.set_title(
        f"Restrained DCDFTBMD {stage_label.lower()}: "
        + "-".join(str(atom) for atom in atoms)
        + f" ({len(grouped)}/{total_windows} windows with data)"
    )
    ax.set_xlim(left=0.0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="best")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def create_us_equil_report(
    cfg: USEquilConfig, yaml_text: str, repo_root: Path
) -> tuple[Path, Path, int, list[int]]:
    samples, missing = collect_equil_samples(cfg, yaml_text, repo_root)
    destination = pull_dir(cfg.pull, repo_root)
    csv_path = destination / "equil_dihedral.csv"
    figure_path = destination / "equil-report.png"
    write_samples_csv(csv_path, samples)
    plot_samples(
        figure_path,
        samples,
        cfg.pull.restraint.atoms,
        len(window_centers(cfg.pull.windows)),
    )
    (destination / "equil_report_spec.yaml").write_text(yaml_text)
    return figure_path, csv_path, len(samples), missing


def run_us_equil_report(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    yaml_text = resolved.read_text()
    cfg = load_config(resolved)
    figure, data, count, missing = create_us_equil_report(
        cfg, yaml_text, find_repo_root(resolved)
    )
    print(f"OK: wrote {count} live equilibration dihedral samples to {data}")
    print(f"OK: wrote US equilibration plot to {figure}")
    if missing:
        names = ", ".join(f"window-{index:03d}" for index in missing)
        print(f"NOTE: no usable samples yet for {names}")
