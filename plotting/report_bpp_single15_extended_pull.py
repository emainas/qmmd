"""Report the completed BPP single15 pull, including its two left extensions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qmmd.us_pull_report import angle_near_target, read_cpptraj_dihedrals, run_cpptraj


ROOT = Path(__file__).resolve().parents[1]
PULL_ROOT = ROOT / "systems/BPP/solv_4.0/us-single15"
TOPOLOGY = ROOT / "systems/BPP/solv_4.0/prep/solv.parm7"
ATOMS = (30, 35, 40, 44)
# Directory indices are deliberately ordered by coordinate for the ladder plot.
WINDOWS = [(20, -20.0), (19, -10.0), *[(index, float(10 * index)) for index in range(19)]]
FRAMES_PER_WINDOW = 10
SAMPLE_INTERVAL_PS = 0.1


def main() -> None:
    trajectories: list[Path] = []
    for index, _ in WINDOWS:
        stage = PULL_ROOT / f"window-{index:03d}" / "pull"
        output = stage / "pull.out"
        trajectory = stage / "pull.nc"
        if not trajectory.is_file() or not output.is_file():
            raise FileNotFoundError(f"Missing completed pull data in {stage}")
        if "Final Performance Info:" not in output.read_text(errors="replace"):
            raise RuntimeError(f"Pull window did not terminate normally: {output}")
        trajectories.append(trajectory)

    data_path = PULL_ROOT / "pull_dihedral_cpptraj_extended.dat"
    input_path = PULL_ROOT / "cpptraj-extended.in"
    log_path = PULL_ROOT / "cpptraj-extended.log"
    lines = [f'parm "{TOPOLOGY}"']
    lines.extend(f'trajin "{trajectory}"' for trajectory in trajectories)
    lines.extend(
        [
            "autoimage",
            "dihedral pulled " + " ".join(f"@{atom}" for atom in ATOMS)
            + f' out "{data_path}"',
            "run",
            "quit",
            "",
        ]
    )
    input_path.write_text("\n".join(lines))
    run_cpptraj(input_path, log_path, "amber/26")
    angles = read_cpptraj_dihedrals(data_path)
    expected = len(WINDOWS) * FRAMES_PER_WINDOW
    if len(angles) != expected:
        raise ValueError(f"Expected {expected} frames, found {len(angles)}")

    rows: list[tuple[float, int, float, float, int, float, float]] = []
    for ordered_index, (window_index, target) in enumerate(WINDOWS):
        for offset in range(FRAMES_PER_WINDOW):
            raw = angles[ordered_index * FRAMES_PER_WINDOW + offset]
            rows.append(
                (
                    (ordered_index * FRAMES_PER_WINDOW + offset + 1) * SAMPLE_INTERVAL_PS,
                    window_index,
                    target,
                    (offset + 1) * SAMPLE_INTERVAL_PS,
                    offset + 1,
                    raw,
                    angle_near_target(raw, target),
                )
            )

    csv_path = PULL_ROOT / "pull_dihedral.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "ordered_time_ps",
                "window_index",
                "target_deg",
                "local_time_ps",
                "frame_in_window",
                "dihedral_raw_deg",
                "dihedral_branch_deg",
            ]
        )
        writer.writerows(
            (
                f"{time:.8f}",
                index,
                f"{target:.8f}",
                f"{local_time:.8f}",
                frame,
                f"{raw:.8f}",
                f"{branch:.8f}",
            )
            for time, index, target, local_time, frame, raw, branch in rows
        )

    times = [row[0] for row in rows]
    measured = [row[-1] for row in rows]
    targets = [target for _, target in WINDOWS]
    boundaries = [float(index) for index in range(len(WINDOWS) + 1)]
    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    axis.plot(times, measured, color="#1769aa", linewidth=1.35, label="Measured dihedral")
    axis.step(
        boundaries,
        [*targets, targets[-1]],
        where="post",
        color="#d1495b",
        linewidth=1.1,
        linestyle="--",
        label="Restraint target",
    )
    axis.set(
        xlabel="Ordered pull time (ps)",
        ylabel="Dihedral (degrees)",
        title="Sequential Amber pull: 30-35-40-44",
        xlim=(0.0, float(len(WINDOWS))),
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure_path = PULL_ROOT / "pull.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {len(rows)} samples to {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
