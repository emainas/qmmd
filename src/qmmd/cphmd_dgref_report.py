from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

from qmmd.cphmd_dgref import DgrefConfig, dgref_dir, find_repo_root, load_config


_EXECUTION_RE = re.compile(
    r"AMBER execution #(\d+): running\s+(\d+)\s+MD steps"
    r"(?: of (equilibration|production))?\s+for DELTAGREF\s*=\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)"
)
_FRACTION_RE = re.compile(
    r"fraction of protonated species is\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)%"
)


@dataclass(frozen=True)
class DgrefSample:
    execution: int
    phase: str
    md_steps: int
    dgref_kcal_mol: float
    protonated_fraction_percent: float


def parse_dgref_log(text: str) -> list[DgrefSample]:
    samples: list[DgrefSample] = []
    pending: tuple[int, str, int, float] | None = None
    for line in text.splitlines():
        execution_match = _EXECUTION_RE.search(line)
        if execution_match is not None:
            phase = execution_match.group(3) or "search"
            pending = (
                int(execution_match.group(1)),
                phase,
                int(execution_match.group(2)),
                float(execution_match.group(4)),
            )
            continue

        fraction_match = _FRACTION_RE.search(line)
        if fraction_match is None or pending is None:
            continue
        fraction = float(fraction_match.group(1))
        if not 0.0 <= fraction <= 100.0:
            raise ValueError(f"Protonated fraction outside 0-100%: {fraction}")
        samples.append(
            DgrefSample(
                execution=pending[0],
                phase=pending[1],
                md_steps=pending[2],
                dgref_kcal_mol=pending[3],
                protonated_fraction_percent=fraction,
            )
        )
        pending = None
    return samples


def write_report_csv(path: Path, samples: list[DgrefSample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "execution",
                "phase",
                "md_steps",
                "dgref_kcal_mol",
                "protonated_fraction_percent",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.execution,
                    sample.phase,
                    sample.md_steps,
                    f"{sample.dgref_kcal_mol:.6f}",
                    f"{sample.protonated_fraction_percent:.4f}",
                ]
            )


def dgref_values_after_discard(
    samples: list[DgrefSample],
    discard_first: int,
) -> list[float]:
    if discard_first < 0:
        raise ValueError("discard_first must be non-negative")
    return [sample.dgref_kcal_mol for sample in samples[discard_first:]]


def summarize_dgref_after_discard(
    samples: list[DgrefSample],
    discard_first: int,
) -> tuple[float, float] | None:
    values = dgref_values_after_discard(samples, discard_first)
    if not values:
        return None
    standard_deviation = stdev(values) if len(values) > 1 else 0.0
    return fmean(values), standard_deviation


def plot_dgref_report(
    path: Path,
    cfg: DgrefConfig,
    samples: list[DgrefSample],
    style_path: Path,
    target_percent: float = 50.0,
    tolerance_percent: float = 5.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if style_path.is_file():
        plt.style.use(style_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    lower = target_percent - tolerance_percent
    upper = target_percent + tolerance_percent
    ax.axhspan(
        lower,
        upper,
        color="#2E9E7E",
        alpha=0.18,
        label=f"Target: {target_percent:g} ± {tolerance_percent:g}%",
        zorder=0,
    )
    ax.axhline(target_percent, color="#2E9E7E", linewidth=1.2, alpha=0.9, zorder=1)

    if samples:
        x = [sample.dgref_kcal_mol for sample in samples]
        y = [sample.protonated_fraction_percent for sample in samples]
        ax.scatter(x, y, s=38, color="#1F3A5F", edgecolors="white", linewidths=0.5,
                   label="Completed evaluations", zorder=3)
        latest = samples[-1]
        ax.scatter(
            [latest.dgref_kcal_mol],
            [latest.protonated_fraction_percent],
            s=75,
            color="#D8481A",
            marker="*",
            label=f"Latest: execution {latest.execution}",
            zorder=4,
        )
        x_span = max(x) - min(x)
        padding = max(1.0, 0.05 * x_span)
        ax.set_xlim(min(x) - padding, max(x) + padding)
    else:
        ax.text(
            0.5,
            0.5,
            "No completed ΔGref evaluations yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel(r"$\Delta G_{\mathrm{ref}}$ (kcal mol$^{-1}$)")
    ax.set_ylabel("Protonated fraction (%)")
    ax.set_title(f"{cfg.system} ΔGref calibration at pH {float(cfg.cntrl['solvph']):g}")
    ax.legend(loc="best")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_dgref_iteration_report(
    path: Path,
    cfg: DgrefConfig,
    samples: list[DgrefSample],
    style_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if style_path.is_file():
        plt.style.use(style_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if samples:
        iterations = [sample.execution for sample in samples]
        dgref = [sample.dgref_kcal_mol for sample in samples]
        ax.plot(
            iterations,
            dgref,
            color="#1F3A5F",
            linewidth=1.4,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            iterations,
            dgref,
            s=38,
            color="#1F3A5F",
            edgecolors="white",
            linewidths=0.5,
            label="Completed evaluations",
            zorder=3,
        )
        latest = samples[-1]
        ax.scatter(
            [latest.execution],
            [latest.dgref_kcal_mol],
            s=75,
            color="#D8481A",
            marker="*",
            label=f"Latest: iteration {latest.execution}",
            zorder=4,
        )
        summary = summarize_dgref_after_discard(samples, cfg.report_discard_first)
        if summary is not None:
            mean_dgref, standard_deviation = summary
            ax.axhline(
                mean_dgref,
                color="#C62828",
                linestyle="--",
                linewidth=1.5,
                label=(
                    f"After first {cfg.report_discard_first}: mean = {mean_dgref:.3f}, "
                    f"SD = {standard_deviation:.3f}"
                ),
                zorder=1,
            )
        ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    else:
        ax.text(
            0.5,
            0.5,
            "No completed ΔGref evaluations yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.set_ylabel(r"$\Delta G_{\mathrm{ref}}$ (kcal mol$^{-1}$)")
    ax.set_title(f"{cfg.system} ΔGref search at pH {float(cfg.cntrl['solvph']):g}")
    ax.legend(loc="best")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def create_dgref_report(
    cfg: DgrefConfig,
    repo_root: Path,
) -> tuple[Path, Path, Path, int]:
    output = dgref_dir(cfg, repo_root)
    if not output.is_dir():
        raise FileNotFoundError(f"Missing prepared dgref directory: {output}")
    log_path = output / "dgref.log"
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    samples = parse_dgref_log(log_text)

    csv_path = output / "dgref-report.csv"
    png_path = output / "dgref-report.png"
    iteration_png_path = output / "dgref-iteration.png"
    style_path = repo_root / "plotting" / "lefteris.mplstyle"
    write_report_csv(csv_path, samples)
    plot_dgref_report(
        png_path,
        cfg,
        samples,
        style_path,
    )
    plot_dgref_iteration_report(iteration_png_path, cfg, samples, style_path)
    return png_path, iteration_png_path, csv_path, len(samples)


def run_cphmd_dgref_report(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_config(resolved)
    png_path, iteration_png_path, csv_path, count = create_dgref_report(
        cfg,
        find_repo_root(resolved),
    )
    print(f"OK: wrote {png_path}")
    print(f"OK: wrote {iteration_png_path}")
    print(f"OK: wrote {csv_path} ({count} completed evaluations)")
