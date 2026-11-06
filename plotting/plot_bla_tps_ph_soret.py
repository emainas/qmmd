"""Compare pH-dependent BLA Soret maxima from target and CpHMD populations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


STATE_FILES = {
    "ppp": "PKK_spectrum.csv",
    "app": "AKK_spectrum.csv",
    "bpp": "BKK_spectrum.csv",
    "cpp": "CKK_spectrum.csv",
    "dpp": "DKK_spectrum.csv",
}
STATES = tuple(STATE_FILES)


@dataclass(frozen=True)
class SpectrumSet:
    wavelength_nm: np.ndarray
    absorbance: np.ndarray


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_soret_spectra(
    directory: Path,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    intensity_column: str,
) -> SpectrumSet:
    wavelengths: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    for state in STATES:
        path = directory / STATE_FILES[state]
        rows = _read_csv(path)
        current_wavelength = np.asarray(
            [float(row["wavelength_nm"]) for row in rows], dtype=float
        )
        current_absorbance = np.asarray(
            [float(row[intensity_column]) for row in rows], dtype=float
        )
        if wavelengths is None:
            wavelengths = current_wavelength
        elif not np.allclose(current_wavelength, wavelengths, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Spectrum wavelength grid differs in {path}")
        spectra.append(current_absorbance)
    assert wavelengths is not None
    mask = (wavelengths >= wavelength_min_nm) & (
        wavelengths <= wavelength_max_nm
    )
    if mask.sum() < 3:
        raise ValueError("Soret search window contains fewer than three samples")
    return SpectrumSet(
        wavelength_nm=wavelengths[mask],
        absorbance=np.vstack(spectra)[:, mask],
    )


def load_population_curve(
    path: Path,
    *,
    sampled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    ph = np.asarray([float(row["pH"]) for row in rows], dtype=float)
    columns = {
        "ppp": "ppp_fraction",
        "app": "app_fraction_total" if sampled else "app_fraction",
        "bpp": "bpp_fraction_total" if sampled else "bpp_fraction",
        "cpp": "cpp_fraction_total" if sampled else "cpp_fraction",
        "dpp": "dpp_fraction_total" if sampled else "dpp_fraction",
    }
    fractions = np.column_stack(
        [
            np.asarray([float(row[columns[state]]) for row in rows], dtype=float)
            for state in STATES
        ]
    )
    if not np.all(np.isfinite(fractions)):
        raise ValueError(f"Non-finite population in {path}")
    if np.any(fractions < -1.0e-10):
        raise ValueError(f"Negative population in {path}")
    if not np.allclose(fractions.sum(axis=1), 1.0, atol=1.0e-7):
        raise ValueError(f"Microstate populations do not sum to one in {path}")
    return ph, fractions


def _quadratic_peak(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float]:
    if index == 0 or index == len(x) - 1:
        return float(x[index]), float(y[index])
    local_x = x[index - 1 : index + 2]
    local_y = y[index - 1 : index + 2]
    a, b, c = np.polyfit(local_x, local_y, deg=2)
    if a >= 0.0:
        return float(x[index]), float(y[index])
    vertex = float(-b / (2.0 * a))
    if vertex < float(local_x[0]) or vertex > float(local_x[-1]):
        return float(x[index]), float(y[index])
    return vertex, float(a * vertex * vertex + b * vertex + c)


def weighted_soret_maxima(
    fractions: np.ndarray,
    spectra: SpectrumSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fractions.ndim != 2 or fractions.shape[1] != spectra.absorbance.shape[0]:
        raise ValueError("Population and spectrum state dimensions do not match")
    mixtures = fractions @ spectra.absorbance
    peak_wavelengths: list[float] = []
    peak_absorbance: list[float] = []
    for mixture in mixtures:
        index = int(np.argmax(mixture))
        wavelength, absorbance = _quadratic_peak(
            spectra.wavelength_nm, mixture, index
        )
        peak_wavelengths.append(wavelength)
        peak_absorbance.append(absorbance)
    return np.asarray(peak_wavelengths), np.asarray(peak_absorbance), mixtures


def write_peak_csv(
    path: Path,
    ph: np.ndarray,
    model_wavelength: np.ndarray,
    model_absorbance: np.ndarray,
    cphmd_wavelength: np.ndarray,
    cphmd_absorbance: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pH",
                "microstate_model_lambda_S_nm",
                "microstate_model_peak_absorbance",
                "cphmd_fit_lambda_S_nm",
                "cphmd_fit_peak_absorbance",
                "cphmd_minus_model_lambda_S_nm",
            ]
        )
        for values in zip(
            ph,
            model_wavelength,
            model_absorbance,
            cphmd_wavelength,
            cphmd_absorbance,
            strict=True,
        ):
            current_ph, model_nm, model_a, cphmd_nm, cphmd_a = values
            writer.writerow(
                [
                    f"{current_ph:.8f}",
                    f"{model_nm:.8f}",
                    f"{model_a:.10f}",
                    f"{cphmd_nm:.8f}",
                    f"{cphmd_a:.10f}",
                    f"{cphmd_nm - model_nm:.8f}",
                ]
            )


def write_sampled_csv(
    path: Path,
    ph: np.ndarray,
    wavelength: np.ndarray,
    absorbance: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pH", "sampled_cphmd_lambda_S_nm", "peak_absorbance"])
        for values in zip(ph, wavelength, absorbance, strict=True):
            writer.writerow([f"{values[0]:.8f}", f"{values[1]:.8f}", f"{values[2]:.10f}"])


def write_state_spectra_csv(path: Path, spectra: SpectrumSet) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm", *(f"{state}_mean_norm" for state in STATES)])
        for index, wavelength in enumerate(spectra.wavelength_nm):
            writer.writerow(
                [
                    f"{wavelength:.10f}",
                    *(f"{spectra.absorbance[state, index]:.10f}" for state in range(len(STATES))),
                ]
            )


def plateau_maxima(
    fractions: np.ndarray,
    spectra: SpectrumSet,
) -> tuple[float, float]:
    """Return the limiting PPP and conditional-deprotonated Soret maxima."""
    low_ph = np.zeros((1, fractions.shape[1]), dtype=float)
    low_ph[0, 0] = 1.0
    conditional = fractions[0, 1:] / fractions[0, 1:].sum()
    high_ph = np.concatenate(([0.0], conditional))[None, :]
    low_nm = float(weighted_soret_maxima(low_ph, spectra)[0][0])
    high_nm = float(weighted_soret_maxima(high_ph, spectra)[0][0])
    return low_nm, high_nm


def write_plateau_summary(
    path: Path,
    summaries: list[tuple[str, float, float]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["population_model", "low_pH_plateau_nm", "high_pH_plateau_nm", "plateau_gap_delta_lambda_S_nm"]
        )
        for name, low_nm, high_nm in summaries:
            writer.writerow(
                [name, f"{low_nm:.8f}", f"{high_nm:.8f}", f"{high_nm - low_nm:.8f}"]
            )


def plot(
    path: Path,
    ph: np.ndarray,
    model_wavelength: np.ndarray,
    cphmd_wavelength: np.ndarray,
    sampled_ph: np.ndarray,
    sampled_wavelength: np.ndarray,
    style: Path,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    model_gap_nm: float,
    cphmd_gap_nm: float,
) -> None:
    if style.is_file():
        plt.style.use(style)
    fig, ax = plt.subplots(figsize=(8.2, 5.5), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.88)
    model_color = "#1F5A94"
    cphmd_color = "#C43C39"
    ax.fill_between(
        ph,
        model_wavelength,
        cphmd_wavelength,
        color="0.75",
        alpha=0.25,
        linewidth=0.0,
        label="Model–CpHMD displacement",
    )
    ax.plot(
        ph,
        model_wavelength,
        color=model_color,
        linewidth=2.4,
        label=rf"Microstate model ($\Delta\lambda_S$={model_gap_nm:.2f} nm)",
    )
    ax.plot(
        ph,
        cphmd_wavelength,
        color=cphmd_color,
        linewidth=2.4,
        label=rf"CpHMD population fit ($\Delta\lambda_S$={cphmd_gap_nm:.2f} nm)",
    )
    ax.scatter(
        sampled_ph,
        sampled_wavelength,
        color=cphmd_color,
        edgecolor="white",
        linewidth=0.8,
        s=43,
        zorder=4,
        label="Sampled CpHMD windows",
    )
    ax.set(
        xlabel="pH",
        ylabel=r"Soret maximum, $\lambda_{S}$ (nm)",
        xlim=(float(ph.min()), float(ph.max())),
    )
    ax.set_title("BLA TPS pH-dependent Soret maximum", fontsize=15)
    ax.legend(frameon=False, loc="best", fontsize=9.5)
    fig.text(
        0.5,
        0.025,
        (
            "Population-weighted mean_norm spectra; quadratic refinement of the "
            f"maximum within {wavelength_min_nm:g}–{wavelength_max_nm:g} nm (Q band excluded)."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.3",
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--soret-min-nm", type=float, default=280.0)
    parser.add_argument("--soret-max-nm", type=float, default=380.0)
    args = parser.parse_args()
    root = args.root.resolve()
    report = (
        root
        / "systems/BLA/solv_10.0/cphmd/titr-tps-ti-tails-deprot-1ns/post/report"
    )
    output = root / "reports/BLA/tps-microstate-model"
    output.mkdir(parents=True, exist_ok=True)

    spectra = load_soret_spectra(
        root / "kai/AAA_PLOTS",
        args.soret_min_nm,
        args.soret_max_nm,
        "mean_norm",
    )
    model_ph, model_fractions = load_population_curve(
        report / "tps-microstate-model.csv", sampled=False
    )
    cphmd_ph, cphmd_fractions = load_population_curve(
        report / "tps-titration-curves.csv", sampled=False
    )
    if not np.allclose(model_ph, cphmd_ph, rtol=0.0, atol=1.0e-10):
        raise ValueError("Configured-model and CpHMD-fit pH grids differ")
    sampled_ph, sampled_fractions = load_population_curve(
        report / "tps-titration-tautomers.csv", sampled=True
    )

    model_nm, model_absorbance, _model_mixtures = weighted_soret_maxima(
        model_fractions, spectra
    )
    cphmd_nm, cphmd_absorbance, _cphmd_mixtures = weighted_soret_maxima(
        cphmd_fractions, spectra
    )
    sampled_nm, sampled_absorbance, _sampled_mixtures = weighted_soret_maxima(
        sampled_fractions, spectra
    )
    model_low_nm, model_high_nm = plateau_maxima(model_fractions, spectra)
    cphmd_low_nm, cphmd_high_nm = plateau_maxima(cphmd_fractions, spectra)

    write_peak_csv(
        output / "lambda_vs_pH.csv",
        model_ph,
        model_nm,
        model_absorbance,
        cphmd_nm,
        cphmd_absorbance,
    )
    write_sampled_csv(
        output / "lambda_vs_pH_sampled.csv",
        sampled_ph,
        sampled_nm,
        sampled_absorbance,
    )
    write_state_spectra_csv(output / "soret-state-spectra.csv", spectra)
    write_plateau_summary(
        output / "soret-plateau-summary.csv",
        [
            ("microstate_model", model_low_nm, model_high_nm),
            ("cphmd_population_fit", cphmd_low_nm, cphmd_high_nm),
        ],
    )
    plot(
        output / "lambda_vs_pH.png",
        model_ph,
        model_nm,
        cphmd_nm,
        sampled_ph,
        sampled_nm,
        root / "plotting/lefteris.mplstyle",
        args.soret_min_nm,
        args.soret_max_nm,
        model_high_nm - model_low_nm,
        cphmd_high_nm - cphmd_low_nm,
    )
    (output / "lambda_vs_pH-provenance.yaml").write_text(
        yaml.safe_dump(
            {
                "state_spectrum_mapping": STATE_FILES,
                "spectrum_intensity_column": "mean_norm",
                "soret_search_window_nm": [
                    args.soret_min_nm,
                    args.soret_max_nm,
                ],
                "peak_method": "three-point local quadratic interpolation around the grid maximum",
                "microstate_model_populations": str(
                    (report / "tps-microstate-model.csv").relative_to(root)
                ),
                "cphmd_fit_populations": str(
                    (report / "tps-titration-curves.csv").relative_to(root)
                ),
                "cphmd_sampled_populations": str(
                    (report / "tps-titration-tautomers.csv").relative_to(root)
                ),
                "figure": "lambda_vs_pH.png",
                "plotted_data": "lambda_vs_pH.csv",
                "sampled_points": "lambda_vs_pH_sampled.csv",
                "plateau_summary": "soret-plateau-summary.csv",
            },
            sort_keys=False,
        )
    )
    print(f"OK: wrote TPS pH-dependent Soret analysis to {output}")


if __name__ == "__main__":
    main()
