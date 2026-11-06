"""Digitize the Rockwell BV spectra figure and compare it with BLA TPS."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class PanelCalibration:
    name: str
    x_left_px: float
    x_right_px: float
    y_top_px: float
    y_bottom_px: float
    ph_min: float = 3.0
    ph_max: float = 12.0
    wavelength_min_nm: float = 350.0
    wavelength_max_nm: float = 390.0


@dataclass(frozen=True)
class SpectralFit:
    low_nm: float
    high_nm: float
    pka: float


PANELS = (
    PanelCalibration("BV (diacid)", 124.5, 804.5, 22.5, 508.5),
    PanelCalibration("BV-DA (diamide)", 124.5, 804.5, 605.5, 1091.5),
)


def _cluster_centers(candidate: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=int))
    centers = ndimage.center_of_mass(candidate, labels, range(1, count + 1))
    return np.asarray([(x, y) for y, x in centers], dtype=float)


def detect_marker_centers(image: np.ndarray) -> np.ndarray:
    """Return centers of the filled blue markers, excluding the thin fit line."""
    rgb = np.asarray(image[..., :3], dtype=float)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    red, green, blue = np.moveaxis(rgb, -1, 0)
    blue_mask = (
        (blue > 0.45)
        & (blue > 1.40 * red)
        & (blue > 1.15 * green)
        & (red < 0.40)
    )
    distance = ndimage.distance_transform_edt(blue_mask)
    local_maximum = distance == ndimage.maximum_filter(distance, size=13)
    centers = _cluster_centers(local_maximum & (distance > 7.0))
    if centers.size == 0:
        raise ValueError("No filled blue markers were detected")
    return centers


def calibrate_panel(
    centers_px: np.ndarray,
    panel: PanelCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = centers_px[:, 0]
    y = centers_px[:, 1]
    inside = (
        (x >= panel.x_left_px)
        & (x <= panel.x_right_px)
        & (y >= panel.y_top_px)
        & (y <= panel.y_bottom_px)
    )
    selected = centers_px[inside]
    ph = panel.ph_min + (selected[:, 0] - panel.x_left_px) * (
        (panel.ph_max - panel.ph_min) / (panel.x_right_px - panel.x_left_px)
    )
    wavelength = panel.wavelength_max_nm - (
        selected[:, 1] - panel.y_top_px
    ) * (
        (panel.wavelength_max_nm - panel.wavelength_min_nm)
        / (panel.y_bottom_px - panel.y_top_px)
    )
    order = np.argsort(ph)
    return ph[order], wavelength[order], selected[order]


def hh_wavelength(
    ph: np.ndarray,
    low_nm: float,
    high_nm: float,
    pka: float,
) -> np.ndarray:
    return low_nm + (high_nm - low_nm) / (1.0 + np.power(10.0, pka - ph))


def fit_spectral_transition(ph: np.ndarray, wavelength: np.ndarray) -> SpectralFit:
    parameters, _covariance = curve_fit(
        hh_wavelength,
        ph,
        wavelength,
        p0=(float(wavelength.min()), float(wavelength.max()), 5.7),
        bounds=((340.0, 360.0, 3.0), (370.0, 390.0, 9.0)),
        maxfev=20000,
    )
    return SpectralFit(*(float(value) for value in parameters))


def read_qmmd_curves(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ph = np.asarray([float(row["pH"]) for row in rows])
    model = np.asarray([float(row["microstate_model_lambda_S_nm"]) for row in rows])
    cphmd = np.asarray([float(row["cphmd_fit_lambda_S_nm"]) for row in rows])
    return ph, model, cphmd


def read_qmmd_plateau_gaps(path: Path) -> tuple[float, float]:
    with path.open(newline="") as handle:
        rows = {row["population_model"]: row for row in csv.DictReader(handle)}
    return tuple(
        float(rows[name]["plateau_gap_delta_lambda_S_nm"])
        for name in ("microstate_model", "cphmd_population_fit")
    )


def read_qmmd_sampled(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ph = np.asarray([float(row["pH"]) for row in rows])
    wavelength = np.asarray(
        [float(row["sampled_cphmd_lambda_S_nm"]) for row in rows]
    )
    return ph, wavelength


def write_points(
    path: Path,
    datasets: list[tuple[PanelCalibration, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["series", "point", "pH", "lambda_max_nm", "source_x_px", "source_y_px"]
        )
        for panel, ph, wavelength, pixels in datasets:
            for index, (current_ph, current_nm, pixel) in enumerate(
                zip(ph, wavelength, pixels, strict=True), start=1
            ):
                writer.writerow(
                    [
                        panel.name,
                        index,
                        f"{current_ph:.6f}",
                        f"{current_nm:.6f}",
                        f"{pixel[0]:.3f}",
                        f"{pixel[1]:.3f}",
                    ]
                )


def write_fits(
    path: Path,
    dense_ph: np.ndarray,
    fits: list[tuple[PanelCalibration, SpectralFit]],
) -> None:
    curves = [
        hh_wavelength(dense_ph, fit.low_nm, fit.high_nm, fit.pka)
        for _panel, fit in fits
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pH", *(f"{panel.name}_lambda_max_nm" for panel, _fit in fits)])
        for index, current_ph in enumerate(dense_ph):
            writer.writerow(
                [f"{current_ph:.6f}", *(f"{curve[index]:.6f}" for curve in curves)]
            )


def write_fit_summary(
    path: Path,
    datasets: list[tuple[PanelCalibration, np.ndarray, np.ndarray, np.ndarray]],
    fits: list[tuple[PanelCalibration, SpectralFit]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "digitized_points",
                "pKa",
                "low_pH_plateau_nm",
                "high_pH_plateau_nm",
                "plateau_gap_delta_lambda_max_nm",
            ]
        )
        for dataset, fit_entry in zip(datasets, fits, strict=True):
            panel, ph, _wavelength, _pixels = dataset
            _fit_panel, fit = fit_entry
            writer.writerow(
                [
                    panel.name,
                    len(ph),
                    f"{fit.pka:.6f}",
                    f"{fit.low_nm:.6f}",
                    f"{fit.high_nm:.6f}",
                    f"{fit.high_nm - fit.low_nm:.6f}",
                ]
            )


def plot_digitized(
    path: Path,
    datasets: list[tuple[PanelCalibration, np.ndarray, np.ndarray, np.ndarray]],
    fits: list[tuple[PanelCalibration, SpectralFit]],
) -> None:
    colors = ("#1F5A94", "#7B4BA1")
    dense_ph = np.linspace(3.0, 12.0, 901)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.8), sharex=True, sharey=True)
    for ax, dataset, fit_entry, color, label in zip(
        axes, datasets, fits, colors, ("A", "B"), strict=True
    ):
        panel, ph, wavelength, _pixels = dataset
        _fit_panel, fit = fit_entry
        ax.scatter(
            ph,
            wavelength,
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label="Digitized points",
        )
        ax.plot(
            dense_ph,
            hh_wavelength(dense_ph, fit.low_nm, fit.high_nm, fit.pka),
            color=color,
            linewidth=2.1,
            label=(
                rf"HH fit: pK$_a$={fit.pka:.2f}; "
                rf"$\Delta\lambda_{{\max}}$={fit.high_nm - fit.low_nm:.2f} nm"
            ),
        )
        ax.text(0.025, 0.93, label, transform=ax.transAxes, weight="bold", fontsize=16)
        ax.set_title(panel.name)
        ax.set_ylabel(r"$\lambda_{\max}$ (nm)")
        ax.set_xlim(3.0, 12.0)
        ax.set_ylim(350.0, 390.0)
        ax.legend(frameon=False, loc="lower right")
    axes[-1].set_xlabel("pH")
    fig.suptitle("Rockwell et al. (2010): digitized BV spectral transitions")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_comparison(
    path: Path,
    qmmd_ph: np.ndarray,
    model_nm: np.ndarray,
    cphmd_nm: np.ndarray,
    datasets: list[tuple[PanelCalibration, np.ndarray, np.ndarray, np.ndarray]],
    fits: list[tuple[PanelCalibration, SpectralFit]],
    model_gap_nm: float,
    cphmd_gap_nm: float,
) -> None:
    experimental_colors = ("#2A8C6F", "#8A5AA6")
    dense_ph = np.linspace(3.0, 12.0, 901)
    fig, (tps_ax, rockwell_ax) = plt.subplots(1, 2, figsize=(11.5, 5.1))
    tps_ax.plot(
        qmmd_ph,
        model_nm,
        color="#1F5A94",
        linewidth=2.5,
        label=rf"TPS target model ($\Delta\lambda_S$={model_gap_nm:.2f} nm)",
    )
    tps_ax.plot(
        qmmd_ph,
        cphmd_nm,
        color="#C43C39",
        linewidth=2.5,
        label=rf"TPS CpHMD fit ($\Delta\lambda_S$={cphmd_gap_nm:.2f} nm)",
    )
    tps_ax.set(
        xlabel="pH",
        ylabel=r"Soret maximum, $\lambda_S$ (nm)",
        xlim=(2.5, 7.0),
        ylim=(318.45, 321.75),
        title="TPS calculated spectra",
    )
    tps_ax.legend(frameon=False, fontsize=9.0, loc="upper left")

    for dataset, fit_entry, color in zip(
        datasets, fits, experimental_colors, strict=True
    ):
        panel, ph, wavelength, _pixels = dataset
        _fit_panel, fit = fit_entry
        rockwell_ax.scatter(
            ph,
            wavelength,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=f"{panel.name} points",
        )
        rockwell_ax.plot(
            dense_ph,
            hh_wavelength(dense_ph, fit.low_nm, fit.high_nm, fit.pka),
            color=color,
            linewidth=1.9,
            linestyle="--",
            label=(
                rf"{panel.name} fit "
                rf"(pK$_a$={fit.pka:.2f}, $\Delta\lambda_{{\max}}$="
                rf"{fit.high_nm - fit.low_nm:.2f} nm)"
            ),
        )
    rockwell_ax.set(
        xlabel="pH",
        ylabel=r"Soret maximum, $\lambda_{\max}$ (nm)",
        xlim=(3.0, 12.0),
        ylim=(350.0, 382.0),
        title="Rockwell et al. (2010)",
    )
    rockwell_ax.legend(
        frameon=False,
        fontsize=8.2,
        loc="lower right",
    )
    fig.suptitle("pH-dependent Soret shifts: calculation and experiment")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_shifted_comparison_data(
    path: Path,
    cphmd_ph: np.ndarray,
    cphmd_nm: np.ndarray,
    sampled_ph: np.ndarray,
    sampled_nm: np.ndarray,
    diamide_ph: np.ndarray,
    diamide_nm: np.ndarray,
    dense_ph: np.ndarray,
    diamide_fit_nm: np.ndarray,
    shift_nm: float,
) -> None:
    series = (
        ("cphmd_population_fit_shifted", cphmd_ph, cphmd_nm + shift_nm),
        ("cphmd_sampled_shifted", sampled_ph, sampled_nm + shift_nm),
        ("rockwell_diamide_points", diamide_ph, diamide_nm),
        ("rockwell_diamide_fit", dense_ph, diamide_fit_nm),
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "pH", "lambda_max_nm", "cphmd_vertical_shift_nm"])
        for name, ph, wavelength in series:
            for current_ph, current_nm in zip(ph, wavelength, strict=True):
                writer.writerow(
                    [
                        name,
                        f"{current_ph:.8f}",
                        f"{current_nm:.8f}",
                        f"{shift_nm:.8f}" if name.startswith("cphmd") else "0.00000000",
                    ]
                )


def plot_shifted_diamide_comparison(
    path: Path,
    cphmd_ph: np.ndarray,
    cphmd_nm: np.ndarray,
    sampled_ph: np.ndarray,
    sampled_nm: np.ndarray,
    diamide_ph: np.ndarray,
    diamide_nm: np.ndarray,
    diamide_fit: SpectralFit,
    shift_nm: float,
    target_nm: float,
    cphmd_gap_nm: float,
) -> None:
    dense_ph = np.linspace(3.0, 12.0, 901)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    ax.scatter(
        diamide_ph,
        diamide_nm,
        s=48,
        color="#7B4BA1",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Rockwell BV-DA (diamide) points",
    )
    ax.plot(
        dense_ph,
        hh_wavelength(
            dense_ph,
            diamide_fit.low_nm,
            diamide_fit.high_nm,
            diamide_fit.pka,
        ),
        color="#7B4BA1",
        linewidth=2.2,
        linestyle="--",
        label=(
            rf"Diamide fit (pK$_a$={diamide_fit.pka:.2f}, "
            rf"$\Delta\lambda_{{\max}}$={diamide_fit.high_nm - diamide_fit.low_nm:.2f} nm)"
        ),
    )
    ax.plot(
        cphmd_ph,
        cphmd_nm + shift_nm,
        color="#C43C39",
        linewidth=2.5,
        label=rf"Shifted CpHMD fit ($\Delta\lambda_S$={cphmd_gap_nm:.2f} nm)",
    )
    ax.scatter(
        sampled_ph,
        sampled_nm + shift_nm,
        s=43,
        color="#C43C39",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="Shifted sampled CpHMD populations",
    )
    ax.axhline(target_nm, color="0.45", linewidth=1.0, linestyle=":")
    ax.text(
        0.02,
        0.035,
        rf"CpHMD rigid offset: +{shift_nm:.2f} nm; low-pH plateau $\rightarrow$ {target_nm:.0f} nm",
        transform=ax.transAxes,
        fontsize=9,
        color="0.3",
    )
    ax.set(
        xlabel="pH",
        ylabel=r"Soret maximum, $\lambda_{\max}$ (nm)",
        xlim=(2.5, 12.0),
        ylim=(350.0, 381.5),
        title="CpHMD Soret shift compared with BV-DA (diamide)",
    )
    ax.legend(frameon=False, fontsize=9.0, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cphmd-shift-target-nm", type=float, default=354.0)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "kai/AAA_PLOTS/nathan_spectra.png"
    output = root / "reports/BLA/tps-microstate-model"
    output.mkdir(parents=True, exist_ok=True)

    centers = detect_marker_centers(mpimg.imread(source))
    datasets = [
        (panel, *calibrate_panel(centers, panel))
        for panel in PANELS
    ]
    expected_counts = (15, 16)
    counts = tuple(len(dataset[1]) for dataset in datasets)
    if counts != expected_counts:
        raise ValueError(f"Expected marker counts {expected_counts}, detected {counts}")
    fits = [
        (panel, fit_spectral_transition(ph, wavelength))
        for panel, ph, wavelength, _pixels in datasets
    ]

    dense_ph = np.linspace(3.0, 12.0, 901)
    write_points(output / "rockwell_2010_digitized.csv", datasets)
    write_fits(output / "rockwell_2010_fits.csv", dense_ph, fits)
    write_fit_summary(output / "rockwell_2010_fit_summary.csv", datasets, fits)
    plot_digitized(output / "rockwell_2010_digitized.png", datasets, fits)
    qmmd_ph, model_nm, cphmd_nm = read_qmmd_curves(output / "lambda_vs_pH.csv")
    model_gap_nm, cphmd_gap_nm = read_qmmd_plateau_gaps(
        output / "soret-plateau-summary.csv"
    )
    plot_comparison(
        output / "comparing_spectra.png",
        qmmd_ph,
        model_nm,
        cphmd_nm,
        datasets,
        fits,
        model_gap_nm,
        cphmd_gap_nm,
    )
    sampled_ph, sampled_nm = read_qmmd_sampled(output / "lambda_vs_pH_sampled.csv")
    with (output / "soret-plateau-summary.csv").open(newline="") as handle:
        plateau_rows = {row["population_model"]: row for row in csv.DictReader(handle)}
    cphmd_low_nm = float(
        plateau_rows["cphmd_population_fit"]["low_pH_plateau_nm"]
    )
    shift_nm = args.cphmd_shift_target_nm - cphmd_low_nm
    diamide_index = next(
        index for index, panel in enumerate(PANELS) if "diamide" in panel.name
    )
    _panel, diamide_ph, diamide_nm, _pixels = datasets[diamide_index]
    _fit_panel, diamide_fit = fits[diamide_index]
    dense_ph = np.linspace(3.0, 12.0, 901)
    diamide_fit_nm = hh_wavelength(
        dense_ph,
        diamide_fit.low_nm,
        diamide_fit.high_nm,
        diamide_fit.pka,
    )
    write_shifted_comparison_data(
        output / "comparing_diamide_cphmd_shifted.csv",
        qmmd_ph,
        cphmd_nm,
        sampled_ph,
        sampled_nm,
        diamide_ph,
        diamide_nm,
        dense_ph,
        diamide_fit_nm,
        shift_nm,
    )
    plot_shifted_diamide_comparison(
        output / "comparing_diamide_cphmd_shifted.png",
        qmmd_ph,
        cphmd_nm,
        sampled_ph,
        sampled_nm,
        diamide_ph,
        diamide_nm,
        diamide_fit,
        shift_nm,
        args.cphmd_shift_target_nm,
        cphmd_gap_nm,
    )

    (output / "rockwell_2010_digitization.yaml").write_text(
        yaml.safe_dump(
            {
                "source": str(source.relative_to(root)),
                "method": "blue-marker segmentation, distance-transform centers, linear axis calibration",
                "axis_calibration": [panel.__dict__ for panel in PANELS],
                "marker_counts": dict(zip((panel.name for panel in PANELS), counts, strict=True)),
                "fit_model": "low + (high - low) / (1 + 10**(pKa - pH))",
                "fits": {
                    panel.name: fit.__dict__ for panel, fit in fits
                },
                "digitized_points": "rockwell_2010_digitized.csv",
                "fitted_curves": "rockwell_2010_fits.csv",
                "fit_summary": "rockwell_2010_fit_summary.csv",
                "digitized_figure": "rockwell_2010_digitized.png",
                "comparison_figure": "comparing_spectra.png",
                "shifted_diamide_comparison": {
                    "figure": "comparing_diamide_cphmd_shifted.png",
                    "data": "comparing_diamide_cphmd_shifted.csv",
                    "cphmd_low_plateau_target_nm": args.cphmd_shift_target_nm,
                    "cphmd_vertical_shift_nm": shift_nm,
                    "note": "Rigid wavelength offset only; no wavelength rescaling",
                },
            },
            sort_keys=False,
        )
    )
    print(f"OK: digitized Rockwell markers and wrote comparison to {output}")
    for panel, fit in fits:
        print(
            f"  {panel.name}: n={counts[PANELS.index(panel)]}, "
            f"pKa={fit.pka:.3f}, low={fit.low_nm:.3f} nm, "
            f"high={fit.high_nm:.3f} nm, "
            f"delta_lambda={fit.high_nm - fit.low_nm:.3f} nm"
        )


if __name__ == "__main__":
    main()
