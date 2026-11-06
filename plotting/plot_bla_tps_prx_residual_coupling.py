"""Plot conditional TPS--PRX coupling from paired BLA REFEP reports."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083
STATES = ("APP", "BPP", "CPP", "DPP")
COLORS = ("#3B6FB6", "#D95F59", "#4E9A73", "#8B6BB1")
METHOD = "TI charge derivative"


def read_ti_estimate(path: Path, state: str) -> tuple[float, float]:
    expected_direction = f"deprotonated-{state.lower()} -> protonated-ppp"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["method"] == METHOD]
    if len(matches) != 1:
        raise ValueError(f"Expected one {METHOD!r} row in {path}")
    row = matches[0]
    if row["direction"] != expected_direction:
        raise ValueError(
            f"Unexpected direction {row['direction']!r} in {path}; "
            f"expected {expected_direction!r}"
        )
    return (
        float(row["delta_g_kcal_mol"]),
        float(row["block_bootstrap_sd_kcal_mol"]),
    )


def collect(root: Path, temperature_k: float) -> list[dict[str, object]]:
    rtln10 = GAS_CONSTANT_KCAL_MOL_K * temperature_k * math.log(10.0)
    rows: list[dict[str, object]] = []
    for state in STATES:
        relative = Path(
            f"systems/BLA/solv_10.0/refep/tps-tails-deprot/{state.lower()}/"
            "prod-5ns/report-igb2/free-energy-summary.csv"
        )
        deprotonated_path = root / relative
        protonated_relative = Path(
            f"systems/BLA/solv_10.0/refep/tps-tails-prot/{state.lower()}/"
            "prod-5ns/report-igb2/free-energy-summary.csv"
        )
        protonated_path = root / protonated_relative
        deprotonated_dg, deprotonated_sd = read_ti_estimate(
            deprotonated_path, state
        )
        protonated_dg, protonated_sd = read_ti_estimate(protonated_path, state)
        coupling = protonated_dg - deprotonated_dg
        coupling_sd = math.hypot(deprotonated_sd, protonated_sd)
        rows.append(
            {
                "tautomer": state,
                "definition": "DG(fixed protonated PRX) - DG(fixed deprotonated PRX)",
                "deprotonated_prx_dg_kcal_mol": deprotonated_dg,
                "deprotonated_prx_bootstrap_sd_kcal_mol": deprotonated_sd,
                "protonated_prx_dg_kcal_mol": protonated_dg,
                "protonated_prx_bootstrap_sd_kcal_mol": protonated_sd,
                "delta_delta_g_kcal_mol": coupling,
                "combined_bootstrap_sd_kcal_mol": coupling_sd,
                "pka_equivalent_shift": coupling / rtln10,
                "pka_equivalent_bootstrap_sd": coupling_sd / rtln10,
                "temperature_k": temperature_k,
                "deprotonated_prx_report": str(relative),
                "protonated_prx_report": str(protonated_relative),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_coupling(
    path: Path,
    rows: list[dict[str, object]],
    style: Path,
    temperature_k: float,
) -> None:
    if style.is_file():
        plt.style.use(style)
    values = np.asarray([float(row["delta_delta_g_kcal_mol"]) for row in rows])
    errors = np.asarray(
        [float(row["combined_bootstrap_sd_kcal_mol"]) for row in rows]
    )
    rtln10 = GAS_CONSTANT_KCAL_MOL_K * temperature_k * math.log(10.0)
    fig, ax = plt.subplots(figsize=(11.2, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.04, top=0.88)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    fig.suptitle(
        "A charge-invariant linker does not eliminate TPS–PRX coupling",
        x=0.5,
        y=0.965,
        fontsize=19,
    )
    ax.text(
        0.5,
        0.98,
        r"Protonating two fixed PRX tails changes the TPS electrostatic reference mutation",
        ha="center",
        va="top",
        fontsize=11.5,
        color="0.25",
    )

    node_width = 0.275
    node_height = 0.145
    left_x = 0.025
    right_x = 0.405
    top_y = 0.625
    bottom_y = 0.205
    node_face = "#F7F7F5"
    node_edge = "#252525"
    linker_color = "#777777"

    def state_node(x: float, y: float, tps: str, prx: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                node_width,
                node_height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=node_face,
                edgecolor=node_edge,
                linewidth=1.4,
            )
        )
        ax.text(
            x + 0.052,
            y + node_height / 2,
            tps,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            x + node_width / 2,
            y + node_height / 2 + 0.014,
            "C2C–CA–CB",
            ha="center",
            va="center",
            fontsize=10.5,
            color=linker_color,
        )
        ax.text(
            x + node_width / 2,
            y + node_height / 2 - 0.028,
            r"$\Delta q=0$",
            ha="center",
            va="center",
            fontsize=10.0,
            color=linker_color,
        )
        ax.text(
            x + node_width - 0.050,
            y + node_height / 2,
            prx,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
        )
        ax.plot(
            [x + 0.085, x + 0.105],
            [y + node_height / 2, y + node_height / 2],
            color=linker_color,
            linewidth=1.4,
        )
        ax.plot(
            [x + node_width - 0.105, x + node_width - 0.083],
            [y + node_height / 2, y + node_height / 2],
            color=linker_color,
            linewidth=1.4,
        )

    state_node(left_x, top_y, "TPS: X", r"2×PRX$^{-}$")
    state_node(right_x, top_y, "TPS: PPP", r"2×PRX$^{-}$")
    state_node(left_x, bottom_y, "TPS: X", "2×PRX–H")
    state_node(right_x, bottom_y, "TPS: PPP", "2×PRX–H")

    arrow_style = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#222222")
    top_center = top_y + node_height / 2
    bottom_center = bottom_y + node_height / 2
    ax.add_patch(
        FancyArrowPatch(
            (left_x + node_width + 0.008, top_center),
            (right_x - 0.008, top_center),
            **arrow_style,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (left_x + node_width + 0.008, bottom_center),
            (right_x - 0.008, bottom_center),
            **arrow_style,
        )
    )
    for x, fixed_state in (
        (left_x + node_width / 2, "X"),
        (right_x + node_width / 2, "PPP"),
    ):
        ax.add_patch(
            FancyArrowPatch(
                (x, top_y - 0.012),
                (x, bottom_y + node_height + 0.012),
                **arrow_style,
            )
        )
        ax.text(
            x,
            0.505,
            "protonate both PRX tails\n" + f"TPS fixed as {fixed_state}",
            ha="center",
            va="center",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )

    ax.text(
        0.3525,
        0.825,
        r"$\Delta G_{\mathrm{ref}}^{\mathrm{PRX^-}}(X)$"
        r" $=$ $\Delta G_{\mathrm{mut,elec}}^{\mathrm{PRX^-}}(X)$",
        ha="center",
        va="center",
        fontsize=12.5,
    )
    ax.text(
        0.3525,
        0.145,
        r"$\Delta G_{\mathrm{ref}}^{\mathrm{PRX-H}}(X)$"
        r" $=$ $\Delta G_{\mathrm{mut,elec}}^{\mathrm{PRX-H}}(X)$",
        ha="center",
        va="center",
        fontsize=12.5,
    )
    ax.text(
        0.3525,
        0.415,
        r"$\Delta G_{\mathrm{ref}}^{\mathrm{PRX-H}}(X)$"
        r" $\ne$ "
        r"$\Delta G_{\mathrm{ref}}^{\mathrm{PRX^-}}(X)$",
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#FFF1C7",
            "edgecolor": "#B27B00",
            "linewidth": 1.1,
        },
    )

    panel_x = 0.735
    panel_y = 0.13
    panel_w = 0.245
    panel_h = 0.72
    ax.add_patch(
        FancyBboxPatch(
            (panel_x, panel_y),
            panel_w,
            panel_h,
            boxstyle="round,pad=0.014,rounding_size=0.02",
            facecolor="#FBFBFB",
            edgecolor="#333333",
            linewidth=1.3,
        )
    )
    ax.text(
        panel_x + panel_w / 2,
        panel_y + panel_h - 0.055,
        "Residual coupling",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        panel_x + panel_w / 2,
        panel_y + panel_h - 0.115,
        r"$\Delta\Delta G_{\mathrm{ref}}(X)$"
        r"$=\Delta G_{\mathrm{ref}}^{\mathrm{PRX-H}}(X)$"
        "\n"
        r"$-\Delta G_{\mathrm{ref}}^{\mathrm{PRX^-}}(X)$",
        ha="center",
        va="center",
        fontsize=10.5,
    )

    row_ys = (0.585, 0.475, 0.365, 0.255)
    for row, value, error, color, y in zip(
        rows, values, errors, COLORS, row_ys, strict=True
    ):
        ax.scatter(
            panel_x + 0.036,
            y,
            s=85,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(
            panel_x + 0.062,
            y + 0.018,
            str(row["tautomer"]),
            ha="left",
            va="center",
            fontsize=11.0,
            fontweight="bold",
        )
        ax.text(
            panel_x + 0.112,
            y + 0.018,
            rf"${value:+.2f}\pm{error:.2f}$ kcal mol$^{{-1}}$",
            ha="left",
            va="center",
            fontsize=9.5,
        )
        ax.text(
            panel_x + 0.112,
            y - 0.026,
            rf"${value / rtln10:+.2f}$ pKa-equivalent",
            ha="left",
            va="center",
            fontsize=9.0,
            color="0.30",
        )

    ax.text(
        0.3525,
        0.025,
        "The cycle closes: a nonzero cycle difference measures genuine TPS–PRX coupling.",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="0.25",
        fontstyle="italic",
    )
    fig.savefig(path, dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/BLA/tps-prx-residual-coupling"),
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--style", type=Path, default=Path("plotting/lefteris.mplstyle")
    )
    args = parser.parse_args()
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    style = args.style if args.style.is_absolute() else root / args.style
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(root, args.temperature)
    csv_path = output / "residual-coupling.csv"
    png_path = output / "residual-coupling.png"
    write_csv(csv_path, rows)
    plot_coupling(png_path, rows, style, args.temperature)
    print(f"OK: wrote {png_path}")
    print(f"OK: wrote {csv_path}")


if __name__ == "__main__":
    main()
