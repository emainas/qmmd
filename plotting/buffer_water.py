#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_dat(dat_path: Path) -> pd.DataFrame:
    rows = []
    with dat_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            system, buf, wat = line.split()[:3]
            if wat.upper() == "NA":
                continue
            rows.append((system, float(buf), int(float(wat))))

    df = pd.DataFrame(rows, columns=["system", "buffer_A", "water_residues"])
    return df.sort_values(["system", "buffer_A"]).reset_index(drop=True)


# -----------------------------
# BAR PLOT
# -----------------------------
def bar_plot(df: pd.DataFrame, reports_dir: Path):
    uniq_buffers = np.array(sorted(df["buffer_A"].unique()))
    systems = sorted(df["system"].unique())

    base_colors = ["blue", "red", "green", "orange", "purple", "black"]
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(systems)}

    wmap = {(r.system, float(r.buffer_A)): int(r.water_residues)
            for r in df.itertuples(index=False)}

    x = np.arange(len(uniq_buffers))
    nsys = len(systems)
    group_width = 0.82
    bar_w = group_width / nsys

    fig, ax = plt.subplots()

    all_waters = []

    for i, sys in enumerate(systems):
        centers = x - group_width / 2 + (i + 0.5) * bar_w

        xs, ys = [], []
        for j, buf in enumerate(uniq_buffers):
            key = (sys, float(buf))
            if key in wmap:
                xs.append(centers[j])
                ys.append(wmap[key])

        bars = ax.bar(
            xs,
            ys,
            width=bar_w * 0.95,
            color=color_map[sys],
            edgecolor="black",
            linewidth=0.6,
            label=sys,
        )

        # annotate bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 2,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        all_waters.extend(ys)

    #ticks = sorted(set(all_waters))
    #ax.set_yticks(ticks)

    ax.set_xlabel("Solvation buffer (Å)")
    ax.set_ylabel("Number of added water residues")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.1f}" for b in uniq_buffers])

    ax.legend(frameon=False)

    ax.set_ylim(0, max(all_waters) * 1.15)

    #fig.tight_layout()
    fig.savefig(reports_dir / "buffer_vs_water_bar.png", dpi=300)
    plt.close(fig)


# -----------------------------
# SCATTER PLOT
# -----------------------------
def scatter_plot(df: pd.DataFrame, reports_dir: Path):

    systems = sorted(df["system"].unique())
    base_colors = ["blue", "red", "green", "orange", "purple", "black"]
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(systems)}

    fig, ax = plt.subplots()

    for sys in systems:
        sub = df[df["system"] == sys]

        ax.scatter(
            sub["buffer_A"],
            sub["water_residues"],
            color=color_map[sys],
            s=70,
            label=sys,
        )

        ax.plot(
            sub["buffer_A"],
            sub["water_residues"],
            color=color_map[sys],
            linewidth=1.5,
        )

    ax.set_xlabel("Solvation buffer (Å)")
    ax.set_ylabel("Number of added water residues")

    ax.legend(frameon=False)

    #fig.tight_layout()
    fig.savefig(reports_dir / "buffer_vs_water_scatter.png", dpi=300)
    plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------
def main():

    script_path = Path(__file__).resolve()
    root = script_path.parents[1]

    dat_path = root / "data" / "prep" / "buffer_vs_water.dat"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    plt.style.use("prl.mplstyle")

    df = read_dat(dat_path)

    # Excel
    pivot = df.pivot_table(
        index="buffer_A",
        columns="system",
        values="water_residues"
    )

    with pd.ExcelWriter(reports_dir / "buffer_vs_water.xlsx",
                        engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        pivot.to_excel(writer, sheet_name="pivot")

    bar_plot(df, reports_dir)
    scatter_plot(df, reports_dir)

    print("✅ Wrote plots + Excel to reports/")


if __name__ == "__main__":
    main()

