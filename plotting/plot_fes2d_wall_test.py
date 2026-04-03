#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FES_PATH = Path("systems/HIST/solv_5.5/dftb/N1T64C1/run-11/wall-test/fes.dat")
OUT_PATH = Path("reports/fes2d_wall_test.png")


def load_last_fes_block(path: Path):
    blocks = []
    current = []
    with path.open() as f:
        for line in f:
            if line.startswith(" ### FREE ENERGY SURFACE"):
                if current:
                    blocks.append(current)
                    current = []
                continue
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            current.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"No FES blocks found in {path}")
    data = np.array(blocks[-1], dtype=float)
    x = data[:, 0]
    y = data[:, 1]
    f = data[:, 2]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = len(x_unique)
    ny = len(y_unique)
    z = f.reshape(ny, nx)
    return x_unique, y_unique, z


def local_minima(z: np.ndarray, nmax: int = 10):
    mins = []
    ny, nx = z.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            val = z[i, j]
            nbrs = z[i - 1 : i + 2, j - 1 : j + 2]
            if np.all(val <= nbrs):
                if np.any(val < nbrs):
                    mins.append((i, j, val))
    mins.sort(key=lambda t: t[2])
    return mins[:nmax]


def main() -> None:
    x, y, z = load_last_fes_block(FES_PATH)
    z = z - np.nanmin(z)

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    levels = 30
    cf = ax.contourf(X, Y, z, levels=levels, cmap="viridis")
    ax.contour(X, Y, z, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("ΔF (shifted)")

    mins = local_minima(z, nmax=8)
    for i, j, val in mins:
        ax.plot(x[j], y[i], "r.")
        ax.annotate(f"{val:.2f}", (x[j], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("CV1")
    ax.set_ylabel("CV2")
    ax.set_title("2D FES (wall-test)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)


if __name__ == "__main__":
    main()
