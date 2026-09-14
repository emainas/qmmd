"""Extract a portable, aligned VMD clip of PRN anti run-2 (read-only source)."""
from pathlib import Path
import argparse
import csv
import shutil

import numpy as np

from prn_anti_dih import ROOT, TIME, box_from_input, minimum_image, dihedral


def proton_hbonds(coords: np.ndarray, symbols: list[str], box: np.ndarray,
                  dh_cutoff: float, ha_cutoff: float | None, angle_cutoff: float,
                  da_cutoff: float | None = None) -> list[tuple]:
    """H11-mediated O-H...O contacts; nearest oxygen within dh_cutoff is donor.

    Returns zero-based donor/acceptor IDs, D-H and H-A distances, and D-H-A angle.
    Solute and solvent oxygens are tested equally, using minimum-image vectors.
    """
    hydrogen = 10
    oxygens = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    dh_vectors = minimum_image(coords[oxygens] - coords[hydrogen], box)
    distances = np.linalg.norm(dh_vectors, axis=1)
    nearest = int(np.argmin(distances))
    donor = oxygens[nearest]
    dh = float(distances[nearest])
    if not 0 < dh <= dh_cutoff:
        return []
    contacts = []
    for acceptor, vector, ha in zip(oxygens, dh_vectors, distances):
        if acceptor == donor or ha <= 0:
            continue
        if ha_cutoff is not None and ha > ha_cutoff:
            continue
        da = np.linalg.norm(minimum_image(coords[acceptor] - coords[donor], box))
        if da_cutoff is not None and da > da_cutoff:
            continue
        cosine = np.dot(dh_vectors[nearest], vector) / (dh * ha)
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        if angle >= angle_cutoff:
            contacts.append((donor, acceptor, dh, float(ha), angle))
    return contacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/PRN-anti/run2_flip_hbonds_vmd")
    parser.add_argument("--dh-cutoff", type=float, default=1.3, help="Maximum donor-H distance, Angstrom")
    distance = parser.add_mutually_exclusive_group()
    distance.add_argument("--da-cutoff", type=float, help="Maximum donor-acceptor distance, Angstrom (default 3.0)")
    distance.add_argument("--ha-cutoff", type=float, help="Use an H-acceptor cutoff instead, Angstrom (legacy definition)")
    parser.add_argument("--angle-cutoff", type=float, default=135, help="Minimum donor-H-acceptor angle, degrees")
    args = parser.parse_args()
    if args.da_cutoff is None and args.ha_cutoff is None:
        args.da_cutoff = 3.0
    if any(value <= 0 for value in (args.dh_cutoff, args.ha_cutoff, args.da_cutoff) if value is not None) or not 0 <= args.angle_cutoff <= 180:
        parser.error("Distance cutoffs must be positive and angle must be in [0,180]")
    source = ROOT / "systems/PRN-anti/solv_100/dftb/N1T48C1/run-2/equil"
    out = args.out.resolve()
    if out.is_relative_to(ROOT / "systems"):
        parser.error("Output must be outside simulation directories")
    if out.exists():
        raise FileExistsError(out)
    box = box_from_input(source / "dftb.inp")
    # Zero-based internal indices; these are the original MOL2 solute bonds.
    bonds = [(0, 1), (0, 5), (0, 6), (0, 7), (1, 2), (1, 8), (1, 9), (2, 3), (2, 4), (3, 10)]
    frames, rows, hbond_rows = [], [], []
    with (source / "traject").open() as handle:
        while line := handle.readline():
            count = int(line)
            comment = handle.readline()
            match = TIME.search(comment)
            if not match:
                raise ValueError("Missing frame timestamp")
            time = float(match[1]) / 1000
            if time > .5 + 1e-8:
                break
            lines = [handle.readline().split() for _ in range(count)]
            if any(len(line) < 4 for line in lines):
                break
            symbols = [line[0] for line in lines]
            coords = np.array([[float(v) for v in line[1:4]] for line in lines])
            assert count == 311
            aligned = coords.copy()
            seen = {2}
            while len(seen) < 11:
                for a, b in bonds:
                    if a in seen and b not in seen:
                        aligned[b] = aligned[a] + minimum_image(coords[b] - coords[a], box)
                        seen.add(b)
                    elif b in seen and a not in seen:
                        aligned[a] = aligned[b] + minimum_image(coords[a] - coords[b], box)
                        seen.add(a)
            # Keep each water whole in its nearest image around the carboxyl carbon.
            for oxygen in range(11, count, 3):
                assert symbols[oxygen:oxygen+3] == ["O", "H", "H"]
                aligned[oxygen] = aligned[2] + minimum_image(coords[oxygen] - coords[2], box)
                for hydrogen in (oxygen + 1, oxygen + 2):
                    delta = minimum_image(coords[hydrogen] - coords[oxygen], box)
                    if np.linalg.norm(delta) > 1.3:
                        raise ValueError("Water proton ownership changed: static movie bonds are invalid")
                    aligned[hydrogen] = aligned[oxygen] + delta
            # CG at origin, CG->O1 along +x, carbonyl oxygen in the +xy half-plane.
            x = aligned[3] - aligned[2]
            x /= np.linalg.norm(x)
            y = aligned[4] - aligned[2]
            y -= np.dot(y, x) * x
            y /= np.linalg.norm(y)
            basis = np.column_stack((x, y, np.cross(x, y)))
            aligned = (aligned - aligned[2]) @ basis
            original_phi = dihedral(coords[[4, 2, 3, 10]], box)
            # A large box makes this an ordinary nonperiodic check after alignment.
            transformed_phi = dihedral(aligned[[4, 2, 3, 10]], np.full(3, 1000.))
            assert abs((original_phi - transformed_phi + 180) % 360 - 180) < 1e-6
            oh = np.linalg.norm(aligned[10] - aligned[3])
            assert oh < 1.3
            contacts = proton_hbonds(coords, symbols, box, args.dh_cutoff, args.ha_cutoff, args.angle_cutoff, args.da_cutoff)
            for donor, acceptor, dh, ha, angle in contacts:
                endpoint = aligned[10] + minimum_image(coords[acceptor] - coords[10], box) @ basis
                np.testing.assert_allclose(np.linalg.norm(endpoint - aligned[10]), ha, atol=1e-8)
                hbond_rows.append((len(rows), time, donor + 1, 11, acceptor + 1,
                                   "solute" if acceptor < 11 else "solvent", dh, ha, angle, *endpoint,
                                   np.linalg.norm(minimum_image(coords[acceptor] - coords[donor], box))))
            rows.append((len(rows), time, original_phi % 360, oh, len(contacts)))
            frames.append((symbols, aligned))
    if not rows or rows[-1][1] < .5 - 1e-8:
        raise ValueError("Full 0–0.5 ps clip is not yet available")
    out.mkdir(parents=True)
    with (out / "flip.xyz").open("w") as handle:
        for (symbols, xyz), row in zip(frames, rows):
            handle.write(f"{len(symbols)}\nrun=2 time_ps={row[1]:.5f} aligned_for_visualization=true\n")
            for symbol, point in zip(symbols, xyz):
                handle.write(f"{symbol} {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
    with (out / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_ps", "dihedral_deg", "O1_H11_distance_A", "H11_hbond_count"])
        writer.writerows(rows)
    with (out / "hbonds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_ps", "donor_id", "hydrogen_id", "acceptor_id", "acceptor_type",
                         "DH_A", "HA_A", "DHA_deg", "acceptor_x_A", "acceptor_y_A", "acceptor_z_A", "DA_A"])
        writer.writerows(hbond_rows)
    distance_definition = (f"donor-acceptor <= {args.da_cutoff:g} A" if args.da_cutoff is not None
                           else f"H-acceptor <= {args.ha_cutoff:g} A")
    (out / "README.txt").write_text(
        "Run: vmd -e view_flip.tcl\nKeep all files in this directory together.\n"
        "PRN-anti run-2; 51 frames over 0–0.5 ps. Flip around 0.13–0.19 ps.\n"
        "Purple: H11; green dashed: H11...solvent O; cyan dashed: H11...solute O.\n"
        "Each frame shows all qualifying contacts (including multiple acceptors), or zero.\n"
        f"Criteria: donor-H <= {args.dh_cutoff:g} A, {distance_definition}, "
        f"donor-H-acceptor angle >= {args.angle_cutoff:g} degrees.\n"
        "Nearest O within donor cutoff supplies donor; all other oxygens are candidates.\n"
        "Contacts use source minimum-image geometry, before camera alignment. Atom IDs are one-based.\n"
        "Regenerate with plotting/build_prn_flip_vmd.py --out NEW_DIR to change cutoffs.\n"
        "frames.csv includes zero-contact frames; hbonds.csv lists individual contacts.\n"
        "Play: animate forward; pause: animate pause; inspect transition: animate goto 15.\n"
        "Coordinates are made whole and rotated for visualization, not simulation restart.\n"
        "No live run files changed. VMD rendering was not available on the preparation host.\n")
    shutil.copy2(ROOT / "scripts/prn_run2_flip.tcl", out / "view_flip.tcl")
    print(f"Saved {len(rows)} complete frames to {out}; source trajectory unchanged")


if __name__ == "__main__":
    main()
