import argparse
from pathlib import Path
from qmmd.prep import run_prep
from qmmd.mdequil import run_mdequil
from qmmd.salt import run_salt
from qmmd.dftb import run_dftb_prep, run_dftb_submit
from qmmd.ncoord import run_ncoord
from qmmd.ncoord2d import run_2dncoord
from qmmd.meta import run_meta_prep, run_meta_submit
from qmmd.wtmeta import run_wtmeta_prep, run_wtmeta_submit
from qmmd.density import run_density
from qmmd.cv_coord import run_cv_coord
from qmmd.cv_dist import run_cv_dist
from qmmd.orb import run_orb_prep, run_orb_submit
from qmmd.rdf import run_rdf
from qmmd.dihedral import run_dihedral
from qmmd.rmsd import run_rmsd
from qmmd.radgyr import run_radgyr
from qmmd.hbond import run_hbond

def main():
    p = argparse.ArgumentParser(prog="qmmd")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep", help="Prepare solvated system with tleap")
    prep.add_argument("yaml", type=Path)

    mdequil = sub.add_parser("mdequil", help="Write MD equil inputs and run (slurm if provided, else local)")
    mdequil.add_argument("yaml", type=Path)

    salt = sub.add_parser("salt", help="Delete the counterion (sodium or chlorind) and turn the furthest water into a hydroxide")
    salt.add_argument("yaml", type=Path)

    dftb_prep = sub.add_parser("dftb-prep", help="Write DCDFTBMD inputs/scripts for one or many equil runs (no submit)")
    dftb_prep.add_argument("yaml", type=Path)

    dftb_submit = sub.add_parser("dftb-submit", help="Submit DCDFTBMD jobs for runs matching the config (no writes)")
    dftb_submit.add_argument("yaml", type=Path)

    ncoord = sub.add_parser("ncoord", help="Write metacv.dat for selected runs based on dftb.inp")
    ncoord.add_argument("yaml", type=Path)

    ncoord2d = sub.add_parser("2dncoord", help="Write 2D metacv.dat for selected runs based on dftb.inp")
    ncoord2d.add_argument("yaml", type=Path)

    meta_prep = sub.add_parser("meta-prep", help="Write metadynamics inputs/scripts for selected runs (no submit)")
    meta_prep.add_argument("yaml", type=Path)

    meta_submit = sub.add_parser("meta-submit", help="Submit metadynamics jobs for runs matching the config (no writes)")
    meta_submit.add_argument("yaml", type=Path)

    wtmeta_prep = sub.add_parser("wtmeta-prep", help="Write well-tempered metadynamics inputs/scripts for selected runs (no submit)")
    wtmeta_prep.add_argument("yaml", type=Path)

    wtmeta_submit = sub.add_parser("wtmeta-submit", help="Submit well-tempered metadynamics jobs for runs matching the config (no writes)")
    wtmeta_submit.add_argument("yaml", type=Path)

    density = sub.add_parser("density", help="Compute solute/box volume from salt outputs")
    density.add_argument("yaml", type=Path)

    cv_coord = sub.add_parser("cv-coord", help="Compute rational coordination CV from XYZ trajectories")
    cv_coord.add_argument("yaml", type=Path)
    cv_coord.add_argument("--validate", action="store_true", help="Validate against biaspot Coordinate series")

    cv_dist = sub.add_parser("cv-dist", help="Compute group distance CV from XYZ trajectories")
    cv_dist.add_argument("yaml", type=Path)

    orb_prep = sub.add_parser("orb-prep", help="Write ORB equil inputs/scripts (no submit)")
    orb_prep.add_argument("yaml", type=Path)

    orb_submit = sub.add_parser("orb-submit", help="Submit ORB equil jobs for runs matching the config (no writes)")
    orb_submit.add_argument("yaml", type=Path)

    rdf = sub.add_parser("rdf", help="Compute RDF with cpptraj for selected runs")
    rdf.add_argument("yaml", type=Path)

    dihedral = sub.add_parser("dihedral", help="Compute dihedral time series with cpptraj for selected runs")
    dihedral.add_argument("yaml", type=Path)

    rmsd = sub.add_parser("rmsd", help="Compute RMSD time series with cpptraj for selected runs")
    rmsd.add_argument("yaml", type=Path)

    radgyr = sub.add_parser("radgyr", help="Compute radius of gyration time series with cpptraj for selected runs")
    radgyr.add_argument("yaml", type=Path)

    hbond = sub.add_parser("hbond", help="Compute hydrogen bond time series and lifetimes with cpptraj")
    hbond.add_argument("yaml", type=Path)

    args = p.parse_args()

    if args.cmd == "prep":
        run_prep(args.yaml)
    elif args.cmd == "mdequil":
        run_mdequil(args.yaml)
    elif args.cmd == "salt":
        run_salt(args.yaml)
    elif args.cmd == "dftb-prep":
        run_dftb_prep(args.yaml)
    elif args.cmd == "dftb-submit":
        run_dftb_submit(args.yaml)
    elif args.cmd == "ncoord":
        run_ncoord(args.yaml)
    elif args.cmd == "2dncoord":
        run_2dncoord(args.yaml)
    elif args.cmd == "meta-prep":
        run_meta_prep(args.yaml)
    elif args.cmd == "meta-submit":
        run_meta_submit(args.yaml)
    elif args.cmd == "wtmeta-prep":
        run_wtmeta_prep(args.yaml)
    elif args.cmd == "wtmeta-submit":
        run_wtmeta_submit(args.yaml)
    elif args.cmd == "density":
        run_density(args.yaml)
    elif args.cmd == "cv-coord":
        run_cv_coord(args.yaml, validate=args.validate)
    elif args.cmd == "cv-dist":
        run_cv_dist(args.yaml)
    elif args.cmd == "orb-prep":
        run_orb_prep(args.yaml)
    elif args.cmd == "orb-submit":
        run_orb_submit(args.yaml)
    elif args.cmd == "rdf":
        run_rdf(args.yaml)
    elif args.cmd == "dihedral":
        run_dihedral(args.yaml)
    elif args.cmd == "rmsd":
        run_rmsd(args.yaml)
    elif args.cmd == "radgyr":
        run_radgyr(args.yaml)
    elif args.cmd == "hbond":
        run_hbond(args.yaml)
