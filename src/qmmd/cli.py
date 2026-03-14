import argparse
from pathlib import Path
from qmmd.prep import run_prep
from qmmd.mdequil import run_mdequil
from qmmd.salt import run_salt
from qmmd.dftb import run_dftb_prep, run_dftb_submit
from qmmd.ncoord import run_ncoord
from qmmd.meta import run_meta_prep, run_meta_submit
from qmmd.density import run_density

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

    meta_prep = sub.add_parser("meta-prep", help="Write metadynamics inputs/scripts for selected runs (no submit)")
    meta_prep.add_argument("yaml", type=Path)

    meta_submit = sub.add_parser("meta-submit", help="Submit metadynamics jobs for runs matching the config (no writes)")
    meta_submit.add_argument("yaml", type=Path)

    density = sub.add_parser("density", help="Compute solute/box volume from salt outputs")
    density.add_argument("yaml", type=Path)

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
    elif args.cmd == "meta-prep":
        run_meta_prep(args.yaml)
    elif args.cmd == "meta-submit":
        run_meta_submit(args.yaml)
    elif args.cmd == "density":
        run_density(args.yaml)
