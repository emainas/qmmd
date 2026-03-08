import argparse
from pathlib import Path
from qmmd.prep import run_prep
from qmmd.mdequil import run_mdequil
from qmmd.salt import run_salt
from qmmd.dftb import run_dftb_prep, run_dftb_submit

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
