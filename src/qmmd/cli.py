import argparse
from pathlib import Path
from qmmd.prep import run_prep
from qmmd.mdequil import run_mdequil

def main():
    p = argparse.ArgumentParser(prog="qmmd")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep", help="Prepare solvated system with tleap")
    prep.add_argument("yaml", type=Path)

    mdequil = sub.add_parser("mdequil", help="Write MD equil inputs and run (slurm if provided, else local)")
    mdequil.add_argument("yaml", type=Path)

    args = p.parse_args()

    if args.cmd == "prep":
        run_prep(args.yaml)
    elif args.cmd == "mdequil":
        run_mdequil(args.yaml)
