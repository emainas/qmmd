import argparse
from pathlib import Path
from qmmd.prep import run_prep

def main():
    p = argparse.ArgumentParser(prog="qmmd")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep", help="Prepare solvated system with tleap")
    prep.add_argument("yaml", type=Path)

    args = p.parse_args()

    if args.cmd == "prep":
        run_prep(args.yaml)
