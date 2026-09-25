import argparse
from pathlib import Path
from qmmd.prep import run_prep
from qmmd.mdequil import run_mdequil
from qmmd.salt import run_salt
from qmmd.dftb import run_dftb_prep, run_dftb_submit
from qmmd.dftb_anneal import run_dftb_anneal_prep, run_dftb_anneal_submit
from qmmd.ncoord import run_ncoord
from qmmd.lcod import run_lcod
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
from qmmd.gif import run_gif_submit, run_gif_run
from qmmd.station import run_station
from qmmd.us_pull import run_us_pull_prep, run_us_pull_submit
from qmmd.us_pull_report import run_us_pull_report
from qmmd.us_equil import run_us_equil_prep, run_us_equil_submit
from qmmd.us_equil_report import run_us_equil_report
from qmmd.us_prod import run_us_prod_prep, run_us_prod_submit
from qmmd.us_prod_report import run_us_prod_report
from qmmd.us_wham import run_us_wham
from qmmd.us_wham_report import run_us_wham_report
from qmmd.us_lcod import (
    run_pull_prep as run_us_lcod_pull_prep,
    run_pull_report as run_us_lcod_pull_report,
    run_equil_prep as run_us_lcod_equil_prep,
    run_equil_submit as run_us_lcod_equil_submit,
)
from qmmd.us_lcod_equil_report import run_us_lcod_equil_report
from qmmd.us_lcod_wham import run_us_lcod_wham
from qmmd.cphmd_prep import run_cphmd_prep
from qmmd.cphmd_dgref import run_cphmd_dgref_prep, run_cphmd_dgref_submit
from qmmd.cphmd_dgref_report import run_cphmd_dgref_report
from qmmd.cphmd_titr import run_cphmd_titr_prep, run_cphmd_titr_submit
from qmmd.cphmd_titr_post import run_cphmd_titr_post
from qmmd.cphmd_titr_report import run_cphmd_titr_report

def main():
    p = argparse.ArgumentParser(prog="qmmd")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep", help="Prepare solvated system with tleap")
    prep.add_argument("yaml", type=Path)

    mdequil = sub.add_parser("mdequil", help="Write MD equil inputs and run (slurm if provided, else local)")
    mdequil.add_argument("yaml", type=Path)

    cphmd_prep = sub.add_parser(
        "cphmd-prep",
        help="Print master/deprotonated MOL2 atom and charge mapping",
    )
    cphmd_prep.add_argument("yaml", type=Path)

    cphmd_dgref_prep = sub.add_parser(
        "cphmd-dgref-prep",
        help="Prepare Amber finddgref inputs without running or submitting",
    )
    cphmd_dgref_prep.add_argument("yaml", type=Path)

    cphmd_dgref_submit = sub.add_parser(
        "cphmd-dgref-submit",
        help="Validate and submit a prepared Amber finddgref job",
    )
    cphmd_dgref_submit.add_argument("yaml", type=Path)

    cphmd_dgref_report = sub.add_parser(
        "cphmd-dgref-report",
        help="Plot completed finddgref evaluations from a live or finished log",
    )
    cphmd_dgref_report.add_argument("yaml", type=Path)

    cphmd_titr_prep = sub.add_parser(
        "cphmd-titr-prep",
        help="Prepare explicit-solvent replica-exchange CpHMD inputs",
    )
    cphmd_titr_prep.add_argument("yaml", type=Path)

    cphmd_titr_submit = sub.add_parser(
        "cphmd-titr-submit",
        help="Validate and submit a prepared RECpHMD titration job",
    )
    cphmd_titr_submit.add_argument("yaml", type=Path)

    cphmd_titr_post = sub.add_parser(
        "cphmd-titr-post",
        help="Sort RECpHMD coordinates and protonation records into fixed-pH ensembles",
    )
    cphmd_titr_post.add_argument("yaml", type=Path)

    cphmd_titr_report = sub.add_parser(
        "cphmd-titr-report",
        help="Plot RECpHMD exchange health, titration, and protonation diagnostics",
    )
    cphmd_titr_report.add_argument("yaml", type=Path)

    salt = sub.add_parser("salt", help="Delete the counterion (sodium or chlorind) and turn the furthest water into a hydroxide")
    salt.add_argument("yaml", type=Path)

    dftb_prep = sub.add_parser("dftb-prep", help="Write DCDFTBMD inputs/scripts for one or many equil runs (no submit)")
    dftb_prep.add_argument("yaml", type=Path)

    dftb_submit = sub.add_parser("dftb-submit", help="Submit DCDFTBMD jobs for runs matching the config (no writes)")
    dftb_submit.add_argument("yaml", type=Path)

    dftb_anneal_prep = sub.add_parser("dftb-anneal", help="Write DCDFTBMD anneal inputs/scripts for one or many equil runs (no submit)")
    dftb_anneal_prep.add_argument("yaml", type=Path)

    dftb_anneal_submit = sub.add_parser("dftb-anneal-submit", help="Submit DCDFTBMD anneal jobs for runs matching the config (no writes)")
    dftb_anneal_submit.add_argument("yaml", type=Path)

    ncoord = sub.add_parser("ncoord", help="Write metacv.dat for selected runs based on dftb.inp")
    ncoord.add_argument("yaml", type=Path)
    lcod = sub.add_parser("lcod", help="Prepare a fixed-atom bond-distance-difference CV")
    lcod.add_argument("yaml", type=Path)

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

    gif_submit = sub.add_parser("gif-submit", help="Generate Slurm script and submit FES flooding GIF jobs")
    gif_submit.add_argument("yaml", type=Path)

    gif_run = sub.add_parser("gif-run", help="Run FES flooding GIF generation from config (internal)")
    gif_run.add_argument("yaml", type=Path)

    station = sub.add_parser("station", help="Plot station FES curves grid from config")
    station.add_argument("yaml", type=Path)

    us_pull_prep = sub.add_parser(
        "us-pull-prep",
        help="Prepare one serial Amber restrained-MD pulling job (no submission)",
    )
    us_pull_prep.add_argument("yaml", type=Path)

    us_pull_submit = sub.add_parser(
        "us-pull-submit",
        help="Submit a prepared serial Amber pulling job matching its config",
    )
    us_pull_submit.add_argument("yaml", type=Path)

    us_pull_report = sub.add_parser(
        "us-pull-report",
        help="Plot the stitched dihedral time series from a completed Amber pull",
    )
    us_pull_report.add_argument("yaml", type=Path)

    us_equil_prep = sub.add_parser(
        "us-equil-prep",
        help="Prepare restrained DCDFTBMD equilibration inputs for pull windows",
    )
    us_equil_prep.add_argument("yaml", type=Path)

    us_equil_submit = sub.add_parser(
        "us-equil-submit",
        help="Submit all prepared restrained DCDFTBMD equilibration windows",
    )
    us_equil_submit.add_argument("yaml", type=Path)

    us_equil_report = sub.add_parser(
        "us-equil-report",
        help="Plot currently available restrained DCDFTBMD dihedral traces",
    )
    us_equil_report.add_argument("yaml", type=Path)

    us_prod_prep = sub.add_parser(
        "us-prod-prep",
        help="Prepare restrained DCDFTBMD production inputs from equilibration restarts",
    )
    us_prod_prep.add_argument("yaml", type=Path)

    us_prod_submit = sub.add_parser(
        "us-prod-submit",
        help="Submit all prepared restrained DCDFTBMD production windows",
    )
    us_prod_submit.add_argument("yaml", type=Path)

    us_prod_report = sub.add_parser(
        "us-prod-report",
        help="Plot restrained DCDFTBMD production dihedral traces",
    )
    us_prod_report.add_argument("yaml", type=Path)

    us_wham = sub.add_parser(
        "us-wham",
        help="Construct and bootstrap a WHAM PMF from umbrella production windows",
    )
    us_wham.add_argument("yaml", type=Path)

    us_wham_report = sub.add_parser(
        "us-wham-report",
        help="Combine pull, production densities, and smooth PMF in one figure",
    )
    us_wham_report.add_argument("yaml", type=Path)

    for name, help_text in (
        ("us-lcod-pull-prep", "Select nearest LCOD frames for handmade umbrella windows"),
        ("us-lcod-pull-report", "Plot targets and selected LCOD seed frames"),
        ("us-lcod-equil-prep", "Prepare restrained DFTB LCOD equilibration windows"),
        ("us-lcod-equil-submit", "Submit all prepared DFTB LCOD equilibration windows"),
        ("us-lcod-equil-report", "Plot live restrained DFTB LCOD equilibration traces"),
        ("us-lcod-wham", "Construct LCOD WHAM PMF and report from equilibration trajectories"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("yaml", type=Path)

    args = p.parse_args()

    if args.cmd == "prep":
        run_prep(args.yaml)
    elif args.cmd == "mdequil":
        run_mdequil(args.yaml)
    elif args.cmd == "cphmd-prep":
        run_cphmd_prep(args.yaml)
    elif args.cmd == "cphmd-dgref-prep":
        run_cphmd_dgref_prep(args.yaml)
    elif args.cmd == "cphmd-dgref-submit":
        run_cphmd_dgref_submit(args.yaml)
    elif args.cmd == "cphmd-dgref-report":
        run_cphmd_dgref_report(args.yaml)
    elif args.cmd == "cphmd-titr-prep":
        run_cphmd_titr_prep(args.yaml)
    elif args.cmd == "cphmd-titr-submit":
        run_cphmd_titr_submit(args.yaml)
    elif args.cmd == "cphmd-titr-post":
        run_cphmd_titr_post(args.yaml)
    elif args.cmd == "cphmd-titr-report":
        run_cphmd_titr_report(args.yaml)
    elif args.cmd == "salt":
        run_salt(args.yaml)
    elif args.cmd == "dftb-prep":
        run_dftb_prep(args.yaml)
    elif args.cmd == "dftb-submit":
        run_dftb_submit(args.yaml)
    elif args.cmd == "dftb-anneal":
        run_dftb_anneal_prep(args.yaml)
    elif args.cmd == "dftb-anneal-submit":
        run_dftb_anneal_submit(args.yaml)
    elif args.cmd == "ncoord":
        run_ncoord(args.yaml)
    elif args.cmd == "lcod":
        run_lcod(args.yaml)
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
    elif args.cmd == "gif-submit":
        run_gif_submit(args.yaml)
    elif args.cmd == "gif-run":
        run_gif_run(args.yaml)
    elif args.cmd == "station":
        run_station(args.yaml)
    elif args.cmd == "us-pull-prep":
        run_us_pull_prep(args.yaml)
    elif args.cmd == "us-pull-submit":
        run_us_pull_submit(args.yaml)
    elif args.cmd == "us-pull-report":
        run_us_pull_report(args.yaml)
    elif args.cmd == "us-equil-prep":
        run_us_equil_prep(args.yaml)
    elif args.cmd == "us-equil-submit":
        run_us_equil_submit(args.yaml)
    elif args.cmd == "us-equil-report":
        run_us_equil_report(args.yaml)
    elif args.cmd == "us-prod-prep":
        run_us_prod_prep(args.yaml)
    elif args.cmd == "us-prod-submit":
        run_us_prod_submit(args.yaml)
    elif args.cmd == "us-prod-report":
        run_us_prod_report(args.yaml)
    elif args.cmd == "us-wham":
        run_us_wham(args.yaml)
    elif args.cmd == "us-wham-report":
        run_us_wham_report(args.yaml)
    elif args.cmd == "us-lcod-pull-prep":
        run_us_lcod_pull_prep(args.yaml)
    elif args.cmd == "us-lcod-pull-report":
        run_us_lcod_pull_report(args.yaml)
    elif args.cmd == "us-lcod-equil-prep":
        run_us_lcod_equil_prep(args.yaml)
    elif args.cmd == "us-lcod-equil-submit":
        run_us_lcod_equil_submit(args.yaml)
    elif args.cmd == "us-lcod-equil-report":
        run_us_lcod_equil_report(args.yaml)
    elif args.cmd == "us-lcod-wham":
        run_us_lcod_wham(args.yaml)


if __name__ == "__main__":
    main()
