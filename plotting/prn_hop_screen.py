#!/usr/bin/env python3
"""Screen PRN solvent proton transfers and reprotonation using explicit H ownership.

All atoms are one-based in outputs. This is a geometric candidate screen, not
an equilibrium/pKa analysis or proof of bulk diffusion. No charge-argmax jumps
are used as evidence of transfer. Complete trajectory snapshots only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from prn_acid_base_analysis import snapshot_bytes, TIME_RE
from prn_anti_dih import box_from_input, run_ids
from plot_cv_grid import infer_offset_ps


def intervals(times: np.ndarray, values: np.ndarray, duration: float) -> list[tuple[int, int, int]]:
    """Constant integer-state intervals with persistence; break at sampling gaps."""
    if np.any(np.diff(times) <= 0):
        raise ValueError("Nonmonotonic timestamps")
    gap = 1.5 * np.median(np.diff(times)) if len(times) > 1 else np.inf
    cuts = np.flatnonzero((values[1:] != values[:-1]) | (np.diff(times) > gap)) + 1
    bounds = np.r_[0, cuts, len(times)]
    return [(int(a), int(b-1), int(values[a])) for a, b in zip(bounds[:-1], bounds[1:])
            if times[b-1]-times[a]+1e-9 >= duration]


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    blob = snapshot_bytes(path)
    lines = blob.splitlines(keepends=True)
    n = int(lines[0]); block = n+2
    nf = len(lines)//block
    if nf and not lines[nf*block-1].endswith(b"\n"):
        nf -= 1
    if nf < 2:
        raise ValueError("Fewer than two complete frames")
    symbols = [lines[2+j].split()[0].decode() for j in range(n)]
    times = np.empty(nf); coords = np.empty((nf,n,3))
    for k in range(nf):
        start = k*block
        if int(lines[start]) != n:
            raise ValueError("Changing atom count")
        match = TIME_RE.search(lines[start+1].decode())
        if match is None:
            raise ValueError("Missing timestamp")
        times[k] = float(match[1])/1000
        coords[k] = np.loadtxt(io.BytesIO(b"".join(lines[start+2:start+block])),
                               usecols=(1,2,3))
    if symbols[:11] != ["C","C","C","O","O"]+["H"]*6:
        raise ValueError("Unexpected PRN atom order")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Restart/stitching required")
    return times, coords, symbols, hashlib.sha256(blob).hexdigest()


def screen_run(run: int, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    folder = args.runs_path/f"run-{run}"/"meta-h"
    if (folder/"metad-restart").exists():
        raise ValueError("Restart directory needs explicit stitching")
    t, xyz, symbols, sha = read_trajectory(folder/"traject")
    offset = infer_offset_ps(folder.parent, "meta-h", float(t[0]))
    box = box_from_input(folder/"dftb.inp")
    h = np.array([i+1 for i,s in enumerate(symbols) if s == "H"])
    heavy = np.array([i+1 for i,s in enumerate(symbols) if s != "H"])
    solvent_o = np.array([i+1 for i,s in enumerate(symbols) if s == "O" and i+1 > 11])
    n = len(t)
    owners = np.zeros((n,len(h)),dtype=np.int16)
    min_oh = np.empty((n,2))
    counts = np.zeros((n,2),dtype=int)
    hydronium = np.zeros(n,dtype=int)
    hydronium_count = np.zeros(n,dtype=int)
    for k,frame in enumerate(xyz):
        tree = cKDTree(frame[heavy-1] % box,boxsize=box)
        distances,nearest = tree.query(frame[h-1] % box,k=2)
        unique = (distances[:,0] <= args.bond_cutoff) & (distances[:,1]-distances[:,0] >= args.margin)
        owners[k] = np.where(unique,heavy[nearest[:,0]],0)
        delta = frame[h-1,None,:]-frame[None,np.array([4,5])-1,:]
        delta -= box*np.round(delta/box)
        min_oh[k] = np.linalg.norm(delta,axis=-1).min(axis=0)
        counts[k] = [(owners[k]==site).sum() for site in (4,5)]
        nh = np.bincount(owners[k],minlength=len(symbols)+1)
        candidates = solvent_o[nh[solvent_o]==3]
        hydronium_count[k] = len(candidates)
        if len(candidates)==1:
            hydronium[k] = candidates[0]
    # Definite vacancy is conservative: not even an ambiguous nearby H <= cutoff.
    free = np.all(min_oh > args.bond_cutoff,axis=1)
    free_intervals = [(a,b) for a,b,v in intervals(t,free.astype(int),args.persistence) if v==1]
    bound = np.zeros((n,2),dtype=bool)
    for col in range(2):
        for a,b,v in intervals(t,(counts[:,col]>0).astype(int),args.persistence):
            if v: bound[a:b+1,col] = True
    loss = next((a for a,b in free_intervals),None)
    stable = [(a,b,v) for a,b,v in intervals(t,hydronium,args.persistence) if v>0]
    events = []
    for prev,nxt in zip(stable[:-1],stable[1:]):
        a,b,donor = prev; c,d,acceptor = nxt
        if donor==acceptor or t[c]-t[b] > args.transfer_gap:
            continue
        moved = h[(owners[b]==donor)&(owners[c]==acceptor)]
        if len(moved)!=1 or (owners[b]==acceptor).sum()!=2 or (owners[c]==donor).sum()!=2:
            continue
        # Both endpoint hydronia persistent; check the explicit bridging H identity.
        clean = bool(free[b:d+1].all())
        after = np.flatnonzero(bound[c:].any(axis=1))
        reprot = c+int(after[0]) if len(after) else None
        free_end = next((j for i,j in free_intervals if i<=b and j>=d),None)
        separation = []
        for k,site in ((b,donor),(c,acceptor)):
            vec = xyz[k,site-1]-xyz[k,np.array([4,5])-1]
            vec -= box*np.round(vec/box)
            separation.append(float(np.linalg.norm(vec,axis=1).min()))
        events.append(dict(run=run,from_O=donor,to_O=acceptor,transferred_H=int(moved[0]),
            before_grid_ps=float(t[b]+offset),after_grid_ps=float(t[c]+offset),
            both_sites_vacant_during_hop=clean,
            continuous_vacancy_after_hop_ps=float(t[free_end]-t[c]) if free_end is not None else 0.,
            subsequent_reprotonation_grid_ps=float(t[reprot]+offset) if reprot is not None else None,
            subsequent_reprotonated_O=(";".join(str(x) for x,y in zip((4,5),bound[reprot]) if y)
                                       if reprot is not None else ""),
            hydronium_min_solute_O_distance_before_A=separation[0],
            hydronium_min_solute_O_distance_after_A=separation[1]))
    clean_events = [e for e in events if e["both_sites_vacant_during_hop"]]
    lasting = [e for e in clean_events if e["subsequent_reprotonation_grid_ps"] is None
               and e["continuous_vacancy_after_hop_ps"] >= args.post_hop]
    row = dict(run=run,status="ok",frames=n,grid_start_ps=float(t[0]+offset),grid_end_ps=float(t[-1]+offset),
        raw_to_grid_offset_ps=float(offset),first_definite_both_O_vacant_grid_ps=float(t[loss]+offset) if loss is not None else None,
        longest_both_O_vacant_ps=max((float(t[b]-t[a]) for a,b in free_intervals),default=0.),
        solvent_hops=len(events),hops_with_both_O_vacant=len(clean_events),
        lasting_clean_hops=len(lasting),first_clean_hop_grid_ps=clean_events[0]["after_grid_ps"] if clean_events else None,
        first_lasting_clean_hop_grid_ps=lasting[0]["after_grid_ps"] if lasting else None,
        O4_reprotonation_after_first_vacancy_grid_ps=None,O5_reprotonation_after_first_vacancy_grid_ps=None,
        source_sha256=sha)
    if loss is not None:
        for col,site in enumerate((4,5)):
            idx = np.flatnonzero(bound[loss:,col])
            if len(idx): row[f"O{site}_reprotonation_after_first_vacancy_grid_ps"] = float(t[loss+idx[0]]+offset)
    target = args.out_dir/f"run-{run}.csv.gz"
    with gzip.open(target,"wt") as f:
        np.savetxt(f,np.column_stack([t,t+offset,min_oh,counts,bound,free,hydronium,hydronium_count,owners[:,np.flatnonzero(h==11)[0]]]),
                   delimiter=",",fmt="%.6g",header="raw_time_ps,grid_time_ps,O4_min_H_A,O5_min_H_A,O4_owned_H_count,O5_owned_H_count,O4_persistent_bound,O5_persistent_bound,both_definitely_vacant,unique_hydronium_O,hydronium_count,H11_owner",comments="")
    print(f"run-{run}: {len(events)} solvent hops; {len(clean_events)} with both O vacant; {len(lasting)} lasting clean",flush=True)
    return row,events


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-path",type=Path,default=Path("systems/PRN-syn/solv_5.5/dftb/N1T48C1"))
    p.add_argument("--out-dir",type=Path,default=Path("reports/PRN-syn/solv_5.5/proton_hop_screen"))
    p.add_argument("--runs",type=run_ids,default=run_ids("1-100"))
    p.add_argument("--bond-cutoff",type=float,default=1.4,help="Angstrom")
    p.add_argument("--margin",type=float,default=.15,help="Nearest-heavy ownership distance advantage, Angstrom")
    p.add_argument("--persistence",type=float,default=.05,help="Stable hydronium/site state duration, ps")
    p.add_argument("--transfer-gap",type=float,default=.15,help="Maximum gap between stable donor and acceptor states, ps")
    p.add_argument("--post-hop",type=float,default=.5,help="Minimum continuous definite vacancy after clean hop, ps")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--review-only",action="store_true",help="Review existing screen, verify candidate Mulliken changes, plot candidates")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.review_only:
        review(args.out_dir)
        return
    def safe(run: int) -> tuple[dict,list[dict]]:
        try: return screen_run(run,args)
        except (ValueError,FileNotFoundError) as exc:
            print(f"run-{run}: ERROR {exc}",flush=True)
            return dict(run=run,status=str(exc)),[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(safe,args.runs))
    rows = [r for r,e in results]; events = [e for r,es in results for e in es]
    for name,data in (("screening.csv",rows),("hop_events.csv",events)):
        with (args.out_dir/name).open("w",newline="") as f:
            fields = list(dict.fromkeys(k for row in data for k in row))
            w = csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    (args.out_dir/"parameters.json").write_text(json.dumps(vars(args),default=str,indent=2)+"\n")


def review(out: Path) -> None:
    """Corroborate geometric candidate transfers, without requiring charge-window selection."""
    import matplotlib.pyplot as plt
    from acid_base_PRN import read_xyz_symbols
    from prn_acid_base_analysis import mulliken_frames
    settings = json.loads((out/"parameters.json").read_text())
    rows = list(csv.DictReader((out/"screening.csv").open()))
    events = list(csv.DictReader((out/"hop_events.csv").open()))
    candidates = [r for r in rows if r["status"]=="ok" and int(r["lasting_clean_hops"])>0]
    reviewed,charge_rows = [],[]
    plt.style.use(Path(__file__).with_name("lefteris.mplstyle"))
    fig,axes = plt.subplots(max(1,len(candidates)),1,figsize=(15,2.4*max(1,len(candidates))),squeeze=False)
    for row,ax in zip(candidates,axes.flat):
        run = int(row["run"])
        data = np.genfromtxt(out/f"run-{run}.csv.gz",delimiter=",",names=True)
        t = data["grid_time_ps"]
        start = float(row["first_lasting_clean_hop_grid_ps"])
        post = t >= start
        selected = [e for e in events if int(e["run"])==run and e["both_sites_vacant_during_hop"]=="True"
                    and not e["subsequent_reprotonation_grid_ps"]
                    and float(e["continuous_vacancy_after_hop_ps"]) >= settings["post_hop"]]
        folder = Path(settings["runs_path"])/f"run-{run}"/"meta-h"
        mt,q = mulliken_frames(folder/"mulliken",read_xyz_symbols(folder/"traject"))
        mt += float(row["raw_to_grid_offset_ps"])
        supported = 0
        for e in selected:
            before,after = float(e["before_grid_ps"]),float(e["after_grid_ps"])
            ib = (mt >= before-.025-1e-8)&(mt <= before+1e-8)
            ia = (mt >= after-1e-8)&(mt <= after+.025+1e-8)
            donor,acceptor = int(e["from_O"]),int(e["to_O"])
            qb = np.mean(q[ib][:,[donor-1,acceptor-1]],axis=0) if ib.any() else np.full(2,np.nan)
            qa = np.mean(q[ia][:,[donor-1,acceptor-1]],axis=0) if ia.any() else np.full(2,np.nan)
            agrees = bool(np.isfinite([*qb,*qa]).all() and qa[0]<qb[0] and qa[1]>qb[1])
            supported += agrees
            charge_rows.append(dict(**e,donor_q_before_e=qb[0],donor_q_after_e=qa[0],
                acceptor_q_before_e=qb[1],acceptor_q_after_e=qa[1],charge_changes_support_transfer=agrees))
        row = dict(row,charge_supported_clean_hops=supported,
                   definite_vacancy_fraction_after_first_clean_hop=float(np.mean(data["both_definitely_vacant"][post])),
                   final_O4_min_H_A=float(data["O4_min_H_A"][-1]),final_O5_min_H_A=float(data["O5_min_H_A"][-1]),
                   no_sustained_reprotonation_since_first_vacancy=(not row["O4_reprotonation_after_first_vacancy_grid_ps"]
                                                               and not row["O5_reprotonation_after_first_vacancy_grid_ps"]))
        reviewed.append(row)
        for site,color in ((4,"#2166ac"),(5,"#b2182b")):
            ax.plot(t,data[f"O{site}_min_H_A"],lw=.8,color=color,label=f"O{site}: closest H")
        ax.axhline(settings["bond_cutoff"],color="0.4",ls="--",lw=1)
        for e in selected:
            ax.axvline(float(e["after_grid_ps"]),color="green",alpha=.5,lw=1)
        ax.set_ylim(.8,3.1); ax.set_ylabel("Closest H (Å)")
        ax.set_title(f"run-{run}: {len(selected)} clean hops ({supported} charge-supported); green = hop",fontsize=11)
        ax.legend(loc="upper right",fontsize=9); ax.set_xlabel("Grid-clock time (ps)")
        print(f"REVIEW run-{run}: {supported}/{len(selected)} charge-supported; never rebind after first vacancy={row['no_sustained_reprotonation_since_first_vacancy']}",flush=True)
    for name,data in (("candidates.csv",reviewed),("candidate_charge_checks.csv",charge_rows)):
        with (out/name).open("w",newline="") as f:
            fields = list(dict.fromkeys(k for r in data for k in r))
            w = csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    fig.tight_layout(); fig.savefig(out/"candidates.png",dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
