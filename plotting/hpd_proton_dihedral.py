"""Persistent dynamic O7-H ownership and N1-C2-O7-H torsion for HPD."""
from pathlib import Path
import numpy as np
from prn_proton_dihedrals import persistent_ids


def oxygen_dihedral(traj: Path, box: np.ndarray, end_ps: float,
                    cutoff: float = 1.4, margin: float = .15,
                    persistence: float = .05) -> dict[str, np.ndarray]:
    from acid_base_HPD import iter_xyz_frames, read_xyz_symbols, _minimum_image, _signed_dihedral_deg
    if cutoff <= 0 or margin < 0 or persistence < 0:
        raise ValueError('Invalid O-H ownership thresholds')
    symbols=read_xyz_symbols(traj)
    heavy=np.array([i for i,s in enumerate(symbols) if s!='H'])
    hs=np.array([i for i,s in enumerate(symbols) if s=='H'])
    times=[]; candidates=[]; distances=[]; angles=[]; counts=[]
    for t,c in iter_xyz_frames(traj):
        if t>end_ps: break
        d=np.linalg.norm(_minimum_image(c[hs,None]-c[heavy],box),axis=2)
        order=np.argsort(d,axis=1)[:,:2]; rr=np.arange(len(hs))
        first=d[rr,order[:,0]]; second=d[rr,order[:,1]]
        matches=np.flatnonzero((heavy[order[:,0]]==6)&(first<=cutoff)&(second-first>=margin))
        h=int(hs[matches[0]]) if len(matches)==1 else None
        times.append(t);counts.append(len(matches))
        candidates.append(h+1 if h is not None else np.nan)
        distances.append(first[matches[0]] if h is not None else np.nan)
        angles.append(_signed_dihedral_deg(c[[0,1,6,h]],box) if h is not None else np.nan)
    t=np.array(times);candidate=np.array(candidates)
    ids=persistent_ids(t,candidate,persistence)
    phi=np.where(np.isfinite(ids),angles,np.nan)
    return dict(time_ps=t,candidate_H_id=candidate,assigned_H_count=np.array(counts),
                O7_H_distance_A=np.array(distances),hydrogen_id=ids,dihedral_deg=phi)


def plot_dihedral(ax, data: dict[str,np.ndarray]) -> None:
    t=data['time_ps'];phi=data['dihedral_deg'].copy();ids=data['hydrogen_id']
    # Insert explicit gaps at identity changes or the angular wrapping seam.
    breaks=np.flatnonzero((np.abs(np.diff(phi))>180)|(np.diff(ids)!=0))+1
    tx=np.insert(t,breaks,np.nan);py=np.insert(phi,breaks,np.nan)
    ax.plot(tx,py,color='darkorange',lw=.8)
    ax.set(title='New O7–H bond',ylabel='N1–C2–O7–Hnew dihedral (°)',
           xlabel='Raw DFTB time (ps)',ylim=(-180,180))
    ax.set_yticks(np.arange(-180,181,60));ax.grid(alpha=.2)
    if not np.isfinite(phi).any():
        ax.text(.5,.5,'No sustained O7–H assignment',ha='center',transform=ax.transAxes)
