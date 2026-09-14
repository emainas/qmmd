"""Prepare a fixed-atom bond-distance-difference metadynamics coordinate."""
from dataclasses import dataclass
import math
from pathlib import Path
import yaml

from qmmd.ncoord import find_repo_root, parse_run_ids, read_symbols_from_dftb_inp


@dataclass(frozen=True)
class LcodConfig:
    system_dir: Path
    dftb_dirname: str
    replica_dirname: str
    cv_dirname: str
    bench_tag: str
    run_ids: tuple[int, ...]
    gaussian_width: float
    atoms: tuple[int, ...]
    grid_min: float | None = None
    grid_max: float | None = None
    grid_step: float | None = None


def load_config(path: Path) -> LcodConfig:
    data = yaml.safe_load(path.read_text())
    coordinate = data['lcod']
    grid_keys = {'grid_min', 'grid_max', 'grid_step'}
    unknown = set(coordinate) - {'type', 'gaussian_width', 'atoms'} - grid_keys
    if unknown:
        raise ValueError(f'Unknown lcod settings: {sorted(unknown)}')
    if coordinate.get('type', 'BONDDISTANCEDIFFERENCE') != 'BONDDISTANCEDIFFERENCE':
        raise ValueError('lcod.type must be BONDDISTANCEDIFFERENCE')
    width = float(coordinate['gaussian_width'])
    if not math.isfinite(width) or width <= 0:
        raise ValueError('lcod.gaussian_width must be positive and finite (Å)')
    grid = (None, None, None)
    if grid_keys & coordinate.keys():
        if not grid_keys <= coordinate.keys():
            raise ValueError('Specify all three lcod grid_min, grid_max and grid_step values')
        try:
            grid = tuple(float(coordinate[k]) for k in ('grid_min', 'grid_max', 'grid_step'))
        except (TypeError, ValueError) as exc:
            raise ValueError('LCOD grid values must be finite numbers (Å)') from exc
        lower, upper, step = grid
        if not all(math.isfinite(v) for v in grid) or lower >= upper or step <= 0:
            raise ValueError('LCOD grid requires finite min < max and positive step (Å)')
        if step > upper - lower:
            raise ValueError('LCOD grid_step must not exceed the grid range')
    atoms = coordinate['atoms']
    if not isinstance(atoms, list) or len(atoms) != 4 or any(type(a) is not int or a < 1 for a in atoms):
        raise ValueError('lcod.atoms requires four positive one-based integer atom IDs')
    if atoms[0] == atoms[1] or atoms[2] == atoms[3]:
        raise ValueError('Each distance requires two distinct atoms')
    if set(atoms[:2]) == set(atoms[2:]):
        raise ValueError('Identical distance pairs give a constant zero CV')
    runs = tuple(parse_run_ids(data['run_ids']))
    if not runs or any(r < 1 for r in runs) or len(set(runs)) != len(runs):
        raise ValueError('run_ids must be nonempty, positive and unique')
    if data.get('system_dir') is not None:
        base = Path(data['system_dir'])
    elif data.get('buffer') is not None:
        base = Path('systems') / data['system'] / f"{data.get('prefix', 'solv')}_{float(data['buffer']):.1f}"
    else:
        raise ValueError('Either system_dir or buffer is required')
    names = [data.get('dftb_dirname', 'dftb'), data.get('replica_dirname', 'equil'),
             data.get('cv_dirname', 'meta-lcod'), data['bench_tag']]
    if any(not isinstance(n, str) or not n or Path(n).name != n or n in {'.', '..'} for n in names):
        raise ValueError('Directory names and bench_tag must be single path components')
    if names[1] == names[2]:
        raise ValueError('cv_dirname must differ from replica_dirname')
    return LcodConfig(base, *names, runs, width, tuple(atoms), *grid)


def render_metacv(cfg: LcodConfig, symbols: list[str]) -> str:
    if max(cfg.atoms) > len(symbols):
        raise ValueError(f'LCOD atom ID exceeds input atom count ({len(symbols)})')
    text = f"BONDDISTANCEDIFFERENCE {cfg.gaussian_width:g} {' '.join(map(str, cfg.atoms))}"
    if cfg.grid_min is not None:
        text += f" {cfg.grid_min:g} {cfg.grid_max:g} {cfg.grid_step:g}"
    return text + '\n'


def run_lcod(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    root = find_repo_root(yaml_path)
    base = cfg.system_dir if cfg.system_dir.is_absolute() else root / cfg.system_dir
    pending = []
    # Validate every new target before creating any output directories.
    for run in cfg.run_ids:
        run_dir = base / cfg.dftb_dirname / cfg.bench_tag / f'run-{run}'
        dest = run_dir / cfg.cv_dirname
        if dest.exists():
            print(f'SKIP: {dest} already exists; not touching')
            continue
        inp = run_dir / cfg.replica_dirname / 'dftb.inp'
        symbols = read_symbols_from_dftb_inp(inp)
        pending.append((dest, render_metacv(cfg, symbols)))
    for dest, text in pending:
        dest.mkdir()  # No overwrite if another process creates the directory.
        (dest / 'metacv.dat').write_text(text)
        (dest / 'spec.yaml').write_text(yaml_path.read_text())
        print(f'OK: wrote {dest / "metacv.dat"}')
