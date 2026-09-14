"""Prepare PRN syn/anti DFTB replicas directly from Packmol XYZ; never submit."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import secrets
import shlex

import numpy as np
import yaml


def read_xyz(path: Path) -> tuple[list[tuple[str, float, float, float]], np.ndarray]:
    lines = path.read_text().splitlines()
    match = re.search(r'Lattice="([^"]+)"', lines[1])
    if match is None:
        raise ValueError(f"Missing XYZ lattice: {path}")
    box = np.asarray([float(x) for x in match[1].split()]).reshape(3, 3)
    atoms = [(p[0], *map(float, p[1:4])) for line in lines[2:] if (p := line.split())]
    if len(atoms) != int(lines[0]) or len(atoms) != 311:
        raise ValueError(f"Expected 311 atoms: {path}")
    if np.linalg.det(box) <= 0:
        raise ValueError("Invalid box")
    return atoms, box


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    template = root / "configs/HPD/dftb/dftb.yaml"
    base = yaml.safe_load(template.read_text())
    executable = Path(base["runtime"]["executable"])
    if not executable.is_file():
        raise FileNotFoundError(executable)
    prepared = []
    for rotamer in ("anti", "syn"):
        source = root / f"systems/PRN-{rotamer}/solv_100/solvated.xyz"
        atoms, box = read_xyz(source)
        destination = source.parent / "dftb/N1T48C1"
        if destination.exists():
            raise FileExistsError(destination)
        prepared.append((rotamer, source, atoms, box, destination))
    elements = [e for e in base["dftb"]["elements"] if e["symbol"] in {"H", "O", "C"}]
    symbols = [e["symbol"] for e in elements]
    params = [root / f"params/{a}-{b}.skf" for a in symbols for b in symbols]
    for path in params:
        if not path.is_file():
            raise FileNotFoundError(path)
    seeds: set[int] = set()
    for rotamer, source, atoms, box, destination in prepared:
        spec = yaml.safe_load(template.read_text())
        # Point submission at the water-count layout rather than a buffer directory.
        for key in ("buffer", "prefix", "salt_dirname"):
            spec.pop(key, None)
        spec.update(system=f"PRN-{rotamer}", source_xyz=str(source.relative_to(root)),
                    system_dir=str(source.parent.relative_to(root)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    template=str(template.relative_to(root)), replicas=20, append=False,
                    bench_tag="N1T48C1", replica_dirname="equil")
        spec["dftb"]["elements"] = elements
        spec["runtime"]["mpirun_np"] = 16
        job = spec["slurm"]["job"]
        job.update(name=f"PRN-{rotamer}", partition="batch", time="1-00:00:00",
                   nodes=1, ntasks=48, cpus_per_task=1, qos="highpri")
        destination.mkdir(parents=True)
        (destination / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        manifest = ["run_id\trandom_seed\tinput"]
        for run_id in range(1, 21):
            seed = secrets.randbelow(2**31 - 1) + 1
            while seed in seeds:
                seed = secrets.randbelow(2**31 - 1) + 1
            seeds.add(seed)
            out = destination / f"run-{run_id}/equil"
            (out / "params").mkdir(parents=True)
            for path in params:
                (out / "params" / path.name).symlink_to(path)
            header = [line.replace("RANDOMSEED=0", f"RANDOMSEED={seed}") for line in spec["dftb"]["header_lines"]]
            lines = header + ["", spec["dftb"]["title"], "", str(len(elements))]
            for element in elements:
                symbol = element["symbol"]
                lines.extend([f"{symbol}   {element['lmax']} {element['hubbard']:.4f}",
                              "  " + " ".join(f"params/{symbol}-{other}.skf" for other in symbols)])
            lines.extend(["", f"{len(atoms):5d}  0  1"])
            lines.extend(f"{symbol:<2s}{x:>15.8f}{y:>15.8f}{z:>15.8f}" for symbol, x, y, z in atoms)
            lines.extend(f"TV{x:>18.8f}{y:>15.8f}{z:>15.8f}" for x, y, z in box)
            (out / "dftb.inp").write_text("\n".join(lines) + "\n\n")
            run_spec = dict(spec, run_id=run_id, random_seed=seed)
            (out / "run_metadata.yaml").write_text(yaml.safe_dump(run_spec, sort_keys=False))
            (out / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
            run = f'''#!/usr/bin/env bash
set -euo pipefail

module purge
module load {shlex.quote(spec['runtime']['module'])}

export PATH="/nas/sycamore/home/emainas/software/dcdftbmd.2.0/:$PATH"

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export OMP_STACKSIZE=1G

ulimit -s unlimited

mpirun -np 16 "{executable}"
'''
            (out / "run.sh").write_text(run)
            slurm = f'''#!/usr/bin/env bash
#SBATCH --qos=highpri
#SBATCH --job-name=PRN-{rotamer}-N1T48C1
#SBATCH --partition={job['partition']}
#SBATCH --time={job['time']}
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem={job['mem']}
#SBATCH --cpus-per-task=1
#SBATCH --output={job['stdout']}
#SBATCH --error={job['stderr']}

bash run.sh
'''
            (out / "slurm.sh").write_text(slurm)
            for name in ("run.sh", "slurm.sh"):
                (out / name).chmod(0o755)
            manifest.append(f"{run_id}\t{seed}\trun-{run_id}/equil/dftb.inp")
        (destination / "runs.tsv").write_text("\n".join(manifest) + "\n")
        print(f"Prepared 20 runs: {destination}")


if __name__ == "__main__":
    main()
