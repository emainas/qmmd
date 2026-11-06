#!/usr/bin/env python3
"""Prepare an expanded Sandercyanin pH-REMD ladder from a completed run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
from pathlib import Path

import yaml
from scipy.io import netcdf_file


EXCHANGE_RE = re.compile(r"^# exchange\s+(\d+)\s*$")
REPLICA_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_final_ph_assignments(path: Path) -> tuple[int, dict[float, int]]:
    """Map final post-exchange pH values onto source replica numbers."""
    exchanges: dict[int, list[tuple[int, float]]] = {}
    current_exchange: int | None = None
    for line in path.read_text(errors="replace").splitlines():
        exchange_match = EXCHANGE_RE.match(line)
        if exchange_match is not None:
            current_exchange = int(exchange_match.group(1))
            exchanges[current_exchange] = []
            continue
        replica_match = REPLICA_RE.match(line)
        if replica_match is None or current_exchange is None:
            continue
        replica = int(replica_match.group(1))
        new_ph = float(replica_match.group(3))
        exchanges[current_exchange].append((replica, new_ph))
    if not exchanges:
        raise ValueError(f"No replica-exchange records found in {path}")
    final_exchange = max(exchanges)
    final_rows = exchanges[final_exchange]
    if not final_rows:
        raise ValueError(f"Exchange {final_exchange} has no replica rows in {path}")
    mapping: dict[float, int] = {}
    for replica, ph in final_rows:
        if ph in mapping:
            raise ValueError(f"Final pH {ph:g} occurs more than once in {path}")
        mapping[ph] = replica
    return final_exchange, mapping


def render_mdin(ph: float, numexchg: int) -> str:
    return f"""Sandercyanin monomer expanded-ladder RECpHMD; pH {ph:.1f}
&cntrl
 iwrap=1, imin=0, irest=1, ntx=5,
 nstlim=1000, numexchg={numexchg}, dt=0.002,
 tempi=300.0, temp0=300.0, ntt=3, gamma_ln=1.0, ig=-1,
 ntb=1, ntp=0, ntc=2, ntf=2, cut=8.0,
 ntpr=500, ntwx=500, ntwr=1000, ioutfm=1,
 icnstph=2, solvph={ph:.1f}, ntcnstph=100, ntrelax=100, saltcon=0.1,
/
"""


def retarget_remd_restart(path: Path, ph: float, replica_index: int) -> None:
    """Reset hidden REMD metadata after cloning a NetCDF restart."""
    required = {
        "time",
        "remd_values",
        "remd_repidx",
        "remd_crdidx",
        "remd_dimtype",
        "remd_indices",
    }
    with netcdf_file(path, mode="a", mmap=False) as restart:
        missing = required.difference(restart.variables)
        if missing:
            raise ValueError(f"Restart {path} lacks REMD variables: {sorted(missing)}")
        restart.variables["time"].data[...] = 0.0
        restart.variables["remd_values"].data[...] = ph
        restart.variables["remd_repidx"].data[...] = replica_index - 1
        restart.variables["remd_crdidx"].data[...] = replica_index - 1
        restart.variables["remd_dimtype"].data[...] = 4
        restart.variables["remd_indices"].data[...] = replica_index
        restart.flush()


def prepare_expanded_ladder(
    source: Path,
    destination: Path,
    ph_values: tuple[float, ...],
    production_ns: float,
    partition: str,
    nodes: int,
    ranks_per_replica: int,
    walltime: str,
    seed_mode: str,
    launcher: str,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if tuple(sorted(set(ph_values))) != ph_values:
        raise ValueError("pH values must be unique and strictly increasing")
    numexchg_float = production_ns * 500.0
    numexchg = int(round(numexchg_float))
    if not math.isclose(numexchg_float, numexchg) or numexchg <= 0 or numexchg % 2:
        raise ValueError("production_ns must yield a positive even number of 2-ps exchanges")

    required = (
        source / "mon.rings-nearby.parm7",
        source / "mon.rings-nearby.cpin",
        source / "mon.rings-nearby.rst7",
    )
    if seed_mode == "final-remd":
        required += (source / "rem.log",)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    final_exchange: int | None = None
    final_ph_to_replica: dict[float, int] = {}
    if seed_mode == "final-remd":
        final_exchange, final_ph_to_replica = parse_final_ph_assignments(
            source / "rem.log"
        )
    source_max_ph = max(final_ph_to_replica) if final_ph_to_replica else math.nan

    destination.mkdir(parents=True)
    for name in (
        "mon.rings-nearby.parm7",
        "mon.rings-nearby.cpin",
        "mon.rings-nearby.rst7",
    ):
        shutil.copy2(source / name, destination / name)

    seed_rows: list[dict[str, object]] = []
    group_lines: list[str] = []
    for target_replica, ph in enumerate(ph_values, start=1):
        target_stem = f"replica-{target_replica:02d}"
        target_cpin = destination / f"{target_stem}.cpin"
        if seed_mode == "equilibrated":
            source_replica: int | str = "equilibrated"
            seed_ph: float | str = "not applicable"
            source_rst7 = source / "mon.rings-nearby.rst7"
            source_cpin = source / "mon.rings-nearby.cpin"
            coordinate_input = "mon.rings-nearby.rst7"
            shutil.copy2(source_cpin, target_cpin)
        else:
            seed_ph = ph if ph in final_ph_to_replica else min(ph, source_max_ph)
            if seed_ph not in final_ph_to_replica:
                seed_ph = min(final_ph_to_replica, key=lambda value: abs(value - ph))
            source_replica = final_ph_to_replica[seed_ph]
            source_stem = f"replica-{source_replica:02d}"
            source_rst7 = source / f"{source_stem}.rst7"
            source_cpin = source / f"{source_stem}.cprestrt"
            if not source_rst7.is_file() or not source_cpin.is_file():
                raise FileNotFoundError(
                    f"Missing restart pair for source replica {source_replica}: "
                    f"{source_rst7}, {source_cpin}"
                )
            target_rst7 = destination / f"{target_stem}.start.rst7"
            shutil.copy2(source_rst7, target_rst7)
            retarget_remd_restart(target_rst7, ph, target_replica)
            shutil.copy2(source_cpin, target_cpin)
            coordinate_input = target_rst7.name
        (destination / f"{target_stem}.mdin").write_text(render_mdin(ph, numexchg))
        group_lines.append(
            f"-O -i {target_stem}.mdin -p mon.rings-nearby.parm7 "
            f"-c {coordinate_input} -cpin {target_stem}.cpin "
            f"-o {target_stem}.mdout -r {target_stem}.rst7 -x {target_stem}.nc "
            f"-cpout {target_stem}.cpout -cprestrt {target_stem}.cprestrt "
            f"-inf {target_stem}.mdinfo"
        )
        seed_rows.append(
            {
                "target_replica": target_replica,
                "target_pH": f"{ph:.1f}",
                "source_replica": source_replica,
                "source_final_pH": (
                    f"{seed_ph:.1f}" if isinstance(seed_ph, float) else seed_ph
                ),
                "source_rst7": str(source_rst7),
                "source_cpin": str(source_cpin),
                "source_rst7_sha256": sha256(source_rst7),
                "source_cpin_sha256": sha256(source_cpin),
            }
        )

    (destination / "groupfile").write_text("\n".join(group_lines) + "\n")
    (destination / "ph-ladder.csv").write_text(
        "replica,pH\n"
        + "".join(f"{index},{ph:.1f}\n" for index, ph in enumerate(ph_values, start=1))
    )
    with (destination / "seed-map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    total_ranks = ranks_per_replica * len(ph_values)
    if total_ranks % nodes:
        raise ValueError(
            f"{total_ranks} total ranks cannot be divided evenly across {nodes} nodes"
        )
    tasks_per_node = total_ranks // nodes
    run_script = destination / "run.sh"
    launch_command = (
        f"mpirun -np {total_ranks}"
        if launcher == "mpirun"
        else f"srun --mpi=pmix --ntasks={total_ranks}"
    )
    run_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmodule purge\nmodule load amber/26\n"
        "export TEMP=300\n"
        f"{launch_command} pmemd.MPI -ng {len(ph_values)} "
        "-groupfile groupfile -rem 4 -remlog rem.log\n"
    )
    run_script.chmod(0o755)
    slurm_script = destination / "slurm.sh"
    slurm_script.write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH -J sc-mon-ring-ph12\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH -N {nodes}\n"
        f"#SBATCH --ntasks={total_ranks}\n"
        f"#SBATCH --ntasks-per-node={tasks_per_node}\n"
        f"#SBATCH -t {walltime}\n"
        "#SBATCH -o slurm.%j.out\n"
        "#SBATCH -e slurm.%j.err\n\n"
        "bash run.sh\n"
    )
    slurm_script.chmod(0o755)

    provenance = {
        "source_run": str(source),
        "source_final_exchange": final_exchange,
        "source_final_ph_to_replica": {
            f"{ph:.1f}": replica for ph, replica in sorted(final_ph_to_replica.items())
        },
        "seed_mode": seed_mode,
        "restart_strategy": (
            "All replicas start from the same equilibrated pre-REMD structure and "
            "master CPIN, matching the successful original run."
            if seed_mode == "equilibrated"
            else "Existing pH values use the source walker ending at that pH; newly "
            "added pH values use the source walker ending at the maximum source pH."
        ),
        "mpi_launcher": launcher,
        "production_ns_per_replica": production_ns,
        "ph_values": list(ph_values),
        "replicas": len(ph_values),
        "slurm": {
            "partition": partition,
            "nodes": nodes,
            "ranks_per_replica": ranks_per_replica,
            "total_ranks": total_ranks,
            "tasks_per_node": tasks_per_node,
            "walltime": walltime,
        },
        "topology_sha256": sha256(destination / "mon.rings-nearby.parm7"),
        "master_cpin_sha256": sha256(destination / "mon.rings-nearby.cpin"),
    }
    (destination / "provenance.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ph-start", type=float, default=2.5)
    parser.add_argument("--ph-stop", type=float, default=12.0)
    parser.add_argument("--ph-step", type=float, default=0.5)
    parser.add_argument("--production-ns", type=float, default=1.0)
    parser.add_argument("--partition", default="small")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--ranks-per-replica", type=int, default=2)
    parser.add_argument("--walltime", default="48:00:00")
    parser.add_argument(
        "--seed-mode",
        choices=("equilibrated", "final-remd"),
        default="equilibrated",
    )
    parser.add_argument("--launcher", choices=("mpirun", "srun"), default="mpirun")
    args = parser.parse_args()
    count_float = (args.ph_stop - args.ph_start) / args.ph_step
    count = int(round(count_float))
    if not math.isclose(count_float, count) or count < 1:
        raise ValueError("pH range must be an integer number of positive pH steps")
    ph_values = tuple(args.ph_start + index * args.ph_step for index in range(count + 1))
    prepare_expanded_ladder(
        args.source_run.resolve(),
        args.output_dir.resolve(),
        ph_values,
        args.production_ns,
        args.partition,
        args.nodes,
        args.ranks_per_replica,
        args.walltime,
        args.seed_mode,
        args.launcher,
    )
    print(f"OK: prepared expanded pH ladder in {args.output_dir.resolve()}")
    print(
        f"OK: {len(ph_values)} replicas, pH {ph_values[0]:.1f}-{ph_values[-1]:.1f}, "
        f"{args.ranks_per_replica * len(ph_values)} MPI ranks"
    )


if __name__ == "__main__":
    main()
