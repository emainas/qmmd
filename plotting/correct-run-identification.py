#!/usr/bin/env python3
"""Screen metadynamics runs for diffusion and Grotthuss proton hopping.

The screening is deliberately hierarchical.  A run must first show a sustained
low-to-recovered transition in the biased N--H coordination.  Only runs that
pass that inexpensive test have their Mulliken files scanned for sustained
changes in the defect-carrying solvent oxygen.
"""

from __future__ import annotations

import argparse
import csv
import mmap
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence


TIME_RE = re.compile(rb"\*\*\* AT T=\s*([0-9.]+)\s*FSEC")
COORD_RE = re.compile(rb"Coordinate\s*=\s*([+-]?[0-9.]+)")
MULLIKEN_RECORD_RE = re.compile(
    rb"(?:\*\*\* AT T=\s*([0-9.]+)\s*FSEC)"
    rb"|(?:^\s*(\d+)\s+O\s+([sp])\s+"
    rb"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?))",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ScreenResult:
    run_id: int
    status: str
    diffusive: bool
    deprotonated_start_ps: float | None
    diffusive_start_ps: float | None
    proton_hopping: bool
    hop_count: int
    defect_oxygen_ids: tuple[int, ...]
    coordination_samples: int
    mulliken_frames: int
    note: str = ""


def parse_run_ids(value: str) -> set[int]:
    """Parse comma-separated positive run IDs and inclusive ranges."""
    run_ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("run IDs must not contain empty entries")
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"invalid run ID range: {item!r}")
            try:
                start, end = (int(part.strip()) for part in parts)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid run ID range: {item!r}"
                ) from exc
            if start < 1 or end < 1:
                raise argparse.ArgumentTypeError("run IDs must be positive integers")
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"run ID range must be ascending: {item!r}"
                )
            run_ids.update(range(start, end + 1))
        else:
            try:
                run_id = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid run ID: {item!r}") from exc
            if run_id < 1:
                raise argparse.ArgumentTypeError("run IDs must be positive integers")
            run_ids.add(run_id)
    return run_ids


def resolve_runs_root(runs_path: Path, cv_dir: str, run_ids: set[int]) -> Path:
    """Resolve a directory containing run-* from either it or an ancestor path."""
    requested_names = {f"run-{run_id}" for run_id in run_ids}
    direct = [p for p in runs_path.iterdir() if p.is_dir() and p.name in requested_names]
    if direct:
        return runs_path

    parents = {
        path.parent
        for path in runs_path.rglob("run-*")
        if path.is_dir()
        and path.name in requested_names
        and (path / cv_dir).is_dir()
    }
    if not parents:
        raise SystemExit(
            f"No requested run-* directories containing {cv_dir!r} found under {runs_path}"
        )
    if len(parents) > 1:
        choices = ", ".join(str(path) for path in sorted(parents))
        raise SystemExit(
            "Multiple run roots match the request; pass one of these as --runs-path: "
            + choices
        )
    return parents.pop()


def parse_coordination(path: Path) -> tuple[list[float], list[float]]:
    """Read biaspot timestamps in ps and their following coordination values."""
    times: list[float] = []
    values: list[float] = []
    pending_time: float | None = None
    with path.open("rb") as handle:
        for line in handle:
            match = TIME_RE.search(line)
            if match:
                pending_time = float(match.group(1)) / 1000.0
                continue
            if pending_time is None:
                continue
            match = COORD_RE.search(line)
            if match:
                times.append(pending_time)
                values.append(float(match.group(1)))
                pending_time = None
    return times, values


def first_sustained_start(
    times: Sequence[float],
    values: Sequence[float],
    predicate: Callable[[float], bool],
    persistence_ps: float,
    start_index: int = 0,
) -> int | None:
    """Find the first continuously qualifying interval of the requested duration."""
    run_start: int | None = None
    for index in range(start_index, len(times)):
        if not predicate(values[index]):
            run_start = None
            continue
        if run_start is None:
            run_start = index
        if times[index] - times[run_start] >= persistence_ps:
            return run_start
    return None


def detect_diffusive_regime(
    times: Sequence[float],
    coordination: Sequence[float],
    deprotonated_s_max: float,
    returned_s_min: float,
    deprotonation_persistence_ps: float,
    recovery_persistence_ps: float,
) -> tuple[int, int] | None:
    """Require sustained deprotonation followed by sustained N--H recovery."""
    deprotonated = first_sustained_start(
        times,
        coordination,
        lambda value: value <= deprotonated_s_max,
        deprotonation_persistence_ps,
    )
    if deprotonated is None:
        return None
    returned = first_sustained_start(
        times,
        coordination,
        lambda value: value >= returned_s_min,
        recovery_persistence_ps,
        start_index=deprotonated + 1,
    )
    if returned is None:
        return None
    return deprotonated, returned


def read_coordinate_symbols(dftb_inp: Path) -> list[str]:
    """Read atom symbols from the DFTB coordinate block."""
    lines = dftb_inp.read_text(encoding="utf-8", errors="ignore").splitlines()
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            natoms = int(parts[0])
            int(parts[1])
            int(parts[2])
        except ValueError:
            continue
        symbols: list[str] = []
        for atom_line in lines[index + 1 : index + 1 + natoms]:
            atom_parts = atom_line.split()
            if len(atom_parts) < 4:
                break
            symbols.append(atom_parts[0].upper())
        if len(symbols) == natoms:
            return symbols
    raise ValueError(f"Could not read the coordinate block from {dftb_inp}")


def infer_solvent_oxygen_ids(dftb_inp: Path) -> set[int]:
    """Infer the first water O-H-H block and select later oxygen atom IDs."""
    symbols = read_coordinate_symbols(dftb_inp)
    solvent_start: int | None = None
    for index in range(len(symbols) - 8):
        if symbols[index : index + 9] == ["O", "H", "H"] * 3:
            solvent_start = index
            break
    if solvent_start is None:
        raise ValueError(
            f"Could not infer a solvent O-H-H block from coordinates in {dftb_inp}"
        )
    return {
        atom_id
        for atom_id, symbol in enumerate(symbols, start=1)
        if atom_id > solvent_start and symbol == "O"
    }


def _selected_defect_id(
    populations: dict[int, float],
    defect_charge_min: float,
    defect_charge_max: float,
) -> int | None:
    candidates = [
        (population, atom_id)
        for atom_id, population in populations.items()
        if defect_charge_min <= population <= defect_charge_max
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def iter_mulliken_defects(
    path: Path,
    solvent_oxygen_ids: set[int],
    defect_charge_min: float,
    defect_charge_max: float,
    start_time_ps: float = 0.0,
    trajectory_end_ps: float | None = None,
) -> Iterator[tuple[float, int | None]]:
    """Stream frame times and Mulliken-assigned solvent defect oxygen IDs."""
    current_time: float | None = None
    populations: dict[int, float] = {}
    with path.open("rb") as handle:
        offset = 0
        if start_time_ps > 0.0 and trajectory_end_ps is not None and trajectory_end_ps > 0.0:
            # Mulliken frames have nearly uniform byte sizes.  Seek to a
            # conservative point before the target, then synchronize using the
            # exact timestamps below.  This avoids multi-GB pre-event scans.
            fraction = min(1.0, start_time_ps / trajectory_end_ps)
            offset = int(path.stat().st_size * max(0.0, fraction - 0.05))
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as data:
            for match in MULLIKEN_RECORD_RE.finditer(data, offset):
                raw_time = match.group(1)
                if raw_time is not None:
                    if current_time is not None and current_time >= start_time_ps:
                        yield current_time, _selected_defect_id(
                            populations, defect_charge_min, defect_charge_max
                        )
                    current_time = float(raw_time) / 1000.0
                    populations = {}
                    continue
                if current_time is None:
                    continue
                atom_id = int(match.group(2))
                if atom_id not in solvent_oxygen_ids:
                    continue
                populations[atom_id] = populations.get(atom_id, 0.0) + float(
                    match.group(4)
                )
    if current_time is not None and current_time >= start_time_ps:
        yield current_time, _selected_defect_id(
            populations, defect_charge_min, defect_charge_max
        )


def stable_defect_states(
    frames: Iterator[tuple[float, int | None]],
    persistence_ps: float,
) -> tuple[list[int], int]:
    """Return sustained identities through the first confirmed hop."""
    states: list[int] = []
    candidate_id: int | None = None
    candidate_start: float | None = None
    frame_count = 0
    for time_ps, defect_id in frames:
        frame_count += 1
        if defect_id is None:
            candidate_id = None
            candidate_start = None
            continue
        if defect_id != candidate_id:
            candidate_id = defect_id
            candidate_start = time_ps
            continue
        if candidate_start is None or time_ps - candidate_start < persistence_ps:
            continue
        if not states or states[-1] != defect_id:
            states.append(defect_id)
            if len(states) == 2:
                break
    return states, frame_count


def screen_run(
    run_id: int,
    run_dir: Path,
    cv_dir: str,
    deprotonated_s_max: float,
    returned_s_min: float,
    deprotonation_persistence_ps: float,
    recovery_persistence_ps: float,
    defect_charge_min: float,
    defect_charge_max: float,
    hop_persistence_ps: float,
) -> ScreenResult:
    cv_path = run_dir / cv_dir
    biaspot = cv_path / "biaspot"
    mulliken = cv_path / "mulliken"
    dftb_inp = cv_path / "dftb.inp"
    missing = [str(path.name) for path in (biaspot, mulliken, dftb_inp) if not path.exists()]
    if missing:
        return ScreenResult(
            run_id, "missing-input", False, None, None, False, 0, (), 0, 0,
            f"missing {', '.join(missing)}",
        )

    try:
        times, coordination = parse_coordination(biaspot)
    except (OSError, ValueError) as exc:
        return ScreenResult(
            run_id, "error", False, None, None, False, 0, (), 0, 0,
            f"coordination read failed: {exc}",
        )
    diffusion = detect_diffusive_regime(
        times,
        coordination,
        deprotonated_s_max,
        returned_s_min,
        deprotonation_persistence_ps,
        recovery_persistence_ps,
    )
    if diffusion is None:
        return ScreenResult(
            run_id, "no-diffusive-regime", False, None, None, False, 0, (),
            len(coordination), 0,
        )

    deprotonated_index, returned_index = diffusion
    try:
        solvent_oxygen_ids = infer_solvent_oxygen_ids(dftb_inp)
        states, frame_count = stable_defect_states(
            iter_mulliken_defects(
                mulliken,
                solvent_oxygen_ids,
                defect_charge_min,
                defect_charge_max,
                start_time_ps=times[deprotonated_index],
                trajectory_end_ps=times[-1],
            ),
            hop_persistence_ps,
        )
    except (OSError, ValueError) as exc:
        return ScreenResult(
            run_id, "error", True, times[deprotonated_index], times[returned_index],
            False, 0, (), len(coordination), 0, f"Mulliken read failed: {exc}",
        )

    hop_count = sum(left != right for left, right in zip(states, states[1:]))
    proton_hopping = hop_count > 0
    status = "candidate" if proton_hopping else "no-proton-hopping"
    return ScreenResult(
        run_id,
        status,
        True,
        times[deprotonated_index],
        times[returned_index],
        proton_hopping,
        hop_count,
        tuple(dict.fromkeys(states)),
        len(coordination),
        frame_count,
    )


def write_results(path: Path, results: Sequence[ScreenResult]) -> None:
    """Write the screening evidence for every requested run."""
    fieldnames = [field for field in ScreenResult.__dataclass_fields__]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result.__dict__)
            row["defect_oxygen_ids"] = ";".join(map(str, result.defect_oxygen_ids))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Identify candidate runs with a diffusive N-H coordination regime "
            "and Mulliken-detected proton hopping."
        )
    )
    parser.add_argument("--runs-path", required=True, type=Path)
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--run-ids", required=True, type=parse_run_ids, metavar="IDS")
    parser.add_argument("--deprotonated-s-max", type=float, default=0.05)
    parser.add_argument("--returned-s-min", type=float, default=0.20)
    parser.add_argument("--deprotonation-persistence-ps", type=float, default=0.25)
    parser.add_argument("--recovery-persistence-ps", type=float, default=0.05)
    parser.add_argument("--defect-charge-min", type=float, default=-0.625)
    parser.add_argument("--defect-charge-max", type=float, default=-0.525)
    parser.add_argument("--hop-persistence-ps", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.deprotonated_s_max >= args.returned_s_min:
        parser.error("--deprotonated-s-max must be less than --returned-s-min")
    if args.defect_charge_min >= args.defect_charge_max:
        parser.error("--defect-charge-min must be less than --defect-charge-max")
    if (
        args.deprotonation_persistence_ps < 0
        or args.recovery_persistence_ps < 0
        or args.hop_persistence_ps < 0
    ):
        parser.error("persistence times must be non-negative")

    runs_root = resolve_runs_root(args.runs_path, args.cv_dir, args.run_ids)
    print(f"Resolved run root: {runs_root}")
    results: list[ScreenResult] = []
    for run_id in sorted(args.run_ids):
        print(f"Screening run-{run_id} ...", flush=True)
        result = screen_run(
            run_id,
            runs_root / f"run-{run_id}",
            args.cv_dir,
            args.deprotonated_s_max,
            args.returned_s_min,
            args.deprotonation_persistence_ps,
            args.recovery_persistence_ps,
            args.defect_charge_min,
            args.defect_charge_max,
            args.hop_persistence_ps,
        )
        results.append(result)
        print(f"  {result.status}", flush=True)

    out = args.out
    if out is None:
        out = Path("reports") / f"{args.cv_dir}_correct_run_identification.csv"
    write_results(out, results)
    candidates = [result.run_id for result in results if result.status == "candidate"]
    diffusive = [result.run_id for result in results if result.diffusive]
    print(f"Diffusive-regime runs ({len(diffusive)}): {diffusive}")
    print(f"Candidate runs with proton hopping ({len(candidates)}): {candidates}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
  
