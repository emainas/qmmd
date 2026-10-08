from __future__ import annotations

import csv
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from qmmd.cphmd_dgref import find_repo_root, read_charge_sets, select_charge_states
from qmmd.cphmd_titr_post import (
    TitrPostConfig,
    load_titr_post_config,
    post_dir,
    run_cpptraj,
    validate_completed_titration,
)


_ROUNDTRIP_RE = re.compile(
    r"^(\d+)\s+(\d+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+(\d+)\s+(\d+)\s*$"
)
_LIFETIME_NAME_RE = re.compile(r"Rep(\d+),Crd(\d+)$")


@dataclass(frozen=True)
class TitrReportConfig:
    post: TitrPostConfig
    output_dir: str
    residue_id: int
    reference_pka: float | None
    fit_hill: bool
    discard_time_ps: float
    block_count: int
    rolling_window_records: int
    slope_window_exchanges: int
    style: Path
    syn_anti: SynAntiConfig | None
    composite: CompositeReportConfig | None


@dataclass(frozen=True)
class SynAntiConfig:
    syn_names: tuple[str, ...]
    anti_names: tuple[str, ...]
    syn_states: tuple[int, ...]
    anti_states: tuple[int, ...]


@dataclass(frozen=True)
class TpsDecompositionConfig:
    residue_id: int
    residue_name: str
    protonated_name: str
    bpp_name: str
    cpp_name: str
    protonated_state: int
    bpp_state: int
    cpp_state: int


@dataclass(frozen=True)
class PrxDecompositionConfig:
    residue_ids: tuple[int, int]
    residue_name: str
    deprotonated_name: str
    syn_names: tuple[str, ...]
    anti_names: tuple[str, ...]
    deprotonated_state: int
    syn_states: tuple[int, ...]
    anti_states: tuple[int, ...]


@dataclass(frozen=True)
class CompositeReportConfig:
    tps: TpsDecompositionConfig | None
    prx: PrxDecompositionConfig | None


@dataclass(frozen=True)
class ProtonationRecord:
    record: int
    md_step: int
    ph: float
    residue: str
    residue_id: int
    state: int
    proton_count: int


@dataclass(frozen=True)
class HHFit:
    pka: float
    hill: float


@dataclass(frozen=True)
class EdgeAcceptance:
    source: int
    target: int
    forward_percent: float
    reverse_percent: float


@dataclass(frozen=True)
class SynAntiPopulation:
    ph: float
    total_records: int
    protonated_count: int
    syn_count: int
    anti_count: int
    syn_fraction: float
    anti_fraction: float
    syn_to_anti_ratio: float


def _child_path(value: object, field: str, default: str) -> str:
    text = default if value is None else str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{field} must be a child path")
    return text


def _name_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list of state names")
    names = tuple(str(item).strip() for item in value)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{field} must contain unique nonempty state names")
    return names


def _site_state_lookup(
    post: TitrPostConfig,
    residue_id: int,
    expected_name: str,
    field: str,
) -> tuple[dict[str, int], tuple[int, ...]]:
    matches = [site for site in post.titr.sites if site.residue_number == residue_id]
    if len(matches) != 1:
        raise ValueError(f"{field}.residue_id must identify exactly one cpin.sites entry")
    site = matches[0]
    if site.residue_name != expected_name:
        raise ValueError(
            f"{field} expects residue {expected_name}:{residue_id}, "
            f"but cpin.sites defines {site.residue_name}:{residue_id}"
        )
    charges = select_charge_states(read_charge_sets(site.charge_sets), site.states)
    return (
        {name: index for index, name in enumerate(charges.state_names)},
        charges.proton_counts,
    )


def _load_composite_report(
    value: object,
    post: TitrPostConfig,
) -> CompositeReportConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("titration_report.composite must be a YAML mapping")
    if not post.titr.sites:
        raise ValueError("titration_report.composite requires cpin.sites")

    tps: TpsDecompositionConfig | None = None
    tps_raw = value.get("tps")
    if tps_raw is not None:
        if not isinstance(tps_raw, dict):
            raise ValueError("titration_report.composite.tps must be a YAML mapping")
        tps_id = int(tps_raw["residue_id"])
        tps_name = str(tps_raw.get("residue_name", "TPS")).strip()
        protonated_name = str(tps_raw.get("protonated_state", "ppp")).strip()
        bpp_name = str(tps_raw.get("bpp_state", "bpp")).strip()
        cpp_name = str(tps_raw.get("cpp_state", "cpp")).strip()
        tps_lookup, tps_counts = _site_state_lookup(
            post, tps_id, tps_name, "titration_report.composite.tps"
        )
        tps_names = {protonated_name, bpp_name, cpp_name}
        unknown = sorted(tps_names - tps_lookup.keys())
        if unknown:
            raise ValueError("Unknown TPS decomposition states: " + ", ".join(unknown))
        if tps_names != set(tps_lookup):
            raise ValueError("TPS decomposition must account for every selected TPS state")
        if not (
            tps_counts[tps_lookup[protonated_name]]
            > tps_counts[tps_lookup[bpp_name]]
            == tps_counts[tps_lookup[cpp_name]]
        ):
            raise ValueError(
                "TPS decomposition requires PPP protonated and BPP/CPP deprotonated"
            )
        tps = TpsDecompositionConfig(
            residue_id=tps_id,
            residue_name=tps_name,
            protonated_name=protonated_name,
            bpp_name=bpp_name,
            cpp_name=cpp_name,
            protonated_state=tps_lookup[protonated_name],
            bpp_state=tps_lookup[bpp_name],
            cpp_state=tps_lookup[cpp_name],
        )

    prx: PrxDecompositionConfig | None = None
    prx_raw = value.get("prx")
    if prx_raw is not None:
        if not isinstance(prx_raw, dict):
            raise ValueError("titration_report.composite.prx must be a YAML mapping")
        raw_ids = prx_raw.get("residue_ids")
        if not isinstance(raw_ids, list) or len(raw_ids) != 2:
            raise ValueError(
                "titration_report.composite.prx.residue_ids must contain two IDs"
            )
        prx_ids = (int(raw_ids[0]), int(raw_ids[1]))
        if prx_ids[0] == prx_ids[1] or min(prx_ids) < 1:
            raise ValueError("PRX residue IDs must be distinct and one-based")
        prx_name = str(prx_raw.get("residue_name", "PRX")).strip()
        deprot_name = str(prx_raw.get("deprotonated_state", "deprot")).strip()
        syn_names = _name_list(prx_raw.get("syn_states"), "composite.prx.syn_states")
        anti_names = _name_list(prx_raw.get("anti_states"), "composite.prx.anti_states")
        if set(syn_names) & set(anti_names):
            raise ValueError("PRX syn and anti state groups must not overlap")
        lookups: list[dict[str, int]] = []
        counts_list: list[tuple[int, ...]] = []
        for residue_id in prx_ids:
            lookup, counts = _site_state_lookup(
                post, residue_id, prx_name, "titration_report.composite.prx"
            )
            lookups.append(lookup)
            counts_list.append(counts)
        if lookups[0] != lookups[1] or counts_list[0] != counts_list[1]:
            raise ValueError("Both PRX sites must use identical selected state definitions")
        prx_lookup = lookups[0]
        configured = {deprot_name, *syn_names, *anti_names}
        unknown = sorted(configured - prx_lookup.keys())
        if unknown:
            raise ValueError("Unknown PRX decomposition states: " + ", ".join(unknown))
        if configured != set(prx_lookup):
            raise ValueError("PRX decomposition must account for every selected PRX state")
        counts = counts_list[0]
        deprot_state = prx_lookup[deprot_name]
        protonated = {prx_lookup[name] for name in (*syn_names, *anti_names)}
        if counts[deprot_state] != min(counts) or protonated != {
            index for index, count in enumerate(counts) if count == max(counts)
        }:
            raise ValueError("PRX syn/anti groups must partition all protonated states")
        prx = PrxDecompositionConfig(
            residue_ids=prx_ids,
            residue_name=prx_name,
            deprotonated_name=deprot_name,
            syn_names=syn_names,
            anti_names=anti_names,
            deprotonated_state=deprot_state,
            syn_states=tuple(prx_lookup[name] for name in syn_names),
            anti_states=tuple(prx_lookup[name] for name in anti_names),
        )

    if tps is None and prx is None:
        raise ValueError("titration_report.composite must define tps, prx, or both")
    return CompositeReportConfig(tps=tps, prx=prx)


def load_titr_report_config(yaml_path: Path) -> TitrReportConfig:
    resolved = yaml_path.resolve()
    post = load_titr_post_config(resolved)
    data = yaml.safe_load(resolved.read_text())
    raw = data.get("titration_report", {})
    if not isinstance(raw, dict):
        raise ValueError("titration_report must be a YAML mapping")

    output_dir = _child_path(
        raw.get("output_dir"), "titration_report.output_dir", "report"
    )
    residue_id = int(raw.get("residue_id", 1))
    if residue_id < 1:
        raise ValueError("titration_report.residue_id must be one-based and positive")

    reference = raw.get("reference_pka")
    reference_pka = None if reference is None else float(reference)
    fit_hill = bool(raw.get("fit_hill", True))
    discard_time_ps = float(raw.get("discard_time_ps", 0.0))
    block_count = int(raw.get("block_count", 5))
    rolling_window_records = int(raw.get("rolling_window_records", 25))
    slope_window = int(raw.get("slope_window_exchanges", 10))
    if discard_time_ps < 0:
        raise ValueError("titration_report.discard_time_ps must be non-negative")
    if block_count < 2:
        raise ValueError("titration_report.block_count must be at least 2")
    if rolling_window_records < 1:
        raise ValueError("titration_report.rolling_window_records must be positive")
    numexchg = int(post.titr.cntrl["numexchg"])
    if slope_window < 2 or slope_window >= numexchg:
        raise ValueError(
            "titration_report.slope_window_exchanges must be at least 2 and "
            "smaller than remd_mdin.cntrl.numexchg"
        )

    style_value = raw.get("style", "plotting/lefteris.mplstyle")
    style = Path(str(style_value))
    if not style.is_absolute():
        style = find_repo_root(resolved) / style
    if not style.is_file():
        raise FileNotFoundError(f"Missing matplotlib style: {style}")

    syn_anti_raw = raw.get("syn_anti")
    syn_anti: SynAntiConfig | None = None
    if syn_anti_raw is not None:
        if not isinstance(syn_anti_raw, dict):
            raise ValueError("titration_report.syn_anti must be a YAML mapping")
        raw_syn = syn_anti_raw.get("syn_states")
        raw_anti = syn_anti_raw.get("anti_states")
        if not isinstance(raw_syn, list) or not raw_syn:
            raise ValueError("titration_report.syn_anti.syn_states must be a nonempty list")
        if not isinstance(raw_anti, list) or not raw_anti:
            raise ValueError("titration_report.syn_anti.anti_states must be a nonempty list")
        syn_names = tuple(str(value).strip() for value in raw_syn)
        anti_names = tuple(str(value).strip() for value in raw_anti)
        if any(not value for value in (*syn_names, *anti_names)):
            raise ValueError("Syn/anti state names must be nonempty")
        if set(syn_names) & set(anti_names):
            raise ValueError("Syn and anti state groups must not overlap")

        charges = read_charge_sets(post.titr.charge_sets)
        state_lookup = {name: index for index, name in enumerate(charges.state_names)}
        unknown = sorted((set(syn_names) | set(anti_names)) - state_lookup.keys())
        if unknown:
            raise ValueError("Unknown syn/anti charge states: " + ", ".join(unknown))
        proton_counts = charges.proton_counts or (
            charges.proton_count_prot,
            charges.proton_count_deprot,
        )
        protonated_states = {
            index for index, count in enumerate(proton_counts) if count == max(proton_counts)
        }
        configured_states = {
            state_lookup[name] for name in (*syn_names, *anti_names)
        }
        if configured_states != protonated_states:
            raise ValueError(
                "Syn and anti groups must partition every protonated charge state"
            )
        syn_anti = SynAntiConfig(
            syn_names=syn_names,
            anti_names=anti_names,
            syn_states=tuple(state_lookup[name] for name in syn_names),
            anti_states=tuple(state_lookup[name] for name in anti_names),
        )

    composite = _load_composite_report(raw.get("composite"), post)

    return TitrReportConfig(
        post=post,
        output_dir=output_dir,
        residue_id=residue_id,
        reference_pka=reference_pka,
        fit_hill=fit_hill,
        discard_time_ps=discard_time_ps,
        block_count=block_count,
        rolling_window_records=rolling_window_records,
        slope_window_exchanges=slope_window,
        style=style,
        syn_anti=syn_anti,
        composite=composite,
    )


def report_dir(cfg: TitrReportConfig, repo_root: Path) -> Path:
    return post_dir(cfg.post, repo_root) / cfg.output_dir


def render_remlog_cpptraj_input(cfg: TitrReportConfig, rem_log: Path) -> str:
    window = cfg.slope_window_exchanges
    return "\n".join(
        [
            f"readdata {rem_log} as remlog",
            "remlog rem.log out walker-replica-index.dat crdidx stats "
            "statsout roundtrip-stats.dat printtrips reptime residence-percent.dat "
            "lifetime residence-lifetime.dat "
            f"reptimeslope {window} reptimeslopeout residence-slope.dat "
            "acceptout exchange-acceptance.dat name CPHREMD",
            "remlog rem.log out replica-walker-index.dat repidx name CPHREMD_REPIDX",
            "runanalysis",
            "quit",
            "",
        ]
    )


def _read_numeric_table(path: Path) -> tuple[list[str], np.ndarray]:
    header: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped.lstrip("#").split()
            continue
        rows.append([float(value) for value in stripped.split()])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Malformed numeric table: {path}")
    return header, array


def parse_roundtrip_stats(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("#CRDIDX"):
            in_table = True
            continue
        if not in_table:
            continue
        match = _ROUNDTRIP_RE.match(line.strip())
        if match:
            rows.append([float(value) for value in match.groups()])
    if not rows:
        raise ValueError(f"No round-trip summary found in {path}")
    return np.asarray(rows, dtype=float)


def parse_edge_acceptance(path: Path, replica_count: int) -> list[EdgeAcceptance]:
    _, data = _read_numeric_table(path)
    if data.shape != (replica_count, 3):
        raise ValueError(f"Expected {replica_count} acceptance rows in {path}")
    edges: list[EdgeAcceptance] = []
    for source in range(1, replica_count + 1):
        target = source + 1 if source < replica_count else 1
        forward = float(data[source - 1, 1])
        reverse = float(data[target - 1, 2])
        edges.append(EdgeAcceptance(source, target, forward, reverse))
    return edges


def read_protonation_records(path: Path) -> list[ProtonationRecord]:
    records: list[ProtonationRecord] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                ProtonationRecord(
                    record=int(row["record"]),
                    md_step=int(row["md_step"]),
                    ph=float(row["pH"]),
                    residue=row["residue"],
                    residue_id=int(row["residue_id"]),
                    state=int(row["state"]),
                    proton_count=int(row["proton_count"]),
                )
            )
    if not records:
        raise ValueError(f"No protonation records found in {path}")
    return records


def fit_henderson_hasselbalch(
    ph: np.ndarray, fractions: np.ndarray, fit_hill: bool = True
) -> HHFit:
    mask = np.isfinite(ph) & np.isfinite(fractions) & (fractions > 0.0) & (fractions < 1.0)
    x = ph[mask]
    y = np.log10((1.0 - fractions[mask]) / fractions[mask])
    minimum = 2 if fit_hill else 1
    if x.size < minimum:
        raise ValueError("Too few non-endpoint fractions for a Henderson-Hasselbalch fit")
    if fit_hill:
        hill, intercept = np.polyfit(x, y, 1)
        if hill <= 0.0:
            raise ValueError(f"Fitted Hill coefficient is not positive: {hill:.6g}")
    else:
        hill = 1.0
        intercept = float(np.mean(y - x))
    return HHFit(pka=float(-intercept / hill), hill=float(hill))


def hh_fraction(ph: np.ndarray, fit: HHFit) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, fit.hill * (ph - fit.pka)))


def bhattacharyya_overlap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("State-probability vectors must be one-dimensional and aligned")
    if not np.isclose(a.sum(), 1.0) or not np.isclose(b.sum(), 1.0):
        raise ValueError("State-probability vectors must each sum to one")
    return float(np.sum(np.sqrt(a * b)))


def _validate_postprocessed_data(cfg: TitrReportConfig, repo_root: Path) -> Path:
    validate_completed_titration(cfg.post, repo_root)
    source = post_dir(cfg.post, repo_root)
    required = [
        source / "postprocess.yaml",
        source / "trajectory-manifest.csv",
        source / "protonation-states.csv",
    ]
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Run cphmd-titr-post first; missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    metadata = yaml.safe_load((source / "postprocess.yaml").read_text())
    if [float(value) for value in metadata.get("pH_values", [])] != list(
        cfg.post.titr.ph_values
    ):
        raise ValueError("Postprocessed pH ladder does not match the current configuration")

    with (source / "trajectory-manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != len(cfg.post.titr.ph_values):
        raise ValueError("Fixed-pH trajectory manifest has the wrong replica count")
    for row, expected_ph in zip(manifest, cfg.post.titr.ph_values):
        trajectory = source / row["trajectory"]
        if not trajectory.is_file() or trajectory.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty fixed-pH trajectory: {trajectory}")
        if not math.isclose(float(row["pH"]), expected_ph, abs_tol=1.0e-6):
            raise ValueError("Fixed-pH trajectory manifest is not ordered by the pH ladder")
    return source


def _selected_series(
    cfg: TitrReportConfig, records: list[ProtonationRecord]
) -> tuple[str, dict[float, list[ProtonationRecord]], int]:
    selected = [record for record in records if record.residue_id == cfg.residue_id]
    if not selected:
        raise ValueError(f"Residue ID {cfg.residue_id} is absent from protonation-states.csv")
    names = {record.residue for record in selected}
    if len(names) != 1:
        raise ValueError(f"Residue ID {cfg.residue_id} is ambiguous: {sorted(names)}")
    max_protons = max(record.proton_count for record in selected)
    dt = float(cfg.post.titr.cntrl["dt"])
    grouped: dict[float, list[ProtonationRecord]] = {}
    for record in selected:
        time_ps = record.md_step * dt
        if time_ps > cfg.discard_time_ps + 1.0e-12:
            grouped.setdefault(record.ph, []).append(record)
    expected = list(cfg.post.titr.ph_values)
    if sorted(grouped) != expected:
        raise ValueError("Filtered protonation records do not cover the configured pH ladder")
    return next(iter(names)), grouped, max_protons


def _analyze_titration(
    cfg: TitrReportConfig,
    grouped: dict[float, list[ProtonationRecord]],
    max_protons: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HHFit, list[HHFit | None]]:
    ph = np.asarray(cfg.post.titr.ph_values, dtype=float)
    states = [
        np.asarray([record.proton_count == max_protons for record in grouped[value]], dtype=float)
        for value in ph
    ]
    fraction = np.asarray([values.mean() for values in states])
    transitions = np.asarray([np.count_nonzero(np.diff(values)) for values in states], dtype=int)

    block_fractions = np.full((cfg.block_count, len(ph)), np.nan)
    for column, values in enumerate(states):
        if len(values) < cfg.block_count:
            raise ValueError(
                f"Only {len(values)} records remain at pH {ph[column]:g}; "
                f"cannot form {cfg.block_count} blocks"
            )
        for block, indices in enumerate(np.array_split(np.arange(len(values)), cfg.block_count)):
            block_fractions[block, column] = float(values[indices].mean())
    sem = np.std(block_fractions, axis=0, ddof=1) / math.sqrt(cfg.block_count)
    fit = fit_henderson_hasselbalch(ph, fraction, cfg.fit_hill)
    block_fits: list[HHFit | None] = []
    for row in block_fractions:
        try:
            block_fits.append(fit_henderson_hasselbalch(ph, row, cfg.fit_hill))
        except ValueError:
            block_fits.append(None)
    return ph, fraction, sem, transitions, fit, block_fits


def _state_probabilities(
    ph: np.ndarray, grouped: dict[float, list[ProtonationRecord]]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    state_values = sorted({record.proton_count for records in grouped.values() for record in records})
    probabilities = np.zeros((len(ph), len(state_values)), dtype=float)
    for i, value in enumerate(ph):
        counts = np.asarray([record.proton_count for record in grouped[float(value)]])
        probabilities[i] = [np.mean(counts == state) for state in state_values]
    overlap = np.empty((len(ph), len(ph)), dtype=float)
    for i in range(len(ph)):
        for j in range(len(ph)):
            overlap[i, j] = bhattacharyya_overlap(probabilities[i], probabilities[j])
    return state_values, probabilities, overlap


def analyze_syn_anti(
    ph: np.ndarray,
    grouped: dict[float, list[ProtonationRecord]],
    syn_states: tuple[int, ...],
    anti_states: tuple[int, ...],
) -> list[SynAntiPopulation]:
    """Calculate syn/anti populations conditional on being protonated."""
    syn_set = set(syn_states)
    anti_set = set(anti_states)
    rows: list[SynAntiPopulation] = []
    for value in ph:
        records = grouped[float(value)]
        syn_count = sum(record.state in syn_set for record in records)
        anti_count = sum(record.state in anti_set for record in records)
        protonated_count = syn_count + anti_count
        if protonated_count:
            syn_fraction = syn_count / protonated_count
            anti_fraction = anti_count / protonated_count
        else:
            syn_fraction = math.nan
            anti_fraction = math.nan
        if anti_count:
            ratio = syn_count / anti_count
        elif syn_count:
            ratio = math.inf
        else:
            ratio = math.nan
        rows.append(
            SynAntiPopulation(
                ph=float(value),
                total_records=len(records),
                protonated_count=protonated_count,
                syn_count=syn_count,
                anti_count=anti_count,
                syn_fraction=syn_fraction,
                anti_fraction=anti_fraction,
                syn_to_anti_ratio=ratio,
            )
        )
    return rows


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _tidy_remlog_outputs(
    cfg: TitrReportConfig, work: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[EdgeAcceptance], np.ndarray, np.ndarray]:
    nrep = len(cfg.post.titr.ph_values)
    _, walker = _read_numeric_table(work / "walker-replica-index.dat")
    _, residence = _read_numeric_table(work / "residence-percent.dat")
    _, slope = _read_numeric_table(work / "residence-slope.dat")
    roundtrips = parse_roundtrip_stats(work / "roundtrip-stats.dat")
    edges = parse_edge_acceptance(work / "exchange-acceptance.dat", nrep)
    expected_exchanges = int(cfg.post.titr.cntrl["numexchg"])
    if walker.shape != (expected_exchanges, nrep + 1):
        raise ValueError(f"Unexpected CPPTRAJ walker-index shape: {walker.shape}")
    if residence.shape != (nrep, nrep + 1):
        raise ValueError(f"Unexpected CPPTRAJ residence shape: {residence.shape}")
    if roundtrips.shape != (nrep, 6):
        raise ValueError(f"Unexpected CPPTRAJ round-trip shape: {roundtrips.shape}")

    ph_values = cfg.post.titr.ph_values
    walker_rows: list[list[object]] = []
    for exchange_row in walker:
        exchange = int(exchange_row[0])
        for index, replica_index in enumerate(exchange_row[1:], start=1):
            walker_rows.append(
                [exchange, index, int(replica_index), f"{ph_values[int(replica_index) - 1]:.6f}"]
            )
    _write_csv(
        work / "walker-pH.csv",
        ["exchange", "walker", "replica_index", "pH"],
        walker_rows,
    )

    _write_csv(
        work / "exchange-acceptance.csv",
        [
            "source_replica",
            "source_pH",
            "target_replica",
            "target_pH",
            "forward_acceptance_percent",
            "reverse_acceptance_percent",
        ],
        [
            [
                edge.source,
                f"{ph_values[edge.source - 1]:.6f}",
                edge.target,
                f"{ph_values[edge.target - 1]:.6f}",
                f"{edge.forward_percent:.6f}",
                f"{edge.reverse_percent:.6f}",
            ]
            for edge in edges
        ],
    )
    _write_csv(
        work / "residence-percent.csv",
        ["pH", *[f"walker_{i}" for i in range(1, nrep + 1)]],
        [
            [f"{ph_values[int(row[0]) - 1]:.6f}", *[f"{value:.6f}" for value in row[1:]]]
            for row in residence
        ],
    )
    _write_csv(
        work / "roundtrip-summary.csv",
        ["walker", "round_trips", "mean_exchanges", "sd_exchanges", "min_exchanges", "max_exchanges"],
        [[int(v) if i in {0, 1, 4, 5} else f"{v:.6f}" for i, v in enumerate(row)] for row in roundtrips],
    )

    lifetime_rows: list[list[object]] = []
    lifetime_weighted = np.zeros(nrep, dtype=float)
    lifetime_frames = np.zeros(nrep, dtype=float)
    for line in (work / "residence-lifetime.dat").read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 6:
            continue
        match = _LIFETIME_NAME_RE.fullmatch(fields[-1])
        if match is None:
            continue
        replica, coordinate = (int(value) for value in match.groups())
        count = int(fields[1])
        maximum = int(fields[2])
        average = float(fields[3])
        frames = int(fields[4])
        lifetime_weighted[replica - 1] += frames
        lifetime_frames[replica - 1] += count
        lifetime_rows.append([replica, f"{ph_values[replica - 1]:.6f}", coordinate, count, maximum, average, frames])
    with np.errstate(divide="ignore", invalid="ignore"):
        lifetime_mean = np.divide(
            lifetime_weighted,
            lifetime_frames,
            out=np.zeros_like(lifetime_weighted),
            where=lifetime_frames > 0,
        )
    _write_csv(
        work / "residence-lifetime.csv",
        ["replica", "pH", "walker", "visits", "max_exchanges", "mean_exchanges", "total_exchanges"],
        lifetime_rows,
    )

    slope_rows: list[list[object]] = []
    for row in slope:
        for walker_index in range(nrep):
            slope_rows.append(
                [int(row[0]), walker_index + 1, f"{row[1 + 2 * walker_index]:.8f}", f"{row[2 + 2 * walker_index]:.8f}"]
            )
    _write_csv(
        work / "residence-slope.csv",
        ["exchange", "walker", "slope_percent_per_exchange", "correlation"],
        slope_rows,
    )
    return walker, residence, roundtrips, edges, lifetime_mean, slope


def _write_titration_outputs(
    cfg: TitrReportConfig,
    work: Path,
    grouped: dict[float, list[ProtonationRecord]],
    max_protons: int,
    ph: np.ndarray,
    fraction: np.ndarray,
    sem: np.ndarray,
    transitions: np.ndarray,
    fit: HHFit,
    block_fits: list[HHFit | None],
    overlap: np.ndarray,
) -> None:
    predicted = hh_fraction(ph, fit)
    _write_csv(
        work / "titration-fit.csv",
        ["pH", "fraction_protonated", "block_sem", "fitted_fraction", "residual", "records", "transitions"],
        [
            [f"{p:.6f}", f"{f:.8f}", f"{s:.8f}", f"{model:.8f}", f"{f-model:.8f}", len(grouped[float(p)]), int(t)]
            for p, f, s, model, t in zip(ph, fraction, sem, predicted, transitions)
        ],
    )
    _write_csv(
        work / "pka-blocks.csv",
        ["block", "pKa", "hill_coefficient"],
        [
            [index, "" if value is None else f"{value.pka:.8f}", "" if value is None else f"{value.hill:.8f}"]
            for index, value in enumerate(block_fits, start=1)
        ],
    )

    dt = float(cfg.post.titr.cntrl["dt"])
    time_rows: list[list[object]] = []
    for p in ph:
        series = grouped[float(p)]
        binary = np.asarray([record.proton_count == max_protons for record in series], dtype=float)
        width = cfg.rolling_window_records
        cumulative = np.cumsum(binary)
        rolling = np.empty_like(binary)
        for index in range(len(binary)):
            start = max(0, index + 1 - width)
            total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
            rolling[index] = total / (index + 1 - start)
        for record, state, value in zip(series, binary, rolling):
            time_rows.append(
                [f"{p:.6f}", record.record, f"{record.md_step * dt:.6f}", int(state), f"{value:.8f}"]
            )
    _write_csv(
        work / "protonation-timeseries.csv",
        ["pH", "record", "time_ps", "protonated", "rolling_fraction"],
        time_rows,
    )
    overlap_rows = [
        [f"{ph[i]:.6f}", f"{ph[j]:.6f}", f"{overlap[i, j]:.8f}"]
        for i in range(len(ph))
        for j in range(len(ph))
    ]
    _write_csv(
        work / "protonation-state-overlap.csv",
        ["pH_i", "pH_j", "bhattacharyya_coefficient"],
        overlap_rows,
    )


def _plot_replica_health(
    cfg: TitrReportConfig,
    path: Path,
    walker: np.ndarray,
    residence: np.ndarray,
    roundtrips: np.ndarray,
    edges: list[EdgeAcceptance],
    lifetime_mean: np.ndarray,
    slope: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    ph = np.asarray(cfg.post.titr.ph_values)
    nrep = len(ph)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, nrep))
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0), constrained_layout=True)

    ax = axes[0, 0]
    for index in range(nrep):
        y = ph[walker[:, index + 1].astype(int) - 1]
        ax.plot(walker[:, 0], y, color=colors[index], lw=0.9, alpha=0.78, label=f"W{index + 1}")
    ax.set(xlabel="Exchange attempt", ylabel="pH", title="Walker diffusion across the pH ladder")
    ax.set_yticks(ph)
    ax.legend(ncol=2, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))

    ax = axes[0, 1]
    image = ax.imshow(residence[:, 1:].T, aspect="auto", origin="lower", cmap="magma", vmin=0)
    ax.set_xticks(range(nrep), [f"{value:g}" for value in ph], rotation=45)
    ax.set_yticks(range(nrep), range(1, nrep + 1))
    ax.set(xlabel="pH replica", ylabel="Walker", title=f"Residence occupancy (ideal {100/nrep:.1f}%)")
    fig.colorbar(image, ax=ax, label="Time (%)")

    ax = axes[1, 0]
    positions = np.arange(nrep)
    labels = [
        f"{ph[e.source-1]:g}↔{ph[e.target-1]:g}" + (" (wrap)" if e.target == 1 else "")
        for e in edges
    ]
    ax.bar(positions - 0.19, [edge.forward_percent for edge in edges], 0.38, label="Forward")
    ax.bar(positions + 0.19, [edge.reverse_percent for edge in edges], 0.38, label="Reverse")
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set(xlabel="Exchange edge", ylabel="Accepted attempts (%)", title="Forward/reverse exchange acceptance")
    ax.legend()

    ax = axes[1, 1]
    walkers = roundtrips[:, 0].astype(int)
    ax.bar(walkers, roundtrips[:, 1], color=colors)
    ax.set_xticks(walkers)
    ax.set(xlabel="Walker", ylabel="Complete round trips", title=f"End-to-end round trips (total {int(roundtrips[:, 1].sum())})")

    ax = axes[2, 0]
    exchange_ps = int(cfg.post.titr.cntrl["nstlim"]) * float(cfg.post.titr.cntrl["dt"])
    ax.plot(ph, lifetime_mean * exchange_ps, marker="o")
    ax.set(xlabel="pH", ylabel="Mean residence lifetime (ps)", title="Residence lifetime by pH")
    ax.set_xticks(ph)

    ax = axes[2, 1]
    for index in range(nrep):
        ax.plot(slope[:, 0], slope[:, 1 + 2 * index], color=colors[index], lw=1.0, marker="o", ms=3)
    ax.axhline(0.0, color="#C43C39", ls="--", lw=1.2)
    ax.set(xlabel="Exchange attempt", ylabel="Residence slope", title=f"Occupancy convergence ({cfg.slope_window_exchanges}-exchange windows; target 0)")
    fig.suptitle(f"{cfg.post.titr.system} RECpHMD replica-exchange health", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_titration_curve(
    cfg: TitrReportConfig,
    path: Path,
    ph: np.ndarray,
    fraction: np.ndarray,
    sem: np.ndarray,
    fit: HHFit,
    block_fits: list[HHFit | None],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    fig, (ax, block_ax) = plt.subplots(2, 1, figsize=(7.5, 7.8), height_ratios=[2.2, 1.0], constrained_layout=True)
    dense = np.linspace(ph.min() - 0.2, ph.max() + 0.2, 400)
    ax.errorbar(ph, fraction, yerr=sem, fmt="o", capsize=3, label="RECpHMD mean ± block SEM")
    ax.plot(dense, hh_fraction(dense, fit), label=f"HH fit: pKa={fit.pka:.2f}, n={fit.hill:.2f}")
    ax.axvline(fit.pka, color="#1F3A5F", ls="--", lw=1.0)
    if cfg.reference_pka is not None:
        ax.axvline(cfg.reference_pka, color="#C43C39", ls=":", lw=1.5, label=f"Reference pKa={cfg.reference_pka:g}")
    ax.axhline(0.5, color="0.55", ls="--", lw=0.8)
    ax.set(xlabel="pH", ylabel="Protonated fraction", ylim=(-0.03, 1.03), title=f"{cfg.post.titr.system} titration curve")
    ax.legend()

    valid = [(index, value) for index, value in enumerate(block_fits, start=1) if value is not None]
    if valid:
        x = [index for index, _ in valid]
        pka = np.asarray([value.pka for _, value in valid])
        block_ax.plot(x, pka, marker="o", label=f"Block pKa: {pka.mean():.2f} ± {pka.std(ddof=1) if len(pka)>1 else 0:.2f} SD")
    block_ax.axhline(fit.pka, color="#1F3A5F", ls="--", label="Full-trajectory fit")
    if cfg.reference_pka is not None:
        block_ax.axhline(cfg.reference_pka, color="#C43C39", ls=":", label="Reference")
    block_ax.set(xlabel="Contiguous time block", ylabel="Fitted pKa")
    block_ax.set_xticks(range(1, cfg.block_count + 1))
    block_ax.legend(fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_protonation_timeseries(
    cfg: TitrReportConfig,
    path: Path,
    grouped: dict[float, list[ProtonationRecord]],
    max_protons: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    ph = list(cfg.post.titr.ph_values)
    columns = 2
    rows = math.ceil(len(ph) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.0, 2.35 * rows), sharex=True, sharey=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    dt = float(cfg.post.titr.cntrl["dt"])
    window = cfg.rolling_window_records
    for ax, value in zip(axes_array, ph):
        records = grouped[value]
        time = np.asarray([record.md_step * dt for record in records])
        binary = np.asarray([record.proton_count == max_protons for record in records], dtype=float)
        rolling = np.convolve(binary, np.ones(window), mode="full")[: len(binary)]
        rolling /= np.minimum(np.arange(1, len(binary) + 1), window)
        ax.step(time, binary, where="post", color="0.72", lw=0.7, alpha=0.65, label="State")
        ax.plot(time, rolling, color="#1F3A5F", lw=1.35, label=f"Rolling fraction ({window} records)")
        ax.axhline(binary.mean(), color="#C43C39", ls="--", lw=1.0, label=f"Mean={binary.mean():.3f}")
        ax.set_title(f"pH {value:g}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for ax in axes_array[len(ph):]:
        ax.set_visible(False)
    for ax in axes_array[-columns:]:
        ax.set_xlabel("Simulation time (ps)")
    for ax in axes_array[::columns]:
        ax.set_ylabel("Protonated / fraction")
    fig.suptitle(f"{cfg.post.titr.system} protonation sampling by fixed pH", fontsize=15)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_state_overlap(
    cfg: TitrReportConfig,
    path: Path,
    ph: np.ndarray,
    overlap: np.ndarray,
    edges: list[EdgeAcceptance],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    fig, (ax, edge_ax) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    image = ax.imshow(overlap, origin="lower", vmin=0, vmax=1, cmap="viridis")
    labels = [f"{value:g}" for value in ph]
    ax.set_xticks(range(len(ph)), labels, rotation=45)
    ax.set_yticks(range(len(ph)), labels)
    ax.set(xlabel="pH", ylabel="pH", title="Protonation-state distribution overlap")
    fig.colorbar(image, ax=ax, label="Bhattacharyya coefficient")

    positions = np.arange(len(edges))
    edge_overlap = [overlap[e.source - 1, e.target - 1] for e in edges]
    acceptance = [(e.forward_percent + e.reverse_percent) / 2.0 / 100.0 for e in edges]
    edge_labels = [
        f"{ph[e.source-1]:g}↔{ph[e.target-1]:g}" + (" (wrap)" if e.target == 1 else "")
        for e in edges
    ]
    edge_ax.plot(positions, edge_overlap, marker="o", label="State overlap")
    edge_ax.plot(positions, acceptance, marker="s", label="Mean exchange acceptance")
    edge_ax.set_xticks(positions, edge_labels, rotation=45, ha="right")
    edge_ax.set_ylim(-0.03, 1.03)
    edge_ax.set(xlabel="Exchange edge", ylabel="Coefficient / probability", title="Overlap versus exchange acceptance")
    edge_ax.legend()
    fig.suptitle("Discrete protonation-state overlap (not Hamiltonian energy overlap)", fontsize=14)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_syn_anti_populations(
    cfg: TitrReportConfig,
    path: Path,
    rows: list[SynAntiPopulation],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    ph = np.asarray([row.ph for row in rows])
    syn = np.asarray([row.syn_fraction for row in rows])
    anti = np.asarray([row.anti_fraction for row in rows])
    ratio = np.asarray([row.syn_to_anti_ratio for row in rows])
    protonated = np.asarray([row.protonated_count for row in rows])

    fig, (fraction_ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.6),
        sharex=True,
        height_ratios=[1.35, 1.0],
        constrained_layout=True,
    )
    fraction_ax.plot(ph, syn, marker="o", label="Syn / total protonated")
    fraction_ax.plot(ph, anti, marker="s", label="Anti / total protonated")
    fraction_ax.axhline(0.5, color="0.55", ls="--", lw=0.9)
    fraction_ax.set(
        ylabel="Conditional molar fraction",
        ylim=(-0.03, 1.03),
        title="Syn and anti populations within the protonated ensemble",
    )
    fraction_ax.legend()

    finite = np.isfinite(ratio)
    ratio_ax.plot(ph[finite], ratio[finite], color="#C43C39", marker="o", label="Syn / anti")
    ratio_ax.axhline(1.0, color="0.35", ls="--", lw=0.9, label="Equal populations")
    ratio_ax.set(xlabel="pH", ylabel="Syn:anti molar ratio")
    ratio_ax.set_xticks(ph)

    count_ax = ratio_ax.twinx()
    spacing = float(np.min(np.diff(ph))) if len(ph) > 1 else 1.0
    count_ax.bar(
        ph,
        protonated,
        width=0.45 * spacing,
        color="0.72",
        alpha=0.28,
        label="Protonated samples",
        zorder=0,
    )
    count_ax.set_ylabel("Protonated samples", color="0.4")
    count_ax.tick_params(axis="y", colors="0.4")
    handles, labels = ratio_ax.get_legend_handles_labels()
    count_handles, count_labels = count_ax.get_legend_handles_labels()
    ratio_ax.legend(handles + count_handles, labels + count_labels, loc="best")

    fig.suptitle(f"{cfg.post.titr.system} protonated microstate balance", fontsize=15)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _group_residue_records(
    cfg: TitrReportConfig,
    records: list[ProtonationRecord],
    residue_name: str,
    residue_id: int,
) -> dict[float, list[ProtonationRecord]]:
    grouped: dict[float, list[ProtonationRecord]] = {}
    dt = float(cfg.post.titr.cntrl["dt"])
    for record in records:
        if record.residue == residue_name and record.residue_id == residue_id:
            if record.md_step * dt > cfg.discard_time_ps + 1.0e-12:
                grouped.setdefault(record.ph, []).append(record)
    if sorted(grouped) != list(cfg.post.titr.ph_values):
        raise ValueError(
            f"Filtered records for {residue_name}:{residue_id} do not cover the pH ladder"
        )
    for values in grouped.values():
        values.sort(key=lambda record: record.record)
    return grouped


def _mean_block_sem(values: np.ndarray, block_count: int) -> tuple[float, float]:
    if len(values) < block_count:
        raise ValueError(
            f"Only {len(values)} records remain; cannot form {block_count} blocks"
        )
    blocks = np.asarray(
        [float(values[index].mean()) for index in np.array_split(np.arange(len(values)), block_count)]
    )
    return float(values.mean()), float(blocks.std(ddof=1) / math.sqrt(block_count))


def _plot_bla_tps(
    cfg: TitrReportConfig,
    path: Path,
    rows: list[dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    ph = np.asarray([row["pH"] for row in rows])
    fig, (titr_ax, taut_ax) = plt.subplots(
        2, 1, figsize=(8.0, 7.8), sharex=True,
        height_ratios=[1.25, 1.0], constrained_layout=True,
    )
    titr_ax.errorbar(
        ph, [row["ppp_fraction"] for row in rows],
        yerr=[row["ppp_sem"] for row in rows], marker="o", capsize=3,
        label="Protonated PPP",
    )
    titr_ax.errorbar(
        ph, [row["deprotonated_fraction"] for row in rows],
        yerr=[row["deprotonated_sem"] for row in rows], marker="s", capsize=3,
        label="Deprotonated BPP + CPP",
    )
    titr_ax.axhline(0.5, color="0.55", ls="--", lw=0.9)
    titr_ax.set(ylabel="Molar fraction", ylim=(-0.03, 1.03), title="TPS protonation equilibrium")
    titr_ax.legend()

    taut_ax.plot(ph, [row["bpp_fraction_of_deprotonated"] for row in rows], marker="o", label="BPP / (BPP + CPP)")
    taut_ax.plot(ph, [row["cpp_fraction_of_deprotonated"] for row in rows], marker="s", label="CPP / (BPP + CPP)")
    taut_ax.axhline(0.5, color="0.55", ls="--", lw=0.9)
    taut_ax.set(xlabel="pH", ylabel="Conditional fraction", ylim=(-0.03, 1.03), title="Deprotonated TPS tautomer balance")
    taut_ax.set_xticks(ph)
    taut_ax.legend()
    fig.suptitle("BLA TPS: PPP ⇌ (BPP + CPP)", fontsize=15)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_bla_prx_titration(
    cfg: TitrReportConfig,
    path: Path,
    rows: list[dict[str, float]],
    residue_ids: tuple[int, int],
    residue_name: str,
) -> list[HHFit | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    ph = np.asarray([row["pH"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    fits: list[HHFit | None] = []
    dense = np.linspace(ph.min() - 0.2, ph.max() + 0.2, 400)
    for index, residue_id in enumerate(residue_ids, start=1):
        fraction = np.asarray([row[f"prx{index}_protonated_fraction"] for row in rows])
        sem = np.asarray([row[f"prx{index}_sem"] for row in rows])
        fit: HHFit | None
        try:
            fit = fit_henderson_hasselbalch(ph, fraction, cfg.fit_hill)
        except ValueError:
            fit = None
        fits.append(fit)
        label = f"{residue_name} tail {index} (residue {residue_id})"
        ax.errorbar(ph, fraction, yerr=sem, marker="o" if index == 1 else "s", capsize=3, ls="none", label=label)
        if fit is not None:
            ax.plot(dense, hh_fraction(dense, fit), label=f"{label} HH fit: pKa={fit.pka:.2f}")
    ax.axhline(0.5, color="0.55", ls="--", lw=0.9)
    ax.set(
        xlabel="pH",
        ylabel="Protonated fraction",
        ylim=(-0.03, 1.03),
        title=f"Individual {residue_name} tail titration curves",
    )
    ax.set_xticks(ph)
    ax.legend()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return fits


def _plot_bla_prx_syn_anti(
    cfg: TitrReportConfig,
    path: Path,
    rows: list[dict[str, float]],
    residue_ids: tuple[int, int],
    residue_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    ph = np.asarray([row["pH"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.4), sharex=True, constrained_layout=True)
    for index, (ax, residue_id) in enumerate(zip(axes, residue_ids), start=1):
        ax.plot(ph, [row[f"prx{index}_syn_fraction"] for row in rows], marker="o", label="Syn / protonated")
        ax.plot(ph, [row[f"prx{index}_anti_fraction"] for row in rows], marker="s", label="Anti / protonated")
        ax.axhline(0.5, color="0.55", ls="--", lw=0.9)
        ax.set(
            ylabel="Conditional fraction", ylim=(-0.03, 1.03),
            title=f"{residue_name} tail {index} (residue {residue_id}) syn/anti balance",
        )
        ax.legend()
    axes[-1].set_xlabel("pH")
    axes[-1].set_xticks(ph)
    fig.suptitle(
        f"{residue_name} conformers within each protonated tail ensemble", fontsize=15
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_bla_joint_tails(
    cfg: TitrReportConfig,
    path: Path,
    rows: list[dict[str, float]],
    residue_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.style)
    ph = np.asarray([row["pH"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    for count, marker in zip(range(3), ("o", "s", "^")):
        ax.errorbar(
            ph,
            [row[f"fraction_{count}H"] for row in rows],
            yerr=[row[f"sem_{count}H"] for row in rows],
            marker=marker, capsize=3, label=f"{count} protonated tail{'s' if count != 1 else ''}",
        )
    ax.set(
        xlabel="pH",
        ylabel="Joint molar fraction",
        ylim=(-0.03, 1.03),
        title=f"Joint protonation of both {residue_name} tails",
    )
    ax.set_xticks(ph)
    ax.legend()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_composite_report(
    cfg: TitrReportConfig,
    work: Path,
    records: list[ProtonationRecord],
) -> dict[str, object]:
    assert cfg.composite is not None
    composite = cfg.composite
    ph_values = np.asarray(cfg.post.titr.ph_values, dtype=float)
    tps_grouped = (
        _group_residue_records(
            cfg, records, composite.tps.residue_name, composite.tps.residue_id
        )
        if composite.tps is not None
        else None
    )
    prx_grouped = (
        [
            _group_residue_records(
                cfg, records, composite.prx.residue_name, residue_id
            )
            for residue_id in composite.prx.residue_ids
        ]
        if composite.prx is not None
        else None
    )

    tps_rows: list[dict[str, float]] = []
    prx_rows: list[dict[str, float]] = []
    syn_rows: list[dict[str, float]] = []
    joint_rows: list[dict[str, float]] = []
    syn_set = set(composite.prx.syn_states) if composite.prx is not None else set()
    anti_set = set(composite.prx.anti_states) if composite.prx is not None else set()
    protonated_set = syn_set | anti_set

    for ph in ph_values:
        if composite.tps is not None and tps_grouped is not None:
            tps_records = tps_grouped[float(ph)]
            tps_states = np.asarray([record.state for record in tps_records])
            ppp = tps_states == composite.tps.protonated_state
            bpp = tps_states == composite.tps.bpp_state
            cpp = tps_states == composite.tps.cpp_state
            deprot = bpp | cpp
            ppp_mean, ppp_sem = _mean_block_sem(ppp.astype(float), cfg.block_count)
            deprot_mean, deprot_sem = _mean_block_sem(
                deprot.astype(float), cfg.block_count
            )
            denominator = int(deprot.sum())
            tps_rows.append(
                {
                    "pH": float(ph),
                    "records": float(len(tps_records)),
                    "ppp_fraction": ppp_mean,
                    "ppp_sem": ppp_sem,
                    "deprotonated_fraction": deprot_mean,
                    "deprotonated_sem": deprot_sem,
                    "bpp_fraction_total": float(bpp.mean()),
                    "cpp_fraction_total": float(cpp.mean()),
                    "bpp_fraction_of_deprotonated": (
                        float(bpp.sum() / denominator) if denominator else math.nan
                    ),
                    "cpp_fraction_of_deprotonated": (
                        float(cpp.sum() / denominator) if denominator else math.nan
                    ),
                    "deprotonated_samples": float(denominator),
                }
            )

        if composite.prx is not None and prx_grouped is not None:
            prx_row: dict[str, float] = {"pH": float(ph)}
            syn_row: dict[str, float] = {"pH": float(ph)}
            aligned: list[dict[int, ProtonationRecord]] = []
            for index, grouped in enumerate(prx_grouped, start=1):
                tail_records = grouped[float(ph)]
                aligned.append({record.record: record for record in tail_records})
                protonated = np.asarray(
                    [record.state in protonated_set for record in tail_records]
                )
                mean, sem = _mean_block_sem(
                    protonated.astype(float), cfg.block_count
                )
                syn_count = sum(record.state in syn_set for record in tail_records)
                anti_count = sum(record.state in anti_set for record in tail_records)
                total = syn_count + anti_count
                prx_row[f"prx{index}_protonated_fraction"] = mean
                prx_row[f"prx{index}_sem"] = sem
                syn_row[f"prx{index}_syn_fraction"] = (
                    syn_count / total if total else math.nan
                )
                syn_row[f"prx{index}_anti_fraction"] = (
                    anti_count / total if total else math.nan
                )
                syn_row[f"prx{index}_protonated_samples"] = float(total)
            prx_rows.append(prx_row)
            syn_rows.append(syn_row)

            if set(aligned[0]) != set(aligned[1]):
                raise ValueError(f"PRX tail records are not aligned at pH {ph:g}")
            proton_counts = np.asarray(
                [
                    int(aligned[0][record].state in protonated_set)
                    + int(aligned[1][record].state in protonated_set)
                    for record in sorted(aligned[0])
                ]
            )
            joint_row: dict[str, float] = {
                "pH": float(ph),
                "records": float(len(proton_counts)),
            }
            for count in range(3):
                fraction, sem = _mean_block_sem(
                    (proton_counts == count).astype(float), cfg.block_count
                )
                joint_row[f"fraction_{count}H"] = fraction
                joint_row[f"sem_{count}H"] = sem
            joint_rows.append(joint_row)

    def write_dict_csv(name: str, rows: list[dict[str, float]]) -> None:
        with (work / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    output_tables: list[str] = []
    summary: dict[str, object] = {}
    if composite.tps is not None:
        write_dict_csv("tps-titration-tautomers.csv", tps_rows)
        _plot_bla_tps(cfg, work / "tps-titration-tautomers.png", tps_rows)
        output_tables.append("tps-titration-tautomers.csv")
        summary["tps_residue_id"] = composite.tps.residue_id

    if composite.prx is not None:
        write_dict_csv("prx-titration.csv", prx_rows)
        write_dict_csv("prx-syn-anti.csv", syn_rows)
        write_dict_csv("prx-joint-protonation.csv", joint_rows)
        fits = _plot_bla_prx_titration(
            cfg,
            work / "prx-titration.png",
            prx_rows,
            composite.prx.residue_ids,
            composite.prx.residue_name,
        )
        _plot_bla_prx_syn_anti(
            cfg,
            work / "prx-syn-anti.png",
            syn_rows,
            composite.prx.residue_ids,
            composite.prx.residue_name,
        )
        _plot_bla_joint_tails(
            cfg,
            work / "prx-joint-protonation.png",
            joint_rows,
            composite.prx.residue_name,
        )
        output_tables.extend(
            ["prx-titration.csv", "prx-syn-anti.csv", "prx-joint-protonation.csv"]
        )
        summary["prx_residue_ids"] = list(composite.prx.residue_ids)
        summary["prx_fitted_pKa"] = [
            None if fit is None else fit.pka for fit in fits
        ]

    included = [
        name
        for name, configured in (("TPS", composite.tps), ("two joint tails", composite.prx))
        if configured is not None
    ]
    summary["definition"] = "Composite analysis of " + " and ".join(included)
    summary["output_tables"] = output_tables
    return summary


def build_titration_report(cfg: TitrReportConfig, repo_root: Path) -> Path:
    post = _validate_postprocessed_data(cfg, repo_root)
    destination = report_dir(cfg, repo_root)
    records = read_protonation_records(post / "protonation-states.csv")
    residue, grouped, max_protons = _selected_series(cfg, records)
    ph, fraction, sem, transitions, fit, block_fits = _analyze_titration(cfg, grouped, max_protons)
    state_values, probabilities, overlap = _state_probabilities(ph, grouped)
    syn_anti_rows = (
        analyze_syn_anti(
            ph,
            grouped,
            cfg.syn_anti.syn_states,
            cfg.syn_anti.anti_states,
        )
        if cfg.syn_anti is not None
        else None
    )

    with tempfile.TemporaryDirectory(prefix=".report-work-", dir=post) as tmp:
        work = Path(tmp)
        cpptraj_input = work / "remlog.cpptraj.in"
        cpptraj_log = work / "remlog.cpptraj.log"
        cpptraj_input.write_text(
            render_remlog_cpptraj_input(cfg, post.parent / "rem.log")
        )
        run_cpptraj(
            cpptraj_input,
            cpptraj_log,
            cfg.post.titr.runtime.module,
            cfg.post.cpptraj_executable,
        )
        required = [
            "walker-replica-index.dat",
            "replica-walker-index.dat",
            "exchange-acceptance.dat",
            "residence-percent.dat",
            "residence-lifetime.dat",
            "residence-slope.dat",
            "roundtrip-stats.dat",
        ]
        missing = [name for name in required if not (work / name).is_file()]
        if missing:
            raise FileNotFoundError(f"CPPTRAJ did not create: {', '.join(missing)}")

        walker, residence, roundtrips, edges, lifetime_mean, slope = _tidy_remlog_outputs(cfg, work)
        _write_titration_outputs(
            cfg, work, grouped, max_protons, ph, fraction, sem, transitions, fit, block_fits, overlap
        )
        _write_csv(
            work / "protonation-state-probabilities.csv",
            ["pH", *[f"proton_count_{value}" for value in state_values]],
            [[f"{p:.6f}", *[f"{value:.8f}" for value in row]] for p, row in zip(ph, probabilities)],
        )

        _plot_replica_health(cfg, work / "replica-health.png", walker, residence, roundtrips, edges, lifetime_mean, slope)
        _plot_titration_curve(cfg, work / "titration-curve.png", ph, fraction, sem, fit, block_fits)
        _plot_protonation_timeseries(cfg, work / "protonation-timeseries.png", grouped, max_protons)
        _plot_state_overlap(cfg, work / "protonation-state-overlap.png", ph, overlap, edges)
        composite_summary = (
            _write_composite_report(cfg, work, records)
            if cfg.composite is not None
            else None
        )
        if syn_anti_rows is not None:
            _write_csv(
                work / "syn-anti-populations.csv",
                [
                    "pH",
                    "total_records",
                    "protonated_count",
                    "syn_count",
                    "anti_count",
                    "syn_fraction_of_protonated",
                    "anti_fraction_of_protonated",
                    "syn_to_anti_ratio",
                ],
                [
                    [
                        f"{row.ph:.6f}",
                        row.total_records,
                        row.protonated_count,
                        row.syn_count,
                        row.anti_count,
                        f"{row.syn_fraction:.8f}",
                        f"{row.anti_fraction:.8f}",
                        f"{row.syn_to_anti_ratio:.8f}",
                    ]
                    for row in syn_anti_rows
                ],
            )
            _plot_syn_anti_populations(
                cfg,
                work / "syn-anti-populations.png",
                syn_anti_rows,
            )

        valid_block_pka = [value.pka for value in block_fits if value is not None]
        adjacent_edges = [edge for edge in edges if edge.target != 1]
        summary = {
            "system": cfg.post.titr.system,
            "residue": residue,
            "residue_id": cfg.residue_id,
            "fraction_definition": f"proton_count == {max_protons}",
            "discard_time_ps": cfg.discard_time_ps,
            "records_per_pH": len(next(iter(grouped.values()))),
            "fit": {
                "model": "1 / (1 + 10**(hill * (pH - pKa)))",
                "pKa": fit.pka,
                "hill_coefficient": fit.hill,
                "fit_hill": cfg.fit_hill,
                "reference_pKa": cfg.reference_pka,
                "block_count": cfg.block_count,
                "block_pKa_mean": float(np.mean(valid_block_pka)) if valid_block_pka else None,
                "block_pKa_sd": float(np.std(valid_block_pka, ddof=1)) if len(valid_block_pka) > 1 else 0.0,
            },
            "replica_exchange": {
                "exchange_attempts": int(cfg.post.titr.cntrl["numexchg"]),
                "total_round_trips": int(roundtrips[:, 1].sum()),
                "adjacent_acceptance_percent_min": min(
                    min(edge.forward_percent, edge.reverse_percent) for edge in adjacent_edges
                ),
                "adjacent_acceptance_percent_max": max(
                    max(edge.forward_percent, edge.reverse_percent) for edge in adjacent_edges
                ),
                "endpoint_edge_acceptance_percent": {
                    "forward": edges[-1].forward_percent,
                    "reverse": edges[-1].reverse_percent,
                },
            },
            "overlap_definition": (
                "Bhattacharyya coefficient between discrete proton-count distributions; "
                "this is not Hamiltonian energy overlap"
            ),
        }
        if cfg.syn_anti is not None and syn_anti_rows is not None:
            summary["syn_anti"] = {
                "definition": "conditional populations within the protonated ensemble",
                "syn_states": list(cfg.syn_anti.syn_names),
                "anti_states": list(cfg.syn_anti.anti_names),
                "per_pH": [
                    {
                        "pH": row.ph,
                        "protonated_count": row.protonated_count,
                        "syn_fraction": row.syn_fraction,
                        "anti_fraction": row.anti_fraction,
                        "syn_to_anti_ratio": row.syn_to_anti_ratio,
                    }
                    for row in syn_anti_rows
                ],
            }
        if composite_summary is not None:
            summary["composite"] = composite_summary
        (work / "report-summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

        destination.mkdir(parents=True, exist_ok=True)
        for path in work.iterdir():
            path.replace(destination / path.name)
    return destination


def run_cphmd_titr_report(yaml_path: Path) -> None:
    resolved = yaml_path.resolve()
    cfg = load_titr_report_config(resolved)
    destination = build_titration_report(cfg, find_repo_root(resolved))
    summary = yaml.safe_load((destination / "report-summary.yaml").read_text())
    fit = summary["fit"]
    exchange = summary["replica_exchange"]
    print(f"OK: wrote RECpHMD report to {destination}")
    print(
        f"OK: fitted pKa = {fit['pKa']:.3f}; Hill coefficient = "
        f"{fit['hill_coefficient']:.3f}"
    )
    print(
        f"OK: {exchange['total_round_trips']} total round trips; adjacent exchange "
        f"acceptance {exchange['adjacent_acceptance_percent_min']:.1f}-"
        f"{exchange['adjacent_acceptance_percent_max']:.1f}%"
    )
    syn_anti = summary.get("syn_anti")
    if syn_anti is not None:
        print("Syn:anti ratio within the protonated ensemble:")
        for row in syn_anti["per_pH"]:
            print(
                f"  pH {row['pH']:g}: {row['syn_to_anti_ratio']:.3f} "
                f"({row['protonated_count']} protonated samples)"
            )
