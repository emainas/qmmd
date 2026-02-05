#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Put this in: <root>/scripts/temp_and_density.sh
# Run from anywhere (but you can run from scripts/ as usual)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SYSTEMS_DIR="$ROOT/systems"
OUT_BASE="$ROOT/data"
mkdir -p "$OUT_BASE"

for sysdir in "$SYSTEMS_DIR"/*/; do
  system="$(basename "$sysdir")"
  mkdir -p "$OUT_BASE/$system"

  for solvdir in "$sysdir"/solv_*/; do
    solv_tag="$(basename "$solvdir")"          # solv_5.0
    solv_val="${solv_tag#solv_}"              # 5.0

    md_dir="$solvdir/mdequil"
    [[ -d "$md_dir" ]] || continue

    heat="$md_dir/heat.out"
    nvt="$md_dir/equil-nvt.out"
    npt="$md_dir/equil-npt.out"

    # ---- temperature: TIME(ps) TEMP(K) appended heat -> nvt -> npt
    temp_out="$OUT_BASE/$system/${solv_tag}_temp.dat"
    {
      echo "# TIME(ps) TEMP(K)"
      [[ -f "$heat" ]] && awk '/NSTEP/ {print $6, $9}' "$heat" | head -n -2 || true
      [[ -f "$nvt"  ]] && awk '/NSTEP/ {print $6, $9}' "$nvt"  | head -n -2 || true
      [[ -f "$npt"  ]] && awk '/NSTEP/ {print $6, $9}' "$npt"  | head -n -2 || true
    } > "$temp_out"

    # ---- density: from NPT only, pair TIME(ps) with Density(g/cm^3)
    dens_out="$OUT_BASE/$system/${solv_tag}_density.dat"
    {
      echo "# TIME(ps) DENSITY(g/cm^3)"
      if [[ -f "$npt" ]]; then
        paste \
          <(awk '/NSTEP/   {print $6}' "$npt" | head -n -2) \
          <(awk '/Density/ {print $3}' "$npt" | head -n -2) \
        | awk 'NF==2 {print $1, $2}'
      fi
    } > "$dens_out"

    echo "Wrote: $temp_out"
    echo "Wrote: $dens_out"
  done
done

