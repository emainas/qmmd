#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SYSTEMS_DIR="$ROOT/systems"
DATA_DIR="$ROOT/data"
OUT="$DATA_DIR/buffer_vs_water.dat"
mkdir -p "$DATA_DIR"

echo "# system buffer_A water_residues" > "$OUT"

# Find every tleap.out matching: systems/<SYS>/solv_*/prep/tleap.out
# Then parse <SYS> and buffer from the path.
while IFS= read -r tleap; do
  # Example path:
  # .../qmmd/systems/MEA/solv_5.0/prep/tleap.out
  rel="${tleap#"$SYSTEMS_DIR"/}"         # MEA/solv_5.0/prep/tleap.out
  system="${rel%%/*}"                   # MEA
  rest="${rel#*/}"                      # solv_5.0/prep/tleap.out
  solvdir="${rest%%/*}"                 # solv_5.0
  buffer="${solvdir#solv_}"             # 5.0

  waters="$(awk '/Added[[:space:]]+[0-9]+[[:space:]]+residues\./ {n=$2} END{print (n=="" ? "NA" : n)}' "$tleap")"

  printf "%-6s %6.2f %6s\n" "$system" "$buffer" "$waters" >> "$OUT"
done < <(find "$SYSTEMS_DIR" -type f -path "*/solv_*/prep/tleap.out" | sort)

# Keep header, sort rows by system then numeric buffer
{ head -n 1 "$OUT"; tail -n +2 "$OUT" | sort -k1,1 -k2,2n; } > "${OUT}.tmp"
mv "${OUT}.tmp" "$OUT"

echo "Wrote: $OUT"
