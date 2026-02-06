#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Parse DFTB performance stats into data/<system>/<solv>/dftb/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SYSTEMS_DIR="$ROOT/systems"
DATA_DIR="$ROOT/data/dftb"
mkdir -p "$DATA_DIR" 

for dftb_out in "$SYSTEMS_DIR"/*/solv_*/dftb/N*T*C*/dftb.out; do
  [[ -f "$dftb_out" ]] || continue

  # Example: systems/MEA/solv_5.0/dftb/N1T64C1/dftb.out
  rel="${dftb_out#"$SYSTEMS_DIR"/}"
  system="${rel%%/*}"
  rest="${rel#*/}"
  solv_tag="${rest%%/*}"
  run_tag="$(basename "$(dirname "$dftb_out")")"

  out_dir="$DATA_DIR/$system/$solv_tag"
  mkdir -p "$out_dir"

  step_out="$out_dir/${run_tag}_step_perf.dat"
  summary_out="$out_dir/${run_tag}_summary.dat"

  awk -v step_out="$step_out" -v summary_out="$summary_out" '
    BEGIN {
      print "# STEP_NO SUBSYSTEMS FINAL_ITERATIONS STEP_SECONDS" > step_out
    }

    /Number of steps[[:space:]]*=/ {
      total_steps=$5
    }

    /Total wall clock time[[:space:]]*:/ {
      total_time=$6
    }

    /Start DC-DFTB-3rd calculation/ {
      # Reset per-step fields at start of each MD step
      subsys=""
      iters=""
      time_sum=""
    }

    /Number of subsystems[[:space:]]*=/ {
      if (match($0, /Number of subsystems[[:space:]]*=[[:space:]]*([0-9]+)/, m)) {
        subsys=m[1]
      }
    }

    /Final DC-DFTB-3rd Energy/ {
      if (match($0, /after[[:space:]]*([0-9]+)[[:space:]]*iterations/, m)) {
        iters=m[1]
      }
    }

    /Step wall clock time \[seconds\]/ {
      in_time=1
      time_sum=0
      next
    }

    in_time {
      # Skip the separator line in the dftb.out file
      if ($0 ~ /^[[:space:]]*-{3,}[[:space:]]*$/) next
    
      if (match($0, /:[[:space:]]*([0-9.]+)[[:space:]]*$/, m)) {
        time_sum += m[1]
        next
      }
      # End of block when line no longer matches time entry
      in_time=0
    }

    /THIS RUN\x27S STEP NO\.=/{
      if (match($0, /STEP NO\.= *([0-9]+)/, m)) {
        stepno=m[1]
        if (time_sum != "" && subsys != "" && iters != "") {
          printf "%d %d %d %.6f\n", stepno, subsys, iters, time_sum >> step_out
        }
      }
    }

    END {
      print "# TOTAL_MD_STEPS TOTAL_WALL_CLOCK_SECONDS" > summary_out
      if (total_steps == "") total_steps="NA"
      if (total_time  == "") total_time="NA"
      print total_steps, total_time >> summary_out
    }
  ' "$dftb_out"

done
