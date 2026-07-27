#!/usr/bin/env bash
# Confound-free convergence sweep: one FRESH MATLAB process per (seed, MaxEpochs).
# Each process does a single trainFromData call (like 6.1) at MaxEpochs=ME, so
# trainFromData's session-carried RNG/cursor cannot leak across configs.
#
# Per code review, two things beyond the single sweep:
#   1. REPRODUCIBILITY check (P0): run (seed=1000, ME=100) an extra time up front.
#      If its two rows match ~4 decimals on qGlobal, trainFromData's internal
#      sampler is effectively deterministic across processes and the curve is valid.
#   2. POWER (P1): >=3 seeds per ME so we can separate sampler noise from
#      non-convergence (week9b showed Q oscillating at n=1). "Converged" iff the
#      ME-to-ME change in mean qGlobal is within the across-seed std at fixed ME.
#
# Header is written ONCE here (race-safe); MATLAB always appends.
set -euo pipefail
cd "$(dirname "$0")/.."

MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
CSV=experiments/logs/week9c_results.csv

echo "seed,ME,qGlobal,qEarly,qLate,margin,stateStd,marginOverStateStd,effActions,evalSharpe,success,evalMddP90,evalTermP10" > "$CSV"

run() {  # run SEED ME
  local SEED=$1 ME=$2
  echo "===== week9c: seed=$SEED MaxEpochs=$ME (fresh process) ====="
  "$MATLAB" -batch "SEED=$SEED; ME=$ME; addpath('src'); week9c_convergence_session" \
    || echo "WARN: process failed for seed=$SEED ME=$ME (continuing)"
}

# 1) P0 reproducibility probe: same config twice (rows 1 and the grid's 1000/100 row)
run 1000 100

# 2) Power sweep: 3 seeds x 4 epoch budgets
for SEED in 1000 2000 3000; do
  for ME in 50 100 200 400; do
    run "$SEED" "$ME"
  done
done

echo "===== sweep done; results ====="
cat "$CSV"
