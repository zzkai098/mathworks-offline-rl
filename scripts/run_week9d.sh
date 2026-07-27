#!/usr/bin/env bash
# week9d: same confound-free sweep as run_week9c.sh, ONE lever changed
# (target update: periodic-every-4 -> Polyak smoothing). Direct A/B against
# experiments/logs/week9c_results.csv to test whether slowing the target bounds
# the Q divergence (1e3-1e5 -> hopefully O(1)).
#
# Two target speeds per subagent review: tau=1e-3 (standard) AND tau=1e-2. The
# second speed is the converged-vs-frozen discriminator: if tau=1e-3 comes back
# "bounded but tiny/frozen", the faster tau=1e-2 tells us whether the target was
# just too slow to finish learning vs the divergence had another cause.
# Header written once here; MATLAB always appends.
set -uo pipefail
cd "$(dirname "$0")/.."

MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
CSV=experiments/logs/week9d_results.csv

echo "tsf,seed,ME,qGlobal,qEarly,qLate,margin,stateStd,marginOverStateStd,effActions,evalSharpe,success,evalMddP90,evalTermP10" > "$CSV"

run() {  # run TSF SEED ME
  local TSF=$1 SEED=$2 ME=$3
  echo "===== week9d: TSF=$TSF seed=$SEED ME=$ME (fresh process) ====="
  "$MATLAB" -batch "TSF=$TSF; SEED=$SEED; ME=$ME; addpath('src'); week9d_slowtarget_session" \
    || echo "WARN: process failed for TSF=$TSF seed=$SEED ME=$ME (continuing)"
}

# Explicit process-reproducibility pair (two identical runs -> rows must match ~4dp)
run 1e-3 1000 100
run 1e-3 1000 100

# Sweep: 2 target speeds x 3 seeds x 4 epoch budgets (matches week9c grid)
for TSF in 1e-3 1e-2; do
  for SEED in 1000 2000 3000; do
    for ME in 50 100 200 400; do
      run "$TSF" "$SEED" "$ME"
    done
  done
done

echo "===== week9d sweep done; results ====="
cat "$CSV"
