#!/usr/bin/env bash
# Train the co-shipped IQL agent (expectile 0.7, tau=1e-3, LR 3e-4, ME=100) at 20 seeds,
# each saved to experiments/models/iql_seed_<seed>.mat (bundle with qNet), for the
# multi-seed return/win-rate distribution + backtest. Skips seeds already trained.
# Parallel pool (xargs -P 3), hang-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
mkdir -p experiments/models

JOBS=()
for SEED in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
  [ -f "experiments/models/iql_seed_${SEED}.mat" ] && continue   # skip already trained
  JOBS+=("SEED=$SEED; ME=100; EXPECTILE=0.7; TSF=1e-3; LRATE=3e-4; OUTCSV='experiments/logs/iql_final_s${SEED}.csv'; MODELPATH='experiments/models/iql_seed_${SEED}.mat'; addpath('src'); week10_iql_session")
done
if [ ${#JOBS[@]} -eq 0 ]; then echo "all 20 IQL seeds already trained."; exit 0; fi
printf '%s\0' "${JOBS[@]}" | xargs -0 -P 3 -I {} "$MATLAB" -batch "{}"
echo "===== trained; models: ====="
ls -la experiments/models/iql_seed_*.mat
