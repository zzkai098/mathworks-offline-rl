#!/usr/bin/env bash
# Train the shipped tuned-DQN (Polyak tau=1e-3, LR 1e-4, ME=100) at 5 seeds, each
# saved to its own experiments/models/dqn_seed_<seed>.mat, for the multi-seed
# cost-aware backtest (across-seed std + worst-Sharpe need >1 seed). Parallel pool
# (xargs -P 3), independent processes, hang-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
mkdir -p experiments/models

JOBS=()
for SEED in 1000 2000 3000 4000 5000; do
  JOBS+=("SEED=$SEED; ME=100; MODELPATH='experiments/models/dqn_seed_${SEED}.mat'; addpath('src'); train_final_agent")
done
printf '%s\0' "${JOBS[@]}" | xargs -0 -P 3 -I {} "$MATLAB" -batch "{}"
echo "===== trained; models: ====="
ls -la experiments/models/dqn_seed_*.mat
