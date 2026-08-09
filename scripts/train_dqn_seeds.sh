#!/usr/bin/env bash
# Train the shipped tuned-DQN (Polyak tau=1e-3, LR 1e-4, ME=100) at 20 seeds, each
# saved to its own experiments/models/dqn_seed_<seed>.mat, for the multi-seed
# return/win-rate distribution + backtest. Skips seeds already trained. Parallel pool
# (xargs -P 3), independent processes, hang-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
mkdir -p experiments/models

JOBS=()
for SEED in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000; do
  [ -f "experiments/models/dqn_seed_${SEED}.mat" ] && continue   # skip already trained
  JOBS+=("SEED=$SEED; ME=100; MODELPATH='experiments/models/dqn_seed_${SEED}.mat'; addpath('src'); train_final_agent")
done
if [ ${#JOBS[@]} -eq 0 ]; then echo "all 20 DQN seeds already trained."; exit 0; fi
printf '%s\0' "${JOBS[@]}" | xargs -0 -P 3 -I {} "$MATLAB" -batch "{}"
echo "===== trained; models: ====="
ls -la experiments/models/dqn_seed_*.mat
