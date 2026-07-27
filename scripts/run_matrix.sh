#!/usr/bin/env bash
# Final matrix: tuned-DQN (Polyak tau=1e-3, LR 1e-4) vs IQL (expectile 0.7),
# 3 seeds x 4 ME, + one IQL@LR1e-4 cell (no-LR-babysitting demo).
# Parallel job pool (xargs -P 3): each cell is an independent fresh MATLAB process
# writing its OWN per-cell CSV, so a hang blocks only one slot (not the whole run)
# and there is no concurrent-CSV-append race. Collect experiments/logs/matrix/*.csv.
set -uo pipefail
cd "$(dirname "$0")/.."
MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
OUT=experiments/logs/matrix
mkdir -p "$OUT"

JOBS=()
for SEED in 1000 2000 3000; do
  for ME in 50 100 200 400; do
    # tuned DQN: week9f with standard reward (BONUS=1.0), Polyak, LR 1e-4
    JOBS+=("OUTCSV='$OUT/dqn_s${SEED}_me${ME}.csv'; BONUS=1.0; TSF=1e-3; LRATE=1e-4; SEED=$SEED; ME=$ME; addpath('src'); week9f_bonus_attribution")
    # IQL: custom loop, natural LR 3e-4
    JOBS+=("OUTCSV='$OUT/iql_s${SEED}_me${ME}.csv'; EXPECTILE=0.7; TSF=1e-3; LRATE=3e-4; SEED=$SEED; ME=$ME; addpath('src'); week10_iql_session")
  done
done
# no-LR-tuning demo: IQL bounds at LR 1e-4 too (same as tuned DQN's LR)
JOBS+=("OUTCSV='$OUT/iql_lr1e4_s1000_me400.csv'; EXPECTILE=0.7; TSF=1e-3; LRATE=1e-4; SEED=1000; ME=400; addpath('src'); week10_iql_session")

echo "launching ${#JOBS[@]} cells, 3 at a time..."
printf '%s\0' "${JOBS[@]}" | xargs -0 -P 3 -I {} "$MATLAB" -batch "{}"
echo "===== matrix done; per-cell CSVs in $OUT ====="
ls "$OUT"
