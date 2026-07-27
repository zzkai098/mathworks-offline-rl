#!/usr/bin/env bash
# week9f: attribute the residual ME=400 divergence (reward-scale vs OOD bootstrap).
# 4 runs, all ME=400 / tau=1e-3:
#   1 CONTROL  (bonus=1.0, seed 1000) -> should reproduce week9d's ~122, confirming
#              week9f == week9d at bonus=1.0 (only the bonus param differs).
#   3 TEST     (bonus=0.05, seeds 1000/2000/3000) -> does the residual vanish?
# Header once here; MATLAB always appends. Kept small on purpose (past 26-run sweep
# was too slow / one process hung); monitor and kill if a run wedges.
set -uo pipefail
cd "$(dirname "$0")/.."

MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
CSV=experiments/logs/week9f_results.csv

echo "bonus,tsf,seed,ME,qGlobal,qEarly,qLate,margin,stateStd,marginOverStateStd,effActions,evalSharpe,success,evalMddP90,evalTermP10" > "$CSV"

run() {  # run BONUS SEED ME
  local BONUS=$1 SEED=$2 ME=$3
  echo "===== week9f: bonus=$BONUS seed=$SEED ME=$ME (fresh process) ====="
  "$MATLAB" -batch "BONUS=$BONUS; TSF=1e-3; SEED=$SEED; ME=$ME; addpath('src'); week9f_bonus_attribution" \
    || echo "WARN: process failed for bonus=$BONUS seed=$SEED ME=$ME (continuing)"
}

# control: verify week9f==week9d at bonus=1.0 (expect ~122)
run 1.0 1000 400

# signal-alive guard: bonus=0.05 at ME=100 -> compare effA/success vs week9d ME=100
# (effA~15, succ~13). If policy stays alive here, "no divergence at ME=400" is a
# real reward-scale result, not "bonus too weak to learn anything".
run 0.05 1000 100

# test: shrunk bonus at ME=400, all seeds (does the residual return?)
run 0.05 1000 400
run 0.05 2000 400
run 0.05 3000 400

echo "===== week9f done; results ====="
cat "$CSV"
