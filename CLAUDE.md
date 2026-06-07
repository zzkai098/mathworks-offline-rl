# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

8-week MathWorks summer project (2026-05 → 2026-07) implementing **offline reinforcement learning for portfolio management** in MATLAB R2025b. The baseline is the official MathWorks GBWM (Goal-Based Wealth Management) demo; the goal is to extend it with CQL and IQL algorithms.

- **Mentor:** Yuchen Dong (MathWorks)
- **Reference paper:** Unnikrishnan, *Financial News-Driven LLM RL for Portfolio Management*

## MATLAB Setup

**Required:** MATLAB R2025b with toolboxes: Reinforcement Learning, Financial, Statistics & Machine Learning, Deep Learning, Optimization.

Open MATLAB, set project root as Current Folder, then open any `.mlx` under `demos/` or `src/`.

## Architecture

### Data flow
`data/SimulatedData.xlsx` → log-returns matrix → sliding-window offline episodes (`.mat` files) → `fileDatastore` → `trainFromData(agent, fds, opts)` → trained DQN agent

### Key design choices in the demo
- **State:** 2D observation `[normalizedWealth, timeFraction]` where wealth is clipped at 5× goal
- **Action space:** 15 discrete efficient frontier portfolios, computed once from the training window via MATLAB's `Portfolio` object
- **Reward:** sparse binary — 1 only at terminal step if `wealthFinal >= goalWealth`
- **Agent:** DQN (`rlDQNAgent`) with a 2-layer MLP critic (`rlVectorQValueFunction`)
- **Offline training:** `trainFromData` + `rlTrainingFromDataOptions` (no environment interaction during training)

### Repo layout
```
demos/          Original MathWorks reference demo (treat as read-only)
src/            Project experiments (.mlx files go here)
data/           SimulatedData.xlsx — 3 columns: Stock1, Stock2, Stock3 prices
experiments/
  logs/         Saved .mat episode files and trained agents (gitignored)
  figures/      Saved plots
docs/weekly_reports/   Progress notes
papers/         7 reference PDFs (see papers/README.md for reading order)
```

## Known Bugs in the Demo

These are documented in `docs/weekly_reports/week2.md` and already fixed in `src/wee2_baseline.mlx`:

1. **`price2ret` dependency** — moved to Econometrics Toolbox in R2025b. Replace with `diff(log(P))` for continuous returns.
2. **`{:, 3:end}` indexing** (demo lines 42, 188) — `SimulatedData.xlsx` has only 3 price columns with no date column, so `3:end` silently drops all assets. Should be `{:, :}` or `{:, 1:end}`.
3. **`numEpisodes = 400` mismatch** — with `trainingRange=252`, `horizonPeriods=60`, and a step of 30, only ~7 episodes fit in the window. The loop exits early; the value is misleading.

## Development Roadmap

| Week | Focus |
|------|-------|
| 2 | Literature review + demo validation (current) |
| 3–4 | Data preprocessing, extend dataset via Datafeed Toolbox |
| 5–6 | Implement CQL regularizer on top of DQN critic; then IQL as alternative |
| 7 | Backtesting, baseline comparison, Sharpe/drawdown evaluation |
| 8 | Final report |

## Algorithm Pointers

- **CQL baseline to implement:** add `α * (mean(Q_all_actions) - Q_behavior_action)` penalty to the Bellman loss inside a custom critic update
- **IQL alternative:** expectile regression on V(s); avoids out-of-distribution action queries entirely
- Full algorithm details in `papers/01_CQL_Kumar2020.pdf` and `papers/03_IQL_Kostrikov2021.pdf`

## Experiment Log

### Week 4 Baseline (`src/week4_baseline_realdata.mlx`)

**Config:** 15 real assets (13 sector stocks + TLT + GLD), 2010-2017 training window
(trainingRange=2000), 130 episodes × 30 trading-day horizon, 2D state `[wealth, time]`,
shaped reward (log return + terminal bonus), DQN 32×32 critic, single seed 1000.
Macro factors fetched into `data/macro_*.csv` but **not used in state**.

**Hold-out results** (30 rolling 30-day windows over test 2022-01 → 2024-08):
- Success rate: **15/30 = 50%**
- Mean terminal wealth: **103,255 (+3.3%)**
- Mean Sharpe: **0.59 (std 2.75)** — bimodal distribution, not stable
- Mean MaxDD: **10.72% (std 5.52%)**

**Learned policy:** State-dependent 3-phase aggressive strategy:
- t=1-5: action 15 (100% NVDA)
- t=6-20: action 12 (UNH 55% + AAPL 19% + NVDA 17%)
- t=21-30: action 13/14 (UNH/NVDA collapse)
- **Never selects actions 1-6** (TLT-heavy defensive portfolios)

**Diagnosis:** Agent cannot perceive market regime because state lacks macro signal.
In 2022 bear-market windows it sticks with the training-regime aggressive playbook
and incurs large drawdowns; in 2023-2024 AI rally windows the same playbook captures
big upside. Mean Sharpe 0.59 + std 2.75 reflects this bimodal exposure.

**Lessons for v2:**

1. **State extension is the highest-priority fix** (not algorithmic changes).
   Extend obs to 6D: `[wealth, time, DGS10_z, T10Y2Y_z, VIXCLS_z, DFF_z]`. Use
   train-segment mean/std for z-score to avoid look-ahead bias.

2. **Success metric should pivot to risk-adjusted, not absolute.** v2 evaluation
   must report Sharpe std, MaxDD P90, terminal wealth P10, and action diversity —
   not just mean success rate. Target: narrower 10-90% wealth band, similar median.

3. **Change one variable at a time.** Add macro state first; only then revisit
   trainingRange extension to 2515 days, multi-seed, or CQL regularization.

4. **CQL/IQL is not the most urgent improvement.** Current problem is information
   starvation, not over-optimism on out-of-distribution actions. CQL helps when
   agent over-trusts under-represented actions; it does not invent missing state
   features. Defer until after the macro-state version is benchmarked.

5. **Technical debt to fix in v2:**
   - `fileDatastore` should restrict to `loggedData*.mat` to avoid TrainedAgent.mat
     pollution on rerun
   - `yline` calls need `DisplayName` to show "Goal"/"Start" instead of "data1/data2"
   - Add per-window action logging and percentile-band wealth plot to standard eval

**Action items locked for next version (`week5_macro_state.mlx`):**
- 6D state with z-scored FRED factors loaded from `data/macro_train.csv` and
  `data/macro_test.csv`
- Standardization params fitted on train only, applied unchanged to test
- Episode generator and hold-out loop both look up macro by `dayIdx + 1`
  (compensating for the row lost to `diff(log)`)
- Keep all other hyperparams identical to week 4 for clean A/B comparison
