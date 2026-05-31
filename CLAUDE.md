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
