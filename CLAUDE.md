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

### Week 4 v2: Macro State (`src/week4_macro_state.mlx`)

**Config delta vs baseline:** state extended 2D → 6D by appending
`[DGS10_z, T10Y2Y_z, VIXCLS_z, DFF_z]`, z-scored on train mean/std only.
Everything else identical: trainingRange=2000, 130 episodes, single seed 1000,
same network width, same reward. Technical debt fixes (fileDatastore filter,
yline DisplayName, P10/P90 metrics, action histogram) also landed here.

**Hold-out results** (30 rolling windows on test):
- Success rate: **10/30 = 33%** (↓ -17pp vs baseline)
- Mean Sharpe: **-0.01 (std 2.95)** — std went UP, not down
- Mean MaxDD: 7.85% (std 8.70%)
- Mean Terminal: 98,275 (-1.7%)
- Sharpe range: **-7.80 to 6.29** (extreme tail got worse)
- Action distribution: **79% on action 7** (TLT 36% + UNH 28% + AAPL 13%),
  remaining split across 10/13/14/15. Never selected actions 1-6.
- Q value converged to 0.20 (15× baseline's 0.014 — signal was absorbed)

**Diagnosis:** Agent learned the WRONG macro→action mapping for the test regime.
In 2010-2017 train data, high VIX / inverted yield curve / rising rates
correlated with "switch to bonds" being protective. In 2022, the regime broke:
TLT itself dropped -36% from $145 to $93 as rates spiked. Agent's macro-triggered
defensive playbook routed it INTO the worst-performing asset. This is classic
distribution shift, just in feature→action mapping rather than feature
distribution itself.

**Why v2 was not a failure even though numbers look worse:**
- Q value rose from 0.014 → 0.20: network successfully absorbed macro signal
- Action diversity emerged (79% on action 7 ≠ random — it's conditional behavior)
- Median MaxDD actually improved (4.83% vs baseline ~10%)
- The cost was tail risk: when the wrong defensive bet was made, it was big

**Lesson:** macro state alone is not enough — training data must contain a
regime where the macro→action relationship resembles the test regime. The
2010-2017 window has no episode of TLT failing as a hedge.

### Week 4 v3: Macro + Full Train + Multi-Seed (`src/week4_macro_state_v3.m`)

**Config delta vs v2:** trainingRange 2000 → 2515 (full train CSV including
2018-Q4 rate-hike mini-bear and 2019 rate-cut reversal); numEpisodes 130 → 165
so episode windows actually reach 2018-2019 data; 5 seeds [1000, 2000, 3000,
4000, 5000] with fresh episode generation, network init, and training per seed.

**Cross-seed summary (5 seeds × 30 eval windows = 150 paths):**
- Success rate: **11.6/30 = 39%** (across-seed std 1.52)
- Mean Sharpe: **0.25** (across-seed std **0.27** — highly reproducible)
- Within-seed Sharpe std (avg): 2.85 (still high; this is regime-driven, not
  strategy-driven)
- Mean MaxDD: **6.19%** (across-seed std 1.08%) — sharply improved
- MaxDD P90: **12.87%** (vs v2's 19.93%) — tail clipped
- Worst Sharpe: **-5.78** (vs v2's -7.80) — extreme losses reduced
- Mean Terminal: 99,625 (across-seed std only 682 — extreme stability)
- Terminal P10: 91,141

**Action histogram across all 4,500 eval steps:**
| Action | Share | Role |
|---|---|---|
| 6  | 27.3% | Deep defensive (TLT-heavy + JNJ) |
| 9  | 21.2% | Mid-risk balanced |
| 12 | 18.9% | Balanced-aggressive (UNH/AAPL/NVDA) |
| 13 | 12.5% | Aggressive (UNH/NVDA) |
| 15 | 5.3%  | All-in NVDA |
| 1-5, 7-8, 10-11, 14 | 14.8% combined | Long tail of conditional choices |

For the first time agent actually uses defensive actions 1-6 (~3% on 1-5,
27% on 6). Compared to v2's single-action monoculture (79% on #7), this is
genuine conditional asset allocation.

**Three-version comparison:**

| Metric | Baseline | v2 | **v3** |
|---|---|---|---|
| Success rate | 15/30 | 10/30 | 11.6/30 |
| Mean Sharpe | 0.59 | -0.01 | **0.25** |
| Within-seed Sharpe std | 2.75 | 2.95 | 2.85 |
| Across-seed std | n/a | n/a | **0.27** |
| Mean MaxDD | 10.72% | 7.85% | **6.19%** |
| MaxDD P90 | n/a | 19.93% | **12.87%** |
| Worst Sharpe | n/a | -7.80 | **-5.78** |
| Mean Terminal Wealth | 103,255 | 98,275 | 99,625 |
| Action diversity | 4 actions | 1 dominates (79%) | **5+ regular** |

**Verdict:** v3 is a **risk-control win, not a mean-return win**.
- ✓ Tail risk significantly improved (MaxDD P90, worst Sharpe)
- ✓ Strategy now genuinely conditional on regime (action histogram)
- ✓ Extremely reproducible across seeds (Sharpe std 0.27)
- ✗ Mean Sharpe and Mean Terminal Wealth slightly below baseline
- ✗ Within-window Sharpe variance unchanged (this is eval-period regime
  diversity, not strategy noise)

**Narrative for paper/mentor:** "Adding macro state and full-period training
transforms the agent from a single-regime aggressive policy (high mean, high
variance, high tail risk) into a multi-regime conditional policy (slightly
lower mean, comparable variance, significantly reduced tail risk and improved
reproducibility)."

**Locked next step — Week 5 CQL:**
- v3 has established that state + data extensions can take the agent to
  regime-aware behavior. What remains is over-confidence on OOD states like
  2022 stock-bond simultaneous crash (which shows up as action 6 still being
  chosen when TLT is failing).
- CQL's `α * (logsumexp(Q_all) - Q_behavior)` penalty in the critic loss
  directly suppresses Q on under-represented (state, action) pairs, which is
  exactly the v3 failure mode.
- Expected effect: mean Sharpe holds at ~0.25, worst Sharpe lifts from -5.78
  toward -4, MaxDD P90 drops below 10%. If mean collapses, CQL was too
  conservative (tune α down).
- Keep v3's multi-seed + full-train protocol; only change the critic update
  rule. Custom training loop required (`trainFromData` does not support
  custom losses; will need a manual minibatch loop with `dlfeval` + custom
  gradient function).
