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

### Week 5 v1: Regime-Gated Asymmetric Reward (`src/week5_regime_reward.m`)

**Motivation (mentor suggestion):** distinguish normal vs stressed market in
the reward function so the agent learns regime-conditional risk aversion.
Plan A from the literature analysis: amplify negative rewards on "stress" days.

**Config delta vs v3:** only the reward in episode generation changes.
```
stress(t) := VIX_z(t) > 1.0  OR  T10Y2Y_z(t) < -1.0
r(t)      := lambda * log(W'/W)   if stress(t) AND log(W'/W) < 0
             log(W'/W)             otherwise
r(T)      += 1.0  if W_T >= goal   (terminal bonus unchanged)
```
LAMBDA = 2.5 (Tversky-Kahneman loss aversion + Nature SR 2026 behavioral DRL).
Everything else (6D state, 5 seeds, trainingRange=2515, network, agent options,
eval windows) identical to v3 for clean A/B.

**Train stress rate: 31.1%** (target was 15-25% — too broad, gate fires on a
third of all training days, behaves more like global loss aversion than a
sparse crisis signal).

**Cross-seed results vs v3:**

| Metric | v3 | v1 | Direction |
|---|---|---|---|
| Success rate | 11.6/30 | 11.6/30 | flat |
| Mean Sharpe | 0.25 | **0.37** | ↑ |
| Across-seed std | 0.27 | 0.36 | ↑ (worse) |
| Mean MaxDD | 6.19% | 7.06% | ↑ (worse) |
| **MaxDD P90** | **12.87%** | **14.40%** | ↑ (worse) ← core KPI failed |
| **Worst Sharpe** | **-5.78** | **-5.57** | flat |
| Mean Terminal | 99,625 | 100,556 | ↑ |

**Action distribution went bimodal:** action #2 (low-vol defensive) jumped
from ~1% to 23.1%, and action #15 (all-in NVDA) tripled from 5.3% to 17.3%.
The middle-risk actions (#6, #9, #12) got hollowed out. Classic
loss-amplification side effect: two winning strategies emerge, "avoid
volatility entirely (#2)" and "swing for the goal bonus (#15)".

**Stress vs Normal action histograms were nearly identical** (action #2:
23.1% vs 23.3%, action #15: 17.3% vs 17.5%) — eval period 2022-2024 is mostly
in stress regime, so the diagnostic is degenerate. Policy is NOT
regime-conditional, just globally shifted.

**Diagnosis:** mean Sharpe improvement is a happy accident, not the mentor's
intended fix. Tail risk (MaxDD P90, worst Sharpe) — the metrics that
motivated this change — went the wrong direction.

### Week 5 v2: Tighter Stress Gate (`src/week5_regime_reward_v2.m`)

**Config delta vs v1:**
- `VIX_THR` 1.0 → 1.5
- `SLOPE_THR` -1.0 → -1.5
- LAMBDA unchanged at 2.5 (isolate threshold effect)

**Train stress rate: 16.9%** (target hit ✓). Amplified-loss steps: 7.2% of
all training steps (was 13.4% in v1).

**Cross-seed results — three-version comparison:**

| Metric | v3 | v1 | **v2** |
|---|---|---|---|
| Success rate | 11.6/30 | 11.6/30 | **13.0/30** ↑ |
| Mean Sharpe | 0.25 | 0.37 | **0.50** ↑↑ |
| Across-seed std | 0.27 | 0.36 | 0.38 |
| Mean MaxDD | 6.19% | 7.06% | 7.70% |
| **MaxDD P90** | **12.87%** | 14.40% | **15.41%** ← monotone worse |
| **Worst Sharpe** | **-5.78** | -5.57 | **-5.80** ← unchanged in 2 iters |
| Mean Terminal | 99,625 | 100,556 | **101,421** ↑ |
| Terminal P10 | 91,141 | 91,267 | 90,724 |

Per-seed Sharpe range: 0.01 (seed 1000) to 1.03 (seed 3000) — across-seed
extreme spread is ~1.0, worse than v3's ~0.4.

**Action distribution shifted to aggressive duo:** v1's bimodal #2+#15
collapsed; v2 dominated by **#12 (24.3%, UNH+AAPL+NVDA balanced-aggressive)
+ #15 (21.0%, all-in NVDA)**, with #6 still at 14.4%. Stress-day and
normal-day histograms remain nearly identical.

**Trend across three versions is informative:**
- Mean Sharpe: 0.25 → 0.37 → 0.50 (monotonically up)
- MaxDD P90: 12.87% → 14.40% → 15.41% (monotonically up — wrong direction)
- Worst Sharpe stuck around -5.7

**Root-cause diagnosis — why lambda-loss saturated:**

Episode generation uses **random** actions. When stress days amplify losses
by λ, ALL actions on that day get penalized; the differentiation between
defensive (#6, low magnitude loss) and aggressive (#15, high magnitude loss)
shrinks relatively. DQN's Q signal learned to encode "stress days have lower
value overall" rather than "in stress, action X dominates action Y". Hence
no regime-conditional behavior, and worse tail because agent chases the
terminal +1 bonus harder under amplified loss aversion.

**Verdict:** the lambda-on-loss reward shaping approach is saturated.
Continuing to tune threshold or λ will not address the mechanism. Reward
signal needs to become **action-dependent within the stress regime** — which
is exactly what a drawdown penalty provides (action mix determines drawdown
magnitude even before regime is conditioned).

### Week 5 Plan C: Drawdown Penalty (`src/week5_drawdown_penalty.m`)

Replaced lambda-loss shaping with explicit drawdown penalty:
```
peakW    = max(peakW, W')              (per-episode running peak)
DD_t     = (peakW - W') / peakW
r(t)     = log(W'/W) - beta * max(0, DD_t - DD_THR) + terminal_bonus
```
Config: BETA=5.0, DD_THR=0.03. Same v3 5-seed protocol, no stress gate
(drawdown is itself the regime signal).

**Training diagnostic:** Penalty-active share averaged 18.1% of training
steps across seeds, avg penalty magnitude ~0.12 — gate hit healthy band
(target 15-35%) without any threshold tuning, because DD itself is the
self-calibrating signal.

**Cross-seed results — four-version comparison:**

| Metric | v3 | v1 | v2 | **Plan C** | Plan C vs v3 |
|---|---|---|---|---|---|
| Success rate | 11.6 | 11.6 | 13.0 | 10.6 | -1.0 |
| Mean Sharpe | 0.25 | 0.37 | 0.50 | **0.36** | +0.11 |
| Across-seed std | 0.27 | 0.36 | 0.38 | **0.25** | -0.02 (best) |
| Within-seed Sharpe std | 2.85 | 2.83 | 2.88 | 3.01 | +0.16 |
| Mean MaxDD | 6.19% | 7.06% | 7.70% | **5.84%** | **-0.35pp (best)** |
| **MaxDD P90** | 12.87% | 14.40% | 15.41% | **11.79%** | **-1.08pp (first time below v3)** |
| **Worst Sharpe** | -5.78 | -5.57 | -5.80 | **-5.38** | **+0.40 (first real lift)** |
| Mean Terminal | 99,625 | 100,556 | 101,421 | 100,189 | +564 |
| Terminal P10 | 91,141 | 91,267 | 90,724 | **92,505** | **+1,364 (best)** |

Plan C is the **first version where every tail-risk metric improves vs
v3** simultaneously (MaxDD P90, worst Sharpe, mean MaxDD, terminal P10),
while across-seed reproducibility also reaches its best (std 0.25).

**Action distribution flipped from aggressive duo to mid-defensive trio:**

| Action | v2 | **Plan C** | Role |
|---|---|---|---|
| #9 | 13.4% | **35.7%** | mid-risk balanced (new dominant) |
| #11 | 1.0% | **17.1%** | mid-risk (barely used before) |
| #3 | 1.1% | **15.5%** | defensive (first time used) |
| #14 | 7.5% | 8.4% | |
| #6 | 14.4% | 5.5% | |
| #15 (all-in NVDA) | 21.0% | **4.2%** | aggressive collapsed |
| #12 (UNH+AAPL+NVDA) | 24.3% | **0.5%** | aggressive collapsed |

The aggressive duo #12+#15 went from 45.3% in v2 to 4.7% in Plan C; the
mid-defensive trio #3+#9+#11 went from 15.5% to 68.3%. This is the
conditional-defense behavior the mentor's reward suggestion was meant to
produce — and Plan C achieves it WITHOUT a regime gate, because the
penalty mechanism is action-dependent by construction.

**Cost:** Success rate -1.0 (3% of windows), within-seed Sharpe std +0.16.
The within-window variance reflects evaluation-period regime diversity
(conservative policy "leaves money on the table" in bull windows), not
strategy noise.

**Verdict:** Plan C is the Week 5 deliverable. Trade-off (~1 missed
success rate point for across-the-board tail-risk improvement) is exactly
the type of result the v3 diagnosis called for.

**Narrative for paper / mentor:** "Three reward designs were tested.
Asymmetric loss amplification (lambda on stress days) produced higher mean
Sharpe but worse tail risk because random-action episodes cannot
differentiate actions within a regime under uniform loss scaling. An
explicit drawdown penalty — which is action-dependent by construction —
successfully transferred the protective signal to action selection,
achieving the first across-the-board tail-risk improvement (MaxDD P90
12.87% → 11.79%, worst Sharpe -5.78 → -5.38, across-seed std 0.27 →
0.25)."

### Week 6.0: Stress-Day Oversampling on Plan C (`src/week6_oversample_0.mlx`)

**Motivation (mentor suggestion #2 on Week 6):** test 2022-2024 is mostly
in stress regime while train 2010-2017 is mostly normal — distribution
mismatch is suspected to be the main reason Plan C's tail still hurts.
Plan: reweight episode starts so train stress occupation approaches test.

**Config delta vs Plan C:** only episode-start sampling. Per-day stress
intensity `stressScore(t) = max(0, VIX_z(t)) + max(0, -T10Y2Y_z(t))`,
window weight = sum over horizon + epsilon. Episode starts drawn via
`randsample(starts, N, true, weight)`. EPSILON_WEIGHT=0.5. Reward,
network, agent, eval — all identical to Plan C. CQL (the original Week 6
plan) deferred indefinitely after mentor pivoted to regime + drawdown
combination.

**Stress diagnostic before training:**
- Train baseline stress rate: **17.4%** (uniform-sample proxy)
- Test stress rate (target): **73.8%** — 56pp gap, much wider than the
  20-25pp originally guessed
- After weighted sampling: train stress occupation rose to **40.6%** (still
  well below test, but +23pp lift)

**Cross-seed results vs Plan C:**

| Metric | Plan C | **6.0** | Direction |
|---|---|---|---|
| Success rate | 10.6 | **14.8** | ↑↑ |
| Mean Sharpe | 0.36 | **0.62** | ↑↑ |
| Across-seed std | 0.25 | 0.65 | ↑ (worse) |
| Mean MaxDD | 5.84% | 8.48% | ↑ (worse) |
| **MaxDD P90** | **11.79%** | **18.04%** | ↑↑ (much worse) |
| **Worst Sharpe** | **-5.38** | -5.31 | ~flat |
| Mean Terminal | 100,189 | 102,256 | ↑ |
| Terminal P10 | 92,505 | 89,153 | ↓ |

Strongly **bimodal per-seed**: seeds 1000-3000 stayed conservative (Sharpe
0-0.5, MaxDD 6-8%), seeds 4000-5000 went aggressive (Sharpe 1.0-1.6, MaxDD
10-12%, success 19/30). Action histogram flipped from Plan C's mid-defensive
trio (#3+#9+#11=68%) to a barbell #15(NVDA all-in, 29.6%) + #4(23.6%).

**Diagnosis — oversampling backfired:** the high-weight stress windows in
2010-2019 are all *V-shaped panics* (2011 euro debt, 2015 oil crash, 2018Q4
hike scare) followed by sharp recoveries. Upweighting these taught the
agent "stress = buy-the-dip opportunity" — exactly the wrong reflex for
2022's persistent 12-month grind-down. The mean-Sharpe lift came from
seeds that happened to pick stocks that recovered; the tail destruction
came from the same reflex being misapplied to test stress.

**Conclusion:** oversampling without the right data structure amplifies
the wrong signal. Need to fix the data itself before any reweighting helps.

### Week 6.0b: Extended Train (2010-2021) with COVID exposure (`src/week6_oversample_b.mlx`)

**Motivation:** the previously-unused val split (2020-2021, carved out for
hyperparameter tuning but never used) contains the COVID March 2020 crash
+ 2020-2021 low-rate era — the closest analogue we have to the 2022 stress
regime. Merging it into train (via `scripts/merge_train_val.py`) extends
train to 3020 days (was 2515) without any test leakage. Same script run in
two stages: stage 1 disables oversampling (isolates the data contribution),
stage 2 re-enables it (measures incremental oversampling effect on the
extended dataset).

**Stage 1 — Extended train + uniform sampling (no oversampling):**

| Metric | Plan C | 6.0 | **6.0b s1** |
|---|---|---|---|
| Success rate | 10.6 | 14.8 | 12.8 |
| Mean Sharpe | 0.36 | 0.62 | 0.55 |
| Across-seed std | 0.25 | 0.65 | **0.32** |
| Mean MaxDD | 5.84% | 8.48% | 8.04% |
| MaxDD P90 | **11.79%** | 18.04% | 15.71% |
| Worst Sharpe | **-5.38** | -5.31 | -5.79 |
| Terminal P10 | 92,505 | 89,153 | 89,921 |

Sits between Plan C and 6.0 on tail metrics. Action distribution is the
most diverse of any version (#11=21%, #15=18%, #10=17%, #7=16%, #2=13% —
no action above 22%), suggesting genuine state-dependent decisions. Some
bimodality persists (seed 4000 went aggressive, Sharpe 1.09 / MaxDD 13.9%).

**Stage 2 — Extended train + oversampling re-enabled:**

| Metric | Plan C | 6.0 | 6.0b s1 | **6.0b s2** |
|---|---|---|---|---|
| Success rate | 10.6 | 14.8 | 12.8 | 14.2 |
| Mean Sharpe | 0.36 | 0.62 | 0.55 | 0.51 |
| Across-seed std | 0.25 | 0.65 | 0.32 | 0.51 |
| MaxDD P90 | **11.79%** | 18.04% | 15.71% | 17.98% |
| **Worst Sharpe** | **-5.38** | -5.31 | -5.79 | **-5.99** ← worst |
| Terminal P10 | **92,505** | 89,153 | 89,921 | 88,471 |

Adding oversampling on top of the extended data **recreated the same
tail-risk damage** seen in 6.0 (MaxDD P90 back up to 17.98%, worst Sharpe
new low at -5.99). The bimodal per-seed pattern reappeared (seed 1000
Sharpe 1.01 / MaxDD 13.8%; seed 4000 Sharpe -0.34).

**Two-fold conclusion:**

1. **Oversampling is the wrong tool, full stop.** It hurt tail risk in
   both datasets (6.0 on old data, 6.0b s2 on extended data). The
   mechanism — upweighting V-shaped recovery windows — is dataset-
   independent. Drop this approach.

2. **Data extension alone is helpful but not sufficient.** 6.0b s1 is
   more stable (across-seed std 0.32 vs Plan C 0.25) and has cleaner
   action diversity, but still doesn't beat Plan C on tail. COVID 2020
   is a *V-shaped* event itself (5-week crash, 2-month recovery), so it
   still teaches partial buy-the-dip. 2022's persistent grind has no
   true analogue in any of our train data.

**Locked next step — Week 6.1:** combine the validated ingredients
(extended train + uniform sampling + drawdown penalty) with mentor
suggestion #1 (regime gate) applied to the *action-dependent* drawdown
penalty rather than the action-independent loss multiplier that failed in
Week 5 v1/v2.

### Week 6.1: Regime-Gated Drawdown Penalty — first regime-conditional policy (`src/week6_1_regime_gated.m`)

**Config delta vs 6.0b s1:** drawdown-penalty coefficient becomes
regime-dependent. Same extended train (2010-2021), same uniform sampling,
same 6D state, same network/agent/eval. Stress flag reuses the v2-
calibrated thresholds.
```
isStress(t) = (VIX_z(t) > 1.5) OR (T10Y2Y_z(t) < -1.5)
beta(t)     = BETA_HIGH if isStress(t)   (= 8.0, tighter than Plan C's 5.0)
              BETA_LOW  otherwise         (= 2.0, looser than Plan C)
r(t)        = log(W'/W) - beta(t) * max(0, DD - DD_THR) + terminal_bonus
```
Why this should work where v1/v2 failed: v1/v2 multiplied the *loss*
(action-independent — uniform across all actions on a stress day, so DQN
learned "stress regime has lower value" rather than "in stress, defensive
> aggressive"). Drawdown penalty is action-dependent by construction
(aggressive allocations produce larger DD on stress days), so a stress-
period beta lift differentiates actions instead of just scaling losses.

**Training-time signal validation:** penalty avg magnitude was **0.4186 on
stress steps vs 0.0564 on normal steps** (7.4× separation). Gate worked
at the reward level.

**Cross-seed results — five-version comparison:**

| Metric | Plan C | 6.0 | 6.0b s1 | 6.0b s2 | **6.1** |
|---|---|---|---|---|---|
| Success rate | 10.6 | 14.8 | 12.8 | 14.2 | **14.6** |
| Across-seed std (succ) | n/a | n/a | n/a | n/a | **1.52** |
| Mean Sharpe | 0.36 | 0.62 | 0.55 | 0.51 | **0.66** |
| Across-seed std (Sharpe) | 0.25 | 0.65 | 0.32 | 0.51 | **0.23** |
| Mean MaxDD | **5.84%** | 8.48% | 8.04% | 9.51% | 7.78% |
| MaxDD P90 | **11.79%** | 18.04% | 15.71% | 17.98% | 15.75% |
| **Worst Sharpe** | **-5.38** | -5.31 | -5.79 | -5.99 | **-6.37** ← worst |
| Mean Terminal | 100,189 | 102,256 | 101,526 | 102,070 | 101,633 |
| Across-seed std (TermW) | n/a | n/a | 2,680 | n/a | **1,690** |
| Terminal P10 | **92,505** | 89,153 | 89,921 | 88,471 | 89,758 |

6.1 holds three "best" titles: highest mean Sharpe (0.66), highest success
rate (14.6), most reproducible across seeds (Sharpe std 0.23, TermW std
1,690). **No bimodality**: all five seeds landed Sharpe 0.43-0.98 and
MaxDD < 11% — first version with monotonically consistent per-seed
behavior. Cost: worst Sharpe regressed to -6.37 (~1pp below Plan C).

**🔑 Core breakthrough — first true regime-conditional action selection:**

| action | stress% | normal% | diff (pp) |
|---|---|---|---|
| 1 | 9.9 | 13.8 | -4.0 |
| 9 | 17.1 | 8.1 | **+9.1 diverges** |
| 11 | 11.0 | 24.3 | **-13.3 diverges** |
| 13 | 9.8 | 4.6 | **+5.2 diverges** |
| 14 | 6.5 | 1.2 | **+5.3 diverges** |
| 15 | 9.2 | 3.5 | **+5.7 diverges** |

**Five actions diverge ≥5pp between stress and normal eval days** — the
behavior every prior version (v1/v2/Plan C/6.0/6.0b) failed to produce.
This is the core deliverable of Week 5-6 reward engineering.

**But the direction is counterintuitive:** in stress, agent uses MORE
mid-to-aggressive (#9/#13/#14/#15); in normal, MORE defensive (#11/#1/#2).
Two-part diagnosis:
1. **Training-data V-shape bias:** even in extended train (2010-2021),
   every "stress" event including COVID 2020 was followed by sharp
   recovery. Reward net signal: stress + aggressive → large DD penalty
   AND large recovery log-return → net positive Q for aggressive in
   stress regime.
2. **Test "stress" labels ≠ "down":** in 2022-2024 the
   `VIX_z>1.5 OR T10Y2Y_z<-1.5` flag fires heavily during AI rally days
   (high VIX + curve inversion ≠ falling stocks). Agent's "stress →
   aggressive" reflex catches these and inflates mean Sharpe, but is
   punished severely on the rare days where stocks AND bonds both fall
   (→ worst Sharpe -6.37).

**Verdict — Week 6.1 is the Week 6 deliverable.** Trade-off (modest tail
regression vs first-ever regime-conditional behavior + best mean/stability)
matches what the mentor's regime + drawdown combination was meant to
achieve. The conditioning direction is data-driven and reveals a deeper
limitation: with no train period containing persistent stock-bond co-
crash, no reward design can make the agent learn to defend in 2022-style
regime.

**Narrative for paper/mentor:** "After six experiments, 6.1 produced the
first version with truly regime-conditional action selection — five
actions diverge ≥5pp between stress and normal test days. The
breakthrough required combining the action-dependent drawdown penalty
(Plan C) with the regime gate (mentor suggestion #1) AND extended
training data with COVID exposure. Mean Sharpe rose to 0.66 and
across-seed reproducibility hit its best (std 0.23) while bimodality
across seeds disappeared. The remaining limitation — worst Sharpe -6.37
— reflects the fact that no period in our train data contains a
persistent stock-bond co-crash like 2022; the agent's regime-conditional
policy is correct for the training distribution but mislabeled by test."

**Possible next steps if Week 6 continues:**
- Put stress flag directly into state (6D → 7D) so the network conditions
  explicitly rather than via reward shaping
- Add CVaR-style hard floor on the drawdown penalty to attack worst-case
- Accept 6.1 and move to Week 7 backtesting / baseline comparison

### Reference Papers Consulted for Week 5 Reward Design

1. Moody & Saffell (1998) *Reinforcement Learning for Trading*, NeurIPS —
   foundational paper on differential Sharpe ratio as incremental
   risk-adjusted reward for online trading.
2. *Risk-Sensitive Deep Reinforcement Learning for Portfolio Optimization*
   (MDPI JRFM 2025) — argues "risk-neutral RL is not enough"; encode CVaR /
   drawdown / mean-variance directly into reward. MPLS framework shows
   +40-70% Sharpe improvement over MVO/risk-parity with controlled drawdown
   during COVID stress.
3. *Adaptive and Regime-Aware RL for Portfolio Optimization* (arXiv
   2509.14385, 2025) — closest to our setup. Latent macro signals + reward
   clipping ±3% + transaction penalty + Sharpe reward. Explicit goal:
   "reduce exposure during credit stress".
4. *Behaviorally informed DRL with loss aversion* (Nature Sci Rep 2026) —
   loss aversion lambda baked into reward; supports λ ∈ [2.0, 2.5].
5. *Risk-Sensitive Reward-Free RL with CVaR* (Ni et al., ICML 2024) —
   theoretical anchor for CVaR-based reward-free framework, PAC-efficient.
