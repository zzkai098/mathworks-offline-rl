# GBWM P(goal) — Offline-RL Sample-Efficiency Study

A MATLAB study of **offline reinforcement learning for Goal-Based Wealth
Management (GBWM)**, extending the MathWorks GBWM demo with **CQL** and **IQL**,
and — crucially — with a **dynamic-programming optimum as a measurable ceiling**
so we can ask the one question offline RL can answer honestly.

## The one-sentence framing (read this first)

> I first tried to make RL "beat 60/40 on Sharpe" on real 2010-2025 returns and
> **proved with a power analysis that this is statistically unachievable on the
> available data** (≈600 test days → Sharpe SE ≈ 0.65; detecting the ~0.1 edge
> that exists would need decades of independent multi-regime data; every CI
> includes 0). So I pivoted to the GBWM-native **P(goal)** game, where a **DP
> optimum gives a measurable ceiling**, and used it to ask: **does offline
> pessimism (CQL / IQL) recover more of that optimum than vanilla RL as the
> logged dataset shrinks?**

**Honesty boundary (non-negotiable):** every result here is under a
return-generating model **calibrated on train (2010-2019) and evaluated by Monte
Carlo over that model**. Significance is over a *self-specified simulator with
controllable N* — it is **never** a claim of real-market outperformance. This is
the standard (and only feasible) evaluation for a 10-year lifecycle strategy —
the demo itself uses simulated scenarios — because a realized historical anchor
needs many independent 10-year paths (≈ centuries). What transfers to reality is
**method-level** (algorithm sample efficiency) and **structural/directional**
(dynamics > static), not the absolute P(goal) *levels*.

## The game

- 10-year horizon, **annual** rebalance (10 decisions) + annual contributions.
- **State** (shared by RL and DP): `(normalized wealth, time fraction, regime)`,
  where `regime ∈ {stress, normal}` from the mentor's gate
  `VIX_z > 1.5 OR T10Y2Y_z < -1.5` — the discrete regime is how macro enters
  while keeping the DP tractable.
- **Actions:** 15 long-only efficient-frontier portfolios (MATLAB `Portfolio`).
- **Objective (primary):** `P(goal) = P(terminal wealth ≥ goal)`, sparse terminal
  reward (the demo's native objective). `goal = 1949` calibrated so the DP
  optimum's P(goal) ≈ 0.75 (a reach-y stretch where dynamics matters).
- **Reference lines:** DP optimum (ceiling) · best static mix · best glide path ·
  regime-agnostic DP (isolates the regime signal) · 60/40 (floor).

## What was found (full numbers in `experiments/ledger.md`)

1. **Pre-flight — the game is winnable by dynamics.** At the stretch goal,
   regime-aware DP 0.751 > regime-agnostic 0.723 > best-static 0.625 > 60/40 0.000.
   Dynamics buys **+0.098** over the best static policy and the regime signal
   **+0.028** — not merely "more equity." (60/40 = 0 ⇒ baseline = best-static.)

2. **PRIMARY — gap-to-DP sample efficiency (`out/gap_to_dp.png`).** All methods
   approach the DP optimum as data grows. **IQL is the most sample-efficient**
   (gap 0.02 at 5000 episodes, best at medium-large data). vanilla is a robust
   baseline. Naive CQL (α=1) is worst — a reward-scale artifact.

3. **CQL α-sensitivity (`out/cql_alpha.png`) — fair shot.** P(goal) rewards ∈
   [0,1] make α=1 ~50× too strong. At the reward-scaled α≈0.01, **CQL becomes
   competitive** (beats vanilla at size 500/2000) but is tuning-sensitive;
   IQL needs no tuning. Matches the offline-RL literature.

4. **Drawdown-shaping variant — mentor's regime-gated penalty works.** Adding the
   Week-6.1 penalty (β=8/2, DD_THR=0.03) cuts mean MaxDD **25-42%** and lifts
   downside termP10 **20-31%** at a modest P(goal) cost. **IQL-shaped's MaxDD
   (0.081) is below even the tail-blind DP optimum (0.101)** — the shaping buys
   tail protection the P(goal)-optimal policy lacks.

5. **Beat-baseline (supporting).** The learned dynamic policy (IQL) beats
   best-static on P(goal) by **+0.093** (attributable to dynamics + regime), and
   IQL-shaped **dominates best-static on P(goal), MaxDD, and termP10 together**.
   Framed as effect size + attribution — the MC p-value is trivially small (cheap
   at large N) and is *not* the claim.

## Salvaged from the prior MATLAB project (Weeks 4-8) — ADOPT / DROP

**Adopted (effective):** `diff(log P)` returns; 15-asset universe + efficient-
frontier action set; macro regime gate `VIX_z>1.5 OR T10Y2Y_z<-1.5`; regime-gated
drawdown penalty (β=8/2, DD_THR=0.03); multi-seed + across-seed reporting; the
Week-8 honesty spine (block-bootstrap CI, Deflated Sharpe, no-hill-climbing-test).

**Dropped (proven ineffective or superseded):** λ-loss reward (saturated);
stress-day oversampling (harmful); stress-flag-in-state 7D (seed-unstable,
redundant with regime); extended-train val-into-train merge (leakage — we re-split
from raw); daily sliding-window offline episodes (→ GBWM MC episodes); single-
segment 2022-25 Sharpe backtest (power-dead → P(goal)+MC).

## Layout & how to run

```
gbwm_pgoal/
  gbwm.m                       library: return model, frontier, DP, MC eval, logging
  dp/preflight.m               Part 1 pre-flight (DP + controls, go/no-go)
  offline/train_offline.m      vanilla / CQL / IQL trainers (dlnetwork + dlfeval)
  experiments/
    sweep.m                    Part 2 gap-to-DP sweep  -> out/gap_to_dp.png
    cql_alpha_sweep.m          Part 3 CQL alpha fairness -> out/cql_alpha.png
    drawdown_variant.m         Part 4 mentor's shaping variant
    ledger.md                  all result tables + beat-baseline summary
    out/                       figures + .mat results
  _prototype_python/           the Python prototype that validated the design
```

Run from the repo root (needs Financial + Deep Learning toolboxes):
```matlab
addpath('gbwm_pgoal'); addpath('gbwm_pgoal/offline'); addpath('gbwm_pgoal/experiments');
preflight            % Part 1
sweep                % Part 2 (~3h: 7 sizes x 5 seeds x 3 methods)
cql_alpha_sweep      % Part 3
drawdown_variant     % Part 4
```

The DP solver + evaluation core is intentionally small and self-contained; a
Python prototype (`_prototype_python/`) cross-validates the pre-flight numbers
(dynamics margin +0.125 identical across MATLAB and NumPy).

## Known limitations / honest next steps

- **Simulator-bound levels.** P(goal) levels are optimistic (2010-2019 bull) and
  unverifiable on real data. The credible path to strengthen this is
  **cross-model robustness** (re-fit on 2000s/GFC or stressed parameters and show
  the *method* and *directional* findings hold) — deferred.
- **β not tuned for this game** (carried from the daily project); a β-sweep would
  trace the P(goal)–tail frontier and likely give IQL-shaped a better point.
- **No realized-history walk-forward** — infeasible for a 10-year horizon (only
  ~1 realized path in 2010-2025); this is a property of the problem, not an
  omission.
