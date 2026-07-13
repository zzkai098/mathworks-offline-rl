# GBWM P(goal) — Results Ledger

All numbers are **under a regime-switching return model fit on TRAIN (2010-2019)
monthly returns only**, evaluated by Monte-Carlo over that model. They are
statements about the calibrated simulator, NOT realized-market forecasts (see
README honesty section). Instance: 10-year annual GBWM, W0=100, annual
contribution 10, **goal=1949** calibrated so the DP optimum's P(goal) ≈ 0.75
(a "reach-y" stretch where dynamic risk-management matters). Actions = 15
long-only efficient-frontier portfolios. Regime = {stress, normal} from the gate
`VIX_z>1.5 OR T10Y2Y_z<-1.5`. State = (normWealth, timeFraction, regime).

---

## Part 1 — Pre-flight: is the game winnable by dynamics?
P(goal) under the true model (MC N=200k). `src/dp/preflight.m` / `gbwm.m`.

| policy | P(goal) |
|---|---|
| **regime-aware DP (ceiling)** | 0.751 |
| regime-agnostic DP | 0.723 |
| best glide path | 0.627 |
| best static (#15) | 0.625 |
| 60/40 | 0.000 |

**Attribution margins:**
- regime signal (aware − agnostic) = **+0.028**
- dynamics (agnostic − best-static) = **+0.098**
- dynamics + regime over best-static (aware − static) = **+0.125**
- risk level (best-static − 60/40) = +0.625

**Read:** the game is winnable by *dynamics*, not just by holding more equity —
dynamic risk-management beats the best static policy by +0.098 and the regime
signal adds +0.028. 60/40 = 0 at this stretch → baseline switched to best-static.
Cross-validated: the Python prototype gave +0.125 identically.

---

## Part 2 — PRIMARY: gap-to-DP sample-efficiency curve
gap-to-DP = ceiling(0.750) − P(goal). Mean over 5 seeds. Behavior policy =
random-over-frontier. `experiments/sweep.m` → `out/gap_to_dp.png`.

| dataset size (episodes) | vanilla | CQL (α=1) | IQL |
|---|---|---|---|
| 50   | 0.165 | 0.402 | 0.149 |
| 100  | 0.126 | 0.257 | 0.157 |
| 200  | 0.118 | 0.185 | 0.109 |
| 500  | 0.115 | 0.133 | **0.072** |
| 1000 | 0.087 | 0.107 | **0.051** |
| 2000 | 0.073 | 0.119 | **0.028** |
| 5000 | 0.033 | 0.062 | **0.021** |

**Read:** all three recover the DP optimum as data grows (sample-efficiency
shape holds). **IQL is the most sample-efficient** (gap 0.02 at 5000, best at
medium-large data). vanilla is a robust baseline. CQL at the naive α=1 is worst
(catastrophic at low data) — a reward-scale artifact, fixed in Part 3. The hoped
"pessimism dominates at LOW data" is NOT clean: IQL ≈ vanilla at 50-100, IQL's
edge grows in the mid-data regime.

---

## Part 3 — CQL α-sensitivity (fair shot, no strawman)
P(goal) rewards ∈ [0,1] ⇒ CQL penalty (logsumexp over 15 actions ≈ 2.7) at α=1
dwarfs the TD loss (~0.05) by ~50×. Reward-scaled α ≈ 0.01. gap-to-DP, 5 seeds.
`experiments/cql_alpha_sweep.m` → `out/cql_alpha.png`.

| size | CQL α=0.01 | CQL α=0.03 | CQL α=0.1 | CQL α=0.3 | CQL α=1 | vanilla | IQL |
|---|---|---|---|---|---|---|---|
| 100  | 0.137 | 0.177 | 0.182 | 0.156 | 0.165 | 0.126 | 0.157 |
| 500  | **0.062** | 0.158 | 0.110 | 0.086 | 0.139 | 0.115 | 0.072 |
| 2000 | 0.050 | 0.062 | 0.102 | 0.092 | 0.068 | 0.073 | 0.028 |

**Read:** at the reward-scaled α≈0.01, CQL becomes competitive — beats vanilla at
size 500/2000, ~ties IQL at 500 — but is α-sensitive and higher-variance. IQL
needs no tuning and stays best at large data. α chosen a-priori by reward-scale
reasoning (not by peeking at the evaluator). Matches the offline-RL literature:
CQL is strong but tuning-sensitive; IQL is robust.

---

## Part 4 — Drawdown-shaping variant (mentor's regime-gated penalty)
Per-step reward += `−β(regime)·max(0, DD−0.03)`, β_stress=8 / β_normal=2 (Week
6.1). Objective becomes P(goal)+tail. size=2000, 5 seeds, MC N=100k with path
MaxDD. `experiments/drawdown_variant.m`.

| policy | P(goal) | mean MaxDD | p90 MaxDD | termP10 |
|---|---|---|---|---|
| DP optimum (tail-blind) | 0.756 | 0.101 | 0.355 | 842 |
| best static | 0.629 | 0.303 | 0.596 | 499 |
| 60/40 | 0.000 | 0.006 | 0.018 | 420 |
| vanilla pure  | 0.677 | 0.169 | 0.413 | 610 |
| vanilla shaped| 0.662 | 0.125 | 0.338 | 743 |
| cql pure  | 0.677 | 0.180 | 0.432 | 607 |
| cql shaped| 0.658 | 0.135 | 0.350 | 755 |
| **iql pure**  | **0.722** | 0.139 | 0.409 | 559 |
| **iql shaped**| 0.658 | **0.081** | **0.272** | **734** |

**Read:** the regime-gated drawdown shaping works — **mean MaxDD down 25-42% and
termP10 up 20-31% across all methods**, at a modest P(goal) cost (−0.015 to
−0.064). **IQL-shaped's MaxDD (0.081) is even below the tail-blind DP optimum
(0.101)** — the shaping buys tail protection the P(goal)-optimal DP does not
have, tracing a P(goal)–tail tradeoff. (β=8/2 is carried from the old daily
project and is likely too strong here; a β-sweep would give a better tradeoff
point — deferred.)

---

## Beat-baseline — formal summary (SUPPORTING result)
Baseline = **best-static** (60/40 = 0 at the stretch goal, a dead reference).

- **P(goal), learned dynamic vs best-static:** IQL (pure, size 2000)
  **0.722 vs 0.629 = +0.093**. Two-proportion test (MC N=100k) is trivially
  "significant" (z≈45, 95% CI [+0.089, +0.097]) — but the tight CI is a *choice
  of N*, not real-world confidence. Report the **effect size (+0.093)** and the
  **attribution**, not the p-value.
- **Attribution:** +0.093 is attributable to *dynamics* (best-static is already
  the best state-blind policy); the *regime signal* adds +0.028 (aware − agnostic
  DP). So the win decomposes into dynamics (+0.093 empirically for IQL, +0.098 at
  the DP optimum) and regime (+0.028).
- **Domination on tail:** IQL-shaped (P 0.658, MaxDD 0.081, termP10 734)
  **dominates best-static (P 0.629, MaxDD 0.303, termP10 499) on all three axes**
  simultaneously.

**Honest bound:** all "beats" are under the calibrated simulator. The real-world
P(goal) *level* is optimistic (2010-2019 bull) and fundamentally unverifiable
(a 10-year horizon needs many independent realized paths → centuries). The
transferable content is method-level (IQL sample efficiency, CQL α-scaling) and
structural/directional (dynamics > static, shaping reduces tail), both of which
are model-calibrated, not realized-market, claims.
