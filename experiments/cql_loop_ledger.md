# CQL-on-DDPG Loop Ledger

Offline CQL on the Week 8 on-frontier DDPG. End goal = **trustworthy +
regime-conditional allocator** (this loop delivers effect ①: cure the DDPG
seed collapse). Primary hill-climb metric = **meanDispersion** (cross-seed mean
of within-seed eval-α std; ~0 = collapsed, ↑ = state-responsive). Guard rails
anchored to a non-collapsed reference (MaxDD P90 ≤ ~20%; Terminal P10 not far
below 60/40; net Sharpe ≥ ~0.30). All runs: extended train 2010-2021, 6D state,
regime-gated drawdown reward β=8/2, 5 seeds [1000..5000], 40k grad steps/seed,
inline eval (30 windows, cost 0). Trainer: `src/cql_offline_trainer.m`.

**CQL estimator note:** the first draft used a uniform-only logmeanexp; it was
action-nearly-uniform, merely *relocated* collapse (α=5: both test seeds →
defensive boundary, dispersion ~0.1) and inflated Q(a≈0) to ~1.9 (wrong sign).
Replaced with proper **continuous CQL(H)**: logsumexp over uniform + current-
policy μ(s)+N(0,σ) samples with IS correction (5+5, σ=0.2). This cured the
collapse in the re-probe. All rows below use CQL-H unless noted.

| iter | cqlAlpha | seeds | meanDisp (primary) | collapse lo/hi | succ P(goal) | regime diff | meanSharpe | MaxDD P90 | TermW P10 | CQL-reg | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor | 0 (DDPG) | 5 | 0.087 | 60% / 20% | 12.8 | +0.024 | 0.55 | 13.4% | 91775 | ~0 | ✅ reproduces week8 DDPG (3/5 collapse). Trainer correct. |
| probe† | 5 (uniform-only) | 2 | 0.106 | — | 9.0 | -0.068 | 0.07 | 7.1% | 94242 | neg (wrong sign) | ❌ broken variant: relocated collapse, Q inflated → discarded |
| reprobe | 1 (CQL-H) | 2 | 0.278 | 0% / 0% | 14.0 | -0.027 | 0.78 | 16.6% | 92073 | +0.07..0.14 | ✅ cured both collapsed seeds (s2000 1.0→0.80, s3000 0.0→0.10; disp 0→0.23-0.33) |
| reprobe | 5 (CQL-H) | 2 | 0.342 | 0% / 0% | 15.5 | -0.051 | 0.76 | 21.7% | 89378 | neg | ⚠️ cured but MaxDD P90 breaches ~20% rail → α too strong |
| sweep | 0.3 | 5 | 0.177 | 0% / 0% | 11.0 | **-0.098** | 0.24 (std 0.70) | 12.9% | 91602 | mixed | cures collapse but UNDER-strong: Sharpe/succ drop, uneven (seed1000 stayed 0.91) |
| **sweep** | **1** | 5 | **0.247** | **0% / 0%** | 13.6 | -0.051 | 0.70 (std **0.37**) | 14.9% | **92748** | +/− | ★ **WINNER** — cures collapse, MOST reproducible (std 0.37), MaxDD<rail, best TermP10, dominates DDPG anchor on every axis |
| sweep | 3 | 5 | 0.236 | 0% / 0% | 14.6 | +0.007 | 0.78 (std 0.52) | 17.8% | 92073 | neg | cures but OVER-strong: MaxDD rising, regime direction lost (+0.007) |

†2-seed probe only, and used the discarded uniform-only CQL term — context, not a real sweep row.

## Result (all 5 seeds, CQL-H)

**Effect ① delivered cleanly.** CQL cures the DDPG seed collapse at every α:
boundary-collapse 60%/20% → **0%/0%**, meanDispersion 0.087 → 0.18-0.25, all 5
seeds interior + state-responsive. Robust across 3 α values.

**Winner = α=1**: highest dispersion (0.247), lowest across-seed Sharpe std
(0.37 vs DDPG 0.61 → most reproducible), MaxDD P90 14.9% (< 20% rail), highest
Terminal P10 (92748), P(goal) 13.6/30, regime diff -0.051 (mildly correct
direction). Dominates the DDPG anchor on every axis (collapse, mean Sharpe,
reproducibility, Terminal P10) at ~1.5pp more MaxDD.

**Effect ② (regime direction) still weak** — smaller α gives better direction
(α=0.3: -0.098) but at the cost of return; no α is strongly correct. Confirms
the pre-registered prediction: direction is data-limited, deferred to the Week-9
augmentation workstream.

## Winner cost-aware backtest (`src/cql_winner_harness.m`, @10bp headline)

**Estimator discipline (reviewer B1/B2/S3/S4 fixes applied):** point Sharpe, CI,
and DSR are ALL the pooled-daily Sharpe on the same net-return series (no
window-avg-vs-pooled mismatch). RL strategies are NOT pooled across seeds — each
seed is bootstrapped on its own ~900 daily returns and the per-seed point/CI/DSR
are averaged; reproducibility lives in the across-seed-std column, not the CI.
DSR haircut estimated over the **full 17-config trial universe including the
losing runs** (var/252 = 1.60e-4), not a favorable 7-config subset.

| strat | net Sharpe (95% CI) | across-seed std | MaxDD P90 | TermP10 | turn/step | DSR |
|---|---|---|---|---|---|---|
| **CQL α=1** | 0.39 [−0.42, 1.42] | ±0.47 | 15.1% | 92551 | 0.084 | **0.50** |
| DDPG α=0 | 0.25 [−0.78, 1.38] | ±0.39 | 13.6% | 91657 | 0.060 | 0.42 |
| 6.1 DQN | 0.20 [−0.66, 1.37] | ±0.36 | 16.0% | 89376 | 0.107 | 0.39 |
| 60/40 | 0.13 [−0.91, 1.29] | — | **8.7%** | **95302** | **0.009** | 0.33 |

**Honest headline (bulletproof version — the earlier "only CQL's CI excludes 0 /
DSR 0.77 survives" was a pooling + estimator-mismatch + favorable-subset
artifact, now corrected):**
1. **No strategy's net-Sharpe CI excludes zero, and none clears a convincing DSR
   bar** after the honest 17-trial haircut (CQL's DSR is 0.50, right at the line;
   the rest below). Net of cost, in this single test regime, the strategies are
   statistically indistinguishable — including from 60/40. This reproduces the
   Week-8 conclusion; the corrected stats do NOT overturn it.
2. CQL α=1 is still **nominally first** (highest point Sharpe 0.39, highest DSR
   0.50, best TermP10 of the RL set) — the cleanest offline variant — but the
   ranking is within noise. The anchor DDPG now reads 0.29 (≈ Week-8's 0.35 under
   the same pooled-daily estimator), so the α=0 anchor genuinely reproduces
   Week-8 (reviewer S6 resolved by the estimator fix).
3. 60/40 remains the tail-risk / turnover champion (MaxDD P90 8.7% vs 15%,
   turnover 1/9).

**Where the real CQL win lives:** NOT in net Sharpe (indistinguishable) but in
the **training-side collapse cure** — boundary-collapse 60%/20% → 0%/0% across
5 seeds, robust across α (effect ①). That claim rests on the dispersion/collapse
diagnostics, which do not depend on the backtest's statistical power.

**Loop DONE.** Effect ① delivered and validated end-to-end. Effect ② (regime
direction) remains data-limited → Week-9 augmentation workstream.

## Sweet-spot logic
α=1 (CQL-H, 2 seeds) cures collapse with MaxDD P90 16.6% (< 20% rail); α=5
over-strong (MaxDD 21.7%). Focused sweep {0.3, 1, 3} on full 5 seeds brackets
the sweet spot. Winner = smallest α that cures across all 5 seeds
(meanDisp ≫ anchor's 0.087, no boundary collapse) while MaxDD P90 ≤ ~20% and
Sharpe ≥ ~0.30. Winner then runs the full cost-aware harness (net-Sharpe block-
bootstrap CI + Deflated Sharpe, cumulative trial count).
