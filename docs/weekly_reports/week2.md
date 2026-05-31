# Week 2 — Literature Review & Demo Validation

**Dates:** 2026-05-25 → 2026-05-31

## Tasks (from mentor)
1. Run the demo code to verify MATLAB installation
2. Stress-test the input dataset (longer date range / more stocks)
3. Read & review the code; compare with reference paper workflows for improvements

## Progress

### ✅ Task 1: Demo runs end-to-end
- `OfflineAgentTrainingDemo.mlx` executes cleanly after replacing `price2ret` (which now requires the Econometrics Toolbox) with manual log-return computation: `diff(log(P))`.
- Training: 100 epochs, ~11 min, converged Q ≈ 0.145.
- Hold-out evaluation: terminal wealth 109,139 vs goal 102,000 → success on the single hold-out episode.

### 🟡 Task 2: Dataset stress test — TODO

### 🟡 Task 3: Code review — TODO (see `docs/code_review.md`)

## Bugs / Issues Found

1. **`price2ret` toolbox dependency** — moved to Econometrics Toolbox in newer MATLAB versions. Replaced with `diff(log(P))`.
2. **`{:, 3:end}` indexing** (lines 42, 188) — dataset has only 3 columns of prices (no date column). `3:end` discards `Stock1` and `Stock2`. Should be `:`.
3. **`numEpisodes = 400` is misleading** — with `trainingRange=252`, `horizonPeriods=60`, sliding step 30, only ~7 episodes can be generated. Loop exits early.

## Literature Review

Seven papers collected under `papers/`:

**Offline RL foundations:**
- 01 CQL (Kumar 2020) — conservative Q-learning to handle extrapolation error
- 02 Offline RL Tutorial (Levine 2020) — comprehensive overview
- 03 IQL (Kostrikov 2021) — implicit Q-learning, avoids querying out-of-distribution actions

**RL in finance:**
- 04 FinRL (Liu 2020) — open-source finance RL library
- 05 Ensemble (Yang 2020) — combining multiple RL agents for trading
- 06 LLM+RL Trading (2025) — news-driven LLM-augmented RL for portfolios
- 07 RL Finance Survey (2024) — recent survey

## Next Steps
- Fix the `3:end` bug and rerun
- Extend dataset (longer range / more stocks via Datafeed)
- Draft `code_review.md` comparing demo's vanilla DQN against CQL/IQL approaches
