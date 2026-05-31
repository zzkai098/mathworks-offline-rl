# MathWorks Summer Project 2026 — Offline RL for Financial Trading

8-week summer project at MathWorks on **offline reinforcement learning for portfolio management**, building on the official MathWorks GBWM (Goal-Based Wealth Management) demo and extending it with modern offline RL methods (CQL, IQL) and richer datasets.

- **Author:** Zhankai Zhang (BU MSMFT)
- **Mentor:** Yuchen Dong (MathWorks)
- **Duration:** 2026-05 → 2026-07
- **Reference paper:** A. Unnikrishnan, *Financial News-Driven LLM Reinforcement Learning for Portfolio Management*

## Repo Layout

```
mathworks-offline-rl/
├── data/                  Input datasets (SimulatedData.xlsx, etc.)
├── demos/                 Original MathWorks demo (read-only reference)
├── src/                   Project code — experiments live here
│   └── utils/             Shared helper functions
├── experiments/
│   ├── logs/              Training logs (gitignored)
│   └── figures/           Saved plots
├── docs/
│   ├── weekly_reports/    Weekly progress notes
│   └── code_review.md     Review of the original demo
└── papers/                Reference papers (see papers/README.md)
```

## Setup

- **MATLAB:** R2025b
- **Required toolboxes:** Reinforcement Learning, Financial, Statistics & Machine Learning, Deep Learning, Optimization

Open MATLAB, set the project root as Current Folder, then open any `.mlx` under `demos/` or `src/`.

## Status

See `docs/weekly_reports/` for the latest progress.
