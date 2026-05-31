# Reference Papers — RL for Financial Trading

Reference materials for MathWorks Summer Project (2026), sorted by relevance.

**Project direction:** Offline Reinforcement Learning for Financial Trading

---

## Tier 1 — Offline RL Core Algorithms (Must Read)

### 01. CQL — Conservative Q-Learning 
**Kumar et al., 2020 (NeurIPS)**
- File: `01_CQL_Kumar2020.pdf`
- arXiv: https://arxiv.org/abs/2006.04779

**Key idea:** Adds a regularization term to the Q-learning loss that **actively lowers Q-values for out-of-distribution actions**, preventing extrapolation error. Currently one of the strongest offline RL baselines.

**Relevance:** Our project requires offline agent training — CQL is the first-choice algorithm. Can be implemented by adding the CQL regularizer on top of MATLAB's built-in AC agent.

---

### 02. Offline RL Tutorial 
**Levine, Kumar, Tucker, Fu, 2020**
- File: `02_OfflineRL_Tutorial_Levine2020.pdf`
- arXiv: https://arxiv.org/abs/2005.01643

**Key idea:** The authoritative survey on offline RL. Explains why offline RL is hard (distribution shift, overestimation) and the major solution families (policy constraints, conservative Q, model-based).

**Relevance:** Read this first to build the big-picture understanding before diving into specific algorithms.

---

### 03. IQL — Implicit Q-Learning
**Kostrikov, Nair, Levine, 2021**
- File: `03_IQL_Kostrikov2021.pdf`
- arXiv: https://arxiv.org/abs/2110.06169

**Key idea:** Uses expectile regression to learn V(s), completely avoiding evaluation of unseen actions and sidestepping the extrapolation problem. More stable training, fewer hyperparameters.

**Relevance:** A strong alternative to CQL — switch to IQL if CQL tuning becomes difficult.

---

## Tier 2 — Financial RL Frameworks

### 04. FinRL — DRL Framework for Quantitative Finance 
**Liu, Yang, Gao, Wang, 2020**
- File: `04_FinRL_Liu2020.pdf`
- arXiv: https://arxiv.org/abs/2011.09607

**Key idea:** The first open-source framework for financial RL. Provides a full pipeline: data ingestion → environment → training → backtesting. Modular design, supports multiple algorithms (PPO/A2C/DDPG/SAC/TD3).

**Relevance:** Environment design, reward function, and backtesting workflow are directly transferable to MATLAB even though the original is in Python.

---

### 05. Ensemble Strategy for Stock Trading 
**Yang, Liu, Zhong, Walid, 2020**
- File: `05_Ensemble_Yang2020.pdf`
- arXiv: https://arxiv.org/abs/2106.06107

**Key idea:** Combines **PPO + A2C + DDPG** agents, dynamically selecting the best one based on Sharpe ratio. Evaluated on 30 Dow Jones stocks.

**Relevance:** Classic baseline. The multi-agent design aligns with the MultiAgent example in our MATLAB demo repo.

---

## Tier 3 — Project-Specific Directions

### 06. LLM-Guided RL in Quantitative Trading 
**2025 (FLLM 2025 preprint)**
- File: `06_LLM_RL_Trading2025.pdf`
- arXiv: https://arxiv.org/abs/2508.02366

**Key idea:** Uses an LLM to extract signals from financial news, feeding them as additional state input to the RL agent to guide trading decisions.

**Relevance:** Closely matches the supervisor's specified reference paper — **Unnikrishnan, "Financial News-Driven LLM RL for Portfolio Management"**. This is the latest direction if the project incorporates news data.

---

### 07. RL in Financial Applications — Survey 
**2024**
- File: `07_RL_Finance_Survey2024.pdf`
- arXiv: https://arxiv.org/abs/2411.12746

**Key idea:** A survey of RL in finance covering stock trading, portfolio management, market making, and order execution, with algorithm comparisons.

**Relevance:** Useful for selecting baselines and writing the Related Work section.

---

## Recommended Reading Order

```
1. Levine Tutorial (02)          ← Build big-picture view of offline RL
       ↓
2. RL Finance Survey (07)        ← Understand the financial RL landscape
       ↓
3. FinRL (04) + Ensemble (05)    ← Learn financial RL frameworks & baselines
       ↓
4. CQL (01)                      ← Core algorithm — read carefully
       ↓
5. IQL (03)                      ← Backup plan
       ↓
6. LLM-RL Trading (06)           ← If project includes news data
```

---

## Mapping to Project Roadmap

| Week | Task | Primary References |
|------|------|--------------------|
| Week 2 | Read papers, decide datasets | 02, 04, 07 |
| Week 3 | Data preprocessing | 04, 05 |
| Week 4 | Build offline dataset | 01, 02 |
| Week 5 | Implement offline RL agent | **01 (CQL)** |
| Week 6 | Training & hyperparameter tuning | 01, 03 |
| Week 7 | Evaluation & baseline comparison | 05 |
| Week 8 | Write final report | 07 |

---

*Last updated: 2026-05-15*
