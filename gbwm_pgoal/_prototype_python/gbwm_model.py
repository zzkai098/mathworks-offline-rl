"""
Shared GBWM P(goal) model — the single source of truth for the return model,
the efficient-frontier action set, the regime-switching dynamics, the DP optimum
(the measurable CEILING), Monte-Carlo P(goal) evaluation, and offline-episode
logging. Fit on TRAIN (2010-2019) only.

Used by:  dp/preflight.py (validation)  and  offline/*.py (the RL study).

State the RL agent and the DP share: (normalized wealth, time-fraction, regime).
Objective (primary study): pure P(goal) = P(W_T >= goal). The DP that solves this
exactly is the ceiling; offline RL (vanilla/CQL/IQL) tries to recover it from a
limited logged dataset. The mentor's regime-gated drawdown penalty is a SEPARATE
reward-shaping variant (see offline/) so the primary gap-to-DP stays clean.
"""

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize

# ---- GBWM instance (frozen) ----
YEARS, W0, CONTRIB = 10, 100.0, 10.0
N_FRONTIER, GH_NODES = 15, 15
WGRID_N, WGRID_MAX_MULT = 500, 12.0
VIX_THR_Z, SLOPE_THR_Z, MONTH_STRESS_FRAC = 1.5, -1.5, 0.25
STRETCH_CEILING = 0.75          # calibrate goal so DP-optimal P(goal) ~= this
EQUITY = ["AAPL","MSFT","NVDA","JPM","BAC","V","JNJ","UNH","XOM","CVX","PG","KO","CAT"]
BOND = "TLT"


def _gh():
    x, w = hermgauss(GH_NODES)
    return np.sqrt(2)*x, w/np.sqrt(np.pi)


def load_monthly():
    px = pd.read_csv("data/prices_train.csv", parse_dates=["Date"]).set_index("Date")
    mac = pd.read_csv("data/macro_train.csv", parse_dates=["Date"]).set_index("Date")
    z = (mac - mac.mean())/mac.std()
    stress_day = (z["VIXCLS"] > VIX_THR_Z) | (z["T10Y2Y"] < SLOPE_THR_Z)
    mret = np.log(px.resample("M").last()).diff().dropna()
    mstress = (stress_day.resample("M").mean().reindex(mret.index) >= MONTH_STRESS_FRAC).fillna(False).astype(bool)
    return list(px.columns), mret, mstress


def _long_only_frontier(mu, Sigma, k):
    n = len(mu); ones = np.ones(n); bnds = [(0.0, 1.0)]*n
    csum = {"type": "eq", "fun": lambda w: w.sum()-1.0}
    def mv(t):
        c = [csum, {"type": "eq", "fun": lambda w, tt=t: w@mu-tt}]
        return minimize(lambda w: w@Sigma@w, ones/n, method="SLSQP", bounds=bnds,
                        constraints=c, options={"maxiter": 300, "ftol": 1e-12}).x
    r_lo = minimize(lambda w: w@Sigma@w, ones/n, method="SLSQP", bounds=bnds,
                    constraints=[csum], options={"maxiter": 300}).x @ mu
    W = np.vstack([mv(t) for t in np.linspace(r_lo, mu.max()*0.999, k)])
    W = np.clip(W, 0, None); W /= W.sum(axis=1, keepdims=True)
    return W


def _ann_params(w, mret, mstress):
    r = mret.values @ w
    out = {}
    for k, mask in [(0, ~mstress.values), (1, mstress.values)]:
        x = r[mask]; out[k] = (12*x.mean(), np.sqrt(12)*x.std(ddof=1))
    return out


def _regime_transition(mstress):
    s = mstress.values.astype(int); P = np.ones((2, 2))
    for a, b in zip(s[:-1], s[1:]): P[a, b] += 1
    P /= P.sum(axis=1, keepdims=True)
    Pann = np.linalg.matrix_power(P, 12)
    vals, vecs = np.linalg.eig(Pann.T)
    stat = np.real(vecs[:, np.argmin(np.abs(vals-1))]); stat /= stat.sum()
    return P, Pann, stat


def solve_dp(M, goal, agnostic=False):
    """Backward induction for P(goal). Returns policy list over t.
    regime-aware: pol[t] shape (2, nW) int;  agnostic: pol[t] shape (nW,) int."""
    Wgrid = np.linspace(1.0, WGRID_MAX_MULT*goal, WGRID_N)
    ghz, ghw = _gh(); acts = M["frontier_ann"]; nA = len(acts); Pann, stat = M["Pann"], M["stat"]
    Vterm = (Wgrid >= goal).astype(float)

    def ev(a, r, Vnext):
        m, s = acts[a][r]
        Wp = (Wgrid[:, None] + CONTRIB) * np.exp(m + s*ghz[None, :])
        if agnostic:
            e = np.interp(Wp, Wgrid, Vnext)
        else:
            e = sum(Pann[r, rp]*np.interp(Wp, Wgrid, Vnext[rp]) for rp in range(2))
        return (e*ghw[None, :]).sum(axis=1)

    if agnostic:
        V = Vterm.copy(); pol = []
        for _ in range(YEARS):
            Q = np.vstack([sum(stat[r]*ev(a, r, V) for r in range(2)) for a in range(nA)])
            pol.append(np.argmax(Q, axis=0)); V = Q.max(axis=0)
        return Wgrid, pol[::-1]
    V = {r: Vterm.copy() for r in range(2)}; pol = []
    for _ in range(YEARS):
        nV, nP = {}, {}
        for r in range(2):
            Q = np.vstack([ev(a, r, V) for a in range(nA)])
            nP[r] = np.argmax(Q, axis=0); nV[r] = Q.max(axis=0)
        pol.append(np.vstack([nP[0], nP[1]])); V = nV
    return Wgrid, pol[::-1]


def mc_eval(M, policy_fn, goal, n_mc=200_000, seed=0):
    """MC P(goal) under the TRUE regime-switching model. policy_fn(W,t,regime)->action idx."""
    rng = np.random.default_rng(seed)
    acts = M["frontier_ann"]; Pm = M["Pm"]
    am = np.array([[acts[a][r][0] for r in range(2)] for a in range(len(acts))])
    as_ = np.array([[acts[a][r][1] for r in range(2)] for a in range(len(acts))])
    W = np.full(n_mc, W0); reg = np.zeros(n_mc, dtype=int)
    for t in range(YEARS):
        a = policy_fn(W, t, reg)
        z = rng.standard_normal(n_mc)
        W = (W + CONTRIB)*np.exp(am[a, reg] + as_[a, reg]*z)
        reg = (rng.random(n_mc) < Pm[reg, 1]).astype(int)
    return float((W >= goal).mean())


def log_episodes(M, goal, behavior, n_episodes, seed=0):
    """Generate offline transitions under a SUBOPTIMAL behavior policy.
    Returns dict of arrays; reward is pure terminal P(goal) indicator.
    behavior: 'random' (uniform frontier) or 'noisy_glide' (glide + eps-random)."""
    rng = np.random.default_rng(seed)
    acts = M["frontier_ann"]
    am = np.array([[acts[a][r][0] for r in range(2)] for a in range(len(acts))])
    as_ = np.array([[acts[a][r][1] for r in range(2)] for a in range(len(acts))])
    Pm = M["Pm"]; nA = len(acts)
    glide = np.round(np.linspace(nA-1, nA//2, YEARS)).astype(int)   # aggressive->mid
    S, T, R, A, RW, SN, RN, D = ([] for _ in range(8))
    for _ in range(n_episodes):
        W = W0; reg = 0
        for t in range(YEARS):
            if behavior == "random":
                a = rng.integers(nA)
            else:  # noisy_glide
                a = glide[t] if rng.random() > 0.3 else rng.integers(nA)
            z = rng.standard_normal()
            Wn = (W + CONTRIB)*np.exp(am[a, reg] + as_[a, reg]*z)
            rn = (rng.random() < Pm[reg, 1]).astype(int)
            done = (t == YEARS-1)
            rew = float(Wn >= goal) if done else 0.0
            S.append(W); T.append(t); RW.append(reg); A.append(a); R.append(rew)
            SN.append(Wn); RN.append(rn); D.append(done)
            W, reg = Wn, rn
    return {k: np.asarray(v) for k, v in
            dict(W=S, t=T, reg=RW, a=A, r=R, Wn=SN, regn=RN, done=D).items()}


def build_gbwm(verbose=False):
    """Load train data, fit the regime-switching return model, calibrate the
    stretch goal (DP-optimal P(goal) ~= STRETCH_CEILING), solve the DP ceiling.
    Returns the model dict M with M['goal'], M['dp_ceiling'], M['dp_pol'], ..."""
    assets, mret, mstress = load_monthly()
    mu, Sigma = mret.mean().values, mret.cov().values
    Wfront = _long_only_frontier(mu, Sigma, N_FRONTIER)
    w6040 = np.zeros(len(assets))
    for i, a in enumerate(assets):
        if a in EQUITY: w6040[i] = 0.60/len(EQUITY)
        elif a == BOND: w6040[i] = 0.40
    frontier_ann = [_ann_params(w, mret, mstress) for w in Wfront]
    ann_6040 = _ann_params(w6040, mret, mstress)
    Pm, Pann, stat = _regime_transition(mstress)
    M = dict(assets=assets, Wfront=Wfront, frontier_ann=frontier_ann, w6040=w6040,
             ann_6040=ann_6040, Pm=Pm, Pann=Pann, stat=stat, mstress_rate=float(mstress.mean()))

    # calibrate stretch goal via bisection on DP-aware P(goal)
    lo, hi = W0 + CONTRIB*YEARS, 8000.0
    for _ in range(20):
        mid = 0.5*(lo+hi)
        Wg, pol = solve_dp(M, mid, agnostic=False)
        idx = lambda W: np.clip(np.searchsorted(Wg, W), 0, len(Wg)-1)
        p = mc_eval(M, lambda W, t, r: pol[t][r, idx(W)], mid, n_mc=50_000, seed=1)
        if p > STRETCH_CEILING: lo = mid
        else: hi = mid
    goal = 0.5*(lo+hi)
    Wg, pol = solve_dp(M, goal, agnostic=False)
    idx = lambda W: np.clip(np.searchsorted(Wg, W), 0, len(Wg)-1)
    M.update(goal=goal, dp_Wgrid=Wg, dp_pol=pol,
             dp_policy_fn=(lambda W, t, r: pol[t][r, idx(W)]))
    M["dp_ceiling"] = mc_eval(M, M["dp_policy_fn"], goal, n_mc=200_000, seed=2)
    if verbose:
        print(f"[gbwm] goal={goal:.1f} dp_ceiling={M['dp_ceiling']:.3f} "
              f"stress_rate={M['mstress_rate']:.1%}")
    return M


if __name__ == "__main__":
    M = build_gbwm(verbose=True)
