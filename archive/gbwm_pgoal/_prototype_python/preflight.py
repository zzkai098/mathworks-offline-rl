"""
GBWM P(goal) — PRE-FLIGHT (no RL).

Goal of this script (the HARD-STOP gate before any RL is written):
prove the 10-year annual GBWM game is *winnable by dynamics*, i.e. that on a
regime-switching return model fit on train (2010-2019) only,

    regime-aware DP  >  regime-agnostic DP  >  best static mix  >  60/40

on P(goal) = P(terminal wealth >= goal), with the goal calibrated so 60/40's
P(goal) ~= 0.60. If that ordering + margin do not hold, the pivot is recalibrated
or killed here — cheaply, before any RL.

Design (all fit on TRAIN ONLY; no test/eval leakage):
  * Actions = 15 long-only efficient-frontier portfolios (unconditional train
    mean/cov), plus 60/40 as a fixed reference policy.
  * Regime in {normal, stress} from the mentor's gate VIX_z>1.5 OR T10Y2Y_z<-1.5,
    z-scored on train; monthly regime = month has >=25% stress days.
  * Per (portfolio, regime): annual log return ~ N(12*m, 12*s^2) from that
    portfolio's regime-split MONTHLY returns (stable 1-D estimation).
  * Annual regime transition = (monthly transition)^12.
  * DP: backward induction for P(goal) on a wealth grid with Gauss-Hermite
    quadrature. Regime-aware state (W,t,regime); regime-agnostic state (W,t) uses
    the stationary-mixture return model. EVERY policy is then evaluated under the
    SAME true regime-switching model by Monte-Carlo, so the comparison is fair.

Run:  conda run -n pyfinance python gbwm_pgoal/dp/preflight.py
"""

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize

RNG = np.random.default_rng(12345)

# ---------------- GBWM instance ----------------
YEARS        = 10           # decision points (annual)
W0           = 100.0        # initial wealth
CONTRIB      = 10.0         # annual contribution (start of each year)
TARGET_6040_PGOAL = 0.60    # calibrate goal to this 60/40 success prob
N_FRONTIER   = 15
GH_NODES     = 15           # Gauss-Hermite quadrature nodes
N_MC         = 200_000      # Monte-Carlo paths for P(goal) evaluation
WGRID_N      = 500
WGRID_MAX_MULT = 12.0       # grid spans up to this * goal

VIX_THR_Z, SLOPE_THR_Z = 1.5, -1.5
MONTH_STRESS_FRAC = 0.25    # a month is "stress" if >=25% of its days are stress

EQUITY = ["AAPL","MSFT","NVDA","JPM","BAC","V","JNJ","UNH","XOM","CVX","PG","KO","CAT"]
BOND, GOLD = "TLT", "GLD"


# ---------------- data / return model ----------------
def load_monthly():
    px = pd.read_csv("data/prices_train.csv", parse_dates=["Date"]).set_index("Date")
    mac = pd.read_csv("data/macro_train.csv", parse_dates=["Date"]).set_index("Date")
    assets = list(px.columns)

    # daily regime label from train-z-scored macro (mentor's gate)
    z = (mac - mac.mean()) / mac.std()
    stress_day = (z["VIXCLS"] > VIX_THR_Z) | (z["T10Y2Y"] < SLOPE_THR_Z)

    # monthly log returns (month-end prices) + monthly regime
    mpx = px.resample("ME").last()
    mret = np.log(mpx).diff().dropna()
    mstress = stress_day.resample("ME").mean().reindex(mret.index) >= MONTH_STRESS_FRAC
    mstress = mstress.fillna(False).astype(bool)
    return assets, mret, mstress


def long_only_frontier(mu, Sigma, k):
    """k long-only min-variance portfolios spanning [gmv return, max asset mean]."""
    n = len(mu)
    ones = np.ones(n)
    cons_sum = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bnds = [(0.0, 1.0)] * n

    def minvar(target):
        cons = [cons_sum, {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: w @ Sigma @ w, ones / n, method="SLSQP",
                       bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        return res.x

    # gmv return
    res_gmv = minimize(lambda w: w @ Sigma @ w, ones / n, method="SLSQP",
                       bounds=bnds, constraints=[cons_sum], options={"maxiter": 300})
    r_lo = res_gmv.x @ mu
    r_hi = mu.max() * 0.999
    targets = np.linspace(r_lo, r_hi, k)
    W = np.vstack([minvar(t) for t in targets])
    W = np.clip(W, 0, None)
    W /= W.sum(axis=1, keepdims=True)
    return W  # (k, n)


def annual_params(w, mret, mstress):
    """Return (m_normal, s_normal, m_stress, s_stress) MONTHLY mean/std of the
    portfolio's return, split by regime."""
    r = mret.values @ w
    out = {}
    for name, mask in [("normal", ~mstress.values), ("stress", mstress.values)]:
        x = r[mask]
        out[name] = (x.mean(), x.std(ddof=1))
    return out


def annual_regime_transition(mstress):
    s = mstress.values.astype(int)
    P = np.ones((2, 2))  # Laplace-smoothed counts
    for a, b in zip(s[:-1], s[1:]):
        P[a, b] += 1
    P /= P.sum(axis=1, keepdims=True)
    Pann = np.linalg.matrix_power(P, 12)
    # stationary distribution
    vals, vecs = np.linalg.eig(Pann.T)
    stat = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    stat = stat / stat.sum()
    return P, Pann, stat


# ---------------- DP + evaluation ----------------
def gh():
    x, w = hermgauss(GH_NODES)           # for N(0,1): E[f] = sum w/sqrt(pi) f(sqrt(2) x)
    return np.sqrt(2) * x, w / np.sqrt(np.pi)


def solve_dp(Wgrid, goal, actions_ann, Pann, agnostic=False, stat=None):
    """Backward induction for P(goal).
    actions_ann: list over actions of dict regime-> (annual_mean, annual_std).
    Returns policy pi[t] : regime-aware -> (2, nW) int; agnostic -> (nW,) int.
    """
    ghz, ghw = gh()
    nW = len(Wgrid)
    nA = len(actions_ann)
    Vterm = (Wgrid >= goal).astype(float)

    def expect_action(a, r, Vnext_by_r):
        m, s = actions_ann[a][r]
        Wp = (Wgrid[:, None] + CONTRIB) * np.exp(m + s * ghz[None, :])   # (nW, nodes)
        if agnostic:
            ev = np.interp(Wp, Wgrid, Vnext_by_r)                        # single V
        else:
            ev = sum(Pann[r, rp] * np.interp(Wp, Wgrid, Vnext_by_r[rp]) for rp in range(2))
        return (ev * ghw[None, :]).sum(axis=1)

    if agnostic:
        V = Vterm.copy()
        pol = []
        for _ in range(YEARS):
            Q = np.empty((nA, nW))
            for a in range(nA):
                # marginal (stationary-mixture) return model, next V has no regime
                Q[a] = sum(stat[r] * expect_action(a, r, V) for r in range(2))
            pol.append(np.argmax(Q, axis=0))
            V = Q.max(axis=0)
        return pol[::-1]
    else:
        V = {r: Vterm.copy() for r in range(2)}
        pol = []
        for _ in range(YEARS):
            newV, newPi = {}, {}
            for r in range(2):
                Q = np.vstack([expect_action(a, r, V) for a in range(nA)])
                newPi[r] = np.argmax(Q, axis=0)
                newV[r] = Q.max(axis=0)
            pol.append(np.vstack([newPi[0], newPi[1]]))
            V = newV
        return pol[::-1]


def mc_eval(policy_fn, Wgrid, goal, actions_ann, P):
    """Monte-Carlo P(goal) under the TRUE regime-switching model for any policy.
    policy_fn(w_array, t, regime_array) -> action-index array."""
    W = np.full(N_MC, W0)
    reg = np.zeros(N_MC, dtype=int)        # start in 'normal'
    ann_m = np.array([[actions_ann[a][r][0] for r in range(2)] for a in range(len(actions_ann))])
    ann_s = np.array([[actions_ann[a][r][1] for r in range(2)] for a in range(len(actions_ann))])
    for t in range(YEARS):
        a = policy_fn(W, t, reg)                          # (N_MC,)
        m = ann_m[a, reg]; s = ann_s[a, reg]
        z = RNG.standard_normal(N_MC)
        W = (W + CONTRIB) * np.exp(m + s * z)
        # regime transition
        u = RNG.random(N_MC)
        reg = (u < P[reg, 1]).astype(int)
    return float((W >= goal).mean())


def main():
    assets, mret, mstress = load_monthly()
    print(f"[data] {mret.shape[0]} months, stress-month rate = {mstress.mean():.1%}")

    mu = mret.mean().values          # monthly
    Sigma = mret.cov().values
    Wfront = long_only_frontier(mu, Sigma, N_FRONTIER)      # (15, nAssets)

    # 60/40 weight
    w6040 = np.zeros(len(assets))
    for i, a in enumerate(assets):
        if a in EQUITY: w6040[i] = 0.60 / len(EQUITY)
        elif a == BOND: w6040[i] = 0.40
    # actions: 15 frontier + 60/40 (index 15) for evaluation convenience
    Wall = np.vstack([Wfront, w6040])
    actions_ann = []
    for w in Wall:
        p = annual_params(w, mret, mstress)
        actions_ann.append({0: (12*p["normal"][0], np.sqrt(12)*p["normal"][1]),
                            1: (12*p["stress"][0], np.sqrt(12)*p["stress"][1])})
    I6040 = N_FRONTIER
    frontier_actions = actions_ann[:N_FRONTIER]

    Pm, Pann, stat = annual_regime_transition(mstress)
    print(f"[regime] monthly P=\n{Pm.round(3)}\n annual P=\n{Pann.round(3)}  stationary={stat.round(3)}")
    print(f"[frontier] annual mean/vol (normal): "
          + ", ".join(f"#{a}:{actions_ann[a][0][0]:+.2f}/{actions_ann[a][0][1]:.2f}" for a in [0, 7, 14]))

    def solve_eval(goal, full=False):
        """Solve both DPs at `goal`, MC-eval every policy under the TRUE model."""
        Wgrid = np.linspace(1.0, WGRID_MAX_MULT * goal, WGRID_N)
        pi_aware = solve_dp(Wgrid, goal, frontier_actions, Pann, agnostic=False)
        pi_agn   = solve_dp(Wgrid, goal, frontier_actions, Pann, agnostic=True, stat=stat)
        idx = lambda W: np.clip(np.searchsorted(Wgrid, W), 0, len(Wgrid) - 1)
        o = {}
        o["aware"] = mc_eval(lambda W, t, r: pi_aware[t][r, idx(W)], Wgrid, goal, frontier_actions, Pm)
        o["agn"]   = mc_eval(lambda W, t, r: pi_agn[t][idx(W)],       Wgrid, goal, frontier_actions, Pm)
        o["6040"]  = mc_eval(lambda W, t, r: np.full(len(W), I6040),  Wgrid, goal, actions_ann, Pm)
        if full:
            sp = {a: mc_eval(lambda W, t, r, aa=a: np.full(len(W), aa),
                             Wgrid, goal, frontier_actions, Pm) for a in range(N_FRONTIER)}
            o["static_a"] = max(sp, key=sp.get); o["static"] = sp[o["static_a"]]
            bg, bp = None, -1
            for i0 in range(N_FRONTIER):
                for i1 in range(N_FRONTIER):
                    sched = np.round(np.linspace(i0, i1, YEARS)).astype(int)
                    p = mc_eval(lambda W, t, r, s=sched: np.full(len(W), s[t]),
                                Wgrid, goal, frontier_actions, Pm)
                    if p > bp: bp, bg = p, sched
            o["glide"] = bp; o["glide_sched"] = bg
        return o

    # ---- calibrate goal to a STRETCH: DP-optimal P(goal) ~= 0.75 (bisection) ----
    # Anchoring to 60/40 saturates the risky policies (all ~0.97) so dynamics
    # can't show; anchoring the CEILING to 0.75 puts the goal in the "reach-y"
    # regime where wealth-dependent risk management actually matters.
    STRETCH = 0.75
    lo, hi = W0 + CONTRIB * YEARS, 8000.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if solve_eval(mid)["aware"] > STRETCH:   # aware P(goal) decreases in goal
            lo = mid
        else:
            hi = mid
    goal = 0.5 * (lo + hi)
    print(f"[calib] goal = {goal:.1f}  (DP-aware ceiling ~= {STRETCH})")

    res = solve_eval(goal, full=True)
    aware, agn, bstat, glide, b6040 = res["aware"], res["agn"], res["static"], res["glide"], res["6040"]
    print("\n================ PRE-FLIGHT P(goal) (MC under true regime model, N=%d) ================" % N_MC)
    print(f"  regime-aware DP (ceiling) : {aware:.3f}")
    print(f"  regime-agnostic DP        : {agn:.3f}")
    print(f"  best glide path           : {glide:.3f}")
    print(f"  best static (#{res['static_a']})          : {bstat:.3f}")
    print(f"  60/40                     : {b6040:.3f}")
    print("\n---- attribution margins (the REAL go/no-go) ----")
    print(f"  regime signal   aware - agnostic : {aware-agn:+.3f}")
    print(f"  dynamics        agnostic - static: {agn-bstat:+.3f}")
    print(f"  risk level      static - 60/40   : {bstat-b6040:+.3f}")
    DYN_MARGIN = aware - bstat   # dynamics+regime over the best STATE-BLIND policy
    print(f"  >>> dynamics+regime over best-static (aware - static) : {DYN_MARGIN:+.3f}")
    ok = (aware > agn > bstat > b6040) and (DYN_MARGIN >= 0.03)
    print(f"\n  VERDICT: {'PASS -> dynamics adds real value, proceed to RL' if ok else 'WEAK -> dynamic edge < 0.03, reconsider instance'}")


if __name__ == "__main__":
    main()
