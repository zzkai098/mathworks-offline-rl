"""
Offline RL trainers for the GBWM P(goal) study: vanilla offline DQN, CQL, IQL.
Small MLP Q-network over state features (normWealth, timeFraction, regime), 15
discrete efficient-frontier actions, pure terminal P(goal) reward, gamma=1
(finite horizon). Everything trains on a FIXED logged dataset (offline).

The point: with a LIMITED / narrow-coverage dataset, vanilla offline DQN
over-estimates Q on out-of-distribution (state,action) pairs (the max in the
bootstrap has nothing to correct it) and its greedy policy drifts off-support.
CQL's conservative penalty and IQL's OOD-free expectile target are designed to
fix exactly that — measured here as gap-to-DP on P(goal).

Self-test (`python offline/train.py`): with a large random dataset vanilla should
recover most of the DP ceiling.
"""

import sys, os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbwm_model import build_gbwm, log_episodes, mc_eval, N_FRONTIER, YEARS


def featurize(W, t, reg, goal):
    """-> (n,3) float32 tensor: [clip(W/goal,0,3), t/YEARS, regime]."""
    W = np.asarray(W, float); t = np.asarray(t, float); reg = np.asarray(reg, float)
    f = np.stack([np.clip(W/goal, 0, 3.0), t/YEARS, reg], axis=-1)
    return torch.as_tensor(f, dtype=torch.float32)


class QNet(nn.Module):
    def __init__(self, nA, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, nA))
    def forward(self, x): return self.net(x)


class VNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _tensors(data, goal):
    s  = featurize(data["W"],  data["t"],   data["reg"],  goal)
    sn = featurize(data["Wn"], data["t"]+1, data["regn"], goal)
    a  = torch.as_tensor(data["a"], dtype=torch.long)
    r  = torch.as_tensor(data["r"], dtype=torch.float32)
    d  = torch.as_tensor(data["done"], dtype=torch.float32)
    return s, a, r, sn, d


def train_offline(M, data, method="vanilla", steps=6000, batch=256, lr=1e-3,
                  cql_alpha=1.0, iql_tau=0.7, seed=0, device="cpu"):
    """Train and return a greedy policy_fn(W,t,reg)->action idx for mc_eval."""
    torch.manual_seed(seed); np.random.seed(seed)
    goal, nA = M["goal"], N_FRONTIER
    s, a, r, sn, d = (x.to(device) for x in _tensors(data, goal))
    N = len(a); rng = np.random.default_rng(seed)

    q = QNet(nA).to(device); qt = QNet(nA).to(device); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    if method == "iql":
        v = VNet().to(device); optv = torch.optim.Adam(v.parameters(), lr=lr)

    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, N, batch), device=device)
        bs, ba, br, bsn, bd = s[idx], a[idx], r[idx], sn[idx], d[idx]
        qsa = q(bs).gather(1, ba[:, None]).squeeze(1)

        if method == "iql":
            # V via expectile regression toward Q_target(s, a_data)
            with torch.no_grad():
                tgt_v = qt(bs).gather(1, ba[:, None]).squeeze(1)
            vpred = v(bs)
            diff = tgt_v - vpred
            w = torch.where(diff > 0, iql_tau, 1 - iql_tau)
            lossv = (w * diff**2).mean()
            optv.zero_grad(); lossv.backward(); optv.step()
            # Q toward r + gamma * V(s') (no OOD max)
            with torch.no_grad():
                y = br + (1 - bd) * v(bsn)
            lossq = ((qsa - y)**2).mean()
            opt.zero_grad(); lossq.backward(); opt.step()
        else:
            with torch.no_grad():
                y = br + (1 - bd) * qt(bsn).max(1).values
            td = ((qsa - y)**2).mean()
            if method == "cql":
                # conservative penalty: push down logsumexp_a Q, up Q(s,a_data)
                cql = (torch.logsumexp(q(bs), dim=1) - qsa).mean()
                loss = td + cql_alpha * cql
            else:  # vanilla
                loss = td
            opt.zero_grad(); loss.backward(); opt.step()

        # soft target update
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(0.995).add_(0.005 * p)

    q.eval()
    def policy_fn(W, t, reg):
        with torch.no_grad():
            f = featurize(W, np.full(len(np.atleast_1d(W)), t), reg, goal).to(device)
            return q(f).argmax(1).cpu().numpy()
    return policy_fn


if __name__ == "__main__":
    M = build_gbwm(verbose=True)
    print("Self-test: vanilla offline DQN on a LARGE random dataset should ~recover DP")
    for ne in [200, 5000]:
        data = log_episodes(M, M["goal"], "random", ne, seed=0)
        pf = train_offline(M, data, method="vanilla", seed=0)
        p = mc_eval(M, pf, M["goal"], n_mc=100_000, seed=7)
        print(f"  n_episodes={ne:5d}  vanilla P(goal)={p:.3f}  (ceiling {M['dp_ceiling']:.3f}, gap {M['dp_ceiling']-p:+.3f})")
