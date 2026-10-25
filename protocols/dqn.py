"""DQN cluster-head selection (centralised). Per-node MLP 4->16->2 (Q_member, Q_head), numpy only.

Evaluation: greedy top-k by Q_head - Q_member with frozen weights from protocols/weights/dqn.npz.
Training: `python -m protocols.dqn` (run from the qhpnet dir), seeds 100000.. only.
"""
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

from engine import Protocol
from protocols.common import local_density, n_heads, nearest_head_structure

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "dqn.npz")
HID = 16


def init_params(rng):
    return {"W1": rng.normal(0, 0.5, (4, HID)), "b1": np.zeros(HID),
            "W2": rng.normal(0, 0.5, (HID, 2)), "b2": np.zeros(2)}


def forward(p, X):
    h = np.maximum(X @ p["W1"] + p["b1"], 0.0)
    return h @ p["W2"] + p["b2"], h


def features(s):
    N = s.cfg.N
    a = s.alive
    f = np.zeros((N, 4))
    f[:, 0] = s.E / s.E_init
    f[:, 1] = s.d_bs / max(s.d_bs[a].max(), 1e-9) if a.any() else 0.0
    dens = local_density(s)
    f[:, 2] = dens / max(dens[a].max(), 1e-9) if a.any() else 0.0
    f[:, 3] = np.where(s.last_head < 0, 1.0, np.clip((s.r - s.last_head) / 20.0, 0.0, 1.0))
    return f


def pick_heads(s, adv, k):
    idx = np.flatnonzero(s.alive)
    top = idx[np.argsort(-adv[idx], kind="stable")[:k]]
    heads = np.zeros(s.cfg.N, bool)
    heads[top] = True
    return heads


class DQN(Protocol):
    name = "DQN"
    control = "centralized"

    def reset(self, world, rng):
        with np.load(WEIGHTS) as z:
            self.p = {k: z[k] for k in ("W1", "b1", "W2", "b2")}

    def choose(self, s, F):
        q, _ = forward(self.p, F)
        return pick_heads(s, q[:, 1] - q[:, 0], n_heads(s))

    def setup(self, s):
        return nearest_head_structure(s, self.choose(s, features(s)))


# ---------------------------------------------------------------- training

class _Trainer(DQN):
    """Same policy as DQN but epsilon-greedy, logging transitions and learning online."""

    def __init__(self, params, eps, buf, rng_np, gamma=0.95, lr=1e-3, batch=64, sync=500):
        self.p, self.eps, self.buf, self.gamma, self.lr, self.batch, self.sync = params, eps, buf, gamma, lr, batch, sync
        self.rng = rng_np
        self.tp = {k: v.copy() for k, v in params.items()}
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.losses = []

    def reset(self, world, rng):
        self.prev = None

    def setup(self, s):
        F = features(s)
        a = s.alive
        if self.prev is not None:
            pF, pa, pE, pact = self.prev
            E0 = s.cfg.N * s.cfg.E0 * 1e-3
            spent = pE.sum() - s.E.sum()
            died = int(pa.sum() - a.sum())
            Ea = s.E[a]
            bal = 1.0 - Ea.std() / Ea.mean() if a.any() and Ea.mean() > 0 else 0.0
            rew = -spent / E0 - 5.0 * died + 0.5 * bal
            for i in np.flatnonzero(pa):
                self.buf.append((pF[i], int(pact[i]), rew, F[i], not a[i]))
            if len(self.buf) > 50000:
                del self.buf[:len(self.buf) - 50000]
            if len(self.buf) >= self.batch:
                self.update()
        k = n_heads(s)
        heads = self.choose(s, F)
        idx = np.flatnonzero(a)
        for h in np.flatnonzero(heads):
            if self.rng.random() < self.eps:
                free = idx[~heads[idx]]
                if free.size:
                    heads[h] = False
                    heads[self.rng.choice(free)] = True
        self.prev = (F, a.copy(), s.E.copy(), heads.astype(int))
        return nearest_head_structure(s, heads)

    def update(self):
        B = [self.buf[j] for j in self.rng.integers(0, len(self.buf), self.batch)]
        X = np.array([b[0] for b in B]); A = np.array([b[1] for b in B])
        R = np.array([b[2] for b in B]); X2 = np.array([b[3] for b in B]); D = np.array([b[4] for b in B], float)
        tgt = R + self.gamma * (1 - D) * forward(self.tp, X2)[0].max(1)
        q, h = forward(self.p, X)
        n = np.arange(self.batch)
        err = q[n, A] - tgt
        self.losses.append(float((err ** 2).mean()))
        dq = np.zeros_like(q)
        dq[n, A] = 2 * err / self.batch
        dh = (dq @ self.p["W2"].T) * (h > 0)
        g = {"W2": h.T @ dq, "b2": dq.sum(0), "W1": X.T @ dh, "b1": dh.sum(0)}
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            self.p[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        if self.t % self.sync == 0:
            self.tp = {k: v.copy() for k, v in self.p.items()}


def train(episodes=30, seed0=100000, max_rounds=3000, train_seed=12345):
    import engine
    from config import Config
    cfg = Config()                                  # lossy
    rng = np.random.default_rng(train_seed)
    params = init_params(rng)
    buf, log = [], []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * episodes) * 0.95)
        tr = _Trainer(params, eps, buf, rng)
        t0 = time.time()
        m = engine.run(tr, engine.World(cfg, seed0 + ep), max_rounds=max_rounds)
        log.append({"episode": ep, "seed": seed0 + ep, "eps": round(eps, 3), "FND": m["FND"], "AUC": m["AUC"],
                    "rounds": m["rounds"], "loss": float(np.mean(tr.losses[-1000:])) if tr.losses else None,
                    "sec": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    np.savez(WEIGHTS, **params)
    with open(WEIGHTS.replace(".npz", "_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return log


if __name__ == "__main__":
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
