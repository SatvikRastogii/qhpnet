"""Interval type-2 fuzzy cluster-head selection (Fuzzy-T2) and HIECF (IT2 fuzzy + Q-learning + multi-hop heads)."""
import numpy as np

from engine import Protocol
from protocols.common import local_density, n_heads, nearest_head_structure

# ---------------------------------------------------------------- IT2 fuzzy system
# Terms Low/Med/High on [0,1]: triangles (a, b, c) for the UPPER membership function (UMF).
# FOU choice: LMF = 0.8 * triangle with the same peak b and feet pulled in to 60% of the UMF half-widths
# (a' = b - 0.6 (b-a), c' = b + 0.6 (c-b)). Shoulder terms (Low peak at 0, High peak at 1) keep UMF = 1 at the edge.
_TERMS = [(-0.5, 0.0, 0.5), (0.0, 0.5, 1.0), (0.5, 1.0, 1.5)]
_LMF_H, _LMF_W = 0.8, 0.6


def _tri(x, a, b, c):
    return np.clip(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0.0, 1.0)


def _memberships(x):
    """x: (N,) -> lower, upper: (N, 3)."""
    lo, up = np.empty((x.size, 3)), np.empty((x.size, 3))
    for t, (a, b, c) in enumerate(_TERMS):
        up[:, t] = _tri(x, a, b, c)
        lo[:, t] = _LMF_H * _tri(x, b - _LMF_W * (b - a), b, b + _LMF_W * (c - b))
    return lo, up


# 27 rules, index = 9*e + 3*d + q with levels 0/1/2 = Low/Med/High for energy, distance, density.
# Consequent chance (crisp centre in [0,1]): energy dominates, near BS (d low) and high density help.
_E, _Dd, _Q = np.meshgrid(range(3), range(3), range(3), indexing="ij")
_Y = ((0.5 * _E + 0.25 * (2 - _Dd) + 0.25 * _Q) / 2.0).ravel()      # in [0,1]
_ORDER = np.argsort(_Y, kind="stable")
_Ys = _Y[_ORDER]                                                     # consequents sorted ascending (same for all nodes)


def _km(fl, fu, right):
    """Karnik-Mendel end point for all nodes at once. fl, fu: (N,R) with columns sorted by consequent."""
    n = fl.shape[0]
    f = 0.5 * (fl + fu)
    y = (f * _Ys).sum(1) / np.maximum(f.sum(1), 1e-12)
    for _ in range(30):
        k = np.searchsorted(_Ys, y, side="right") - 1           # y_k <= y < y_{k+1}
        left = np.arange(_Ys.size)[None, :] <= k[:, None]
        # left end: upper firing for i<=k, lower for i>k; right end: the opposite
        f = np.where(left, fl, fu) if right else np.where(left, fu, fl)
        yn = (f * _Ys).sum(1) / np.maximum(f.sum(1), 1e-12)
        if np.allclose(yn, y, atol=1e-9, rtol=0):
            y = yn
            break
        y = yn
    return y


def fuzzy_scores(s, alive_only=True):
    """Chance score S_i in [0,1] for every node (dead nodes get -1)."""
    a = s.alive
    e = np.clip(s.E / s.E_init, 0.0, 1.0)
    dmax = max(float(s.d_bs[a].max()), 1e-9) if a.any() else 1.0
    d = s.d_bs / dmax
    q = local_density(s)
    q = q / max(float(q[a].max()), 1e-9) if a.any() else q
    (el, eu), (dl, du), (ql, qu) = _memberships(e), _memberships(d), _memberships(q)
    # min t-norm over the three antecedents -> firing interval [fl, fu] per rule
    fl = np.minimum(np.minimum(el[:, :, None, None], dl[:, None, :, None]), ql[:, None, None, :]).reshape(e.size, 27)
    fu = np.minimum(np.minimum(eu[:, :, None, None], du[:, None, :, None]), qu[:, None, None, :]).reshape(e.size, 27)
    fl, fu = fl[:, _ORDER], fu[:, _ORDER]
    S = 0.5 * (_km(fl, fu, False) + _km(fl, fu, True))
    return np.where(a, S, -1.0)


class FuzzyT2(Protocol):
    name = "Fuzzy-T2"
    control = "centralized"

    def setup(self, s):
        S = fuzzy_scores(s)
        k = min(n_heads(s), int(s.alive.sum()))
        heads = np.zeros(s.cfg.N, bool)
        heads[np.argsort(-S, kind="stable")[:k]] = True
        return nearest_head_structure(s, heads)


# ---------------------------------------------------------------- HIECF
class HIECF(Protocol):
    """Candidates: S_i above the 60th percentile of alive scores each round (K ~ 40% of alive nodes; the paper's
    fixed S_th is not transferable across energy scales). Q-learning picks one of 3 candidate rankings."""
    name = "HIECF"
    control = "centralized"
    W = (0.4, 0.3, 0.3)
    ALPHA, GAMMA, EPS0, EPS1, EPS_T = 0.1, 0.9, 0.1, 0.01, 1000
    # ranking weights on (residual energy ratio, closeness to BS, fuzzy score): energy-heavy, distance-heavy, balanced
    ACTIONS = np.array([[0.7, 0.1, 0.2], [0.1, 0.7, 0.2], [0.35, 0.35, 0.3]])

    def reset(self, world, rng):
        self.rng = rng
        self.Q = np.zeros((25, 3))
        self.prev = None        # (state, action, total residual, heads)
        self.Emax_init = float(world.E_init.max())

    @staticmethod
    def _state(s):
        a = s.alive
        er = float((s.E[a] / s.E_init[a]).mean()) if a.any() else 0.0
        af = float(a.mean())
        return 5 * min(int(er * 5), 4) + min(int(af * 5), 4)

    def _reward(self, s):
        _, _, Eprev, hprev = self.prev
        a = s.alive
        Ea = s.E[a]
        bal = 1.0 - float(Ea.std() / Ea.mean()) if a.any() and Ea.mean() > 0 else 0.0
        h = np.flatnonzero(hprev)
        pdr = float(np.mean(1.0 - s.per[h, s.cfg.N])) if h.size else 0.0
        ec = float(np.clip((Eprev - s.E.sum()) / (0.01 * s.cfg.N * s.E_init.mean()), 0.0, 1.0))
        return self.W[0] * bal + self.W[1] * pdr - self.W[2] * ec

    def setup(self, s):
        cfg, N = s.cfg, s.cfg.N
        st = self._state(s)
        if self.prev is not None:                                   # online Q update for the previous round
            ps, pa = self.prev[0], self.prev[1]
            self.Q[ps, pa] += self.ALPHA * (self._reward(s) + self.GAMMA * self.Q[st].max() - self.Q[ps, pa])
        eps = max(self.EPS1, self.EPS0 + (self.EPS1 - self.EPS0) * s.r / self.EPS_T)
        if self.rng.random() < eps:
            act = int(self.rng.integers(3))
        else:
            q = self.Q[st]
            act = int(self.rng.choice(np.flatnonzero(q == q.max())))

        S = fuzzy_scores(s)
        a = s.alive
        k = min(n_heads(s), int(a.sum()))
        cand = a & (S > np.percentile(S[a], 60))
        if cand.sum() < k:                                          # too few candidates: top-S fill
            cand = np.zeros(N, bool)
            cand[np.argsort(-S, kind="stable")[:k]] = True
        e = np.clip(s.E / s.E_init, 0.0, 1.0)
        close = 1.0 - s.d_bs / max(float(s.d_bs[a].max()), 1e-9)
        w = self.ACTIONS[act]
        rank = np.where(cand, w[0] * e + w[1] * close + w[2] * S, -np.inf)
        heads = np.zeros(N, bool)
        heads[np.argsort(-rank, kind="stable")[:k]] = True

        # cooperative multi-hop: forward only to heads strictly closer to the BS -> acyclic
        hp = np.full(N, -1, dtype=np.int64)
        hi = np.flatnonzero(heads)
        bs_cost = s.d_bs ** 2 / (10.0 * self.Emax_init)
        for i in hi:
            best, bc = -1, bs_cost[i]
            for j in hi[s.d_bs[hi] < s.d_bs[i]]:
                c = s.D[i, j] ** 2 / max(s.E[j], 1e-12)
                if c < bc:
                    best, bc = int(j), c
            hp[i] = best
        self.prev = (st, act, float(s.E.sum()), heads.copy())
        return nearest_head_structure(s, heads, hp)
