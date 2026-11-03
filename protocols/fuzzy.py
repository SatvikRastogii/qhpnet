"""Interval type-2 fuzzy cluster-head selection (Fuzzy-T2)."""
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

