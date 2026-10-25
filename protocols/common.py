"""Helpers shared by protocol implementations (read-only use of State)."""
import numpy as np

from engine import BS, NONE, Structure


def n_heads(s):
    return max(1, int(round(s.cfg.p_head * int(s.alive.sum()))))


def nearest_head_structure(s, heads, head_parent=None):
    """Members join their nearest head (Euclidean). Heads send to head_parent (default BS)."""
    N = s.cfg.N
    heads = np.asarray(heads, dtype=bool) & s.alive
    parent = np.full(N, NONE, dtype=np.int64)
    h = np.flatnonzero(heads)
    if h.size == 0:
        parent[s.alive] = BS
        return Structure(parent=parent, heads=heads)
    m = s.alive & ~heads
    mi = np.flatnonzero(m)
    if mi.size:
        parent[mi] = h[np.argmin(s.D[np.ix_(mi, h)], axis=1)]
    parent[h] = BS if head_parent is None else np.asarray(head_parent)[h]
    return Structure(parent=parent, heads=heads)


def local_density(s, radius=None):
    """Alive neighbours within the density radius, normalised by alive count."""
    r = s.cfg.density_radius if radius is None else radius
    A = (s.D[:-1, :-1] < r) & s.alive[None, :] & s.alive[:, None]
    np.fill_diagonal(A, False)
    return A.sum(1) / max(int(s.alive.sum()) - 1, 1)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
