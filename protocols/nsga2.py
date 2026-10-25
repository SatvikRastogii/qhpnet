"""NSGA-II cluster-head selection (centralised, run at the sink)."""
import numpy as np

from engine import Protocol, rx_cost, tx_cost
from protocols.common import n_heads, nearest_head_structure


def _ranks(F):
    """Fast non-dominated sorting (minimisation). Returns front index per row."""
    le = ge = True
    for o in range(F.shape[1]):
        a, b = F[:, o, None], F[None, :, o]
        le, ge = le & (a <= b), ge & (a >= b)
    dom = le & ~ge                                  # dom[i, j]: i dominates j (<= everywhere, not equal everywhere)
    cnt = dom.sum(0)
    rank = np.zeros(len(F), dtype=np.int64)
    cur, f = np.flatnonzero(cnt == 0), 0
    while cur.size:
        rank[cur] = f
        cnt = cnt - dom[cur].sum(0)
        cnt[cur] = -1
        cur = np.flatnonzero(cnt == 0)
        f += 1
    return rank


def _crowding(F, rank):
    cd = np.zeros(len(F))
    n = len(F)
    for o in range(F.shape[1]):
        order = np.lexsort((F[:, o], rank))             # sorted by value inside each front
        r, v = rank[order], F[order, o]
        first = np.r_[True, r[1:] != r[:-1]]
        last = np.r_[r[:-1] != r[1:], True]
        span = np.zeros(rank.max() + 1)
        span[r[last]] = v[last]
        span[r[first]] -= v[first]
        cd[order[first | last]] = np.inf
        i = np.flatnonzero(~first & ~last)
        if i.size:
            sp = span[r[i]]
            gap = (v[i + 1] - v[i - 1]) / np.where(sp > 0, sp, 1.0)
            cd[order[i]] += np.where(sp > 0, gap, 0.0)
    return cd


class NSGA2(Protocol):
    name = "NSGA-II"
    control = "centralized"
    POP = 40
    GEN = 30
    WARM = False        # fairness variant: seed one individual with the previous epoch's surviving heads

    def __init__(self, warm=False, name=None):
        self.WARM = warm
        if name:
            self.name = name

    def reset(self, world, rng):
        self.rng = rng
        self.heads = None
        self.last = -10 ** 9

    # ---- objectives for a whole population; P holds positions into the alive list
    def _eval(self, P, Da, Ea, dbs):
        cfg, kd = self._cfg, self._cfg.data_bits
        pop, k = P.shape
        d = Da[:, P].transpose(1, 0, 2)              # (pop, m, k)
        a = d.argmin(2)
        dmin = np.take_along_axis(d, a[:, :, None], 2)[:, :, 0]
        is_head = np.zeros(a.shape, bool)
        np.put_along_axis(is_head, P, True, 1)
        mem = ~is_head
        n = np.bincount((a + k * np.arange(pop)[:, None])[mem], minlength=pop * k).reshape(pop, k)
        e_mem = (tx_cost(cfg, kd, dmin) * mem).sum(1)
        e_head = (n * (rx_cost(cfg, kd) + cfg.E_DA * kd) + tx_cost(cfg, kd, dbs[P])).sum(1)
        return np.column_stack([e_mem + e_head, n.var(1), -Ea[P].min(1)])

    def _select(self, rank, cd, m):
        i = self.rng.integers(0, len(rank), size=(m, 2))
        a, b = i[:, 0], i[:, 1]
        a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] >= cd[b]))
        return np.where(a_wins, a, b)

    def _offspring(self, P, rank, cd, m):
        rng, (pop, k) = self.rng, P.shape
        p1, p2 = self._select(rank, cd, pop), self._select(rank, cd, pop)
        rows = np.arange(pop)[:, None]
        in_u = np.zeros((pop, m), bool)                     # union of both parents' heads
        in_u[rows, P[p1]] = True
        in_u[rows, P[p2]] = True
        C = np.argsort(np.where(in_u, rng.random((pop, m)), 2.0), 1)[:, :k]     # random k of the union
        if m > k:
            sel = np.zeros((pop, m), bool)
            sel[rows, C] = True
            free = np.argsort(np.where(sel, 2.0, rng.random((pop, m))), 1)      # distinct non-selected, random order
            mut = rng.random((pop, k)) < 1.0 / k
            j = np.cumsum(mut, 1) - 1                       # j-th mutated gene takes j-th free node
            C = np.where(mut, np.take_along_axis(free, np.clip(j, 0, m - 1), 1), C)
        return C

    def _optimise(self, s):
        cfg, rng = s.cfg, self.rng
        self._cfg = cfg
        al = np.flatnonzero(s.alive)
        m, k = al.size, min(n_heads(s), al.size)
        Da, Ea, dbs = s.D[np.ix_(al, al)], s.E[al], s.d_bs[al]
        P = np.array([rng.choice(m, size=k, replace=False) for _ in range(self.POP)])
        if self.WARM and self.heads is not None:
            prev = np.flatnonzero(self.heads[al])[:k]          # positions of surviving heads in the alive list
            rest = np.setdiff1d(np.arange(m), prev)
            P[0] = np.concatenate([prev, rng.choice(rest, size=k - prev.size, replace=False)])
        F = self._eval(P, Da, Ea, dbs)
        rank = _ranks(F)
        cd = _crowding(F, rank)
        for _ in range(self.GEN):
            C = self._offspring(P, rank, cd, m)
            PC, FC = np.vstack([P, C]), np.vstack([F, self._eval(C, Da, Ea, dbs)])
            rank = _ranks(FC)
            cd = _crowding(FC, rank)
            keep = np.lexsort((-cd, rank))[:self.POP]       # (mu+lambda): rank, then crowding
            P, F, rank, cd = PC[keep], FC[keep], rank[keep], cd[keep]   # cd as in DEAP: from the merged sort; kept members' fronts are unchanged by truncation
        front = np.flatnonzero(rank == 0)
        G = F[front]
        span = np.where(np.ptp(G, 0) > 0, np.ptp(G, 0), 1.0)
        # final pick: min-max normalise each objective over the rank-0 front, take the smallest equal-weight sum
        best = front[np.argmin(((G - G.min(0)) / span).sum(1))]
        heads = np.zeros(cfg.N, bool)
        heads[al[P[best]]] = True
        return heads

    def setup(self, s):
        redo = (self.heads is None or s.r - self.last >= s.cfg.recluster_interval
                or bool((self.heads & ~s.alive).any()))
        if redo:
            self.heads, self.last = self._optimise(s), s.r
        st = nearest_head_structure(s, self.heads)
        st.reclustered = bool(redo)
        return st
