"""PEGASIS and TEEN."""
import numpy as np

from engine import BS, NONE, Protocol
from protocols.common import nearest_head_structure
from engine import Structure


class PEGASIS(Protocol):
    """Lindsey & Raghavendra 2002: greedy chain from the node farthest from the BS, rotating leader."""
    name = "PEGASIS"
    control = "chain"

    def reset(self, world, rng):
        self.chain, self.mask = None, None

    def _build(self, s):
        left = set(np.flatnonzero(s.alive).tolist())
        cur = max(left, key=lambda i: s.d_bs[i])
        chain = [cur]
        left.discard(cur)
        while left:
            l = list(left)
            cur = l[int(np.argmin(s.D[cur, l]))]
            chain.append(cur)
            left.discard(cur)
        return chain

    def setup(self, s):
        rebuilt = self.chain is None or not np.array_equal(s.alive, self.mask)
        if rebuilt:
            self.chain, self.mask = self._build(s), s.alive.copy()
        c, N = self.chain, s.cfg.N
        L = s.r % len(c)
        parent = np.full(N, NONE, dtype=np.int64)
        for k in range(len(c)):
            parent[c[k]] = BS if k == L else (c[k + 1] if k < L else c[k - 1])
        heads = np.zeros(N, bool)
        heads[c[L]] = True
        return Structure(parent=parent, heads=heads, reclustered=rebuilt)


class TEEN(Protocol):
    """Manjeshwar & Agrawal 2001: LEACH clustering, event-driven sends gated by hard/soft thresholds."""
    name = "TEEN"
    control = "distributed"

    def reset(self, world, rng):
        self.rng = rng
        self.G = np.ones(world.cfg.N, bool)
        self.last = np.full(world.cfg.N, np.nan)

    def setup(self, s):
        c = s.cfg
        p = c.p_head
        epoch = int(round(1 / p))
        if s.r % epoch == 0:
            self.G[:] = True
        T = p / (1 - p * (s.r % epoch))
        u = self.rng.random(c.N)
        heads = s.alive & self.G & (u < T)
        self.G &= ~heads
        S = nearest_head_structure(s, heads)
        x = s.reading
        send = s.alive & (x >= c.ou_mu + 0.78 * c.ou_std) & (np.isnan(self.last) | (np.abs(x - self.last) >= 0.16 * c.ou_std))
        self.last[send] = x[send]
        S.send = send
        return S
