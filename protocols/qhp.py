"""QHP-Net: Quorum-sensing candidacy -> Hippopotamus-optimised clustering -> Pheromone-learned routing.

Layer 1 (CH candidacy, quorum sensing): every node secretes an autoinducer proportional to its energy,
  suppressed by a refractory term after serving as head. A node expresses the candidate phenotype when
  its own secretion beats the kernel-weighted concentration around it (local quorum), so candidates are
  healthy, rested and spread over the field. Beacons are charged as real control packets.
Layer 2 (CH selection + clustering, Hippopotamus Optimization, Amiri et al. 2024): continuous positions
  decoded to the nearest unused candidate. Fitness = bottleneck stress max_i(drain_i/E_i) plus mean
  stress, where drain is the predicted per-round energy including expected ARQ retries from link
  quality. Members join the head (or the sink directly) that minimises their stress contribution, which
  yields unequal, energy-proportional clusters and switches clustering off where it does not pay.
Layer 3 (inter-cluster routing, ant-pheromone link memory): the protocol is never told link quality.
  Every ARQ attempt evaporates the link's pheromone and every ack deposits on it, so the trail converges
  to the per-attempt delivery probability (no prior: every link starts at 1). Heads forward to the sink or
  to a head strictly closer to it, minimising hop stress tx*ETX/E_i + rx*ETX/E_j under the learned ETX;
  routes are acyclic by construction. With ideal fusion a head's packet costs one hop, so per-hop stress
  is the exact marginal cost. physarum=True swaps in a congestion-aware Physarum flux router (Tero et
  al.), kept as a measured negative result.
"""
import math

import numpy as np

from engine import BS, NONE, Protocol, Structure, rx_cost, tx_cost
from protocols.common import n_heads


def levy(rng, shape, beta=1.5):
    s = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
         / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0, s, shape) / np.abs(rng.normal(0, 1, shape)) ** (1 / beta)


class QHPNet(Protocol):
    name = "QHP-Net"
    control = "centralized"
    POP, ITERS = 20, 25           # hippopotamus herd size and iterations per epoch
    TAU = 20.0                    # refractory time constant (rounds)
    CAND = (3, 8)                 # candidate pool kept within [3k, 8k]
    QS_EVERY = 4                  # autoinducer beacons every QS_EVERY re-clustering epochs
    MU = 0.5                      # weight of mean stress against bottleneck stress
    P_HEAD = None                 # head fraction; None = the shared nominal p
    PHY_IT, PHY_MU, PHY_DT = 10, 2.0, 0.5   # Physarum iterations, flux exponent, adaptation rate
    RHO = 0.1                     # pheromone evaporation / deposit rate
    GAMMA = 1.0                   # weight of receiver congestion in tube length

    def __init__(self, qs=True, ho=True, physarum=False, direct=True, stress_join=True, channel="learned",
                 name=None, **params):
        for key, v in params.items():
            if not hasattr(type(self), key):
                raise AttributeError(key)
            setattr(self, key, v)
        self.qs, self.ho, self.physarum = qs, ho, physarum
        self.direct, self.stress_join, self.channel = direct, stress_join, channel
        if name:
            self.name = name

    # ------------------------------------------------------------ lifecycle
    def reset(self, world, rng):
        cfg = world.cfg
        self.rng = rng
        self.N = cfg.N
        self.txd = tx_cost(cfg, cfg.data_bits, world.D)              # J per attempt, per link
        self.crx = rx_cost(cfg, cfg.data_bits) + cfg.E_DA * cfg.data_bits
        if self.channel == "oracle":                                 # true per-link PER (upper reference)
            self.tau = 1.0 - world.per_data
        else:                                                        # no prior: every link starts perfect
            self.tau = np.ones_like(world.D)                         # link pheromone = P(ack per attempt)
        self._refresh_links()
        rho = cfg.density_radius
        Dn = world.D[:-1, :-1]
        self.K = np.exp(-(Dn / rho) ** 2 / 2.0) * (Dn < rho)         # autoinducer kernel, beacon range only
        self.nbr = [np.flatnonzero((Dn[i] < rho) & (np.arange(cfg.N) != i)) for i in range(cfg.N)]
        self.struct = None
        self.t_last = -10 ** 9
        self.epoch = 0
        self.quorum = np.ones(cfg.N)
        self.best_pos = None

    def _refresh_links(self):
        p = 1.0 - self.tau
        self.etx = 1.0 + p + p * p                                   # expected attempts, <= 3 tries
        self.txj = self.txd * self.etx                               # expected tx joules per packet
        self.rxj = self.crx * self.etx                               # expected rx + fusion joules

    def feedback(self, acks):
        """Ant-pheromone link memory: every attempt evaporates the link's trail, every ack deposits."""
        if self.channel != "learned" or not acks:
            return
        a = np.asarray(acks, dtype=np.int64)
        i, j, n, ok = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        for u, v in ((i, j), (j, i)):                               # links are reciprocal
            self.tau[u, v] = self.tau[u, v] * (1.0 - self.RHO) ** n + self.RHO * ok

    def setup(self, s):
        cfg = s.cfg
        st = self.struct
        if (st is not None and s.r - self.t_last < cfg.recluster_interval
                and np.all(s.alive[np.flatnonzero(st.heads)])):
            return Structure(parent=st.parent, heads=st.heads, reclustered=False)
        self.t_last = s.r
        self.struct = self._recluster(s)
        self.epoch += 1
        return self.struct

    # ------------------------------------------------------------ layer 1: quorum sensing
    def _fatigue(self, s, idx):
        lh = np.asarray(s.last_head)[idx]
        return np.where(lh >= 0, np.exp(-(s.r - lh) / self.TAU), 0.0)

    def _candidates(self, s, k):
        A = np.flatnonzero(s.alive)
        E = np.asarray(s.E)
        if not self.qs:
            return A[np.argsort(-E[A], kind="stable")], []    # whole alive set, ranked by energy
        beacons = []
        if self.epoch % self.QS_EVERY == 0:
            a = E[A] / s.E_init.mean() * (1.0 - self._fatigue(s, A))  # autoinducer secretion
            Ka = self.K[np.ix_(A, A)]
            conc = Ka @ a / Ka.sum(1)                                 # local concentration
            self.quorum[:] = 0.0
            self.quorum[A] = a / np.maximum(conc, 1e-12)              # >1: local quorum reached
            rho = s.cfg.density_radius
            beacons = [(i, self.nbr[i], rho) for i in A]
        # the sink assigned past heads, so it applies the refractory term between beacons itself
        score = self.quorum[A] * (1.0 - self._fatigue(s, A))
        lo, hi = min(self.CAND[0] * k, A.size), min(self.CAND[1] * k, A.size)
        n = int(np.clip((score >= 1.0).sum(), lo, hi))
        return A[np.argsort(-score, kind="stable")[:n]], beacons

    # ------------------------------------------------------------ stress model (fitness)
    def _context(self, s, cand):
        """Tables for one epoch, indexed by alive node (rows) and candidate (columns)."""
        N, E = self.N, np.asarray(s.E)
        A = np.flatnonzero(s.alive)
        c = dict(A=A, cand=cand, EA=E[A], Ec=E[cand])
        c["tx_m"] = self.txj[np.ix_(A, cand)]                         # member -> candidate head
        c["rx_m"] = self.rxj[np.ix_(A, cand)]
        if self.stress_join:
            c["cost_m"] = c["tx_m"] / E[A][:, None] + c["rx_m"] / E[cand][None, :]
            c["cost_d"] = self.txj[A, N] / E[A]
        else:
            c["cost_m"] = s.D[np.ix_(A, cand)]
            c["cost_d"] = None
        c["tx_d"] = self.txj[A, N]
        c["tot_m"] = self.txd[np.ix_(A, cand)] + self.crx
        # head -> head and head -> sink
        c["tx_hh"] = self.txj[np.ix_(cand, cand)]
        c["rx_hh"] = self.rxj[np.ix_(cand, cand)]
        c["cost_hh"] = c["tx_hh"] / E[cand][:, None] + c["rx_hh"] / E[cand][None, :]
        dc = s.d_bs[cand]
        c["cost_hh"] = np.where(dc[None, :] < dc[:, None], c["cost_hh"], np.inf)
        c["tx_hb"] = self.txj[cand, N]
        c["cost_hb"] = c["tx_hb"] / E[cand]
        pos_in_A = np.full(N, -1)
        pos_in_A[A] = np.arange(A.size)
        c["candA"] = pos_in_A[cand]
        return c

    def _route(self, c, Hc, base):
        """Next hop of every head, only toward heads strictly closer to the sink or the sink itself.
        Hc: (P,k) candidate positions; base: (P,k) drain already put on each head by its members.
        Physarum: each head is a flux source, every allowed next hop a grounded tube. Tube length is
        the hop's stress plus GAMMA times the receiver's congestion stress, so tubes into overloaded
        heads lengthen as flux arrives; conductance follows flux (dD/dt = Q^mu - D) and with mu > 1 one
        tube per head survives. GAMMA = 0 or physarum=False gives the greedy per-hop rule.
        Returns (P,k) candidate position of the next hop, -1 = sink."""
        P, k = Hc.shape
        Ei = c["Ec"][Hc]
        a, b = Hc[:, :, None], Hc[:, None, :]
        marg = c["cost_hh"][a, b]                                     # inf where not closer
        L0 = np.concatenate([marg, c["cost_hb"][Hc][:, :, None]], axis=2)
        if self.physarum:
            ok = np.isfinite(L0)
            Dt = np.ones_like(L0)
            for _ in range(self.PHY_IT):
                G = np.where(ok, Dt / np.where(ok, L0, 1.0), 0.0)
                if _ > 0:
                    infl = Q[:, :, :k].sum(1)                         # flux arriving at each head
                    cong = (base + self.crx * infl) / Ei
                    L = L0 + self.GAMMA * np.concatenate([np.broadcast_to(cong[:, None, :], marg.shape),
                                                          np.zeros((P, k, 1))], axis=2)
                    G = np.where(ok, Dt / np.where(ok, L, 1.0), 0.0)
                Q = G / G.sum(2, keepdims=True)
                Dt += self.PHY_DT * (Q ** self.PHY_MU - Dt)
            j = np.where(ok, Dt, -1.0).argmax(2)
        else:
            j = L0.argmin(2)
        return np.where(j == k, -1, np.take_along_axis(Hc, np.minimum(j, k - 1), axis=1))

    def _members(self, c, Hc):
        """Every alive node's choice in candidate positions (-1 = sink): nearest-stress head or sink."""
        cm = c["cost_m"][:, Hc]                                       # (n, P, k)
        best = cm.argmin(2)
        choice = np.take_along_axis(Hc, best.T, axis=1)               # (P, n)
        if self.direct:
            if self.stress_join:
                cb = np.take_along_axis(cm, best[:, :, None], axis=2)[:, :, 0].T
                choice = np.where(c["cost_d"][None, :] < cb, -1, choice)
            else:
                tot = np.take_along_axis(c["tot_m"][:, Hc], best[:, :, None], axis=2)[:, :, 0].T
                choice = np.where(self.txd[c["A"], self.N][None, :] < tot, -1, choice)
        return choice

    def _drain(self, c, choice):
        P, n = choice.shape
        C = c["cand"].size
        sink = choice < 0
        ch = np.maximum(choice, 0)
        drain = np.where(sink, c["tx_d"][None, :], c["tx_m"][np.arange(n)[None, :], ch])
        rx = np.where(sink, 0.0, c["rx_m"][np.arange(n)[None, :], ch])
        flat = (np.arange(P)[:, None] * C + ch).ravel()
        recv = np.bincount(flat, weights=rx.ravel(), minlength=P * C).reshape(P, C)
        drain[:, c["candA"]] += recv
        return drain, recv

    def _assign(self, c, Hc):
        P, k = Hc.shape
        choice = self._members(c, Hc)
        rows = np.repeat(np.arange(P), k)
        hA = c["candA"][Hc].ravel()
        choice[rows, hA] = -1                                         # heads: route decided below
        _, recv = self._drain(c, choice)
        base = np.take_along_axis(recv, Hc, axis=1)
        choice[rows, hA] = self._route(c, Hc, base).ravel()
        return choice

    def _fitness(self, c, Hc):
        drain, _ = self._drain(c, self._assign(c, Hc))
        stress = drain / c["EA"][None, :]
        return stress.max(1) + self.MU * stress.mean(1)

    # ------------------------------------------------------------ layer 2: hippopotamus optimisation
    def _decode(self, X, c, pos):
        """X: (P, k, 2) continuous points -> (P, k) distinct candidate positions (nearest unused)."""
        P, k, _ = X.shape
        d = ((X[:, :, None, :] - pos[c["cand"]][None, None, :, :]) ** 2).sum(3)    # (P,k,C)
        used = np.zeros((P, c["cand"].size), bool)
        H = np.empty((P, k), dtype=np.int64)
        ar = np.arange(P)
        for j in range(k):
            cj = np.where(used, np.inf, d[:, j, :]).argmin(1)
            used[ar, cj] = True
            H[:, j] = cj
        return H

    def _hippo(self, s, c, k):
        rng, W = self.rng, s.cfg.W
        pos = s.pos
        P, T = self.POP, self.ITERS
        lb, ub = 0.0, W
        X = rng.uniform(lb, ub, (P, k, 2))
        # seed one hippo on the strongest candidates, one on last epoch's best (elitist memory)
        X[0] = pos[c["cand"][:k]]
        if self.best_pos is not None and self.best_pos.shape == (k, 2):
            X[1] = self.best_pos
        f = lambda Z: self._fitness(c, self._decode(Z, c, pos))
        F = f(X)
        half = P // 2

        def accept(Xn, idx):
            Xn = np.clip(Xn, lb, ub)
            Fn = f(Xn)
            better = Fn < F[idx]
            X[idx[better]] = Xn[better]
            F[idx[better]] = Fn[better]

        i1, i2 = np.arange(half), np.arange(half, P)
        m = i2.size
        for t in range(1, T + 1):
            Dh = X[F.argmin()]                                         # dominant hippopotamus
            # phase 1: position update in the river or pond (exploration)
            y1 = rng.random((half, 1, 1))
            I1 = rng.integers(1, 3, (half, 1, 1))
            accept(X[i1] + y1 * (Dh[None] - I1 * X[i1]), i1)
            MG = np.stack([X[rng.choice(P, rng.integers(2, P + 1), replace=False)].mean(0) for _ in range(m)])
            I2 = rng.integers(1, 3, (m, 1, 1))
            Q = rng.integers(0, 2, (m, 1, 1))
            h = [I2 * rng.random((m, 1, 1)) + (1 - Q), 2 * rng.random((m, 1, 1)) - 1,
                 rng.random((m, k, 2)), I2 * rng.random((m, k, 2)) + (1 - Q),
                 rng.random((m, 1, 1))][int(rng.integers(5))]
            if math.exp(-t / T) > 0.6:
                Xn = X[i2] + h * (Dh[None] - I2 * MG)
            elif rng.random() > 0.5:
                Xn = X[i2] + h * (MG - Dh[None])
            else:
                Xn = rng.uniform(lb, ub, (m, k, 2))
            accept(Xn, i2)
            # phase 2: defence against a predator (exploration with Levy flight)
            pred = rng.uniform(lb, ub, (m, k, 2))
            Fp = f(pred)
            dist = np.abs(pred - X[i2]) + 1e-9
            b = rng.uniform(2, 4, (m, 1, 1)); cc = rng.uniform(1, 1.5, (m, 1, 1))
            dd = rng.uniform(2, 3, (m, 1, 1)); g = rng.uniform(-1, 1, (m, 1, 1))
            coef = b / (cc - dd * np.cos(2 * np.pi * g))
            RL = 0.05 * levy(rng, (m, k, 2))
            near = (Fp < F[i2])[:, None, None]
            accept(RL * pred + coef * np.where(near, 1.0 / dist, 1.0 / (2 * dist + rng.random((m, 1, 1)))), i2)
            # phase 3: escaping the predator (exploitation inside shrinking local bounds)
            lbl, ubl = lb / t, ub / t
            s1 = [2 * rng.random((P, k, 2)) - 1, rng.random((P, 1, 1)),
                  rng.normal(0, 1, (P, 1, 1))][int(rng.integers(3))]
            accept(X + rng.random((P, 1, 1)) * (lbl + s1 * (ubl - lbl)), np.arange(P))
        best = X[F.argmin()]
        self.best_pos = best.copy()
        return self._decode(best[None], c, pos)[0]

    # ------------------------------------------------------------ epoch
    def _recluster(self, s):
        N = self.N
        alive_n = int(s.alive.sum())
        k = n_heads(s) if self.P_HEAD is None else max(1, int(round(self.P_HEAD * alive_n)))
        k = min(k, alive_n)
        if self.channel == "learned":
            self._refresh_links()
        cand, beacons = self._candidates(s, k)
        c = self._context(s, cand)
        if self.ho:
            hc = self._hippo(s, c, k)
        else:
            hc = np.arange(k)                       # candidates are ranked: take the strongest k
        choice = self._assign(c, hc[None, :])
        parent = np.full(N, NONE, dtype=np.int64)
        ch = choice[0]
        parent[c["A"]] = np.where(ch < 0, BS, cand[np.maximum(ch, 0)])
        heads = np.zeros(N, bool)
        heads[cand[hc]] = True
        extra = list(beacons)
        if not s.cfg.charge_uplink:                     # state of the candidates reaches the sink
            extra += [(int(i), np.empty(0, np.int64), float(s.d_bs[i])) for i in cand]
        return Structure(parent=parent, heads=heads, reclustered=True, extra_ctrl=extra)
