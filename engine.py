"""Simulation engine. The engine owns energy, channel, packet sizes, fusion and delivery counting.

A protocol only returns a Structure (who is head, who forwards to whom). Every array a protocol
can see is a read-only view, so no protocol can write energy or liveness.
"""
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from config import Config

BS = -1      # parent value meaning "the base station"
NONE = -2    # parent value meaning "does not transmit this round"


# ---------------------------------------------------------------- radio

def tx_cost(cfg, bits, d):
    d = np.asarray(d, dtype=float)
    amp = np.where(d < cfg.d0, cfg.e_fs * d ** 2, cfg.e_mp * d ** 4)
    return bits * (cfg.E_elec + amp)


def rx_cost(cfg, bits):
    return bits * cfg.E_elec


def per_from(cfg, d, shadow_dB, bits):
    """Packet error rate of a link of length d with fixed shadowing, non-coherent BFSK."""
    d = np.maximum(np.asarray(d, dtype=float), 1.0)
    pl = cfg.PL1m_dB + 10.0 * cfg.n_pl * np.log10(d) + shadow_dB
    snr = 10.0 ** ((cfg.Pt_dBm - pl - cfg.noise_dBm) / 10.0)
    ber = 0.5 * np.exp(-snr / 2.0)
    return -np.expm1(bits * np.log1p(-ber))


# ---------------------------------------------------------------- world

class World:
    """Everything physical about one trial. Seeded only by the trial index, so every protocol
    at index i faces byte-identical positions, energies, interference and readings."""

    def __init__(self, cfg: Config, seed: int):
        self.cfg, self.seed = cfg, seed
        rng = np.random.default_rng(seed)
        N = cfg.N
        self.pos = rng.uniform(0.0, cfg.W, size=(N, 2))
        E0 = np.full(N, cfg.E0)
        perm = rng.permutation(N)
        n_adv, n_sup = int(round(cfg.het_m * N)), int(round(cfg.het_b * N))
        E0[perm[:n_adv]] *= 1.0 + cfg.het_alpha
        E0[perm[n_adv:n_adv + n_sup]] *= 1.0 + cfg.het_beta
        self.E_init = E0
        # node N is the base station in every (N+1)x(N+1) matrix
        P = np.vstack([self.pos, np.asarray(cfg.bs, dtype=float)[None, :]])
        self.D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        X = rng.normal(0.0, cfg.sigma_dB, size=(N + 1, N + 1))
        X = np.triu(X, 1)
        self.shadow = X + X.T                       # symmetric, fixed for the topology
        if cfg.lossy:
            self.per_data = per_from(cfg, self.D, self.shadow, cfg.data_bits)
        else:
            self.per_data = np.zeros_like(self.D)
        np.fill_diagonal(self.per_data, 0.0)
        # sensed stream: OU process started from its stationary law
        T = cfg.max_rounds
        a = 1.0 - cfg.ou_theta
        x = np.empty((T, N))
        x[0] = cfg.ou_mu + cfg.ou_std * rng.standard_normal(N)
        z = rng.standard_normal((T, N)) * cfg.ou_sigma
        for t in range(1, T):
            x[t] = cfg.ou_mu + a * (x[t - 1] - cfg.ou_mu) + z[t]
        self.readings = x
        for arr in (self.pos, self.E_init, self.D, self.shadow, self.per_data, self.readings):
            arr.setflags(write=False)


def keyed_rng(seed, name, stream=0):
    return np.random.default_rng([seed, zlib.crc32(name.encode()), stream])


# ---------------------------------------------------------------- protocol contract

@dataclass
class Structure:
    parent: np.ndarray                  # int, len N: node index, BS (-1) or NONE (-2)
    heads: np.ndarray                   # bool, len N
    reclustered: bool = True            # was the cluster structure rebuilt this round
    send: np.ndarray = None             # bool, len N: which nodes inject their own reading (None = all alive)
    extra_ctrl: list = field(default_factory=list)  # [(src, dst_index_array, tx_distance)] priced by the engine


class State:
    """Read-only view handed to protocols."""

    def __init__(self, world, E, alive, duty, last_head, r):
        self.cfg = world.cfg
        self.pos = world.pos
        self.D = world.D
        self.d_bs = world.D[:-1, -1]
        self.per = world.per_data           # TRUE per-link PER (oracle); QHP-Net learns from acks instead
        self.E_init = world.E_init
        self.E = E
        self.alive = alive
        self.duty = duty                    # rounds served as head so far
        self.last_head = last_head          # last round served as head (-1 never)
        self.r = r
        self.reading = world.readings[min(r, world.cfg.max_rounds - 1)]


def _ro(a):
    v = a.view()
    v.setflags(write=False)
    return v


class Protocol:
    name = "base"
    control = "none"        # none | distributed | centralized | chain

    def reset(self, world, rng):
        pass

    def setup(self, s: State) -> Structure:
        raise NotImplementedError

    def feedback(self, acks):
        """Optional. acks: list of (sender, receiver, attempts, delivered) for this round's data packets,
        receiver N = base station. This is what a sender learns from its own ARQ acknowledgements."""


# ---------------------------------------------------------------- run

CATS = ("tx", "rx", "ctrl", "agg", "retry")


def run(protocol: Protocol, world: World, max_rounds=None, trace=False):
    cfg = world.cfg
    N = cfg.N
    T = max_rounds or cfg.max_rounds
    E = world.E_init.copy()
    alive = E > 0
    duty = np.zeros(N, dtype=np.int64)
    last_head = np.full(N, -1, dtype=np.int64)
    death = np.full(N, -1, dtype=np.int64)
    spent = dict.fromkeys(CATS, 0.0)
    chan = keyed_rng(world.seed, protocol.name, 1)
    protocol.reset(world, keyed_rng(world.seed, protocol.name, 0))

    D, PER = world.D, world.per_data
    Dl = D.tolist()
    PERl = PER.tolist()
    kd = cfg.data_bits
    rxd = rx_cost(cfg, kd)
    fuse = cfg.E_DA * kd
    tx_tab = tx_cost(cfg, kd, D).tolist()       # data tx cost per link
    ctx = lambda d: float(tx_cost(cfg, cfg.ctrl_bits, d))
    crx = rx_cost(cfg, cfg.ctrl_bits)
    t_tx = kd / cfg.bitrate

    alive_curve = np.zeros(T, dtype=np.int32)
    energy_curve = np.zeros(T)
    head_dist_hist = np.zeros(260, dtype=np.int64)    # 1 m bins of sink-hop length
    packets_bs = delivered = injected = taken = lost = 0
    lat_sum = 0.0
    setup_t = 0.0
    n_setup = 0
    r = 0

    def charge_vec(amount, cat):
        """Charge a per-node vector; a node spends at most what it holds."""
        nonlocal E
        s = np.minimum(amount, E)
        E -= s
        spent[cat] += float(s.sum())

    while r < T and alive.any():
        st = State(world, _ro(E), _ro(alive), _ro(duty), _ro(last_head), r)
        t0 = time.perf_counter()
        S = protocol.setup(st)
        setup_t += time.perf_counter() - t0
        n_setup += 1
        parent = np.asarray(S.parent, dtype=np.int64)
        heads = np.asarray(S.heads, dtype=bool) & alive
        parent = np.where(alive, parent, NONE)
        if parent.shape != (N,):
            raise ValueError(f"{protocol.name}: parent has shape {parent.shape}")
        if np.any((parent >= N) | (parent < NONE)) or np.any(parent == np.arange(N)):
            raise ValueError(f"{protocol.name}: invalid parent array")

        duty[heads] += 1
        last_head[heads] = r

        # ---------------- control traffic, priced by the engine from the declared class
        ctrl = np.zeros(N)
        kc = cfg.ctrl_bits
        members = (parent >= 0) & ~heads & alive
        mi = np.flatnonzero(members)
        mp = parent[mi]
        radius = np.zeros(N)                             # distance from each head to its farthest member
        np.maximum.at(radius, mp, D[mi, mp])
        nmem_h = np.bincount(mp, minlength=N)
        h_idx = np.flatnonzero(heads)
        if protocol.control == "distributed":
            # advertisement priced at the farthest joiner, heard by every alive non-head sensor
            ctrl[h_idx] += tx_cost(cfg, kc, radius[h_idx])
            ctrl[alive & ~heads] += crx * h_idx.size
            ctrl[mi] += tx_cost(cfg, kc, D[mi, mp])      # join request
            np.add.at(ctrl, mp, crx)
        if protocol.control in ("distributed", "centralized"):
            hs = h_idx[nmem_h[h_idx] > 0]                # TDMA schedule, every round
            ctrl[hs] += tx_cost(cfg, kc, radius[hs])
            ctrl[mi] += crx
            ctrl[alive & ~heads & (parent == BS)] += crx  # sink-bound non-heads get their slot from the BS
        if protocol.control == "centralized" and S.reclustered:
            ctrl[alive] += crx                           # sink downlink of assignments
            if cfg.charge_uplink:
                ctrl[alive] += tx_cost(cfg, kc, world.D[:-1, -1][alive])
        if protocol.control == "chain":
            ci = np.flatnonzero(alive & (parent >= 0))
            ctrl[ci] += tx_cost(cfg, kc, D[ci, parent[ci]])   # leader token, hop by hop
            np.add.at(ctrl, parent[ci], crx)
            if S.reclustered:
                ctrl[alive] += crx                       # sink broadcast of the rebuilt chain
        for src, dst, dist in S.extra_ctrl:
            if alive[src]:
                ctrl[src] += ctx(dist)
                dst = np.asarray(dst, dtype=np.int64)
                dst = dst[alive[dst]]
                ctrl[dst] += crx
        charge_vec(ctrl, "ctrl")
        alive_ctrl = E > 0

        # ---------------- data phase
        send = alive_ctrl if S.send is None else (np.asarray(S.send, dtype=bool) & alive_ctrl)
        taken += int(alive_ctrl.sum())
        cnt = send.astype(np.int64).tolist()
        injected += int(send.sum())
        par = parent.tolist()
        # depth: hops to the BS; cycles are a protocol bug
        depth = [0] * N
        for i in range(N):
            if par[i] == NONE or not alive_ctrl[i]:
                depth[i] = -1
                continue
            j, h, seen = i, 0, 0
            while j >= 0:
                j = par[j]
                h += 1
                seen += 1
                if seen > N:
                    raise ValueError(f"{protocol.name}: routing cycle through node {i}")
                if j == NONE:
                    h = -1
                    break
            depth[i] = h
        slot = [0] * N
        nmem = np.bincount(parent[members], minlength=N) if members.any() else np.zeros(N, dtype=np.int64)
        order = sorted((i for i in range(N) if depth[i] > 0), key=lambda i: (-depth[i], i))
        El = E.tolist()
        nxt_slot = [0] * N
        for i in order:
            p = par[i]
            if p >= 0:
                slot[i] = nxt_slot[p]
                nxt_slot[p] += 1
        u = chan.random(len(order) * (cfg.max_retries + 1) + 1).tolist()
        ui = 0
        acks = []
        for i in order:
            c = cnt[i]
            if c == 0:
                continue
            p = par[i]
            t = N if p == BS else p
            cost = tx_tab[i][t]
            pe = PERl[i][t]
            ok = False
            tries = 0
            for attempt in range(cfg.max_retries + 1):
                cat_tx = "tx" if attempt == 0 else "retry"
                cat_rx = "rx" if attempt == 0 else "retry"
                if El[i] <= 0.0:
                    break
                if El[i] < cost:
                    spent[cat_tx] += El[i]
                    El[i] = 0.0
                    break
                El[i] -= cost
                spent[cat_tx] += cost
                tries += 1
                if t < N:
                    if El[t] <= 0.0:
                        continue                       # receiver dead: no ack, sender retries
                    if El[t] < rxd:
                        spent[cat_rx] += El[t]
                        El[t] = 0.0
                        continue
                    El[t] -= rxd
                    spent[cat_rx] += rxd
                x = u[ui]
                ui += 1
                if x >= pe:
                    ok = True
                    break
            if tries:
                acks.append((i, t, tries, ok))
            if ok:
                if t == N:
                    delivered += c
                    packets_bs += 1
                    lat_sum += c * ((nmem[i] if heads[i] else slot[i]) * cfg.t_slot
                                    + depth[i] * (t_tx + cfg.t_proc))
                    dd = int(Dl[i][N])
                    head_dist_hist[min(dd, 259)] += 1
                else:
                    cnt[t] += c
                    f = min(fuse, El[t])
                    El[t] -= f
                    spent["agg"] += f
            else:
                lost += c
        E[:] = El
        protocol.feedback(acks)
        new_alive = E > 0
        died = alive & ~new_alive
        death[died] = r + 1
        alive = new_alive
        alive_curve[r] = int(alive.sum())
        energy_curve[r] = float(E.sum())
        r += 1

    rounds = r
    consumed = float(world.E_init.sum() - E.sum())
    acct = sum(spent.values())
    if abs(consumed - acct) > 1e-9:
        raise AssertionError(f"energy not conserved: {consumed} vs {acct}")
    dead_sorted = np.sort(death[death > 0])
    censored = bool(alive.any())

    def nth_death(k):
        return int(dead_sorted[k - 1]) if dead_sorted.size >= k else T
    fnd = nth_death(1)
    hnd = nth_death((N + 1) // 2)
    lnd = nth_death(N) if not censored else T
    return {
        "protocol": protocol.name, "seed": world.seed, "rounds": rounds,
        "FND": fnd, "HND": hnd, "LND": lnd, "censored": censored,
        "AUC": int(alive_curve.sum()),
        "packets_bs": packets_bs, "delivered": delivered, "injected": injected,
        "taken": taken, "lost": lost,
        "PDR": delivered / injected if injected else 0.0,
        "yield": delivered / taken if taken else 0.0,
        "energy": spent, "consumed": consumed,
        "J_per_reading": consumed / delivered if delivered else float("inf"),
        "latency_ms": 1e3 * lat_sum / delivered if delivered else 0.0,
        "setup_ms": 1e3 * setup_t / max(n_setup, 1),
        "duty": duty, "head_dist_hist": head_dist_hist,
        "alive_curve": alive_curve, "energy_curve": energy_curve,
    }
