"""Single frozen configuration shared by every protocol. Nothing here is protocol specific."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Config:
    # topology and energy
    N: int = 100
    W: float = 100.0                 # square field side (m)
    bs: tuple = (50.0, 150.0)        # base station, outside the field
    E0: float = 1.0                  # J per normal node
    max_rounds: int = 7000           # cap; runs reaching it are censored
    # heterogeneity (SEP 3-level): fraction m advanced (E0*(1+alpha)), fraction b super (E0*(1+beta))
    het_m: float = 0.0
    het_b: float = 0.0
    het_alpha: float = 1.0
    het_beta: float = 2.0
    # packets
    data_bits: int = 4000
    ctrl_bits: int = 200
    # first order radio
    E_elec: float = 50e-9
    e_fs: float = 10e-12
    e_mp: float = 0.0013e-12
    E_DA: float = 5e-9               # fusion, per bit received
    # channel (log-distance + per-link shadowing, non-coherent BFSK)
    lossy: bool = True
    Pt_dBm: float = 12.0
    PL1m_dB: float = 40.0
    n_pl: float = 3.0
    sigma_dB: float = 4.0
    noise_dBm: float = -105.0
    max_retries: int = 2
    # sensed data, Ornstein-Uhlenbeck
    ou_mu: float = 50.0
    ou_theta: float = 0.05
    ou_sigma: float = 2.0
    # protocol agnostic
    p_head: float = 0.05
    density_radius: float = 25.0
    recluster_interval: int = 5
    charge_uplink: bool = False      # centralised uplink of node state; False = previous-paper convention
    # timing (latency is analytic, a lower bound)
    bitrate: float = 250e3
    t_slot: float = 20e-3
    t_guard: float = 4e-3
    t_proc: float = 1e-3

    @property
    def d0(self):
        return math.sqrt(self.e_fs / self.e_mp)

    @property
    def ou_std(self):
        a = 1.0 - self.ou_theta
        return self.ou_sigma / math.sqrt(1.0 - a * a)


def cfg_for(N=100, W=100.0, **kw):
    """Sweep helper: base station and density radius scale with the field, as in the previous study."""
    return replace(Config(), N=N, W=float(W), bs=(W / 2.0, 1.5 * W), density_radius=0.25 * W, **kw)
