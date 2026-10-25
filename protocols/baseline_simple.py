"""Direct transmission and LEACH."""
import numpy as np

from engine import BS, NONE, Protocol, Structure
from protocols.common import nearest_head_structure


class Direct(Protocol):
    name = "Direct"
    control = "none"

    def setup(self, s):
        parent = np.where(s.alive, BS, NONE)
        return Structure(parent=parent, heads=np.zeros(s.cfg.N, bool))


class LEACH(Protocol):
    """Heinzelman et al. 2000: threshold T(n) = p / (1 - p (r mod 1/p)) over nodes not yet head this epoch."""
    name = "LEACH"
    control = "distributed"

    def reset(self, world, rng):
        self.rng = rng
        self.G = np.ones(world.cfg.N, bool)

    def setup(self, s):
        p = s.cfg.p_head
        epoch = int(round(1 / p))
        if s.r % epoch == 0:
            self.G[:] = True
        T = p / (1 - p * (s.r % epoch))
        u = self.rng.random(s.cfg.N)
        heads = s.alive & self.G & (u < T)
        self.G &= ~heads
        return nearest_head_structure(s, heads)
