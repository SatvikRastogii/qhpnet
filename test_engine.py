"""Engine gates: radio by hand on both sides of d0, PER curve, conservation, read-only state, pairing."""
import math
import time

import numpy as np

from config import Config
from engine import World, per_from, run, rx_cost, tx_cost
from protocols.baseline_simple import LEACH, Direct


def test_radio():
    c = Config()
    assert abs(c.d0 - 87.7) < 0.1
    assert math.isclose(float(tx_cost(c, 4000, 50)), 4000 * (50e-9 + 10e-12 * 2500))
    assert math.isclose(float(tx_cost(c, 4000, 100)), 4000 * (50e-9 + 0.0013e-12 * 1e8))
    assert math.isclose(rx_cost(c, 4000), 2e-4)


def test_per_curve():
    c = Config()
    p = lambda d: float(per_from(c, d, 0.0, 4000))
    assert 1e-8 < p(100) < 1e-7          # paper: 2.6e-8
    assert 5e-4 < p(120) < 2e-3          # paper: 1.0e-3
    assert 0.1 < p(140) < 0.3            # paper: 0.19
    assert p(158) > 0.9                  # paper: 0.97
    d = np.linspace(1, 250, 500)
    assert np.all(np.diff(per_from(c, d, 0.0, 4000)) >= 0)


def test_pairing_and_readonly():
    c = Config()
    a, b = World(c, 3), World(c, 3)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.shadow, b.shadow)
    assert np.array_equal(a.readings, b.readings)
    try:
        a.pos[0, 0] = 1.0
        raise AssertionError("world arrays must be read-only")
    except ValueError:
        pass


def test_runs():
    c = Config()
    w = World(c, 0)
    for P in (Direct, LEACH):
        t = time.perf_counter()
        m = run(P(), w)          # conservation is asserted inside run()
        print(f"{m['protocol']:8s} FND {m['FND']:5d} HND {m['HND']:5d} LND {m['LND']:5d} "
              f"AUC {m['AUC']/1e3:6.1f}k PDR {m['PDR']:.3f} ctrl% "
              f"{100*m['energy']['ctrl']/m['consumed']:.2f} retry% {100*m['energy']['retry']/m['consumed']:.2f} "
              f"{time.perf_counter()-t:.1f}s")
    m = run(LEACH(), w, max_rounds=900)                   # before first death (FND 972 on seed 0)
    assert m["FND"] == 900 and np.all(m["duty"] == 45)   # epoch elects every node once per 20 rounds


if __name__ == "__main__":
    test_radio(); test_per_curve(); test_pairing_and_readonly(); test_runs()
    print("all engine gates pass")
