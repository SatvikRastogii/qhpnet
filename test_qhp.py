"""QHP-Net self-checks: the stress model predicts the engine's data energy, routes are acyclic,
and every ablation variant runs. Tuning seed 500 only."""
from dataclasses import replace

import numpy as np

from config import Config
from engine import BS, State, World, run
from protocols.qhp import QHPNet


def predicted_vs_engine(lossy):
    cfg = replace(Config(), lossy=lossy)
    w = World(cfg, 500)
    P = QHPNet(channel="oracle")
    P.reset(w, np.random.default_rng(0))
    E = w.E_init.copy()
    s = State(w, E, E > 0, np.zeros(cfg.N, int), np.full(cfg.N, -1), 0)
    st = P.setup(s)
    cand = P._candidates(s, 5)[0]
    c = P._context(s, cand)
    # rebuild the chosen structure in candidate positions and price it with the stress model
    pos = {int(n): j for j, n in enumerate(cand)}
    choice = np.array([[-1 if st.parent[i] == BS else pos[int(st.parent[i])] for i in c["A"]]])
    pred = P._drain(c, choice)[0].sum()
    m = run(QHPNet(channel="oracle"), w, max_rounds=1)
    actual = sum(m["energy"][k] for k in ("tx", "rx", "agg", "retry"))
    return pred, actual


def test_stress_model():
    for lossy in (False, True):
        pred, actual = predicted_vs_engine(lossy)
        err = abs(pred - actual) / actual
        print(f"lossy={lossy}: predicted {pred*1e3:.3f} mJ, engine {actual*1e3:.3f} mJ, error {100*err:.2f}%")
        assert err < (0.01 if not lossy else 0.10), err


def test_variants_run():
    w = World(Config(), 500)
    for kw in ({}, {"qs": False}, {"ho": False}, {"direct": False}, {"stress_join": False},
               {"channel": "none"}, {"physarum": True}):
        m = run(QHPNet(**kw), w, max_rounds=60)     # engine raises on any routing cycle
        assert m["delivered"] > 0, kw


if __name__ == "__main__":
    test_stress_model()
    test_variants_run()
    print("QHP-Net checks pass")
