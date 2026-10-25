"""Paired Monte Carlo experiments. Usage: python montecarlo.py <experiment> [workers]

Every job is (experiment, variant, config, seed); the world depends only on (config, seed), so every
protocol at a given seed faces the same topology, interference and readings. Results append to
results/<experiment>.jsonl and finished jobs are skipped on restart. Test seeds are 0..29; protocol
parameters were fixed on seeds 500..505 beforehand.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import numpy as np

from config import Config, cfg_for
from engine import World, run
from protocols.common import gini

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(30))

# name -> (module, class, kwargs)
P = {
    "Direct": ("protocols.baseline_simple", "Direct", {}),
    "LEACH": ("protocols.baseline_simple", "LEACH", {}),
    "PEGASIS": ("protocols.classic", "PEGASIS", {}),
    "TEEN": ("protocols.classic", "TEEN", {}),
    "NSGA-II": ("protocols.nsga2", "NSGA2", {}),
    "Fuzzy-T2": ("protocols.fuzzy", "FuzzyT2", {}),
    "DQN": ("protocols.dqn", "DQN", {}),
    "HIECF": ("protocols.fuzzy", "HIECF", {}),
    "QHP-Net": ("protocols.qhp", "QHPNet", {}),
    "NSGA-II (warm start)": ("protocols.nsga2", "NSGA2", {"warm": True, "name": "NSGA-II (warm start)"}),
}
ABLATION = {
    "QHP-Net": {},
    "w/o quorum sensing": {"qs": False},
    "w/o hippopotamus opt.": {"ho": False},
    "w/o direct regime": {"direct": False},
    "w/o stress-aware join": {"stress_join": False},
    "w/o pheromone (no ETX)": {"channel": "none"},
    "oracle link quality": {"channel": "oracle"},
    "+ Physarum router": {"physarum": True},
}
ALL = [p for p in P if p != "NSGA-II (warm start)"]
SWEEP_P = ["Direct", "LEACH", "PEGASIS", "NSGA-II", "DQN", "HIECF", "QHP-Net"]
CENTRAL = ["NSGA-II", "Fuzzy-T2", "DQN", "HIECF", "QHP-Net"]


def jobs(exp):
    base = Config()
    if exp == "headline":
        for lossy in (True, False):
            for name in ALL:
                for sd in SEEDS:
                    yield name, name, replace(base, lossy=lossy), sd
    elif exp == "ablation":
        for lossy in (True, False):
            for v in ABLATION:
                for sd in SEEDS:
                    yield v, "QHP-Net", replace(base, lossy=lossy), sd
    elif exp == "sweep":
        for N in (50, 100, 150):
            for W in (50, 100, 150):
                for name in SWEEP_P:
                    for sd in range(15):
                        yield name, name, cfg_for(N, W), sd
    elif exp == "hetero":
        c = replace(base, het_m=0.2, het_b=0.1, het_alpha=1.0, het_beta=2.0)
        for name in ["LEACH", "PEGASIS", "NSGA-II", "Fuzzy-T2", "DQN", "HIECF", "QHP-Net"]:
            for sd in SEEDS:
                yield name, name, c, sd
    elif exp == "warm":
        for lossy in (True, False):
            for sd in SEEDS:
                yield "NSGA-II (warm start)", "NSGA-II (warm start)", replace(base, lossy=lossy), sd
    elif exp == "uplink":
        for name in CENTRAL:
            for sd in SEEDS:
                yield name, name, replace(base, charge_uplink=True), sd
    else:
        raise SystemExit(f"unknown experiment {exp}")


def key(variant, cfg, seed):
    return json.dumps([variant, asdict(cfg), seed], sort_keys=True)


def work(variant, proto, cfg, seed):
    import importlib
    mod, cls, kw = P[proto]
    kw = dict(kw)
    if variant in ABLATION and proto == "QHP-Net":
        kw.update(ABLATION[variant], name=variant)
    prot = getattr(importlib.import_module(mod), cls)(**kw)
    t = time.perf_counter()
    m = run(prot, World(cfg, seed))
    hist = m["head_dist_hist"]
    out = {k: m[k] for k in ("FND", "HND", "LND", "censored", "AUC", "rounds", "packets_bs", "delivered",
                             "injected", "taken", "lost", "PDR", "yield", "consumed", "J_per_reading",
                             "latency_ms", "setup_ms")}
    out.update(variant=variant, protocol=proto, seed=seed, N=cfg.N, W=cfg.W, lossy=cfg.lossy,
               het=cfg.het_m > 0 or cfg.het_b > 0, uplink=cfg.charge_uplink,
               energy=m["energy"], gini=gini(m["duty"]), duty_max=int(m["duty"].max()),
               sink_hops=int(hist.sum()), sink_hops_120=int(hist[120:].sum()),
               wall_s=time.perf_counter() - t,
               alive_curve=m["alive_curve"][::10].tolist(),
               energy_curve=np.round(m["energy_curve"][::10], 5).tolist(),
               key=key(variant, cfg, seed))
    return out


def main():
    exp = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    path = os.path.join(HERE, "results", f"{exp}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [j for j in jobs(exp) if key(j[0], j[2], j[3]) not in done]
    # longest jobs first so the pool drains evenly
    slow = {"QHP-Net": 3, "NSGA-II": 2}
    todo.sort(key=lambda j: -(slow.get(j[1], 1) * j[2].N))
    print(f"{exp}: {len(todo)} jobs to run, {len(done)} already done", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex, open(path, "a") as f:
        futs = [ex.submit(work, *j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  {time.time() - t0:7.0f}s", flush=True)


if __name__ == "__main__":
    main()
