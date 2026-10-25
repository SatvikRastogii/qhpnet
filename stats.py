"""Paired statistics and LaTeX tables from results/*.jsonl. Usage: python stats.py"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from config import Config
from engine import World, tx_cost

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "paper", "tables")
ORDER = ["Direct", "LEACH", "PEGASIS", "TEEN", "NSGA-II", "Fuzzy-T2", "DQN", "HIECF", "QHP-Net"]
PROPOSED = "QHP-Net"


def load(exp):
    p = os.path.join(RES, f"{exp}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def by(rows, *keys):
    g = defaultdict(dict)
    for r in rows:
        g[tuple(r[k] for k in keys)][r["seed"]] = r
    return g


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (p.size - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(a, b):
    """a, b aligned per seed. Returns dict of mean diff, 95% CI, t, p_t, d_z, Shapiro p, Wilcoxon p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    m, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return dict(n=n, diff=m, lo=m, hi=m, t=np.inf if m else 0.0, p=0.0 if m else 1.0,
                    dz=np.inf if m else 0.0, sw=1.0, wil=0.0 if m else 1.0)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    t, p = stats.ttest_rel(a, b)
    sw = stats.shapiro(d).pvalue if n >= 3 else 1.0
    wil = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(n=n, diff=m, lo=m - half, hi=m + half, t=t, p=p, dz=m / sd, sw=sw, wil=wil)


def fnd_upper_bound(cfg, seed):
    """Bound on FND for any protocol in which every alive node transmits every round (no suppression),
    ignoring control, retries and the sink hop chain: each such transmission costs at least the
    electronics plus either the amplifier to the sink or a reception and fusion elsewhere."""
    w = World(cfg, seed)
    k = cfg.data_bits
    direct = tx_cost(cfg, k, w.D[:-1, -1])
    relay = k * cfg.E_elec + k * cfg.E_elec + k * cfg.E_DA
    c = np.minimum(direct, relay)
    return float(w.E_init.sum() / c.sum())


def fmt_p(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def ms(x, d=0):
    x = np.asarray(x, float)
    return f"{x.mean():,.{d}f} $\\pm$ {x.std(ddof=1):,.{d}f}"


def write(name, text):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def headline():
    rows = load("headline")
    if not rows:
        return
    global s_cap
    s_cap = Config().max_rounds
    g = by(rows, "protocol", "lossy")
    out = []
    for lossy in (True, False):
        print(f"\n=== headline, {'lossy' if lossy else 'ideal'} channel ===")
        lines = []
        for p in ORDER:
            R = list(g.get((p, lossy), {}).values())
            if not R:
                continue
            f = lambda k: np.array([r[k] for r in R], float)
            cens = sum(r["censored"] for r in R)
            e = lambda c: np.array([r["energy"][c] / r["consumed"] for r in R])
            lnd = (f"$>${s_cap:,}$^c$" if cens == len(R) else ms(f("LND")) + (f"$^{{c{cens}}}$" if cens else ""))
            print(f"{p:9s} n={len(R):2d} FND {f('FND').mean():7.0f}±{f('FND').std(ddof=1):4.0f} "
                  f"HND {f('HND').mean():6.0f} LND {f('LND').mean():6.0f} cens {cens:2d} AUC {f('AUC').mean()/1e3:6.1f}k "
                  f"yield {f('yield').mean():.3f} PDR {f('PDR').mean():.3f} J/rd {1e3*f('J_per_reading').mean():.3f}mJ "
                  f"ctrl {100*e('ctrl').mean():4.1f}% retry {100*e('retry').mean():4.1f}% gini {f('gini').mean():.2f} "
                  f"setup {f('setup_ms').mean():6.2f}ms lat {f('latency_ms').mean():5.0f}ms")
            name = f"\\textbf{{{p}}}" if p == PROPOSED else p
            lines.append(f"{name} & {ms(f('FND'))} & {ms(f('HND'))} & {lnd} & {f('AUC').mean()/1e3:.1f} & "
                         f"{f('yield').mean():.3f} & {1e3*f('J_per_reading').mean():.2f} \\\\")
        out.append((lossy, lines))
    for lossy, lines in out:
        write(f"lifetime_{'lossy' if lossy else 'ideal'}.tex", "\n".join(lines) + "\n")

    # paired t-tests: proposed against every opponent, Holm across opponents per metric and channel
    for lossy in (True, False):
        Q = g.get((PROPOSED, lossy), {})
        lines = []
        print(f"\n--- paired tests vs {PROPOSED}, {'lossy' if lossy else 'ideal'} ---")
        for metric in ("FND", "HND", "AUC", "yield"):
            opps = [p for p in ORDER if p != PROPOSED and (p, lossy) in g]
            res = []
            for p in opps:
                X = g[(p, lossy)]
                s = sorted(set(Q) & set(X))
                res.append(paired([Q[i][metric] for i in s], [X[i][metric] for i in s]))
            adj = holm([r["p"] for r in res])
            for p, r, pa in zip(opps, res, adj):
                r["p_holm"] = pa
                print(f"{metric:5s} vs {p:9s} n={r['n']} diff {r['diff']:9.3f} CI [{r['lo']:9.3f},{r['hi']:9.3f}] "
                      f"t={r['t']:8.2f} p={r['p']:.2e} pHolm={pa:.2e} dz={r['dz']:6.2f} SW={r['sw']:.3f} Wil={r['wil']:.2e}")
                if metric in ("FND", "AUC") and p in ("NSGA-II", "DQN", "HIECF", "PEGASIS", "Fuzzy-T2", "LEACH"):
                    sc = 1e-3 if metric == "AUC" else 1
                    unit = " ($\\times10^3$)" if sc != 1 else ""
                    lines.append(f"{metric}{unit} & {p} & {r['diff']*sc:+,.1f} & "
                                 f"[{r['lo']*sc:+,.1f}, {r['hi']*sc:+,.1f}] & {r['t']:.2f} & {fmt_p(pa)} & "
                                 f"{r['dz']:.2f} & {fmt_p(r['wil'])} \\\\")
        write(f"ttest_{'lossy' if lossy else 'ideal'}.tex", "\n".join(lines) + "\n")

    # compact table: FND and AUC differences on both channels
    opps = ["LEACH", "PEGASIS", "TEEN", "NSGA-II", "Fuzzy-T2", "DQN", "HIECF"]
    cols = (("FND", True, 1), ("FND", False, 1), ("AUC", True, 1e-3), ("AUC", False, 1e-3))
    res = {}
    for metric, lossy, sc in cols:                     # Holm per column, across opponents
        rr = []
        for p in opps:
            Q, X = g[(PROPOSED, lossy)], g[(p, lossy)]
            sd = sorted(set(Q) & set(X))
            rr.append(paired([Q[i][metric] for i in sd], [X[i][metric] for i in sd]))
        for p, r, pt, pw in zip(opps, rr, holm([r["p"] for r in rr]), holm([r["wil"] for r in rr])):
            r["ph"], r["pw"] = pt, pw
            res[(p, metric, lossy)] = r
    lines = []
    for p in opps:
        cells = []
        for metric, lossy, sc in cols:
            r = res[(p, metric, lossy)]
            mark = "" if max(r["ph"], r["pw"]) < 0.05 else "$^{ns}$"
            cells.append(f"{r['diff']*sc:+,.0f} [{r['lo']*sc:+,.0f}, {r['hi']*sc:+,.0f}]{mark} & {r['dz']:.1f}"
                         if sc == 1 else f"{r['diff']*sc:+,.1f}{mark} & {r['dz']:.1f}")
        lines.append(f"{p} & " + " & ".join(cells) + " \\\\")
    write("ttest_combined.tex", "\n".join(lines) + "\n")
    sig = [max(r["ph"], r["pw"]) for r in res.values() if max(r["ph"], r["pw"]) < 0.05]
    print(f"combined table: max Holm p among significant cells {max(sig):.2e}; "
          f"non-significant: {[k for k, r in res.items() if max(r['ph'], r['pw']) >= 0.05]}")

    # physical bound
    ub = np.array([fnd_upper_bound(Config(), s) for s in range(30)])
    Q = g.get((PROPOSED, True), {})
    if Q:
        f = np.array([Q[s]["FND"] for s in sorted(Q)])
        print(f"\nFND upper bound (lossy cfg): {ub.mean():.0f} ± {ub.std(ddof=1):.0f}; "
              f"{PROPOSED} reaches {100*np.mean(f/ub[sorted(Q)]):.1f}% of it")
        write("bound.tex", f"\\newcommand{{\\fndbound}}{{{ub.mean():,.0f}}}\n"
                           f"\\newcommand{{\\fndfrac}}{{{100*np.mean(f/ub[sorted(Q)]):.1f}}}\n")


def ablation():
    rows = load("ablation")
    if not rows:
        return
    g = by(rows, "variant", "lossy")
    for lossy in (True, False):
        full = g.get(("QHP-Net", lossy), {})
        if not full:
            continue
        print(f"\n=== ablation, {'lossy' if lossy else 'ideal'} ===")
        vs = [v for (v, l) in g if l == lossy and v != "QHP-Net"]
        res = {}
        for metric in ("FND", "AUC"):
            rr = []
            for v in vs:
                X = g[(v, lossy)]
                s = sorted(set(full) & set(X))
                rr.append(paired([X[i][metric] for i in s], [full[i][metric] for i in s]))
            for v, r, pa in zip(vs, rr, holm([r["p"] for r in rr])):
                r["p_holm"] = pa
                res[(v, metric)] = r
        lines = []
        F = lambda v, k: np.array([r[k] for r in g[(v, lossy)].values()], float)
        print(f"{'QHP-Net':26s} FND {F('QHP-Net','FND').mean():7.0f} AUC {F('QHP-Net','AUC').mean()/1e3:6.1f}k")
        lines.append(f"\\textbf{{QHP-Net (full)}} & {F('QHP-Net','FND').mean():,.0f} & -- & "
                     f"{F('QHP-Net','AUC').mean()/1e3:.1f} & -- & {F('QHP-Net','yield').mean():.3f} \\\\")
        for v in vs:
            a, b = res[(v, "FND")], res[(v, "AUC")]
            print(f"{v:26s} FND {F(v,'FND').mean():7.0f} dFND {a['diff']:+7.1f} [{a['lo']:+.1f},{a['hi']:+.1f}] "
                  f"pHolm {a['p_holm']:.2e} | dAUC {b['diff']/1e3:+6.2f}k pHolm {b['p_holm']:.2e} yield {F(v,'yield').mean():.3f}")
            lines.append(f"{v} & {F(v,'FND').mean():,.0f} & {a['diff']:+,.0f} ({fmt_p(a['p_holm'])}) & "
                         f"{F(v,'AUC').mean()/1e3:.1f} & {b['diff']/1e3:+.1f} ({fmt_p(b['p_holm'])}) & "
                         f"{F(v,'yield').mean():.3f} \\\\")
        write(f"ablation_{'lossy' if lossy else 'ideal'}.tex", "\n".join(lines) + "\n")


def simple(exp, title):
    rows = load(exp)
    if not rows:
        return
    g = by(rows, "protocol")
    print(f"\n=== {title} ===")
    lines = []
    Q = g.get((PROPOSED,), {})
    opps = [p for p in ORDER if (p,) in g and p != PROPOSED]
    tests = []
    for p in opps:
        s = sorted(set(Q) & set(g[(p,)]))
        tests.append(paired([Q[i]["FND"] for i in s], [g[(p,)][i]["FND"] for i in s]) if Q else None)
    adj = holm([t["p"] for t in tests]) if Q else [None] * len(opps)
    for p in ORDER:
        if (p,) not in g:
            continue
        R = list(g[(p,)].values())
        f = lambda k: np.array([r[k] for r in R], float)
        extra = ""
        if p != PROPOSED and Q:
            t = tests[opps.index(p)]
            extra = f"  dFND(QHP-x) {t['diff']:+.0f} pHolm {adj[opps.index(p)]:.2e}"
        print(f"{p:9s} FND {f('FND').mean():7.0f}±{f('FND').std(ddof=1):4.0f} HND {f('HND').mean():6.0f} "
              f"AUC {f('AUC').mean()/1e3:6.1f}k yield {f('yield').mean():.3f}{extra}")
        lines.append(f"{p} & {ms(f('FND'))} & {ms(f('HND'))} & {f('AUC').mean()/1e3:.1f} & {f('yield').mean():.3f} \\\\")
    write(f"{exp}.tex", "\n".join(lines) + "\n")


def warm():
    W = by(load("warm"), "protocol", "lossy")
    H = by(load("headline"), "protocol", "lossy")
    if not W:
        return
    print("\n=== NSGA-II warm-start fairness check ===")
    for lossy in (True, False):
        X, N0, Q = W[("NSGA-II (warm start)", lossy)], H[("NSGA-II", lossy)], H[(PROPOSED, lossy)]
        s = sorted(set(X) & set(Q))
        a = paired([X[i]["FND"] for i in s], [N0[i]["FND"] for i in s])
        b = paired([Q[i]["FND"] for i in s], [X[i]["FND"] for i in s])
        print(f"{'lossy' if lossy else 'ideal'}: warm FND {np.mean([X[i]['FND'] for i in s]):.0f}, "
              f"warm-cold {a['diff']:+.1f} p={a['p']:.3f}; QHP-warm {b['diff']:+.1f} [{b['lo']:+.1f},{b['hi']:+.1f}] p={b['p']:.2e} dz={b['dz']:.1f}")


def sweep():
    rows = load("sweep")
    if not rows:
        return
    g = by(rows, "protocol", "N", "W")
    prots = [p for p in ORDER if any(k[0] == p for k in g)]
    cells = [(N, W) for W in (50.0, 100.0, 150.0) for N in (50, 100, 150)]
    print("\n=== sweep FND (15 paired runs per cell) ===")
    print(" " * 9 + " ".join(f"{int(W)}x{int(W)}/N{N:<4d}" for N, W in cells))
    lines = []
    for p in prots:
        vals = [np.mean([r["FND"] for r in g[(p, N, W)].values()]) if (p, N, W) in g else np.nan for N, W in cells]
        print(f"{p:9s} " + " ".join(f"{v:13.0f}" for v in vals))
        name = f"\\textbf{{{p}}}" if p == PROPOSED else p
        lines.append(name + " & " + " & ".join(f"{v:.0f}" for v in vals) + " \\\\")
    # proposed vs the best other protocol in each cell
    print("\nQHP-Net vs best other protocol per cell (paired t, FND):")
    for N, W in cells:
        Q = g.get((PROPOSED, N, W))
        if not Q:
            continue
        others = [(np.mean([r["FND"] for r in g[(p, N, W)].values()]), p) for p in prots
                  if p != PROPOSED and (p, N, W) in g]
        _, best = max(others)
        X = g[(best, N, W)]
        s = sorted(set(Q) & set(X))
        r = paired([Q[i]["FND"] for i in s], [X[i]["FND"] for i in s])
        print(f"  {int(W)}x{int(W)} N={N}: best other {best:8s} diff {r['diff']:+7.0f} p={r['p']:.2e}")
    write("sweep.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    headline()
    ablation()
    warm()
    simple("hetero", "three-level heterogeneity (m=0.2, b=0.1), lossy")
    simple("uplink", "uplink charged for every centralised protocol, lossy")
    sweep()
