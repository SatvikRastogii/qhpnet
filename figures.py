"""IEEE single-column figures from results/*.jsonl. Usage: python figures.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stats import ORDER, PROPOSED, by, load, paired

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical slots 1-3 (all-pairs safe); everything else recedes to gray with dashes
COL = {PROPOSED: "#2a78d6", "NSGA-II": "#eb6834", "DQN": "#1baf7a"}
GRAY = {"LEACH": ("#52514e", (0, (4, 2))), "PEGASIS": ("#7a7974", (0, (1, 1.5))),
        "HIECF": ("#3d3c39", (0, (6, 2, 1, 2))), "Fuzzy-T2": ("#9b9a94", (0, (3, 1, 1, 1))),
        "Direct": ("#b5b4ad", (0, (1, 3)))}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
W1 = 3.45   # IEEE column width (in)


def style(p):
    if p in COL:
        return dict(color=COL[p], lw=1.6 if p == PROPOSED else 1.2, ls="-", zorder=3 if p == PROPOSED else 2)
    c, ls = GRAY.get(p, ("#9b9a94", "-"))
    return dict(color=c, lw=0.9, ls=ls, zorder=1)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def curves(rows, key, lossy, prots):
    g = by(rows, "protocol", "lossy")
    out = {}
    for p in prots:
        R = list(g.get((p, lossy), {}).values())
        if R:
            out[p] = np.mean([r[key] for r in R], axis=0)
    return out


def fig_alive(rows):
    prots = ["Direct", "LEACH", "PEGASIS", "Fuzzy-T2", "HIECF", "DQN", "NSGA-II", PROPOSED]
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 2.05, 1.9), sharey=True)
    for ax, lossy in zip(axes, (True, False)):
        cv = curves(rows, "alive_curve", lossy, prots)
        for p, y in cv.items():
            x = np.arange(y.size) * 10
            ax.plot(x, y, label=p, **style(p))
        ax.set_xlim(0, 3500)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Round")
        ax.set_title("Lossy channel" if lossy else "Ideal channel", fontsize=7.5, color=INK)
    axes[0].set_ylabel("Alive nodes")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, handlelength=2.4,
               bbox_to_anchor=(0.5, -0.17), columnspacing=1.2)
    save(fig, "alive")


def fig_energy(rows):
    prots = ["LEACH", "PEGASIS", "HIECF", "DQN", "NSGA-II", PROPOSED]
    fig, ax = plt.subplots(figsize=(W1, 1.8))
    for p, y in curves(rows, "energy_curve", True, prots).items():
        ax.plot(np.arange(y.size) * 10, y, label=p, **style(p))
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Round")
    ax.set_ylabel("Residual network energy (J)")
    ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=2.6)
    save(fig, "energy")


def fig_strip(rows):
    g = by(rows, "protocol", "lossy")
    prots = [p for p in ORDER if (p, True) in g and p not in ("TEEN",)]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    rng = np.random.default_rng(0)
    for i, p in enumerate(prots):
        y = np.array([r["FND"] for r in g[(p, True)].values()])
        c = COL.get(p, "#7a7974")
        ax.scatter(y, i + rng.uniform(-0.18, 0.18, y.size), s=9, color=c, alpha=0.75, lw=0, zorder=2)
        ax.plot([y.mean()] * 2, [i - 0.32, i + 0.32], color=INK, lw=1.0, zorder=3)
    ax.set_yticks(range(len(prots)), prots)
    ax.set_xlabel("First node death, lossy channel (rounds)")
    ax.grid(axis="y", visible=False)
    save(fig, "fnd_strip")


def fig_paired(rows):
    g = by(rows, "protocol", "lossy")
    opps = ["LEACH", "PEGASIS", "Fuzzy-T2", "DQN", "HIECF", "NSGA-II"]
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 2.05, 1.7), sharey=True)
    for ax, lossy in zip(axes, (True, False)):
        Q = g.get((PROPOSED, lossy), {})
        for i, p in enumerate(opps):
            X = g.get((p, lossy), {})
            s = sorted(set(Q) & set(X))
            if not s:
                continue
            d = np.array([Q[k]["FND"] - X[k]["FND"] for k in s])
            r = paired([Q[k]["FND"] for k in s], [X[k]["FND"] for k in s])
            ax.scatter(d, np.full(d.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, d.size),
                       s=7, color=COL.get(p, "#7a7974"), alpha=0.7, lw=0)
            ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="|",
                        color=INK, ms=8, capsize=2, lw=1)
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.set_yticks(range(len(opps)), opps)
        ax.set_xlabel(f"Paired FND gain of {PROPOSED} (rounds)")
        ax.set_title("Lossy channel" if lossy else "Ideal channel", fontsize=7.5, color=INK)
        ax.grid(axis="y", visible=False)
    save(fig, "paired")


def fig_ablation(rows):
    g = by(rows, "variant", "lossy")
    vs = [v for v in dict.fromkeys(r["variant"] for r in rows) if v != PROPOSED]
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 2.05, 1.9), sharey=True)
    for ax, lossy in zip(axes, (True, False)):
        full = g.get((PROPOSED, lossy), {})
        for i, v in enumerate(vs):
            X = g.get((v, lossy), {})
            s = sorted(set(full) & set(X))
            if not s:
                continue
            r = paired([X[k]["FND"] for k in s], [full[k]["FND"] for k in s])
            ax.barh(i, r["diff"], height=0.6, color="#2a78d6" if r["diff"] >= 0 else "#9b9a94")
            ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="none",
                        color=INK, capsize=2, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.set_yticks(range(len(vs)), vs)
        ax.invert_yaxis()
        ax.set_xlabel("Change in FND vs full QHP-Net (rounds, 95% CI)")
        ax.set_title("Lossy channel" if lossy else "Ideal channel", fontsize=7.5, color=INK)
        ax.grid(axis="y", visible=False)
    save(fig, "ablation")


if __name__ == "__main__":
    h = load("headline")
    if h:
        fig_alive(h); fig_energy(h); fig_strip(h); fig_paired(h)
    a = load("ablation")
    if a:
        fig_ablation(a)
    print("figures written to", FIG)
