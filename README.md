# QHP-Net: simulation code and paper

Paired Monte Carlo study of QHP-Net (quorum-sensing candidacy, Hippopotamus-optimised clustering, pheromone-learned routing) against eight WSN clustering protocols in one engine.

## Layout
| Path | What |
|---|---|
| `config.py` | the single frozen parameter set (Table II of the paper) |
| `engine.py` | world generation, radio/channel model, ARQ, fusion, control pricing, metrics; energy conservation asserted every run |
| `protocols/` | `baseline_simple.py` (Direct, LEACH), `classic.py` (PEGASIS, TEEN), `nsga2.py`, `fuzzy.py` (Fuzzy-T2, HIECF), `dqn.py` (+ trainer, `weights/dqn.npz`), `qhp.py` (proposed) |
| `montecarlo.py` | experiments: `headline`, `ablation`, `hetero`, `uplink`, `sweep` → `results/*.jsonl` (resumable) |
| `stats.py` | paired t-tests, Holm, Cohen's d_z, Shapiro–Wilk, Wilcoxon, LaTeX tables → `paper/tables/` |
| `figures.py` | figures → `paper/figures/` |
| `test_engine.py`, `test_qhp.py` | gates: radio by hand, PER curve, pairing, read-only state, LEACH epoch, stress model vs engine |
| `paper/` | IEEE conference LaTeX (`main.tex`, `sections/`, `refs.bib`) |

## Reproduce
```
python test_engine.py && python test_qhp.py
python -m protocols.dqn 40          # optional: retrain DQN (weights are included)
for e in headline ablation hetero uplink sweep; do python montecarlo.py $e 15; done
python stats.py > results/stats.txt
python figures.py
```
Test seeds are 0–29 (sweep: 0–14 per cell). QHP-Net parameters were fixed on seeds 500–505; DQN trained on 100000+.

## Build the paper
No TeX on this machine: upload `paper/` to Overleaf (pdfLaTeX, IEEEtran is built in) or run `latexmk -pdf main.tex` inside `paper/`.
