"""Check the simulator against queueing theory and plot the result.

1. WS1 mean wait against the exact Pollaczek-Khinchine (M/G/1) formula over its input range.
2. Simulated utilisation of every station against rho = lambda * E[S].
Usage: python scripts/validate_simulator.py   (writes results/simulator_validation.png)
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from queue_sim import simulate, exact_ws1_wait, LAM

m1s = np.arange(2, 7.01, 0.5); sim, se = [], []
t = time.time()
for j, m1 in enumerate(m1s):
    w = simulate((m1, 4.5, 5.0, 3, 2, 0.3), runs=1000, seed=j)['ws1_wait']
    sim.append(w.mean()); se.append(w.std(ddof=1) / np.sqrt(len(w)))
print(f'{len(m1s) * 1000} runs in {time.time() - t:.1f} s')
for m1, s in zip(m1s, sim):
    print(f'm1 = {m1:.1f}: simulated {s:.3f}, exact {exact_ws1_wait(m1):.3f} ({100 * (s - exact_ws1_wait(m1)) / exact_ws1_wait(m1):+.2f}%)')

pts = {'low load': (2, 2, 2.5, 1, 1, .1), 'medium': (4, 4, 5, 3, 2, .3), 'high load': (7, 7, 7.5, 6, 3, .5)}
for k, (name, x) in enumerate(pts.items()):
    m1, m3, c, a, b, p = x
    theory = [LAM * m1, LAM * c, LAM * p * m3, LAM * (1 - p) * (a + b / 2)]
    r = simulate(x, runs=200, seed=100 + k)
    print(name, 'utilisation sim/theory:', ' '.join(f'{r[f"rho{i + 1}"].mean():.3f}/{theory[i]:.3f}' for i in range(4)))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6, 3.6))
mm = np.linspace(2, 7, 200)
ax.plot(mm, exact_ws1_wait(mm), color='#0b0b0b', lw=2, label='Exact M/G/1 (Pollaczek-Khinchine)')
ax.errorbar(m1s, sim, yerr=1.96 * np.array(se), fmt='o', color='#2a78d6', ms=5, capsize=3, label='Simulated (1,000 runs, 95% CI)')
ax.set_xlabel('m1 (mean service time at WS1)'); ax.set_ylabel('Mean wait at WS1'); ax.legend(frameon=False)
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
fig.savefig(os.path.join(ROOT, 'results', 'simulator_validation.png'), dpi=200, bbox_inches='tight')
