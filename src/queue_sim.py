"""Fast simulator of a 4-station stochastic queueing network.

Network: Poisson arrivals (rate 0.1) -> WS1 -> WS2 -> fork: WS3 with probability p, else WS4.
Every station is a single-server first-come-first-served queue, so departure times follow exactly from
    D_n = max(A_n, D_(n-1)) + S_n  =  C_n + max_{k<=n} (A_k - C_(k-1)),   C_n = S_1 + ... + S_n,
computed with cumulative sums instead of an event list (hundreds of times faster than a
process-based simulator).

Service times:  WS1 ~ Uniform(m1-1, m1+1), WS2 ~ Triangular(c-2, c, c+2),
                WS3 ~ Uniform(m3-1, m3+1), WS4 ~ a + b * Beta(2, 2).

Each run starts empty, discards customers arriving during a warm-up period, counts customers arriving
in the following window, and follows each counted customer until it leaves (no end-of-run truncation).
Outputs per run: mean wait at each station, mean time in system, their 90th percentiles, and utilisation.
"""
import numpy as np

LAM = 0.1
OUTPUTS = ['ws1_wait', 'ws2_wait', 'ws3_wait', 'ws4_wait', 'los']


def fifo(arrivals, service):
    """Exact departures and waits of a single-server FIFO queue (arrivals sorted)."""
    c = np.cumsum(service)
    dep = c + np.maximum.accumulate(arrivals - (c - service))
    return dep, np.maximum(dep - service - arrivals, 0.0)


def one_run(m1, m3, c, a, b, p, rng, warmup=2000.0, window=8000.0):
    t_end = warmup + window
    n = int(t_end * LAM * 1.3 + 100)
    arr = np.cumsum(rng.exponential(1 / LAM, n))
    while arr[-1] < t_end:
        arr = np.concatenate([arr, arr[-1] + np.cumsum(rng.exponential(1 / LAM, n))])
    arr = arr[arr < t_end]
    k = len(arr)
    s1 = rng.uniform(m1 - 1, m1 + 1, k)
    s2 = rng.triangular(c - 2, c, c + 2, k)
    u = rng.uniform(0, 1, k)
    s3 = rng.uniform(m3 - 1, m3 + 1, k)
    s4 = a + b * rng.beta(2, 2, k)
    d1, w1 = fifo(arr, s1)
    d2, w2 = fifo(d1, s2)
    to3 = u < p
    d3, w3 = fifo(d2[to3], s3[to3])
    d4, w4 = fifo(d2[~to3], s4[~to3])
    exit_t = np.empty(k); exit_t[to3] = d3; exit_t[~to3] = d4
    w34 = np.empty(k); w34[to3] = w3; w34[~to3] = w4
    keep = arr >= warmup
    vals = {'ws1_wait': w1[keep], 'ws2_wait': w2[keep], 'ws3_wait': w34[keep & to3],
            'ws4_wait': w34[keep & ~to3], 'los': (exit_t - arr)[keep]}
    res = {}
    for o, v in vals.items():
        res[o] = v.mean() if len(v) else np.nan
        res[o + '_p90'] = np.percentile(v, 90) if len(v) else np.nan
    busy = [s1[keep].sum(), s2[keep].sum(), s3[to3 & keep].sum(), s4[~to3 & keep].sum()]
    for i, bt in enumerate(busy):
        res[f'rho{i + 1}'] = bt / window
    return res


def simulate(x, runs, seed=0, **kw):
    """x = (m1, m3, c, a, b, p). Returns dict output -> array of per-run values."""
    out = [one_run(*x, rng=np.random.default_rng(np.random.SeedSequence([seed, r])), **kw) for r in range(runs)]
    return {k: np.array([o[k] for o in out]) for k in out[0]}


def exact_ws1_wait(m1):
    """Pollaczek-Khinchine mean wait at WS1 (an M/G/1 queue with Uniform(m1-1, m1+1) service)."""
    return LAM * (m1 ** 2 + 1 / 3) / (2 * (1 - LAM * m1))
