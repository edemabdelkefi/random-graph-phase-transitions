from pathlib import Path
from collections import Counter
from itertools import combinations
import csv
import hashlib
import json
import math
import platform
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "results"
SEED = 20260930

PROTOCOL = {'seed': 20260930, 'phase': {'n': [250, 1000, 4000], 'c': [0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0], 'replicates': 200}, 'scaling': {'n': [250, 500, 1000, 2000, 4000, 8000], 'c': 1.0, 'replicates': 250, 'bootstrap_replicates': 2000}, 'connectivity': {'n': 1000, 's': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], 'replicates': 400}, 'cycles': {'n': 500, 'c': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 'replicates': 300, 'k': [3, 4]}, 'reporting': {'giant_component_mae': 'all phase-grid c >= 1.4 at n=4000', 'scaling_fit': 'OLS log(mean largest component) against log(n), all six sizes', 'scaling_ci': 'within-size nonparametric bootstrap, percentile 95%, 2000 draws', 'connectivity_rmse': 'all eight s values', 'cycle_reference': 'exact finite-n expected counts, not just asymptotic counts', 'geometry_error': 'maximum relative length error at 512 intervals, all 1000 pairs'}}

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def er_edges(n, p, rng):
    """Exact G(n,p): binomial edge count, then a uniform subset of all pairs.

    Every edge set S has probability p**|S| * (1-p)**(N-|S|).
    No fixed-edge-count approximation is used.
    """
    possible = n * (n - 1) // 2
    count = int(rng.binomial(possible, p))
    flat = rng.choice(possible, size=count, replace=False)
    starts = np.arange(n, dtype=np.int64)
    starts = starts * (2 * n - starts - 1) // 2
    u = np.searchsorted(starts, flat, side='right') - 1
    v = u + 1 + flat - starts[u]
    return u, v


def graph_stats(n, u, v):
    parent, sizes = list(range(n)), [1] * n
    largest, components = 1, n
    for a, b in zip(u.tolist(), v.tolist()):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]
            largest = max(largest, sizes[a])
            components -= 1
    degrees = np.bincount(np.concatenate([u, v]).astype(np.intp), minlength=n)
    return largest, components, int(np.sum(degrees == 0))


def short_cycles(n, u, v):
    adj = [set() for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].add(b)
        adj[b].add(a)
    triangles = sum(sum(w > b for w in adj[a] & adj[b]) for a, b in zip(u.tolist(), v.tolist()))
    common = Counter()
    for neighbors in adj:
        for a, b in combinations(sorted(neighbors), 2):
            common[a * n + b] += 1
    squares = sum(x * (x - 1) // 2 for x in common.values()) // 2
    return triangles, squares


def alpha(c):
    if c <= 1:
        return 0.
    lo, hi = 1e-12, 1.
    for _ in range(70):
        mid = (lo + hi) / 2
        if 1 - mid - math.exp(-c * mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z / (1 + z*z/n) * math.sqrt(p*(1-p)/n + z*z / (4*n*n))
    return max(0., center-half), min(1., center+half)


def mean_se(x):
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


def validate():
    cases = [
        (3, [(0, 1), (1, 2), (0, 2)], (1, 0)),
        (4, list(combinations(range(4), 2)), (4, 3)),
        (4, [(0, 1), (1, 2), (2, 3), (0, 3)], (0, 1)),
        (4, [(0, 1), (1, 2), (2, 3)], (0, 0)),
    ]
    for n, edges, expected in cases:
        u, v = np.array(edges, dtype=np.int64).T
        assert short_cycles(n, u, v) == expected
        assert graph_stats(n, u, v)[:2] == (n, 1)
    empty = np.array([], dtype=np.int64)
    assert graph_stats(4, empty, empty) == (1, 4, 4)
    rng = np.random.default_rng(100)
    u, v = er_edges(20, 1., rng)
    assert len(u) == 190 and np.all(u < v) and len(set(zip(u, v))) == 190
    u, v = er_edges(20, 0., rng)
    assert len(u) == 0
    counts, isolated = [], []
    for _ in range(2000):
        u, v = er_edges(20, .2, rng)
        counts.append(len(u))
        isolated.append(graph_stats(20, u, v)[2])
    expected_m, sd_m = 190 * .2, math.sqrt(190 * .2 * .8 / 2000)
    assert abs(np.mean(counts) - expected_m) < 6 * sd_m
    expected_i = 20 * .8**19
    assert abs(np.mean(isolated) - expected_i) < 6 * np.std(isolated, ddof=1) / math.sqrt(2000)
    print('Validation: graph edge cases, known cycle counts, sampler moments passed.', flush=True)


def graph_experiments(streams):
    raw, phase_rows = [], []
    rng = streams[0]
    spec = PROTOCOL['phase']
    for n in spec['n']:
        for c in spec['c']:
            fractions, isolates = [], []
            for rep in range(spec['replicates']):
                u, v = er_edges(n, c/n, rng)
                largest, components, isolated = graph_stats(n, u, v)
                fractions.append(largest/n)
                isolates.append(isolated)
                raw.append(dict(experiment='phase', n=n, parameter=c, replicate=rep, edges=len(u), largest=largest, components=components, isolated=isolated))
            m, se = mean_se(fractions)
            phase_rows.append(dict(n=n, c=c, mean_fraction=m, standard_error=se, theoretical_fraction=alpha(c), mean_isolated=float(np.mean(isolates)), theoretical_mean_isolated=n*(1-c/n)**(n-1)))
        print(f'Phase sweep: n={n} completed.', flush=True)
    write_csv('phase.csv', phase_rows)
    chosen = [r for r in phase_rows if r['n'] == 4000 and r['c'] >= 1.4]
    giant_mae = float(np.mean([abs(r['mean_fraction'] - r['theoretical_fraction']) for r in chosen]))

    spec, rng = PROTOCOL['scaling'], streams[1]
    scaling_samples, scaling_rows = [], []
    for n in spec['n']:
        samples = []
        for rep in range(spec['replicates']):
            u, v = er_edges(n, 1/n, rng)
            largest, components, isolated = graph_stats(n, u, v)
            samples.append(largest)
            raw.append(dict(experiment='scaling', n=n, parameter=1., replicate=rep, edges=len(u), largest=largest, components=components, isolated=isolated))
        scaling_samples.append(np.array(samples))
        m, se = mean_se(samples)
        scaling_rows.append(dict(n=n, mean_largest=m, standard_error=se))
    x = np.log(spec['n'])
    beta, intercept = np.polyfit(x, np.log([r['mean_largest'] for r in scaling_rows]), 1)
    boot = []
    for _ in range(spec['bootstrap_replicates']):
        means = [rng.choice(s, size=len(s), replace=True).mean() for s in scaling_samples]
        boot.append(float(np.polyfit(x, np.log(means), 1)[0]))
    beta_ci = np.quantile(boot, [.025, .975]).tolist()
    write_csv('scaling.csv', scaling_rows)
    write_csv('scaling_bootstrap.csv', [dict(replicate=i, beta=v) for i, v in enumerate(boot)])
    print(f'Critical-size scaling: exponent={beta:.4f}, CI={beta_ci}.', flush=True)

    spec, rng, conn_rows = PROTOCOL['connectivity'], streams[2], []
    for s in spec['s']:
        n, successes = spec['n'], 0
        p = (math.log(n) + s) / n
        for rep in range(spec['replicates']):
            u, v = er_edges(n, p, rng)
            largest, components, isolated = graph_stats(n, u, v)
            successes += components == 1
            raw.append(dict(experiment='connectivity', n=n, parameter=s, replicate=rep, edges=len(u), largest=largest, components=components, isolated=isolated))
        low, high = wilson(successes, spec['replicates'])
        conn_rows.append(dict(n=n, s=s, successes=successes, trials=spec['replicates'], probability=successes/spec['replicates'], ci95_low=low, ci95_high=high, asymptotic_probability=math.exp(-math.exp(-s))))
    conn_rmse = math.sqrt(np.mean([(r['probability'] - r['asymptotic_probability'])**2 for r in conn_rows]))
    write_csv('connectivity.csv', conn_rows)
    print(f'Connectivity sweep: RMSE={conn_rmse:.5f}.', flush=True)

    spec, rng, cycle_rows, cycle_raw = PROTOCOL['cycles'], streams[3], [], []
    for c in spec['c']:
        n, samples = spec['n'], {3: [], 4: []}
        for rep in range(spec['replicates']):
            u, v = er_edges(n, c/n, rng)
            largest, components, isolated = graph_stats(n, u, v)
            counts = short_cycles(n, u, v)
            raw.append(dict(experiment='cycles', n=n, parameter=c, replicate=rep, edges=len(u), largest=largest, components=components, isolated=isolated))
            cycle_raw.append(dict(n=n, c=c, replicate=rep, triangles=counts[0], squares=counts[1]))
            for k, count in zip([3, 4], counts):
                samples[k].append(count)
        for k in [3, 4]:
            m, se = mean_se(samples[k])
            exact = math.prod(n-i for i in range(k)) * (c/n)**k / (2*k)
            cycle_rows.append(dict(n=n, c=c, k=k, mean_count=m, standard_error=se, exact_expectation=exact, asymptotic_expectation=c**k/(2*k)))
    write_csv('cycles.csv', cycle_rows)
    write_csv('cycles_raw.csv', cycle_raw)
    write_csv('graphs_raw.csv', raw)
    print(f'Graph experiments completed: {len(raw):,} independent realizations.', flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for n in PROTOCOL['phase']['n']:
        rows = [r for r in phase_rows if r['n'] == n]
        ax[0].errorbar([r['c'] for r in rows], [r['mean_fraction'] for r in rows], yerr=[1.96*r['standard_error'] for r in rows], label=f'n={n}', fmt='.-', capsize=2)
    cs = np.linspace(.4, 3, 200)
    ax[0].plot(cs, [alpha(c) for c in cs], 'k--', label='Asymptotic fixed point')
    ax[0].set(xlabel='c = np', ylabel='Mean largest component / n', title='Giant-component transition')
    ax[0].legend(fontsize=8)
    ns = np.array(PROTOCOL['scaling']['n'])
    ax[1].loglog(ns, [r['mean_largest'] for r in scaling_rows], 'o', label='Monte Carlo mean')
    ax[1].loglog(ns, np.exp(intercept)*ns**beta, '--', label=f'OLS exponent {beta:.3f}')
    ax[1].set(xlabel='n', ylabel='Mean largest component', title='Critical scaling: p = 1/n')
    ax[1].legend(fontsize=8)
    ss = np.array([r['s'] for r in conn_rows])
    ps = np.array([r['probability'] for r in conn_rows])
    ax[2].errorbar(ss, ps, yerr=[ps - np.array([r['ci95_low'] for r in conn_rows]), np.array([r['ci95_high'] for r in conn_rows])-ps], fmt='o', capsize=3, label='400 trials / point; Wilson 95%')
    dense = np.linspace(-3, 4, 200)
    ax[2].plot(dense, np.exp(-np.exp(-dense)), 'k--', label='Asymptotic prediction')
    ax[2].set(xlabel='s, where p=(log n+s)/n', ylabel='P(connected)', title='Connectivity transition: n = 1000')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'phase_transitions.png', dpi=180)
    fig.savefig(OUT / 'phase_transitions.pdf')
    plt.close(fig)
    return dict(graph_realizations=len(raw), max_vertices=max(PROTOCOL['scaling']['n']), giant_fraction_mae=giant_mae, critical_exponent=float(beta), critical_exponent_ci95=beta_ci, connectivity_rmse=float(conn_rmse))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    validate()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(4)]
    metrics = graph_experiments(streams)
    results = {"protocol": PROTOCOL, "graphs": metrics, "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    results["csv_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.csv"))}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return results


if __name__ == "__main__":
    run()
