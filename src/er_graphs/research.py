"""Critical-window susceptibility and clustered systemic-loss experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from .experiment import HERE, er_edges
from .clearing import financial_network, clear_payments, default_sensitivity

PROTOCOL = {
    'seed': 20261001,
    'critical': {'n': [250, 1000, 4000], 'lambda': [-2., -1., 0., 1., 2.], 'replicates': 200},
    'financial': {'banks': 80, 'mean_degrees': [1., 2., 4., 8.], 'networks_per_degree': 32,
                  'shocks_per_network': 8, 'interbank_share': .65, 'relative_equity_buffer': .08,
                  'asset_loss_severities': [0., .05, .1, .2, .35, .5],
                  'asset_shock_correlation': .35, 'asset_shock_dispersion': .12,
                  'expected_shortfall_probability': .95, 'bootstrap_draws': 2000},
    'uncertainty': 'critical means use independent graphs; financial bootstrap resamples whole networks with all shocks',
    'scope': 'synthetic symmetric exposures, heterogeneous debt sizes, proportional payment priority, one outside creditor',
}


def write_csv(path, rows):
    import csv
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def component_moments(n, u, v):
    parent = np.arange(n)
    size = np.ones(n, dtype=int)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in zip(u, v):
        a, b = root(a), root(b)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
    sizes = np.sort(size[parent == np.arange(n)])[::-1]
    susceptibility = float(np.sum(sizes.astype(float)**2)/n)
    remainder = n-int(sizes[0])
    finite_susceptibility = float(np.sum(sizes[1:].astype(float)**2)/remainder) if remainder else 0.
    return sizes, susceptibility, finite_susceptibility


def expected_shortfall(losses, probability=.95):
    """Average of exactly ceil((1-q)*N) largest empirical losses."""
    losses = np.asarray(losses, dtype=float)
    if not 0 < probability < 1 or not len(losses) or not np.isfinite(losses).all():
        raise ValueError('A finite nonempty loss sample and a probability in (0, 1) are required.')
    count = max(1, int(np.ceil((1-probability)*len(losses)-1e-12)))
    return float(np.sort(losses)[-count:].mean()), count


def critical_experiment(out, spec, rng):
    raw, summary = [], []
    for n in spec['n']:
        for lam in spec['lambda']:
            probability = 1/n+lam*n**(-4/3)
            group = []
            for replicate in range(spec['replicates']):
                u, v = er_edges(n, probability, rng)
                sizes, susceptibility, finite = component_moments(n, u, v)
                row = {'n': n, 'lambda': lam, 'replicate': replicate, 'p': probability,
                       'largest': int(sizes[0]), 'second_largest': int(sizes[1]) if len(sizes) > 1 else 0,
                       'scaled_largest': float(sizes[0]/n**(2/3)),
                       'susceptibility': susceptibility, 'scaled_susceptibility': susceptibility/n**(1/3),
                       'finite_susceptibility': finite, 'components': len(sizes),
                       'cycle_surplus': len(u)-n+len(sizes)}
                raw.append(row)
                group.append(row)
            for observable in ['scaled_largest', 'scaled_susceptibility', 'finite_susceptibility', 'cycle_surplus']:
                values = np.array([row[observable] for row in group])
                summary.append({'n': n, 'lambda': lam, 'observable': observable,
                                'replicates': len(values), 'mean': float(values.mean()),
                                'standard_error': float(values.std(ddof=1)/np.sqrt(len(values))),
                                'q05': float(np.quantile(values, .05)), 'q95': float(np.quantile(values, .95))})
        print(f'Critical window n={n}: completed.', flush=True)
    write_csv(out/'critical_window_raw.csv', raw)
    write_csv(out/'critical_window_summary.csv', summary)
    return summary, len(raw)


def cluster_bootstrap(losses, defaults, rng, draws, probability):
    # Rows are independent networks; columns are conditionally related shocks.
    indices = rng.integers(0, len(losses), size=(draws, len(losses)))
    resampled_loss = losses[indices].reshape(draws, -1)
    count = expected_shortfall(losses.ravel(), probability)[1]
    tail = np.partition(resampled_loss, resampled_loss.shape[1]-count, axis=1)[:, -count:]
    es_interval = np.quantile(tail.mean(axis=1), [.025, .975])
    default_interval = np.quantile(defaults[indices].mean(axis=(1, 2)), [.025, .975])
    return es_interval.tolist(), default_interval.tolist()


def systemic_experiment(out, spec, streams):
    raw, summary, sensitivities = [], [], []
    maximum_certificate, maximum_disagreement, maximum_iterations = 0., 0., 0
    rng_network, rng_shock, rng_bootstrap = [np.random.default_rng(seed) for seed in streams]
    # Common standardized shocks allow paired severity comparisons on each network.
    shocks = rng_shock.normal(size=(spec['networks_per_degree'], spec['shocks_per_network'], spec['banks']+1))
    for degree in spec['mean_degrees']:
        degree_records = []
        for network in range(spec['networks_per_degree']):
            debt, relative, baseline_assets, edge_count = financial_network(
                spec['banks'], degree, spec['interbank_share'], spec['relative_equity_buffer'], rng_network)
            outside_share = 1-relative.sum(axis=1)
            outside_debt = float(outside_share @ debt)
            standardized = shocks[network]
            rho = spec['asset_shock_correlation']
            factors = np.sqrt(rho)*standardized[:, :1]+np.sqrt(1-rho)*standardized[:, 1:]
            asset_scenarios = []
            for severity in spec['asset_loss_severities']:
                fractions = np.clip(severity+spec['asset_shock_dispersion']*factors, 0, .95)
                # Severity zero is a deterministic balance-sheet solvency control.
                if severity == 0:
                    fractions[:] = 0
                asset_scenarios.append(baseline_assets[:, None]*(1-fractions.T))
            assets = np.concatenate(asset_scenarios, axis=1)
            payments, certificate = clear_payments(debt, relative, assets)
            lower, lower_certificate = clear_payments(debt, relative, assets, greatest=False)
            disagreement = float(np.max(np.abs(payments-lower).sum(axis=0)))
            maximum_disagreement = max(maximum_disagreement, disagreement)
            maximum_certificate = max(maximum_certificate, certificate['maximum_payment_error_bound_l1'], lower_certificate['maximum_payment_error_bound_l1'])
            maximum_iterations = max(maximum_iterations, certificate['iterations'], lower_certificate['iterations'])
            direct_payments = np.minimum(debt[:, None], assets+(relative.T @ debt)[:, None])
            for i, severity in enumerate(spec['asset_loss_severities']):
                for shock in range(spec['shocks_per_network']):
                    index = i*spec['shocks_per_network']+shock
                    payment = payments[:, index]
                    direct = direct_payments[:, index]
                    direct_defaults = int(np.sum(direct < debt-1e-8))
                    defaults = int(np.sum(payment < debt-1e-8))
                    sink_loss = float(outside_share @ (debt-payment)/outside_debt)
                    direct_sink_loss = float(outside_share @ (debt-direct)/outside_debt)
                    row = {'mean_degree': degree, 'network': network, 'shock': shock, 'severity': severity,
                           'edges': edge_count, 'outside_debt': outside_debt,
                           'asset_loss_fraction_of_debt': float((baseline_assets-assets[:, index]).sum()/debt.sum()),
                           'direct_default_fraction': direct_defaults/spec['banks'],
                           'clearing_default_fraction': defaults/spec['banks'],
                           'contagion_default_fraction': (defaults-direct_defaults)/spec['banks'],
                           'outside_creditor_loss_fraction': sink_loss,
                           'direct_outside_loss_fraction': direct_sink_loss,
                           'contagion_outside_loss_fraction': sink_loss-direct_sink_loss}
                    raw.append(row)
                    degree_records.append(row)
                    # One representative shock per nonzero severity and network.
                    if shock == 0 and severity > 0:
                        jacobian, diag = default_sensitivity(debt, relative, assets[:, index], payment)
                        marginal = outside_share @ jacobian
                        sensitivities.append({'mean_degree': degree, 'network': network, 'severity': severity,
                                              **diag, 'max_marginal_outside_loss_per_asset_dollar': float(marginal.max())})
        for severity in spec['asset_loss_severities']:
            group = [row for row in degree_records if row['severity'] == severity]
            losses = np.array([row['outside_creditor_loss_fraction'] for row in group]).reshape(spec['networks_per_degree'], spec['shocks_per_network'])
            defaults = np.array([row['clearing_default_fraction'] for row in group]).reshape(losses.shape)
            es, tail_count = expected_shortfall(losses.ravel(), spec['expected_shortfall_probability'])
            interval, default_interval = cluster_bootstrap(losses, defaults, rng_bootstrap, spec['bootstrap_draws'], spec['expected_shortfall_probability'])
            summary.append({'mean_degree': degree, 'severity': severity,
                            'independent_networks': spec['networks_per_degree'], 'scenarios': losses.size,
                            'mean_outside_loss_fraction': float(losses.mean()), 'es95_outside_loss_fraction': es,
                            'es_tail_count': tail_count, 'es95_ci_low': interval[0], 'es95_ci_high': interval[1],
                            'mean_default_fraction': float(defaults.mean()),
                            'default_ci_low': default_interval[0], 'default_ci_high': default_interval[1],
                            'mean_contagion_default_fraction': float(np.mean([row['contagion_default_fraction'] for row in group]))})
        print(f'Financial network mean degree={degree}: completed.', flush=True)
    write_csv(out/'systemic_scenarios.csv', raw)
    write_csv(out/'systemic_summary.csv', summary)
    write_csv(out/'default_sensitivities.csv', sensitivities)
    return summary, {'networks': spec['networks_per_degree']*len(spec['mean_degrees']),
                     'scenarios': len(raw), 'maximum_payment_error_certificate': maximum_certificate,
                     'maximum_lower_upper_disagreement_l1': maximum_disagreement,
                     'maximum_clearing_iterations': maximum_iterations,
                     'zero_severity_always_solvent': all(row['clearing_default_fraction'] == 0 for row in raw if row['severity'] == 0)}


def run(output=None, quick=False):
    out = Path(output) if output else HERE/'results/research'
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(json.dumps(PROTOCOL))
    if quick:
        protocol['critical'].update(n=[250], replicates=10)
        protocol['financial'].update(networks_per_degree=3, shocks_per_network=3, bootstrap_draws=100)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    streams = np.random.SeedSequence(protocol['seed']).spawn(4)
    critical, graph_count = critical_experiment(out, protocol['critical'], np.random.default_rng(streams[0]))
    systemic, diagnostics = systemic_experiment(out, protocol['financial'], streams[1:])
    metrics = {'protocol': protocol, 'critical_graphs': graph_count, 'financial': diagnostics,
               'systemic_summary': systemic,
               'code_sha256': {str(p.relative_to(HERE)).replace('\\', '/'): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for p in sorted((HERE/'src/er_graphs').glob('*.py'))},
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'csv_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for n in protocol['critical']['n']:
        for observable, axis in [('scaled_largest', axes[0, 0]), ('scaled_susceptibility', axes[0, 1])]:
            group = [row for row in critical if row['n'] == n and row['observable'] == observable]
            axis.errorbar([row['lambda'] for row in group], [row['mean'] for row in group],
                          yerr=[1.96*row['standard_error'] for row in group], fmt='o-', capsize=2, label=f'n={n}')
    axes[0, 0].set(title='Critical window: largest-component scaling', xlabel='lambda', ylabel='E[L1] / n^(2/3)')
    axes[0, 1].set(title='Critical susceptibility scaling', xlabel='lambda', ylabel='E[sum |C|^2 / n] / n^(1/3)')
    for degree in protocol['financial']['mean_degrees']:
        group = [row for row in systemic if row['mean_degree'] == degree]
        x = np.array([row['severity'] for row in group])
        axes[1, 0].plot(x, [row['mean_default_fraction'] for row in group], 'o-', label=f'degree={degree:g}')
        y = np.array([row['es95_outside_loss_fraction'] for row in group])
        axes[1, 1].plot(x, y, 'o-', label=f'degree={degree:g}')
        axes[1, 1].fill_between(x, [row['es95_ci_low'] for row in group], [row['es95_ci_high'] for row in group], alpha=.12)
    axes[1, 0].set(title='Synthetic payment contagion', xlabel='External asset-loss severity', ylabel='Mean default fraction')
    axes[1, 1].set(title='Outside creditor tail loss: cluster bootstrap', xlabel='External asset-loss severity', ylabel='95% expected shortfall, debt fraction')
    for axis in axes.ravel():
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'network_risk.png', dpi=180)
    fig.savefig(out/'network_risk.pdf')
    plt.close(fig)
    print(json.dumps({'critical_graphs': graph_count, 'financial': diagnostics}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run(args.output, args.quick)


if __name__ == '__main__':
    main()
