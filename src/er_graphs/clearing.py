"""Limited-liability payment clearing with an explicit outside creditor."""
import numpy as np
from scipy import sparse


def validate_system(obligations, relative_liabilities, assets):
    obligations = np.asarray(obligations, dtype=float)
    assets = np.asarray(assets, dtype=float)
    relative = np.asarray(relative_liabilities, dtype=float)
    if obligations.ndim != 1 or assets.ndim not in [1, 2] or not len(obligations):
        raise ValueError('Nonempty debt and asset vectors are required.')
    n = len(obligations)
    if relative.shape != (n, n) or assets.shape[0] != n:
        raise ValueError('Incompatible financial system dimensions.')
    if not all(np.isfinite(value).all() for value in [obligations, assets, relative]):
        raise ValueError('Financial inputs must be finite.')
    if np.any(obligations <= 0) or np.any(assets < 0) or np.any(relative < 0) or np.any(np.diag(relative) != 0):
        raise ValueError('Positive debt, nonnegative assets and liabilities, and no self claims are required.')
    contraction = float(relative.sum(axis=1).max())
    if contraction >= 1:
        raise ValueError('Every bank must owe a positive share to the outside creditor.')
    return obligations, relative, assets, contraction


def clear_payments(obligations, relative_liabilities, assets, greatest=True, tolerance=1e-10, max_iterations=2000):
    """Monotone iteration with an L1 residual certificate for the unique fixed point."""
    obligations, relative, assets, contraction = validate_system(obligations, relative_liabilities, assets)
    vector = assets.ndim == 1
    assets = assets[:, None] if vector else assets
    debt = obligations[:, None]
    payments = np.repeat(debt, assets.shape[1], axis=1) if greatest else np.zeros_like(assets)
    operator = sparse.csr_matrix(relative.T)
    for iteration in range(1, max_iterations+1):
        updated = np.minimum(debt, assets+operator @ payments)
        payments = updated
        residual = np.abs(np.minimum(debt, assets+operator @ payments)-payments).sum(axis=0)
        error_bound = residual/(1-contraction)
        if np.max(error_bound) <= tolerance:
            return (payments[:, 0] if vector else payments), {
                'iterations': iteration, 'contraction_bound': contraction,
                'maximum_fixed_point_residual_l1': float(residual.max()),
                'maximum_payment_error_bound_l1': float(error_bound.max()),
            }
    raise RuntimeError('Payment clearing did not meet the residual certificate.')


def default_sensitivity(obligations, relative_liabilities, assets, payments, margin=1e-7):
    """Derivative on an unchanged default set; undefined at a switching boundary."""
    obligations, relative, assets, _ = validate_system(obligations, relative_liabilities, assets)
    if assets.ndim != 1:
        raise ValueError('Local sensitivities require one scenario.')
    resources = assets+relative.T @ payments
    if np.any(np.abs(resources-obligations) < margin):
        raise ValueError('Scenario lies too close to a default-set boundary.')
    default = payments < obligations-margin
    indices = np.flatnonzero(default)
    jacobian = np.zeros_like(relative)
    if len(indices):
        subsystem = relative[np.ix_(indices, indices)].T
        inverse = np.linalg.solve(np.eye(len(indices))-subsystem, np.eye(len(indices)))
        jacobian[np.ix_(indices, indices)] = inverse
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(subsystem))))
        amplification = float(np.linalg.norm(inverse, 1))
    else:
        spectral_radius, amplification = 0., 1.
    return jacobian, {'default_banks': len(indices), 'default_subsystem_spectral_radius': spectral_radius,
                      'default_resolvent_l1_norm': amplification}


def financial_network(n, mean_degree, interbank_share, equity_buffer, rng):
    """Symmetric random exposures with debt scaled by weighted degree.

    Nonisolated banks have the same interbank debt share and relative equity.
    Isolated banks owe exclusively to the outside creditor. Bank sizes vary.
    """
    if n < 2 or not 0 <= mean_degree <= n-1 or not 0 < interbank_share < 1 or equity_buffer <= 0:
        raise ValueError('Invalid network or balance-sheet parameters.')
    upper = np.triu(rng.uniform(size=(n, n)) < mean_degree/(n-1), 1)
    weighted = np.triu(rng.gamma(2., .5, size=(n, n))*upper, 1)
    weighted += weighted.T
    strength = weighted.sum(axis=1)
    connected = strength > 0
    reference = strength[connected].mean() if connected.any() else 1.
    obligations = np.where(connected, strength/reference, 1.)
    liabilities = interbank_share*weighted/reference
    relative = liabilities/obligations[:, None]
    incoming = liabilities.sum(axis=0)
    assets = obligations-incoming+equity_buffer*obligations
    if np.min(assets) <= 0:
        raise RuntimeError('The baseline balance sheet is infeasible.')
    return obligations, relative, assets, int(upper.sum())
