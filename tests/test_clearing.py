from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from er_graphs.clearing import clear_payments, default_sensitivity, financial_network
from er_graphs.research import component_moments, expected_shortfall


class ClearingTests(unittest.TestCase):
    def setUp(self):
        self.debt = np.ones(2)
        self.relative = np.array([[0., .4], [.2, 0.]])
        self.assets = np.array([.1, .2])

    def test_all_default_analytic_solution_and_certificate(self):
        expected = np.linalg.solve(np.eye(2)-self.relative.T, self.assets)
        actual, diag = clear_payments(self.debt, self.relative, self.assets)
        error = np.linalg.norm(actual-expected, 1)
        self.assertLessEqual(error, diag['maximum_payment_error_bound_l1']+1e-14)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        lower, _ = clear_payments(self.debt, self.relative, self.assets, greatest=False)
        np.testing.assert_allclose(actual, lower, atol=1e-10)

    def test_liability_priority_conservation_and_monotonicity(self):
        payment, _ = clear_payments(self.debt, self.relative, self.assets)
        resources = self.assets+self.relative.T @ payment
        np.testing.assert_allclose(payment, np.minimum(self.debt, resources), atol=1e-10)
        equity = resources-payment
        outside_payments = (1-self.relative.sum(axis=1)) @ payment
        self.assertAlmostEqual(outside_payments+equity.sum(), self.assets.sum(), places=10)
        improved, _ = clear_payments(self.debt, self.relative, self.assets+.1)
        self.assertTrue(np.all(improved >= payment))

    def test_sensitivity_against_independent_finite_difference(self):
        payment, _ = clear_payments(self.debt, self.relative, self.assets)
        derivative, diag = default_sensitivity(self.debt, self.relative, self.assets, payment)
        for column in range(2):
            shifted, _ = clear_payments(self.debt, self.relative, self.assets+np.eye(2)[column]*1e-5)
            np.testing.assert_allclose((shifted-payment)/1e-5, derivative[:, column], atol=1e-6)
        self.assertAlmostEqual(diag['default_subsystem_spectral_radius'], np.sqrt(.08))
        self.assertLessEqual(diag['default_resolvent_l1_norm'], 1/(1-.4))

    def test_batch_payments_and_boundary_controls(self):
        assets = np.column_stack([self.assets, np.ones(2)*2, np.zeros(2)])
        payments, _ = clear_payments(self.debt, self.relative, assets)
        np.testing.assert_array_equal(payments[:, 1], np.ones(2))
        np.testing.assert_allclose(payments[:, 2], 0, atol=1e-10)
        with self.assertRaises(ValueError):
            clear_payments(self.debt, np.array([[0., 1.], [1., 0.]]), self.assets)

    def test_financial_network_has_solvent_baseline(self):
        for degree in [0., 2., 9.]:
            debt, relative, assets, _ = financial_network(10, degree, .65, .08, np.random.default_rng(9))
            payment, _ = clear_payments(debt, relative, assets)
            np.testing.assert_allclose(payment, debt, atol=1e-10)
            equity = assets+relative.T @ debt-debt
            np.testing.assert_allclose(equity, .08*debt, atol=1e-12)

    def test_component_susceptibility_and_surplus(self):
        sizes, susceptibility, finite = component_moments(6, np.array([0, 1, 3]), np.array([1, 2, 4]))
        np.testing.assert_array_equal(sizes, [3, 2, 1])
        self.assertAlmostEqual(susceptibility, 14/6)
        self.assertAlmostEqual(finite, 5/3)

    def test_expected_shortfall_tail_definition(self):
        es, count = expected_shortfall(np.arange(100))
        self.assertEqual(count, 5)
        self.assertEqual(es, 97.)


if __name__ == '__main__':
    unittest.main()
