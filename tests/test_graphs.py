from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from er_graphs.experiment import er_edges, graph_stats, short_cycles, alpha


class GraphTests(unittest.TestCase):
    def test_complete_graph_cycle_counts(self):
        u, v = er_edges(4, 1., np.random.default_rng(10))
        self.assertEqual(short_cycles(4, u, v), (4, 3))
        self.assertEqual(graph_stats(4, u, v), (4, 1, 0))

    def test_tree_has_no_cycles(self):
        u, v = np.array([0, 1, 2]), np.array([1, 2, 3])
        self.assertEqual(short_cycles(4, u, v), (0, 0))

    def test_empty_graph_and_no_duplicate_edges(self):
        rng = np.random.default_rng(20)
        u, v = er_edges(30, 0., rng)
        self.assertEqual(graph_stats(30, u, v), (1, 30, 30))
        u, v = er_edges(30, .4, rng)
        self.assertTrue(np.all(u < v))
        self.assertEqual(len(u), len(set(zip(u.tolist(), v.tolist()))))

    def test_binomial_edge_count_distribution(self):
        rng = np.random.default_rng(123)
        values = [len(er_edges(20, .2, rng)[0]) for _ in range(2000)]
        self.assertLess(abs(np.mean(values)-38), 6*np.sqrt(190*.2*.8/2000))
        self.assertLess(abs(np.var(values, ddof=1)-30.4), 5.)

    def test_positive_fixed_point(self):
        self.assertEqual(alpha(.8), 0.)
        for c in [1.1, 1.4, 2., 3.]:
            a = alpha(c)
            self.assertGreater(a, 0)
            self.assertAlmostEqual(1-a, np.exp(-c*a), places=12)


if __name__ == '__main__':
    unittest.main()
