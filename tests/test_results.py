from pathlib import Path
import csv
import hashlib
import json
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/research'


class ResultTests(unittest.TestCase):
    def test_frozen_artifact_checksums(self):
        metrics = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
        for name, digest in metrics['csv_sha256'].items():
            self.assertEqual(hashlib.sha256((OUT/name).read_bytes()).hexdigest(), digest, name)
        for name, digest in metrics['code_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), digest, name)

    def test_systemic_tail_and_default_summary_reconstructed(self):
        with (OUT/'systemic_scenarios.csv').open(encoding='utf-8', newline='') as handle:
            raw = list(csv.DictReader(handle))
        with (OUT/'systemic_summary.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                group = [r for r in raw if float(r['mean_degree']) == float(row['mean_degree']) and float(r['severity']) == float(row['severity'])]
                losses = np.sort([float(r['outside_creditor_loss_fraction']) for r in group])
                count = int(row['es_tail_count'])
                self.assertEqual(len(group), int(row['scenarios']))
                self.assertAlmostEqual(losses[-count:].mean(), float(row['es95_outside_loss_fraction']), places=12)
                defaults = np.mean([float(r['clearing_default_fraction']) for r in group])
                self.assertAlmostEqual(defaults, float(row['mean_default_fraction']), places=12)
        self.assertTrue(all(float(r['contagion_default_fraction']) >= 0 for r in raw))
