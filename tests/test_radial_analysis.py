import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_feecho4d_ablation import audit_and_summarize, paired


class RadialAnalysisTests(unittest.TestCase):
    def test_incomplete_runs_cannot_produce_a_final_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "spread3").mkdir()
            (root / "spread3/runs.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "2080 missing"):
                audit_and_summarize(root)

    def test_duplicate_subject_method_seed_is_rejected_before_aggregation(self):
        row = {"subject": "Patient001", "layout": "spread3", "method": "inr", "seed": 17}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "spread3").mkdir()
            (root / "spread3/runs.jsonl").write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit_and_summarize(root)

    def test_pairs_are_matched_by_subject_and_preserve_difference_sign(self):
        result = paired({"B": .4, "A": .9, "extra": 1}, {"A": .7, "B": .5})
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mean_difference"], .05)
        self.assertEqual(result["first_better"], 1)
        self.assertEqual(result["ties"], 0)
        self.assertEqual(paired({"A": .5, "B": .5}, {"B": .5, "A": .5})["wilcoxon_p"], 1)


if __name__ == "__main__":
    unittest.main()
