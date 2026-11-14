"""Synthetic-only tests for read-only cross-cohort edge statistics."""
import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import gammaln
from scipy.stats import ttest_ind

from PR_EC_changePH.analysis import analyze_table_r7 as analysis


class TableR7Tests(unittest.TestCase):
    def test_effects_welch_and_bh_match_independent_calculation(self):
        rng = np.random.default_rng(42)
        ec = rng.normal(size=(25, 4, 4))
        labels = np.asarray([1] * 12 + [0] * 13)
        for index in range(4):
            ec[:, index, index] = 0
        ec[labels == 1, 0, 1] += 2
        ec[labels == 1, 1, 0] -= 1
        rows = analysis.edge_statistics("synthetic", ec, labels, np.asarray(list("ABCD")))
        self.assertEqual(len(rows), 12)
        lookup = {(row["SourceIndex"], row["TargetIndex"]): row for row in rows}
        source, target = np.where(~np.eye(4, dtype=bool))
        independent_p = []
        for source_index, target_index in zip(source, target):
            asd = ec[labels == 1, source_index, target_index]
            tc = ec[labels == 0, source_index, target_index]
            pooled = np.sqrt((11 * asd.var(ddof=1) + 12 * tc.var(ddof=1)) / 23)
            correction = math.exp(gammaln(23 / 2) - 0.5 * math.log(23 / 2) - gammaln(22 / 2))
            row = lookup[(source_index, target_index)]
            self.assertAlmostEqual(row["Hedges_g"], (asd.mean() - tc.mean()) / pooled * correction, places=12)
            test = ttest_ind(asd, tc, equal_var=False)
            self.assertAlmostEqual(row["PValue"], test.pvalue, places=12)
            independent_p.append(test.pvalue)
        ordered = sorted(zip(independent_p, range(12)))
        expected_q = np.empty(12)
        running = 1.0
        for rank in range(12, 0, -1):
            value, index = ordered[rank - 1]
            running = min(running, value * 12 / rank)
            expected_q[index] = running
        np.testing.assert_allclose([lookup[(s, t)]["QValue"] for s, t in zip(source, target)], expected_q)
        self.assertGreater(lookup[(0, 1)]["Hedges_g"], 0)
        self.assertLess(lookup[(1, 0)]["Hedges_g"], 0)

    def test_retention_requires_significance_and_original_direction_separately(self):
        def row(source, target, g, replicated):
            return {"SourceIndex": source, "TargetIndex": target, "Source": str(source), "Target": str(target),
                    "Replicated": replicated, "ReplicationClass": "enhanced" if g > 0 else "reduced",
                    "ABIDE_I_Hedges_g": g, "ABIDE_II_Hedges_g": g,
                    "ABIDE_I_QValue": 0.01 if replicated else 0.1, "ABIDE_II_QValue": 0.01}
        baseline = [row(0, 1, 1, True), row(1, 0, -1, True), row(0, 2, 1, True)]
        candidate = [row(0, 1, 1, True), row(1, 0, 1, True), row(0, 2, 1, False)]
        result = analysis.retained_edges(baseline, candidate)
        self.assertEqual(sum(item["RetainedReplicated"] for item in result), 2)
        self.assertEqual(sum(item["RetainedAndOriginalDirection"] for item in result), 1)
        self.assertEqual(sum(item["SameOriginalDirectionBothCohorts"] for item in result), 2)

    def test_subject_and_direction_mismatch_fail_closed(self):
        with self.assertRaises(ValueError):
            analysis.check_same(np.asarray(["a", "b"]), np.asarray(["b", "a"]), "IDs")
        with self.assertRaises(ValueError):
            analysis.check_same(np.asarray([[0, 1], [2, 0]]), np.asarray([[0, 2], [1, 0]]), "transpose")

    def test_canonical_diagnosis_mapping_and_missing_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phenotype.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([[" FILE_ID  ", " DX_GROUP "], [" a ", 2], [" b ", 1]])
            np.testing.assert_array_equal(analysis.canonical_labels(path, ["b", "a"]), [0, 1])
            with self.assertRaises(ValueError):
                analysis.canonical_labels(path, ["missing"])

    def test_nonfinite_off_diagonal_tests_fail(self):
        ec = np.zeros((8, 3, 3))
        with self.assertRaises(ValueError):
            analysis.edge_statistics("synthetic", ec, np.asarray([0] * 4 + [1] * 4), np.asarray(list("ABC")))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=analysis.OUTPUT_ROOT) as directory:
            path = Path(directory)
            (path / "existing.csv").write_text("preserve")
            with self.assertRaises(FileExistsError):
                analysis.run(path)
            self.assertEqual((path / "existing.csv").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
