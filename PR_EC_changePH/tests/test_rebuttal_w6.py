"""Synthetic-only checks: no dataset loading, model fitting or real-data analysis."""
from __future__ import annotations

import contextlib
import csv
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from PR_EC.data.adhd200 import prepare_adhd_fold_arrays
from PR_EC.model.mpr.population_similarity import load_phenotypes as original_load_phenotypes
from PR_EC.utils.folds import make_stratified_splits, prepare_fold_arrays
from PR_EC.workflow import build_fold_reference as original_build_fold_reference
from PR_EC_changePH import data as local_data, workflow
from PR_EC_changePH.analysis import summarize_table_r6
from PR_EC_changePH.analysis.summarize_table_r6 import classification_display, validate_oof_pair
from PR_EC_changePH.diagnostics import (
    neighbor_consistency_rows, save_classification, save_neighbor_diagnostics,
    save_run_manifest, validate_partition, write_csv,
)
from PR_EC_changePH.model.mpr.population_similarity import (
    fused_graph, load_phenotypes, topk_row_normalize,
)
from PR_EC_changePH.utils.output import save_qsr_ec_archive, save_results


def parsed(module, *arguments):
    with patch("sys.argv", ["test", *arguments]):
        return importlib.import_module(module).parse_args()


def synthetic_data(n_subjects=40):
    rng = np.random.default_rng(7)
    return {
        "ec": rng.normal(size=(n_subjects, 3, 3)).astype(np.float32),
        "continuous": rng.normal(size=(n_subjects, 1)).astype(np.float32),
        "categorical_raw": np.asarray([[str(index % 2), str(index % 3)] for index in range(n_subjects)]),
        "fmri_features": rng.normal(size=(n_subjects, 6)).astype(np.float32),
        "subject_ids": np.asarray([f"synthetic-{index}" for index in range(n_subjects)]),
        "labels": np.arange(n_subjects) % 2,
        "site_ids": np.asarray([f"site-{index % 3}" for index in range(n_subjects)]),
        "qsr_qc": np.zeros((n_subjects, 3), dtype=np.float32),
        "qsr_confound_values": np.zeros((n_subjects, 4), dtype=np.float32),
    }


class ConfigurationTests(unittest.TestCase):
    def test_original_defaults_and_age_only_difference(self):
        ignored = {"output_dir", "refined_ec_path", "qsr_refined_ec_path", "phenotype_feature_set", "diagnostics_only"}
        for dataset in ("abide_i", "abide_ii", "adhd200"):
            old = parsed(f"PR_EC.main_{dataset}", "--input-mode", "ec")
            original = parsed(f"PR_EC_changePH.main_{dataset}", "--input-mode", "ec")
            alternate = parsed(f"PR_EC_changePH.main_{dataset}", "--input-mode", "ec", "--phenotype-feature-set", "age_sex_site")
            self.assertEqual(original.phenotype_feature_set, "original")
            for name, value in vars(old).items():
                if name not in ignored:
                    self.assertEqual(value, getattr(original, name), name)
            for name, value in vars(original).items():
                if name not in ignored | {"continuous_weights"}:
                    self.assertEqual(value, getattr(alternate, name), name)
            self.assertEqual(alternate.continuous_weights, [1.0])
            self.assertNotEqual(original.output_dir, alternate.output_dir)
            self.assertIn("PR_EC_changePH", str(alternate.qsr_refined_ec_path))

    def test_invalid_weights_raw_diagnostics_and_unsafe_outputs_rejected(self):
        cases = (
            ["--input-mode", "ec", "--phenotype-feature-set", "age_sex_site", "--continuous-weights", "1", "0.3"],
            ["--diagnostics-only"],
            ["--input-mode", "ec", "--diagnostics-only", "--graph-mode", "phenotype"],
            ["--input-mode", "ec", "--output-dir", "/tmp/not-local"],
            ["--input-mode", "ec", "--diagnostics-only", "--reference-k", "10"],
        )
        for arguments in cases:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parsed("PR_EC_changePH.main_abide_i", *arguments)

    def test_feature_selection_and_unchanged_original_loader(self):
        for dataset in ("abide_i", "abide_ii", "adhd200"):
            profile = importlib.import_module(f"PR_EC_changePH.main_{dataset}").DATASET_CONFIG
            age = "Age" if dataset == "adhd200" else "AGE_AT_SCAN"
            fields = list(dict.fromkeys((profile.phenotype_id_column, profile.sex_column, profile.site_column,
                                         age, *profile.continuous_columns, "DX", "questionnaire")))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "synthetic.csv"
                write_csv(path, [{key: value for key, value in zip(fields, ["s1", "1", "site-a", "12", *(["99"] * (len(fields) - 4))])}])
                baseline = original_load_phenotypes(path, ["s1"], ["archive-site"], profile)
                current = load_phenotypes(path, ["s1"], ["archive-site"], profile)
                for name in baseline:
                    np.testing.assert_array_equal(baseline[name], current[name])
                alternate = load_phenotypes(path, ["s1"], ["archive-site"], profile, "age_sex_site")
                np.testing.assert_array_equal(alternate["continuous"], [[12]])
                np.testing.assert_array_equal(alternate["categorical_raw"], [["1", "site-a"]])

    def test_ec_diagnostics_loader_skips_qsr_and_encoder(self):
        args = parsed("PR_EC_changePH.main_abide_ii", "--input-mode", "ec", "--diagnostics-only")
        data = synthetic_data()
        with patch.object(local_data, "load_ec_archive", return_value=data.copy()), \
             patch.object(local_data, "_aligned_graph_time_series", return_value=[]), \
             patch.object(local_data, "subject_fc_features", return_value=data["fmri_features"]), \
             patch.object(local_data, "load_phenotypes", return_value={
                 "continuous": data["continuous"], "categorical_raw": data["categorical_raw"]}), \
             patch.object(local_data, "load_aligned_qc", side_effect=AssertionError("QC loading forbidden")), \
             patch.object(local_data, "load_aligned_phenotypes", side_effect=AssertionError("Confound loading forbidden")):
            loaded, training = local_data.load_pipeline_data(args, "cpu")
        self.assertIsNone(training)
        np.testing.assert_array_equal(loaded["subject_ids"], data["subject_ids"])


class GraphAndStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.args = parsed("PR_EC_changePH.main_abide_i", "--input-mode", "ec", "--phenotype-feature-set", "age_sex_site")

    def test_fold_preprocessing_training_only(self):
        train_ec = np.ones((4, 3, 3), dtype=np.float32)
        for prepare in (prepare_fold_arrays, prepare_adhd_fold_arrays):
            arrays = prepare(
                train_ec, train_ec[:1], train_ec[:1],
                np.asarray([[10.], [12.], [np.nan], [14.]]),
                np.asarray([[1000.]]), np.asarray([[np.nan]]),
                np.asarray([["M", "a"], ["F", "b"], ["M", "a"], ["F", "b"]]),
                np.asarray([["F", "val-only"]]), np.asarray([["new-sex", "test-only"]]),
            )
            np.testing.assert_array_equal(arrays["continuous_scaler"]["median"], [12])
            np.testing.assert_array_equal(arrays["continuous_scaler"]["mean"], [12])
            np.testing.assert_array_equal(arrays["test_cont"], [[0]])
            np.testing.assert_array_equal(arrays["test_cat"], [[2, 2]])
            self.assertGreater(arrays["val_cont"][0, 0], 100)

    def test_original_fold_graph_bitwise_unchanged(self):
        rng = np.random.default_rng(5)
        for dimensions in (2, 3):
            self.args.continuous_weights = [1.0] + [0.3] * (dimensions - 1)
            sizes = (28, 6, 6)
            ec = [rng.normal(size=(size, 3, 3)).astype(np.float32) for size in sizes]
            continuous = [rng.normal(size=(size, dimensions)) for size in sizes]
            categorical = [rng.integers(0, 2, size=(size, 1)).astype(str) for size in sizes]
            prepare = prepare_fold_arrays if dimensions == 2 else prepare_adhd_fold_arrays
            arrays = prepare(*ec, *continuous, *categorical)
            fmri = tuple(rng.normal(size=(size, 5)).astype(np.float32) for size in sizes)
            old = original_build_fold_reference(self.args, arrays, fmri)
            new = workflow.build_fold_reference(self.args, arrays, fmri)
            for name in old:
                np.testing.assert_array_equal(old[name], new[name], err_msg=name)

    def test_final_fusion_indices_and_unweighted_counts(self):
        fmri = np.asarray([[0.9, 0.1, 0, 0], [0, 0, 0.2, 0.8]], dtype=np.float32)
        phenotype = np.asarray([[0, 0, 0.9, 0.1], [0.8, 0.2, 0, 0]], dtype=np.float32)
        weights, selected = fused_graph(fmri, phenotype, 0.6, 2, return_indices=True)
        np.testing.assert_array_equal(weights, fused_graph(fmri, phenotype, 0.6, 2))
        self.assertEqual(set(selected[0]), {0, 2})
        self.assertEqual(set(selected[1]), {0, 3})
        self.args.reference_k = 2
        labels = np.asarray([1, 1, 0, 1, 1, 0, 0])
        rows = neighbor_consistency_rows(self.args, 1, list("abcdefg"), labels, [0, 1, 2, 3], [4, 5], selected)
        self.assertEqual([row["same_diagnosis_count"] for row in rows], [1, 0])
        self.assertEqual([row["same_diagnosis_pct"] for row in rows], [50.0, 0.0])
        self.assertEqual([row["base_rate_pct"] for row in rows], [75.0, 25.0])
        self.assertNotAlmostEqual(100 * weights[0, 0], rows[0]["same_diagnosis_pct"])
        with self.assertRaises(ValueError):
            neighbor_consistency_rows(self.args, 1, list("abcdefg"), labels, [0, 1, 2, 3], [4, 5], [[0, 4], [0, 1]])

    def test_zero_score_ties_and_small_reference_k(self):
        weights, selected = topk_row_normalize(np.zeros((2, 3)), 20, return_indices=True)
        self.assertEqual(selected.shape, (2, 3))
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.args.reference_k = 20
        rows = neighbor_consistency_rows(self.args, 1, list("abcde"), [1, 0, 1, 1, 0], [0, 1, 2], [3, 4], selected)
        self.assertEqual(rows[0]["K"], 3)
        self.assertAlmostEqual(rows[0]["same_diagnosis_pct"], 200 / 3)

    def test_same_splits_no_validation_or_test_references(self):
        data = synthetic_data()
        first = list(make_stratified_splits(data["labels"], 10, 42, 0.2))
        second = list(make_stratified_splits(data["labels"].copy(), 10, 42, 0.2))
        seen = []
        for left, right in zip(first, second):
            self.assertEqual(left[0], right[0])
            for old_index, new_index in zip(left[1:], right[1:]):
                np.testing.assert_array_equal(old_index, new_index)
            validate_partition(data["subject_ids"], *left[1:])
            seen.extend(left[-1])
        self.assertEqual(sorted(seen), list(range(40)))
        with self.assertRaises(ValueError):
            validate_partition(data["subject_ids"], [0, 1], [1], list(range(2, 40)))

    def test_diagnostics_never_trains_and_matches_full_fold_retrieval(self):
        data = synthetic_data()
        self.args.diagnostics_only = True
        with patch.object(workflow, "train_qsr_refiner", side_effect=AssertionError("QSR forbidden")), \
             patch.object(workflow, "train_pgr_refiner", side_effect=AssertionError("MPR training forbidden")), \
             patch.object(workflow, "train_classifier", side_effect=AssertionError("Classifier forbidden")), \
             patch.object(workflow, "set_seed"), contextlib.redirect_stdout(io.StringIO()):
            result = workflow.run_cross_validation(self.args, data, "cpu")
        self.assertEqual(len(result["neighbor_rows"]), 40)
        self.assertEqual(len({row["subject_id"] for row in result["neighbor_rows"]}), 40)
        fold, train, val, test = next(make_stratified_splits(data["labels"], 10, 42, 0.2))
        self.args.diagnostics_only = False
        self.args.representations = ["qc_refined"]
        mock_metrics = {name: 0.5 for name in ("ACC", "Recall", "AUC", "Precision", "F1")}
        with patch.object(workflow, "train_qsr_refiner", return_value=(None, data["ec"][train], None, {})), \
             patch.object(workflow, "apply_qsr_refiner", side_effect=lambda model, ec, *rest: ec), \
             patch.object(workflow, "train_classifier", return_value=(mock_metrics, None)), \
             patch.object(workflow, "set_seed"):
            full = workflow.run_fold(self.args, fold, data, train, val, test, "cpu")
        self.assertEqual(full["neighbor_rows"], result["neighbor_rows"][:len(test)])
        altered = dict(data, labels=1 - data["labels"])
        self.args.diagnostics_only = True
        with patch.object(workflow, "set_seed"):
            swapped = workflow.run_fold(self.args, fold, altered, train, val, test, "cpu")
        self.assertEqual([row["neighbor_subject_ids"] for row in full["neighbor_rows"]],
                         [row["neighbor_subject_ids"] for row in swapped["neighbor_rows"]])


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.args = parsed("PR_EC_changePH.main_abide_i", "--input-mode", "ec")
        self.args.output_dir = Path(self.temporary.name)

    def test_subject_weighted_summary_and_no_output_cleanup(self):
        rows = [{"subject_id": f"s{index}", "fold": fold,
                 "same_diagnosis_pct": value, "base_rate_pct": 50.0}
                for index, (fold, value) in enumerate(((1, 0), (2, 100), (2, 100)))]
        save_neighbor_diagnostics(self.args, rows)
        with (self.args.output_dir / "neighbor_consistency_summary.csv").open() as handle:
            summary = list(csv.DictReader(handle))[0]
        self.assertAlmostEqual(float(summary["same_diagnosis_pct"]), 200 / 3)
        sentinel = self.args.output_dir / "keep.txt"
        sentinel.write_text("unchanged")
        metric = {name: 0.5 for name in ("ACC", "Recall", "AUC", "Precision", "F1")}
        folds = [{"qc_refined": metric} for _ in range(10)]
        save_results(self.args, folds, {})
        save_classification(self.args, folds)
        self.assertEqual(sentinel.read_text(), "unchanged")
        self.assertTrue((self.args.output_dir / "neighbor_consistency_subjects.csv").exists())
        with (self.args.output_dir / "classification_summary.csv").open() as handle:
            summary = list(csv.DictReader(handle))[0]
        self.assertEqual(summary["ACC_display"], "50.00 ± 0.00")
        self.assertEqual(summary["std_ddof"], "0")

    def test_manifest_blocks_mixed_settings_but_allows_diagnostics_then_full(self):
        data = synthetic_data()
        save_run_manifest(self.args, data)
        self.args.diagnostics_only = True
        save_run_manifest(self.args, data)
        self.args.phenotype_feature_set = "age_sex_site"
        with self.assertRaises(ValueError):
            save_run_manifest(self.args, data)

    def test_qsr_archive_preserves_ids_labels_roi_and_order(self):
        data = synthetic_data()
        data["roi_names"] = np.asarray(["ROI-C", "ROI-A", "ROI-B"])
        fold_ids = np.arange(40) % 10 + 1
        path = self.args.output_dir / "synthetic_qsr.npz"
        save_qsr_ec_archive(path, data, data["ec"], fold_ids, "synthetic_input.npz")
        with np.load(path, allow_pickle=False) as archive:
            for name in ("subject_ids", "labels", "roi_names", "site_ids"):
                np.testing.assert_array_equal(archive[name], data[name])
            np.testing.assert_array_equal(archive["fold_ids"], fold_ids)
            np.testing.assert_array_equal(archive["ec"], data["ec"])

    def test_summary_reuses_original_fold_metrics_and_rejects_incomplete(self):
        original = self.args.output_dir / "original"
        alternate = self.args.output_dir / "alternate"
        rows = [{"fold": fold, "qc_refined_ACC": 0.8, "qc_refined_AUC": 0.9} for fold in range(1, 11)]
        write_csv(original / "experiment_summary.csv", rows)
        (original / "summary.json").write_text(json.dumps({"config": {"seed": 42, "n_splits": 10}}))
        display, source = classification_display(alternate, original)
        self.assertEqual(display, {"ACC": "80.00 ± 0.00", "AUC": "90.00 ± 0.00"})
        self.assertTrue(source.endswith("experiment_summary.csv"))
        with self.assertRaises(FileNotFoundError):
            classification_display(alternate)
        membership = [{"fold": 1, "seed": 42, "subject_id": "synthetic", "role": "test"}]
        write_csv(original / "split_membership.csv", membership)
        write_csv(alternate / "split_membership.csv", [{**membership[0], "subject_id": "wrong"}])
        manifest = {"config": {}, "subjects_labels_sha256": "synthetic"}
        (original / "run_manifest.json").write_text(json.dumps(manifest))
        (alternate / "run_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            validate_oof_pair(original, alternate)

    def test_complete_table_r6_on_synthetic_outputs_only(self):
        root = self.args.output_dir / "synthetic-w6"
        paper_root = self.args.output_dir / "synthetic-paper"
        metric = {name: 0.5 for name in ("ACC", "Recall", "AUC", "Precision", "F1")}
        with patch.object(workflow, "train_qsr_refiner", side_effect=AssertionError("QSR forbidden")), \
             patch.object(workflow, "train_pgr_refiner", side_effect=AssertionError("MPR forbidden")), \
             patch.object(workflow, "train_classifier", side_effect=AssertionError("Classifier forbidden")), \
             patch.object(workflow, "set_seed"), contextlib.redirect_stdout(io.StringIO()):
            for module, dataset in (("abide_i", "abide-i"), ("abide_ii", "abide-ii"), ("adhd200", "adhd200")):
                for setting in ("original", "age_sex_site"):
                    args = parsed(f"PR_EC_changePH.main_{module}", "--input-mode", "ec", "--diagnostics-only",
                                  "--phenotype-feature-set", setting)
                    args.output_dir = root / dataset / setting
                    data = synthetic_data()
                    if setting == "original":
                        data["continuous"] = np.repeat(data["continuous"], len(args.continuous_weights), axis=1)
                        data["categorical_raw"] = data["categorical_raw"][:, :1]
                    save_run_manifest(args, data)
                    result = workflow.run_cross_validation(args, data, "cpu")
                    save_neighbor_diagnostics(args, result["neighbor_rows"])
                    write_csv(args.output_dir / "split_membership.csv", result["split_rows"])
                    if setting == "age_sex_site":
                        save_classification(args, [{"qc_refined": metric} for _ in range(10)])
                original = paper_root / dataset
                write_csv(original / "experiment_summary.csv", [
                    {"fold": fold, "qc_refined_ACC": 0.8, "qc_refined_AUC": 0.9} for fold in range(1, 11)])
                (original / "summary.json").write_text(json.dumps({"config": {"seed": 42, "n_splits": 10}}))
            output = root / "synthetic-table.csv"
            with patch.object(summarize_table_r6, "LOCAL_ROOT", self.args.output_dir), \
                 patch("sys.argv", ["summary", "--root", str(root), "--original-results-root", str(paper_root), "--output", str(output)]):
                summarize_table_r6.main()
        rows = summarize_table_r6.read_rows(output)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["ACC (%)"], "80.00 ± 0.00")
        self.assertEqual(rows[1]["ACC (%)"], "50.00 ± 0.00")
        subjects_path = root / "abide-i/age_sex_site/neighbor_consistency_subjects.csv"
        subjects = summarize_table_r6.read_rows(subjects_path)
        subjects[0]["neighbor_subject_ids"] = json.dumps([subjects[0]["subject_id"]] * 20)
        write_csv(subjects_path, subjects)
        with self.assertRaises(ValueError):
            validate_oof_pair(root / "abide-i/original", root / "abide-i/age_sex_site")


if __name__ == "__main__":
    unittest.main()
