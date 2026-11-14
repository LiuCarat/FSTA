"""CPU-only synthetic tensor and mocked-pipeline tests; never load cohort data."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from PR_EC.model.individual_ec.encoder import STFEncoder as LegacyEncoder
from PR_EC.utils import make_stratified_splits
from PR_EC.workflow import run_cross_validation as original_cross_validation
from PR_EC_E2E import data as local_data, w7_runner
from PR_EC_E2E.analysis import summarize_table_r8
from PR_EC_E2E.model.individual_ec.encoder import STFEncoder
from PR_EC_E2E.model.individual_ec.utils import STFWindowLoss, extract_subject_ec
from PR_EC_E2E.utils.output import save_qsr_ec_archive, save_results
from PR_EC_E2E.w7 import (
    audit_original_configuration, initial_ec_metadata, initialize_w7_run, jsonable,
    save_split_membership, save_w7_classification,
    validate_initial_ec_mode, validate_reference_subjects, write_json,
)


def setUpModule():
    torch.set_num_threads(1)


def parsed(suffix, *arguments):
    with patch("sys.argv", ["synthetic-test", *arguments]):
        return importlib.import_module(f"PR_EC_E2E.main_{suffix}").parse_args()


def options(mode="batch_mean", nodes=4):
    return argparse.Namespace(
        nodes_num=nodes, time_num=6, d_model=4, num_hidden_layers=1,
        num_attention_heads=2, hidden_act="gelu", attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2, initializer_range=0.02, no_filters=False,
        stf_fusion_mode=mode,
    )


def encoder(mode="batch_mean", nodes=4, legacy=False):
    constructor = LegacyEncoder if legacy else STFEncoder
    return constructor(options(mode, nodes), 6, 4, 8, 2, 2, 2, dropout=0.2)


def cohort(size=40):
    rng = np.random.default_rng(11)
    labels = np.arange(size) % 2
    fold_ids = np.full(size, -1, dtype=np.int64)
    for fold, _, _, test in make_stratified_splits(labels, 10, 42, 0.2):
        fold_ids[test] = fold
    return {
        "ec": rng.normal(size=(size, 3, 3)).astype(np.float32),
        "labels": labels, "subject_ids": np.asarray([f"synthetic-{index}" for index in range(size)]),
        "site_ids": np.asarray([f"site-{index % 2}" for index in range(size)]),
        "roi_names": np.asarray(["roi-C", "roi-A", "roi-B"]), "fold_ids": fold_ids,
        "reconstruction_mse": np.zeros(size, dtype=np.float32),
    }


class AttentionTests(unittest.TestCase):
    def test_batch_mean_matches_legacy_outputs_and_gradients(self):
        torch.manual_seed(4)
        old = encoder(legacy=True).train()
        current = encoder().train()
        current.load_state_dict(old.state_dict(), strict=True)
        inputs = torch.randn(3, 6, 4)
        torch.manual_seed(19)
        expected, expected_attention = old(inputs)
        torch.manual_seed(19)
        actual, actual_attention = current(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_attention, expected_attention, rtol=0, atol=0)
        criterion = STFWindowLoss(node_count=4)
        criterion(expected, expected_attention, inputs)[0].backward()
        criterion(actual, actual_attention, inputs)[0].backward()
        for (name, old_parameter), (new_name, new_parameter) in zip(old.named_parameters(), current.named_parameters()):
            self.assertEqual(name, new_name)
            if old_parameter.grad is None:
                self.assertIsNone(new_parameter.grad)
            else:
                torch.testing.assert_close(old_parameter.grad, new_parameter.grad, rtol=0, atol=0)

    def test_subject_reduction_temporal_unchanged_and_parameter_count(self):
        batch = encoder().eval()
        personal = encoder("subject_specific").eval()
        personal.load_state_dict(batch.state_dict(), strict=True)
        self.assertEqual(sum(parameter.numel() for parameter in batch.parameters()),
                         sum(parameter.numel() for parameter in personal.parameters()))
        self.assertEqual(list(batch.state_dict()), list(personal.state_dict()))
        inputs = torch.randn(2, 6, 4)
        for expected, actual in zip(batch._encode(inputs), personal._encode(inputs)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        captured = []
        hook = personal.slf_attn.slf_attn.register_forward_hook(lambda module, arguments, output: captured.append(output[1]))
        try:
            reconstruction, attention, mean_attention = personal(inputs, return_batch_mean=True)
        finally:
            hook.remove()
        raw = captured[-1]
        self.assertEqual(reconstruction.shape, inputs.shape)
        self.assertEqual(attention.shape, (2, 4, 4))
        torch.testing.assert_close(attention, raw.mean(dim=1).mean(dim=1), rtol=0, atol=0)
        torch.testing.assert_close(mean_attention, raw.mean(dim=0).mean(dim=0).mean(dim=0), rtol=0, atol=0)
        self.assertFalse(torch.allclose(attention[0], attention[1]))
        self.assertEqual(captured[0].shape[-2:], (6, 6))

    def test_personal_fusion_broadcast_and_no_cross_subject_sharing(self):
        model = encoder("subject_specific").eval()
        spatial = torch.randn(2, 3, 4, 4)
        temporal = torch.randn(2, 3, 4, 4)
        attention = torch.softmax(torch.randn(2, 4, 4), dim=-1)
        actual = model._decode_with_spatial_attention(spatial, temporal, attention)
        expected = torch.cat([
            model._decode_with_spatial_attention(spatial[index:index + 1], temporal[index:index + 1], attention[index])
            for index in range(2)
        ])
        torch.testing.assert_close(actual, expected)
        changed = attention.clone()
        changed[1] = changed[1].flip(-1)
        changed_output = model._decode_with_spatial_attention(spatial, temporal, changed)
        torch.testing.assert_close(actual[0], changed_output[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(actual[1], changed_output[1]))

    def test_batch_size_one_is_equivalent_in_train_and_eval(self):
        batch = encoder()
        personal = encoder("subject_specific")
        personal.load_state_dict(batch.state_dict(), strict=True)
        inputs = torch.randn(1, 6, 4)
        for training in (False, True):
            batch.train(training)
            personal.train(training)
            torch.manual_seed(67)
            batch_output, batch_attention = batch(inputs)
            torch.manual_seed(67)
            personal_output, personal_attention = personal(inputs)
            torch.testing.assert_close(batch_output, personal_output, rtol=0, atol=0)
            torch.testing.assert_close(batch_attention, personal_attention[0], rtol=0, atol=0)

    def test_gradients_reach_each_personal_attention(self):
        model = encoder("subject_specific").train()
        inputs = torch.randn(2, 6, 4, requires_grad=True)
        output, attention, mean_attention = model(inputs, return_batch_mean=True)
        attention.retain_grad()
        loss, _, _ = STFWindowLoss(node_count=4)(output, mean_attention, inputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(attention.grad).all())
        self.assertTrue((attention.grad.flatten(1).abs().sum(dim=1) > 0).all())
        for projection in (model.slf_attn.slf_attn.w_qs, model.slf_attn.slf_attn.w_ks):
            self.assertIsNotNone(projection.weight.grad)
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_identical_batch_mean_entropy_value_and_gradients(self):
        batch = encoder().train()
        personal = encoder("subject_specific").train()
        personal.load_state_dict(batch.state_dict(), strict=True)
        inputs = torch.randn(3, 6, 4)
        criterion = STFWindowLoss(node_count=4)
        torch.manual_seed(17)
        output_batch, _, mean_batch = batch(inputs, return_batch_mean=True)
        torch.manual_seed(17)
        output_personal, attention_personal, mean_personal = personal(inputs, return_batch_mean=True)
        entropy_batch = criterion(output_batch, mean_batch, inputs)[2]
        entropy_personal = criterion(output_personal, mean_personal, inputs)[2]
        torch.testing.assert_close(entropy_batch, entropy_personal, rtol=0, atol=0)
        entropy_batch.backward()
        entropy_personal.backward()
        for (_, expected), (_, actual) in zip(batch.named_parameters(), personal.named_parameters()):
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                torch.testing.assert_close(expected.grad, actual.grad, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            criterion(output_personal, attention_personal, inputs)

    def test_real_encoder_extracts_individual_90_by_90_ec_identically(self):
        batch = encoder(nodes=90).eval()
        personal = encoder("subject_specific", nodes=90).eval()
        personal.load_state_dict(batch.state_dict(), strict=True)
        rng = np.random.default_rng(9)
        series = [rng.normal(size=(8, 90)).astype(np.float32) for _ in range(2)]
        records = [SimpleNamespace(subject_id=f"synthetic-{index}") for index in range(2)]
        with contextlib.redirect_stdout(io.StringIO()):
            expected = extract_subject_ec(batch, records, series, 6, 2, "cpu")
            actual = extract_subject_ec(personal, records, series, 6, 2, "cpu")
        self.assertEqual(actual["ec"].shape, (2, 90, 90))
        np.testing.assert_array_equal(actual["ec"], expected["ec"])
        np.testing.assert_array_equal(np.diagonal(actual["ec"], axis1=1, axis2=2), 0)

    def test_extraction_preserves_window_mean_transpose_and_subject_isolation(self):
        class WindowAttention(torch.nn.Module):
            def __init__(self, personal):
                super().__init__()
                self.personal = personal
                self.batch_sizes = []

            def forward(self, window):
                self.batch_sizes.append(window.shape[0])
                attention = torch.arange(8100, dtype=window.dtype).reshape(90, 90) + window[0, 0, 0]
                return window, attention[None] if self.personal else attention

        series = [np.full((8, 90), value, dtype=np.float32) for value in (1, 100)]
        records = [SimpleNamespace(subject_id=str(index)) for index in range(2)]
        for personal in (False, True):
            model = WindowAttention(personal)
            with contextlib.redirect_stdout(io.StringIO()):
                extracted = extract_subject_ec(model, records, series, 6, 2, "cpu")
            self.assertEqual(model.batch_sizes, [1, 1, 1, 1])
            for index, value in enumerate((1, 100)):
                expected = (np.arange(8100).reshape(90, 90) + value).T.astype(np.float32)
                np.fill_diagonal(expected, 0)
                np.testing.assert_array_equal(extracted["ec"][index], expected)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, suffix="abide_ii", stage="pretrain"):
        args = parsed(suffix, "--input-mode", "ec" if stage == "downstream" else "raw",
                      "--stf-fusion-mode", "subject_specific", "--w7-stage", stage)
        args.output_dir = self.root / "subject_specific"
        args.ec_path = args.output_dir / "synthetic_initial.npz"
        args.refined_ec_path = args.output_dir / "synthetic_refined.npz"
        args.qsr_refined_ec_path = args.output_dir / "synthetic_qsr.npz"
        args.reference_ec_path = self.root / "original/initial.npz"
        args.reference_oof_path = self.root / "original/qsr.npz"
        args.gpu_id = "cpu"
        args.reference_ec_path.parent.mkdir(parents=True, exist_ok=True)
        with patch("sys.argv", ["synthetic-original"]):
            source = importlib.import_module(f"PR_EC.main_{suffix}").parse_args()
        source.input_mode = "raw"
        write_json(args.reference_ec_path.parent / "summary.json", {
            "config": jsonable(vars(source)), "stf_ec_training": {"stf_ec": {"epoch": source.epochs}},
        })
        data = cohort()
        np.savez_compressed(args.reference_ec_path, **{key: value for key, value in data.items() if key != "fold_ids"})
        np.savez_compressed(args.reference_oof_path, **data)
        return args, data

    def test_default_is_legacy_batch_mean_and_controlled_hyperparameters_unchanged(self):
        for suffix, epochs in (("abide_i", 81), ("abide_ii", 81), ("adhd200", 141)):
            legacy = parsed(suffix)
            self.assertEqual(legacy.stf_fusion_mode, "batch_mean")
            self.assertFalse(legacy.w7_enabled)
            self.assertEqual(legacy.output_dir, legacy.profile.output_dir)
            self.assertEqual(legacy.epochs, epochs)
            self.assertEqual(legacy.stf_checkpoint, "final")
            personal = parsed(suffix, "--input-mode", "raw", "--stf-fusion-mode", "subject_specific", "--w7-stage", "pretrain")
            changed = {"stf_fusion_mode", "w7_enabled", "w7_stage", "input_mode", "representations", "output_dir",
                       "ec_path", "refined_ec_path", "qsr_refined_ec_path", "reference_ec_path", "reference_oof_path"}
            for key, value in vars(legacy).items():
                if key not in changed:
                    self.assertEqual(value, getattr(personal, key), key)
            self.assertEqual(personal.representations, ["original", "refined", "qc_refined"])

    def test_wrong_stage_checkpoint_seed_other_ablations_and_outputs_rejected(self):
        cases = (
            ["--w7-stage", "pretrain"],
            ["--w7-stage", "downstream", "--seed", "1"],
            ["--w7-stage", "downstream", "--stf-checkpoint", "best"],
            ["--w7-stage", "downstream", "--qsr-disable-perturbation-restoration"],
            ["--w7-stage", "downstream", "--output-dir", "/tmp/not-w7"],
            ["--stf-fusion-mode", "subject_specific", "--ec-path", "PR_EC/outputs/abide-i/abide_subject_ec.npz"],
        )
        for arguments in cases:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parsed("abide_i", *arguments)

    def test_provenance_rejects_old_or_mismatched_ec_and_hyperparameters(self):
        args, data = self.fixture()
        with self.assertRaises(ValueError):
            validate_initial_ec_mode(args, data)
        data.update(initial_ec_metadata(args, {"epoch": args.epochs}))
        validate_initial_ec_mode(args, data)
        args.loss_alpha = 0.02
        with self.assertRaises(ValueError):
            validate_initial_ec_mode(args, data)
        with self.assertRaises(ValueError):
            audit_original_configuration(args)

    def test_cohort_and_splits_match_original_and_reject_changed_order(self):
        args, data = self.fixture()
        reference = validate_reference_subjects(args, data)
        folds = save_split_membership(args, data, reference)
        np.testing.assert_array_equal(folds, data["fold_ids"])
        shuffled = dict(data, subject_ids=data["subject_ids"][::-1])
        with self.assertRaises(ValueError):
            validate_reference_subjects(args, shuffled)
        changed_reference = dict(reference, fold_ids=reference["fold_ids"] % 10 + 1)
        with self.assertRaises(ValueError):
            save_split_membership(args, data, changed_reference)

    def test_pretrain_then_cached_downstream_without_real_training(self):
        args, source = self.fixture()
        subjects = {key: value for key, value in source.items() if key != "fold_ids"}
        subjects.update(records=[SimpleNamespace(subject_id=subject) for subject in source["subject_ids"]],
                        time_series=[np.zeros((8, 3), dtype=np.float32) for _ in source["subject_ids"]])
        generated = {**source, **initial_ec_metadata(args, {"epoch": args.epochs})}
        with patch.object(local_data, "load_subject_dataset", return_value=subjects), \
             patch("PR_EC_E2E.model.individual_ec.generate_subject_ec", return_value=(generated, {"epoch": args.epochs})) as generate, \
             patch.object(local_data, "subject_fc_features", side_effect=AssertionError("FC forbidden during pretrain-only")), \
             patch.object(local_data, "load_aligned_qc", side_effect=AssertionError("QC forbidden during pretrain-only")), \
             patch.object(w7_runner, "run_cross_validation", side_effect=AssertionError("CV forbidden during pretrain-only")), \
             patch.object(w7_runner, "set_seed"), contextlib.redirect_stdout(io.StringIO()):
            w7_runner.run_w7(args)
        generate.assert_called_once()
        self.assertTrue(args.ec_path.exists())
        self.assertTrue((args.output_dir / "pretraining_complete.json").exists())
        self.assertFalse((args.output_dir / "classification_folds.csv").exists())
        with self.assertRaises(FileExistsError):
            initialize_w7_run(args)
        args.input_mode, args.w7_stage = "ec", "downstream"
        metrics = {key: 0.5 for key in ("ACC", "AUC", "Recall", "Precision", "F1")}
        experiment = {"fold_ids": source["fold_ids"], "oof_pgr": source["ec"], "oof_qc": source["ec"],
                      "fold_results": [{representation: metrics for representation in args.representations} for _ in range(10)],
                      "refinement_metrics": []}
        with patch("PR_EC_E2E.model.individual_ec.generate_subject_ec", side_effect=AssertionError("Pretraining forbidden in cached mode")), \
             patch.object(local_data, "_aligned_graph_time_series", return_value=[]), \
             patch.object(local_data, "subject_fc_features", return_value=np.zeros((40, 3))), \
             patch.object(local_data, "load_phenotypes", return_value={"continuous": np.zeros((40, 2)), "categorical_raw": np.zeros((40, 1))}), \
             patch.object(local_data, "load_aligned_qc", return_value=np.zeros((40, 3))), \
             patch.object(local_data, "load_aligned_phenotypes", return_value=np.zeros((40, 4))), \
             patch.object(w7_runner, "run_cross_validation", return_value=experiment), \
             patch.object(w7_runner, "set_seed"), contextlib.redirect_stdout(io.StringIO()):
            w7_runner.run_w7(args)
        self.assertTrue((args.output_dir / "pretraining_config.json").exists())
        self.assertTrue((args.output_dir / "pretraining_metrics.json").exists())
        self.assertTrue((args.output_dir / "downstream_complete.json").exists())
        rows = summarize_table_r8.read_csv(args.output_dir / "classification_folds.csv")
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["ACC_pct"] == "50.0" for row in rows))
        measured = summarize_table_r8.measured_result(args.output_dir, "abide-ii")
        self.assertEqual(measured, ("50.00 ± 0.00",) * 3)
        with np.load(args.qsr_refined_ec_path, allow_pickle=False) as archive:
            for key in ("subject_ids", "labels", "site_ids", "fold_ids", "roi_names"):
                np.testing.assert_array_equal(archive[key], source[key])

    def test_downstream_function_is_the_original_not_e2e_alternative(self):
        self.assertIs(w7_runner.run_cross_validation, original_cross_validation)

    def test_output_protocol_blocks_cross_mode_or_hyperparameter_mixing(self):
        args, _ = self.fixture()
        initialize_w7_run(args)
        args.stf_fusion_mode = "batch_mean"
        with self.assertRaises(ValueError):
            initialize_w7_run(args)

    def test_table_r8_all_datasets_uses_only_completed_synthetic_outputs(self):
        table_root = self.root / "synthetic-w7"
        original_root = self.root
        for suffix, dataset in (("abide_i", "abide-i"), ("abide_ii", "abide-ii"), ("adhd200", "adhd200")):
            self.root = table_root / dataset
            args, source = self.fixture(suffix, stage="downstream")
            initialize_w7_run(args)
            generated = {**source, **initial_ec_metadata(args, {"epoch": args.epochs})}
            np.savez_compressed(args.ec_path, **generated)
            save_qsr_ec_archive(args.qsr_refined_ec_path, generated, generated["ec"], source["fold_ids"], args.ec_path)
            metrics = [{representation: {name: 0.5 + fold / 100 for name in ("ACC", "AUC", "Recall", "Precision", "F1")}
                        for representation in args.representations} for fold in range(10)]
            save_w7_classification(args, metrics)
            write_json(args.output_dir / "downstream_complete.json", {
                "stf_fusion_mode": args.stf_fusion_mode, "seed": 42, "n_splits": 10,
            })
        self.root = original_root
        output = table_root / "synthetic-table-r8.csv"
        with patch.object(summarize_table_r8, "OUTPUT_ROOT", table_root), \
             patch("sys.argv", ["synthetic-summary", "--root", str(table_root), "--output", str(output)]), \
             contextlib.redirect_stdout(io.StringIO()):
            summarize_table_r8.main()
        rows = summarize_table_r8.read_csv(output)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["Initial EC AUC (%)"], "66.20 ± 4.32")
        self.assertEqual(rows[1]["Initial EC AUC (%)"], "54.50 ± 2.87")
        self.assertTrue(output.with_suffix(".sources.json").exists())
        incomplete = table_root / "missing"
        with self.assertRaises(FileNotFoundError):
            summarize_table_r8.measured_result(incomplete, "abide-i")

    def test_output_writer_preserves_configuration_and_other_artifacts(self):
        args, _ = self.fixture()
        args.output_dir.mkdir(parents=True)
        sentinel = args.output_dir / "pretraining_config.json"
        sentinel.write_text('{"synthetic": true}')
        metrics = {name: 0.5 for name in ("ACC", "AUC", "Recall", "Precision", "F1")}
        save_results(args, [{"original": metrics, "qc_refined": metrics}], {})
        self.assertEqual(sentinel.read_text(), '{"synthetic": true}')


if __name__ == "__main__":
    unittest.main()
