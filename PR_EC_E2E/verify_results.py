"""Recompute saved metrics and audit all completed ten-fold comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PR_EC.downstream.metrics import classification_metrics
from PR_EC.utils.folds import fit_ec_scaler, make_stratified_splits
from PR_EC_E2E.run_e2e import DATASETS, METRICS, digest, write_json


def verify_dataset(output):
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(manifest["source_summary"])
    assert digest(source) == manifest["source_summary_sha256"], "Source results changed"
    config = manifest["source_config"]
    assert digest(Path(config["ec_path"])) == manifest["initial_ec_sha256"], "Initial EC changed"
    assert digest(Path(config["phenotype_csv"])) == manifest["phenotype_sha256"], "Phenotype changed"
    for path, checksum in manifest["implementation_sha256"].items():
        assert digest(ROOT / path) == checksum, f"Implementation changed: {path}"
    assert not manifest["qc_used"] and not manifest["stf_training"]
    original = json.loads(source.read_text())
    original_folds = {row["fold"]: row for row in original["folds"]}
    with np.load(config["ec_path"]) as archive:
        initial_ec = archive["ec"]
        subject_ids = archive["subject_ids"]
    with np.load(config["refined_ec_path"]) as archive:
        labels = archive["labels"]
        source_fold_ids = archive["fold_ids"]
        assert np.array_equal(subject_ids, archive["subject_ids"])
    splits = make_stratified_splits(labels, config["n_splits"], config["seed"], config["validation_size"])
    covered = np.zeros(len(labels), dtype=np.int64)
    repeat_probabilities = np.full((config["classifier_repeats"], len(labels)), np.nan)
    fold_ids = np.zeros(len(labels), dtype=np.int64)
    rows = []
    for fold, train, validation, test in splits:
        fold_dir = output / f"fold_{fold:02d}"
        row = json.loads((fold_dir / "metrics.json").read_text())
        assert row["fold"] == fold
        with np.load(fold_dir / "neighbors.npz") as archive:
            for name, expected in (("train", train), ("val", validation), ("test", test)):
                assert np.array_equal(archive[f"{name}_index"], expected)
                indices = archive[f"{name}_neighbor_index"]
                mask = archive[f"{name}_neighbor_mask"]
                assert np.isin(indices[mask], train).all()
                assert not np.isin(indices[mask], np.concatenate((validation, test))).any()
                assert mask.any(axis=1).all()
                if name == "train":
                    assert not ((indices == train[:, None]) & mask).any()
                support = archive[f"{name}_mpr_weights"] > 0
                global_support = np.zeros((len(expected), len(labels)), dtype=bool)
                global_support[:, train] = support
                reconstructed = np.zeros_like(global_support)
                np.put_along_axis(reconstructed, indices, mask, axis=1)
                assert np.array_equal(global_support, reconstructed)
            mean, std = fit_ec_scaler(initial_ec[train])
            assert np.array_equal(mean, archive["ec_mean"])
            assert np.array_equal(std, archive["ec_std"])
        assert np.array_equal(np.flatnonzero(source_fold_ids == fold), np.sort(test))
        with np.load(fold_dir / "predictions.npz") as archive:
            assert np.array_equal(archive["test_index"], test)
            assert np.array_equal(archive["subject_ids"], subject_ids[test])
            assert np.array_equal(archive["labels"], labels[test])
            probabilities = archive["test_probabilities"]
            assert probabilities.shape == (config["classifier_repeats"], len(test))
            assert np.isfinite(probabilities).all()
            assert ((probabilities >= 0) & (probabilities <= 1)).all()
            recomputed = []
            for repeat in range(config["classifier_repeats"]):
                audit = json.loads((fold_dir / f"audit_repeat_{repeat + 1}.json").read_text())
                assert all(audit["leakage"].values())
                training = audit["training"]
                assert all(np.isfinite(value) and value > 0 for value in training["gradient_norms"].values())
                assert training["attention_weight_max_change"] > 0
                assert training["epochs_run"] <= config["classifier_epochs"]
                best = min(training["history"], key=lambda entry: entry["validation_bce"])
                assert best["epoch"] == training["best_epoch"]
                assert best["validation_bce"] == training["best_validation_bce"]
                assert audit["validation_threshold"] == archive["thresholds"][repeat]
                metrics = classification_metrics(labels[test], probabilities[repeat], archive["thresholds"][repeat])
                assert metrics == audit["test_metrics"]
                recomputed.append(metrics)
            for metric in METRICS:
                assert np.isclose(row[metric], np.mean([entry[metric] for entry in recomputed]), atol=1e-12)
            repeat_probabilities[:, test] = probabilities
        covered[test] += 1
        fold_ids[test] = fold
        rows.append(row)
    assert (covered == 1).all() and np.isfinite(repeat_probabilities).all()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete_ten_fold"] and summary["fold_count"] == 10
    for metric in METRICS:
        key = "SEN" if metric == "Recall" else metric
        values = [row[metric] for row in rows]
        assert np.isclose(summary["methods"]["E2E Neighbor Aggregation"][key]["mean"], np.mean(values))
        assert np.isclose(summary["methods"]["E2E Neighbor Aggregation"][key]["std"], np.std(values))
    for row in summary["folds"]:
        assert row["Full_PR_EC_AUC"] == original_folds[row["fold"]]["qc_refined_AUC"]
        assert row["E2E_minus_Full_PR_EC_AUC"] == row["AUC"] - row["Full_PR_EC_AUC"]
    np.savez_compressed(output / "oof_predictions.npz", subject_ids=subject_ids, labels=labels,
                        fold_ids=fold_ids, test_probabilities=repeat_probabilities)
    return {"dataset": manifest["dataset"], "folds": 10, "subjects": len(labels),
            "source_unchanged": True, "oof_coverage_exactly_once": True,
            "saved_metrics_recomputed": True, "train_only_neighbors": True,
            "standardization_train_only": True, "joint_gradients_all_folds": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "PR_EC_E2E/outputs/neighbor_attention_v1")
    options = parser.parse_args()
    report = [verify_dataset(options.output_root / dataset) for dataset in DATASETS]
    write_json(options.output_root / "verification.json", report)
    lines = ["# Ten-fold comparison", "", "Values: mean ± population SD (%).", "",
             "| Dataset | Method | ACC | SEN | AUC | Precision | F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        summary = json.loads((options.output_root / dataset / "summary.json").read_text())
        for name, metrics in summary["methods"].items():
            lines.append("| " + dataset + " | " + name + " | " + " | ".join(
                f"{100 * value['mean']:.2f} ± {100 * value['std']:.2f}" for value in metrics.values()
            ) + " |")
    lines.extend(["", "## Fold-paired AUC differences", "", "Direction: E2E minus Full PR-EC (percentage points).", ""])
    for dataset in DATASETS:
        summary = json.loads((options.output_root / dataset / "summary.json").read_text())
        delta = summary["paired_auc_delta"]
        lines.append(f"- {dataset}: {100 * delta['mean']:+.2f} ± {100 * delta['std']:.2f}")
    lines.extend(["", "Control results are archived PR-EC fold metrics, not newly retrained controls.",
                  "Full PR-EC subject probabilities are unavailable in the source results; no DeLong test is claimed.",
                  "All saved E2E probabilities reproduce the reported metrics; each subject is held out once."])
    (options.output_root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
