"""Independent, cache-only E2E comparison using the original PR-EC protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PR_EC.data import _aligned_graph_time_series, _labels_from_abide_phenotype
from PR_EC.data.adhd200 import prepare_adhd_fold_arrays
from PR_EC.data.common import load_ec_archive
from PR_EC.dataset_configs import get_profile
from PR_EC.downstream.metrics import classification_metrics, select_youden_threshold
from PR_EC.model.mpr import load_phenotypes, subject_fc_features
from PR_EC.utils import make_stratified_splits, prepare_fold_arrays, select_device
from PR_EC.workflow import build_fold_reference
from PR_EC_E2E.downstream.neighbor_attention import fit_neighbor_classifier, predict

DATASETS = {"abide-i": "abide", "abide-ii": "abide_ii", "adhd200": "adhd200"}
METRICS = ("ACC", "Recall", "AUC", "Precision", "F1")
METHODS = {"Initial EC": "original", "MPR-only": "refined", "Full PR-EC": "qc_refined"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def source_args(dataset):
    source = ROOT / "PR_EC" / "outputs" / dataset / "summary.json"
    summary = json.loads(source.read_text())
    config = dict(summary["config"])
    for name in ("ec_path", "data_root", "phenotype_csv", "refined_ec_path",
                 "qsr_refined_ec_path", "output_dir"):
        config[name] = Path(config[name])
    config["profile"] = get_profile(DATASETS[dataset])
    config["input_mode"] = "ec"
    if config["n_splits"] != 10 or config["graph_mode"] != "fusion":
        raise ValueError("Expected the original ten-fold PH+FC protocol")
    return SimpleNamespace(**config), summary, source


def load_data(args):
    data = load_ec_archive(args.ec_path)
    if args.profile.name == "abide":
        data["labels"] = _labels_from_abide_phenotype(
            args.phenotype_csv, data["subject_ids"], args.profile,
        )
    data.update(load_phenotypes(
        args.phenotype_csv, data["subject_ids"], data["site_ids"], args.profile,
    ))
    data["fmri_features"] = subject_fc_features(_aligned_graph_time_series(args, data, None))
    data["ec"] = np.asarray(data["ec"], dtype=np.float32)
    if not np.isfinite(data["ec"]).all() or len(set(data["subject_ids"])) != len(data["ec"]):
        raise ValueError("Initial EC must be finite and subject IDs unique")
    return data


def candidates(weights, neighbors):
    count = min(neighbors, weights.shape[1])
    indices = np.argsort(-weights, axis=1, kind="stable")[:, :count].copy()
    mask = np.take_along_axis(weights, indices, axis=1) > 0
    recovered = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(recovered, indices, mask, axis=1)
    if not np.array_equal(recovered, weights > 0) or not mask.any(axis=1).all():
        raise RuntimeError("Candidate support differs from the original MPR graph")
    return indices, mask


def prepare(args, data, splits):
    prepare_arrays = prepare_adhd_fold_arrays if args.profile.name == "adhd200" else prepare_fold_arrays
    arrays = prepare_arrays(*(
        data[key][index] for key in ("ec", "continuous", "categorical_raw") for index in splits
    ))
    reference = build_fold_reference(
        args, arrays, tuple(data["fmri_features"][index] for index in splits),
    )
    candidate_sets = {
        split: candidates(reference[f"{split}_weights"], args.reference_k)
        for split in ("train", "val", "test")
    }
    return arrays, reference, candidate_sets


def leakage_audit(args, data, splits, arrays, reference, candidate_sets):
    train, validation, test = splits
    if any(np.intersect1d(first, second).size for first, second in (
        (train, validation), (train, test), (validation, test),
    )):
        raise RuntimeError("Split overlap")
    if np.any(np.diag(reference["train_weights"]) != 0):
        raise RuntimeError("Training subjects retrieve themselves")
    for split, query in zip(("train", "val", "test"), splits):
        indices, mask = candidate_sets[split]
        if not np.isin(train[indices[mask]], train).all():
            raise RuntimeError("Non-training neighbor")
        if split == "train" and np.any((train[indices] == query[:, None]) & mask):
            raise RuntimeError("Self neighbor")
    changed = dict(data)
    changed["ec"] = data["ec"].copy()
    changed["continuous"] = data["continuous"].copy()
    changed["categorical_raw"] = data["categorical_raw"].copy()
    changed["fmri_features"] = data["fmri_features"].copy()
    heldout = np.concatenate((validation, test))
    changed["ec"][heldout] = 100
    changed["continuous"][heldout] = 100
    changed["categorical_raw"][heldout] = "__AUDIT__"
    changed["fmri_features"][heldout] = 100
    changed_arrays, changed_reference, _ = prepare(args, changed, splits)
    for name in ("train_ec", "train_cont", "train_cat", "ec_mean", "ec_std"):
        if not np.array_equal(arrays[name], changed_arrays[name]):
            raise RuntimeError(f"Held-out data influences {name}")
    if not np.array_equal(reference["train_weights"], changed_reference["train_weights"]):
        raise RuntimeError("Held-out data influences training candidates")
    return {"disjoint_splits": True, "train_only_neighbors": True,
            "no_self_neighbors": True, "candidate_support_matches_mpr": True,
            "heldout_feature_noninterference": True,
            "fit_interface_excludes_test_inputs_and_labels": True}


def run_fold(args, data, fold, splits, output, device, hidden_dim):
    train, validation, test = splits
    arrays, reference, candidate_sets = prepare(args, data, splits)
    audit = leakage_audit(args, data, splits, arrays, reference, candidate_sets)
    fold_dir = output / f"fold_{fold:02d}"
    fold_dir.mkdir()
    np.savez_compressed(
        fold_dir / "neighbors.npz", train_index=train, val_index=validation, test_index=test,
        ec_mean=arrays["ec_mean"], ec_std=arrays["ec_std"],
        **{f"{split}_neighbor_index": train[indices]
           for split, (indices, mask) in candidate_sets.items()},
        **{f"{split}_neighbor_mask": mask for split, (indices, mask) in candidate_sets.items()},
        **{f"{split}_mpr_weights": reference[f"{split}_weights"] for split in candidate_sets},
    )
    runs, probabilities, thresholds = [], [], []
    for repeat in range(args.classifier_repeats):
        seed = args.seed + fold * 1000 + repeat + 1
        model, bank, training_audit = fit_neighbor_classifier(
            arrays["train_ec"], data["labels"][train],
            arrays["val_ec"], data["labels"][validation],
            *candidate_sets["train"], *candidate_sets["val"],
            device=device, seed=seed, max_epochs=args.classifier_epochs,
            patience=args.classifier_patience, batch_size=args.batch_size,
            learning_rate=args.classifier_lr, hidden_dim=hidden_dim,
        )
        validation_prob = predict(
            model, arrays["val_ec"], *candidate_sets["val"], bank, device, args.batch_size,
        ).sigmoid().numpy()
        threshold = select_youden_threshold(data["labels"][validation], validation_prob)
        test_prob = predict(
            model, arrays["test_ec"], *candidate_sets["test"], bank, device, args.batch_size,
        ).sigmoid().numpy()
        metrics = classification_metrics(data["labels"][test], test_prob, threshold)
        runs.append(metrics)
        probabilities.append(test_prob)
        thresholds.append(threshold)
        torch.save(model.cpu().state_dict(), fold_dir / f"model_repeat_{repeat + 1}.pt")
        write_json(fold_dir / f"audit_repeat_{repeat + 1}.json", {
            "seed": seed, "leakage": audit, "training": training_audit,
            "validation_threshold": threshold, "test_metrics": metrics,
        })
        print(f"  fold={fold} repeat={repeat + 1} best_epoch={training_audit['best_epoch']} "
              f"epochs={training_audit['epochs_run']} AUC={metrics['AUC']:.4f}", flush=True)
    np.savez_compressed(
        fold_dir / "predictions.npz", test_index=test, subject_ids=data["subject_ids"][test],
        labels=data["labels"][test], test_probabilities=np.asarray(probabilities),
        thresholds=np.asarray(thresholds),
    )
    result = {"fold": fold, **{key: float(np.mean([run[key] for run in runs])) for key in METRICS}}
    write_json(fold_dir / "metrics.json", result)
    return result


def summarize(output, original, all_splits):
    results = []
    for fold, *_ in all_splits:
        path = output / f"fold_{fold:02d}" / "metrics.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    original_folds = {row["fold"]: row for row in original["folds"]}
    rows = []
    for row in results:
        source = original_folds[row["fold"]]
        rows.append({**row, "Full_PR_EC_AUC": source["qc_refined_AUC"],
                     "E2E_minus_Full_PR_EC_AUC": row["AUC"] - source["qc_refined_AUC"]})
    summary = {"fold_count": len(rows), "complete_ten_fold": len(rows) == 10,
               "std_ddof": 0, "comparison_source": "archived PR_EC fold metrics; not retrained",
               "folds": rows, "methods": {}}
    for name, prefix in {**METHODS, "E2E Neighbor Aggregation": None}.items():
        values = {}
        for metric in METRICS:
            samples = [row[metric] if prefix is None else
                       original_folds[row["fold"]][f"{prefix}_{metric}"] for row in results]
            values["SEN" if metric == "Recall" else metric] = {
                "mean": float(np.mean(samples)), "std": float(np.std(samples)),
            }
        summary["methods"][name] = values
    deltas = [row["E2E_minus_Full_PR_EC_AUC"] for row in rows]
    summary["paired_auc_delta"] = {"direction": "E2E minus Full PR-EC",
                                   "mean": float(np.mean(deltas)), "std": float(np.std(deltas))}
    write_json(output / "summary.json", summary)
    with (output / "fold_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# E2E comparison ({len(rows)}/10 folds)", "", "Values: mean ± population SD (%); SEN = Recall.", "",
             "| Method | ACC | SEN | AUC | Precision | F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, values in summary["methods"].items():
        lines.append("| " + name + " | " + " | ".join(
            f"{100 * value['mean']:.2f} ± {100 * value['std']:.2f}" for value in values.values()
        ) + " |")
    lines.extend(["", f"Paired ΔAUC (E2E − Full PR-EC): {100 * np.mean(deltas):+.2f} ± {100 * np.std(deltas):.2f} percentage points.",
                  "", "Controls use archived fold metrics; Full PR-EC probabilities were not saved in the source experiment.",
                  "Fold membership is checked against the original OOF archive; no test-driven selection is used."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-root", type=Path, default=ROOT / "PR_EC_E2E/outputs/neighbor_attention_v1")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 11))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    options = parser.parse_args()
    if options.threads < 1 or options.hidden_dim < 1:
        parser.error("threads and hidden-dim must be positive")
    if (ROOT / "PR_EC/outputs").resolve() in (options.output_root.resolve(), *options.output_root.resolve().parents):
        parser.error("Output must not be inside the original experiment outputs")
    torch.set_num_threads(options.threads)
    device = select_device(options.gpu_id)
    for dataset in options.datasets:
        args, original, source = source_args(dataset)
        output = options.output_root / dataset
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"dataset": dataset, "source_summary": str(source),
                    "source_summary_sha256": digest(source), "initial_ec_sha256": digest(args.ec_path),
                    "phenotype_sha256": digest(args.phenotype_csv),
                    "source_config": original["config"], "hidden_dim": options.hidden_dim,
                    "device": str(device), "torch_version": torch.__version__, "threads": options.threads,
                    "implementation_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
                        Path(__file__), Path(__file__).parent / "downstream/neighbor_attention.py")},
                    "qc_used": False, "stf_training": False,
                    "model_selection": "minimum validation BCE; threshold from validation Youden"}
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not options.resume or json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Refusing to overwrite or mix configurations: {output}")
        else:
            write_json(manifest_path, manifest)
        print(f"Dataset={dataset} device={device}; loading existing Initial EC, phenotype and FC only", flush=True)
        data = load_data(args)
        splits = list(make_stratified_splits(data["labels"], args.n_splits, args.seed, args.validation_size))
        with np.load(args.refined_ec_path) as archive:
            for key in ("subject_ids", "labels"):
                if not np.array_equal(data[key], archive[key]):
                    raise RuntimeError(f"Original OOF {key} mismatch")
            for fold, train, validation, test in splits:
                if not np.array_equal(np.flatnonzero(archive["fold_ids"] == fold), np.sort(test)):
                    raise RuntimeError(f"Original test fold {fold} mismatch")
        for fold, train, validation, test in splits:
            if options.folds and fold not in options.folds:
                continue
            fold_dir = output / f"fold_{fold:02d}"
            if fold_dir.exists():
                if options.resume and (fold_dir / "metrics.json").exists():
                    print(f"Reusing completed {dataset} fold {fold}", flush=True)
                    continue
                raise RuntimeError(f"Incomplete or existing fold: {fold_dir}; use a fresh output root")
            run_fold(args, data, fold, (train, validation, test), output, device, options.hidden_dim)
        summarize(output, original, splits)


if __name__ == "__main__":
    main()
