"""Fold-local neighborhood-stratified downstream performance analysis."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from PR_EC_MPR.data import load_pipeline_data
from PR_EC_MPR.data.adhd200 import (
    apply_numeric_imputer,
    fit_numeric_imputer,
    prepare_adhd_fold_arrays,
)
from PR_EC_MPR.downstream.classifier import train_classifier
from PR_EC_MPR.model.ablation.mpr_refinement import (
    apply_pgr_refiner,
    train_pgr_refiner,
)
from PR_EC_MPR.model.qsr.qsr_refiner import apply_qsr_refiner, train_qsr_refiner
from PR_EC_MPR.utils.folds import make_stratified_splits, prepare_fold_arrays
from PR_EC_MPR.utils.runtime import set_seed
from PR_EC_MPR.workflow import build_fold_reference


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = Path(__file__).resolve().parent
DATASET_MODULES = {
    "abide-i": "PR_EC_MPR.main_abide_i",
    "abide-ii": "PR_EC_MPR.main_abide_ii",
    "adhd200": "PR_EC_MPR.main_adhd200",
}
ARCHIVE_NAMES = {
    "abide-i": "abide_subject_ec.npz",
    "abide-ii": "abide_ii_subject_ec.npz",
    "adhd200": "adhd200_subject_ec.npz",
}
REPRESENTATION_NAMES = ("initial_ec", "mpr_only", "full_pr_ec")
GROUP_ORDER = (
    "neighbor_match_high",
    "neighbor_match_low",
    "neighbor_motion_low",
    "neighbor_motion_high",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_MODULES, "all"],
        default="all",
    )
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--output-root", type=Path, default=ANALYSIS_ROOT / "outputs")
    return parser.parse_args()


def require_cuda(gpu_id):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError(
            "CUDA is required for this analysis; CPU fallback is intentionally disabled."
        )
    if gpu_id == "auto":
        free_memory = [
            torch.cuda.mem_get_info(index)[0]
            for index in range(torch.cuda.device_count())
        ]
        gpu_id = int(np.argmax(free_memory))
    device = torch.device(f"cuda:{int(gpu_id)}")
    torch.cuda.set_device(device)
    return device


def load_original_config(dataset_key):
    summary_path = ROOT / "PR_EC_MPR" / "outputs" / dataset_key / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing original summary: {summary_path}")
    with summary_path.open(encoding="utf-8") as handle:
        config = json.load(handle)["config"]
    module = importlib.import_module(DATASET_MODULES[dataset_key])
    profile = module.DATASET_CONFIG
    config["profile"] = profile
    config["dataset"] = profile.name
    config["input_mode"] = "ec"
    config["representations"] = ["original", "refined", "qc_refined"]
    config["ec_path"] = (
        ROOT / "PR_EC_MPR" / "outputs" / dataset_key / ARCHIVE_NAMES[dataset_key]
    )
    config["phenotype_csv"] = profile.phenotype_path
    config["data_root"] = profile.data_root
    config["output_dir"] = ROOT / "PR_EC_MPR" / "outputs" / dataset_key
    config["asd_label"] = config["patient_label"]
    config["tc_label"] = config["control_label"]
    return SimpleNamespace(**config), summary_path


def standardized_fc_similarity(train_features, query_features):
    train_features = np.asarray(train_features, dtype=np.float32)
    query_features = np.asarray(query_features, dtype=np.float32)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    train_scaled = (train_features - mean) / std
    query_scaled = (query_features - mean) / std
    train_norm = train_scaled / np.maximum(
        np.linalg.norm(train_scaled, axis=1, keepdims=True), 1e-8
    )
    query_norm = query_scaled / np.maximum(
        np.linalg.norm(query_scaled, axis=1, keepdims=True), 1e-8
    )
    return np.clip(query_norm @ train_norm.T, 0.0, 1.0).astype(np.float32)


def weighted_metric(weights, values):
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    denominator = weights.sum(axis=1)
    if np.any(denominator <= 0):
        raise RuntimeError("At least one neighborhood has zero final aggregation weight")
    return ((weights * values).sum(axis=1) / denominator).astype(np.float32)


def mean_edge_variance(ec):
    ec = np.asarray(ec, dtype=np.float64)
    if ec.ndim != 3 or ec.shape[1] != ec.shape[2] or len(ec) < 2:
        return float("nan")
    edge_variance = np.var(ec, axis=0)
    mask = ~np.eye(ec.shape[1], dtype=bool)
    return float(edge_variance[mask].mean())


def safe_auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return float("nan"), "empty_group"
    if np.unique(labels).size < 2:
        return float("nan"), "single_class_group"
    return float(roc_auc_score(labels, probabilities)), ""


def make_neighbor_rows(
    dataset_key,
    fold,
    split,
    query_indices,
    train_indices,
    subject_ids,
    weights,
    similarities,
    train_fd_raw,
    train_fd_filled,
):
    rows = []
    for query_row, query_index in enumerate(query_indices):
        neighbor_rows = np.flatnonzero(weights[query_row] > 0)
        for rank, neighbor_row in enumerate(
            neighbor_rows[np.argsort(weights[query_row, neighbor_rows])[::-1]], 1
        ):
            neighbor_index = train_indices[neighbor_row]
            rows.append(
                {
                    "dataset": dataset_key,
                    "fold": fold,
                    "split": split,
                    "query_subject_id": str(subject_ids[query_index]),
                    "neighbor_rank": rank,
                    "neighbor_subject_id": str(subject_ids[neighbor_index]),
                    "aggregation_weight": float(weights[query_row, neighbor_row]),
                    "fc_similarity_pre_normalization": float(
                        similarities[query_row, neighbor_row]
                    ),
                    "neighbor_mean_fd_raw": float(train_fd_raw[neighbor_row])
                    if np.isfinite(train_fd_raw[neighbor_row])
                    else np.nan,
                    "neighbor_mean_fd_fold_imputed": float(
                        train_fd_filled[neighbor_row]
                    ),
                }
            )
    return rows


def train_representations(args, arrays, reference, data, train_index, val_index, test_index, device, fold):
    fold_seed = args.seed + fold * 1000
    set_seed(fold_seed)
    pgr_model, train_pgr, _ = train_pgr_refiner(
        args, arrays["train_ec"], reference["train_neighbor"], device
    )
    val_pgr = apply_pgr_refiner(
        pgr_model, arrays["val_ec"], reference["val_neighbor"], device
    )
    test_pgr = apply_pgr_refiner(
        pgr_model, arrays["test_ec"], reference["test_neighbor"], device
    )

    if args.profile.name == "adhd200":
        categorical_indices = tuple(
            index
            for index, column in enumerate(args.profile.confound_columns)
            if column in ("Gender",)
        )
        confound_fills = fit_numeric_imputer(
            data["qsr_confound_values"][train_index], categorical_indices
        )
        qsr_train_confound = apply_numeric_imputer(
            data["qsr_confound_values"][train_index], confound_fills
        )
    else:
        qsr_train_confound = data["qsr_confound_values"][train_index]

    set_seed(fold_seed + 1)
    qsr_model, train_qsr, sensitive_map, _ = train_qsr_refiner(
        args,
        arrays["train_ec"],
        reference["train_neighbor"],
        data["qsr_qc"][train_index],
        qsr_train_confound,
        data["site_ids"][train_index],
        device,
        fold_seed + 1,
    )
    val_qsr = apply_qsr_refiner(
        qsr_model,
        arrays["val_ec"],
        reference["val_neighbor"],
        sensitive_map,
        device,
    )
    test_qsr = apply_qsr_refiner(
        qsr_model,
        arrays["test_ec"],
        reference["test_neighbor"],
        sensitive_map,
        device,
    )
    return {
        "initial_ec": {
            "train": arrays["train_ec"],
            "val": arrays["val_ec"],
            "test": arrays["test_ec"],
        },
        "mpr_only": {"train": train_pgr, "val": val_pgr, "test": test_pgr},
        "full_pr_ec": {"train": train_qsr, "val": val_qsr, "test": test_qsr},
    }


def predict_representations(args, representations, labels, device, fold):
    probability_runs = {name: [] for name in REPRESENTATION_NAMES}
    for repeat in range(args.classifier_repeats):
        classifier_seed = args.seed + fold * 1000 + repeat + 1
        for name in REPRESENTATION_NAMES:
            _, _, predictions = train_classifier(
                representations[name]["train"],
                labels["train"],
                representations[name]["val"],
                labels["val"],
                representations[name]["test"],
                labels["test"],
                device=device,
                seed=classifier_seed,
                max_epochs=args.classifier_epochs,
                patience=args.classifier_patience,
                batch_size=args.batch_size,
                learning_rate=args.classifier_lr,
                return_predictions=True,
            )
            probability_runs[name].append(predictions["test_probabilities"])
    return {
        name: np.mean(np.stack(runs, axis=0), axis=0).astype(np.float32)
        for name, runs in probability_runs.items()
    }


def group_definitions(match_values, motion_values, match_threshold, motion_threshold):
    return {
        "neighbor_match_high": match_values >= match_threshold,
        "neighbor_match_low": match_values < match_threshold,
        "neighbor_motion_low": motion_values < motion_threshold,
        "neighbor_motion_high": motion_values >= motion_threshold,
    }


def analyze_fold(args, data, fold, train_index, val_index, test_index, device, dataset_key):
    prepare = prepare_adhd_fold_arrays if args.profile.name == "adhd200" else prepare_fold_arrays
    arrays = prepare(
        data["ec"][train_index],
        data["ec"][val_index],
        data["ec"][test_index],
        data["continuous"][train_index],
        data["continuous"][val_index],
        data["continuous"][test_index],
        data["categorical_raw"][train_index],
        data["categorical_raw"][val_index],
        data["categorical_raw"][test_index],
    )
    fmri_arrays = tuple(
        data["fmri_features"][indices]
        for indices in (train_index, val_index, test_index)
    )
    reference = build_fold_reference(args, arrays, fmri_arrays)
    if np.any(np.diag(reference["train_weights"]) != 0):
        raise RuntimeError(f"Fold {fold}: training self-neighbor leakage detected")
    for split in ("train", "val", "test"):
        if reference[f"{split}_weights"].shape[1] != len(train_index):
            raise RuntimeError(f"Fold {fold}: {split} neighbors do not index the training fold")

    train_similarity = standardized_fc_similarity(fmri_arrays[0], fmri_arrays[0])
    test_similarity = standardized_fc_similarity(fmri_arrays[0], fmri_arrays[2])
    np.fill_diagonal(train_similarity, 0.0)
    train_match = weighted_metric(reference["train_weights"], train_similarity)
    test_match = weighted_metric(reference["test_weights"], test_similarity)

    mean_fd_column = tuple(args.qsr_qc_columns).index("func_mean_fd")
    train_fd_raw = np.asarray(
        data["qsr_qc"][train_index, mean_fd_column], dtype=np.float64
    )
    finite_fd = train_fd_raw[np.isfinite(train_fd_raw)]
    if not len(finite_fd):
        raise RuntimeError(f"Fold {fold}: no finite training mean FD values")
    fd_fill = float(np.median(finite_fd))
    train_fd_filled = np.where(np.isfinite(train_fd_raw), train_fd_raw, fd_fill)
    train_motion = weighted_metric(reference["train_weights"], train_fd_filled[None, :])
    test_motion = weighted_metric(reference["test_weights"], train_fd_filled[None, :])
    match_threshold = float(np.median(train_match))
    motion_threshold = float(np.median(train_motion))

    representations = train_representations(
        args, arrays, reference, data, train_index, val_index, test_index, device, fold
    )
    labels = {
        "train": data["labels"][train_index],
        "val": data["labels"][val_index],
        "test": data["labels"][test_index],
    }
    probabilities = predict_representations(args, representations, labels, device, fold)
    groups = group_definitions(
        test_match, test_motion, match_threshold, motion_threshold
    )

    subject_rows = []
    for row, subject_index in enumerate(test_index):
        subject_rows.append(
            {
                "subject_id": str(data["subject_ids"][subject_index]),
                "dataset": dataset_key,
                "fold": fold,
                "true_label": int(data["labels"][subject_index]),
                "neighbor_match": float(test_match[row]),
                "neighbor_motion_mean_fd": float(test_motion[row]),
                "match_threshold_train_median": match_threshold,
                "motion_threshold_train_median": motion_threshold,
                "match_group": "high" if groups["neighbor_match_high"][row] else "low",
                "motion_group": "high" if groups["neighbor_motion_high"][row] else "low",
                "initial_ec_probability": float(probabilities["initial_ec"][row]),
                "mpr_only_probability": float(probabilities["mpr_only"][row]),
                "full_pr_ec_probability": float(probabilities["full_pr_ec"][row]),
            }
        )

    fold_rows = []
    for group_name, mask in groups.items():
        group_labels = labels["test"][mask]
        aucs = {}
        reason = ""
        for name in REPRESENTATION_NAMES:
            aucs[name], current_reason = safe_auc(group_labels, probabilities[name][mask])
            reason = reason or current_reason
        variances = {
            name: mean_edge_variance(representations[name]["test"][mask])
            for name in REPRESENTATION_NAMES
        }
        initial_variance = variances["initial_ec"]
        fold_rows.append(
            {
                "dataset": dataset_key,
                "fold": fold,
                "group": group_name,
                "n_subjects": int(mask.sum()),
                "n_patients": int(np.sum(group_labels == args.patient_label)),
                "n_controls": int(np.sum(group_labels == args.control_label)),
                "initial_ec_auc": aucs["initial_ec"],
                "mpr_only_auc": aucs["mpr_only"],
                "full_pr_ec_auc": aucs["full_pr_ec"],
                "mpr_minus_initial_auc": aucs["mpr_only"] - aucs["initial_ec"],
                "full_pr_ec_minus_initial_auc": aucs["full_pr_ec"] - aucs["initial_ec"],
                "initial_ec_mean_edge_variance": variances["initial_ec"],
                "mpr_only_mean_edge_variance": variances["mpr_only"],
                "full_pr_ec_mean_edge_variance": variances["full_pr_ec"],
                "mpr_variance_retention": variances["mpr_only"] / initial_variance
                if np.isfinite(initial_variance) and initial_variance > 0
                else np.nan,
                "full_pr_ec_variance_retention": variances["full_pr_ec"] / initial_variance
                if np.isfinite(initial_variance) and initial_variance > 0
                else np.nan,
                "auc_unavailable_reason": reason,
                "train_match_threshold": match_threshold,
                "train_motion_threshold": motion_threshold,
                "train_mean_fd_missing_count": int(np.sum(~np.isfinite(train_fd_raw))),
                "train_mean_fd_imputation_median": fd_fill,
            }
        )

    neighbor_rows = make_neighbor_rows(
        dataset_key,
        fold,
        "train",
        train_index,
        train_index,
        data["subject_ids"],
        reference["train_weights"],
        train_similarity,
        train_fd_raw,
        train_fd_filled,
    )
    neighbor_rows.extend(
        make_neighbor_rows(
            dataset_key,
            fold,
            "test",
            test_index,
            train_index,
            data["subject_ids"],
            reference["test_weights"],
            test_similarity,
            train_fd_raw,
            train_fd_filled,
        )
    )
    return subject_rows, fold_rows, neighbor_rows


def summarize_folds(fold_frame):
    rows = []
    metrics = (
        "initial_ec_auc",
        "mpr_only_auc",
        "full_pr_ec_auc",
        "mpr_minus_initial_auc",
        "full_pr_ec_minus_initial_auc",
        "mpr_variance_retention",
        "full_pr_ec_variance_retention",
    )
    for dataset in fold_frame["dataset"].drop_duplicates():
        dataset_values = fold_frame[fold_frame["dataset"] == dataset]
        for group in GROUP_ORDER:
            values = dataset_values[dataset_values["group"] == group]
            if values.empty:
                continue
            row = {
                "dataset": dataset,
                "group": group,
                "n_subjects": int(values["n_subjects"].sum()),
                "n_patients": int(values["n_patients"].sum()),
                "n_controls": int(values["n_controls"].sum()),
                "n_folds": int(len(values)),
                "n_auc_valid_folds": int(values["initial_ec_auc"].notna().sum()),
            }
            for metric in metrics:
                row[f"{metric}_mean"] = float(values[metric].mean(skipna=True))
                row[f"{metric}_std"] = float(values[metric].std(ddof=0, skipna=True))
            rows.append(row)
    return pd.DataFrame(rows)


def save_plot(summary_frame, output_path):
    import matplotlib.pyplot as plt

    labels = {
        "neighbor_match_high": "Match high",
        "neighbor_match_low": "Match low",
        "neighbor_motion_low": "Motion low",
        "neighbor_motion_high": "Motion high",
    }
    frame = summary_frame.set_index("group").loc[list(GROUP_ORDER)].reset_index()
    ordered = [
        "neighbor_match_high",
        "neighbor_match_low",
        "neighbor_motion_low",
        "neighbor_motion_high",
    ]
    frame = summary_frame.set_index("group").reindex(ordered)
    x = np.arange(len(frame))
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for offset, (column, label) in zip(
        (-width, 0.0, width),
        (
            ("initial_ec_auc", "Initial EC"),
            ("mpr_only_auc", "MPR-only"),
            ("full_pr_ec_auc", "Full PR-EC"),
        ),
    ):
        axes[0].bar(
            x + offset,
            frame[f"{column}_mean"],
            width,
            yerr=frame[f"{column}_std"],
            capsize=3,
            label=label,
        )
    for offset, (column, label) in zip(
        (-width / 2, width / 2),
        (
            ("mpr_minus_initial_auc", "MPR - Initial"),
            ("full_pr_ec_minus_initial_auc", "PR-EC - Initial"),
        ),
    ):
        axes[1].bar(
            x + offset,
            frame[f"{column}_mean"],
            width,
            yerr=frame[f"{column}_std"],
            capsize=3,
            label=label,
        )
    for axis in axes:
        axis.set_xticks(x, [labels[item] for item in ordered], rotation=20, ha="right")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Fold AUC mean ± SD")
    axes[1].set_ylabel("Paired fold AUC difference mean ± SD")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def validate_fold_auc_reproduction(subject_frame, summary_path):
    with Path(summary_path).open(encoding="utf-8") as handle:
        original_folds = json.load(handle)["folds"]
    columns = {
        "initial_ec_probability": "original_AUC",
        "mpr_only_probability": "refined_AUC",
        "full_pr_ec_probability": "qc_refined_AUC",
    }
    differences = {}
    for probability_column, original_column in columns.items():
        reproduced = []
        for _, values in subject_frame.groupby("fold", sort=True):
            reproduced.append(roc_auc_score(values["true_label"], values[probability_column]))
        expected = [fold[original_column] for fold in original_folds]
        maximum = float(np.max(np.abs(np.asarray(reproduced) - np.asarray(expected))))
        if maximum > 1e-7:
            raise RuntimeError(
                f"Fold AUC reproduction failed for {probability_column}: {maximum:.3e}"
            )
        differences[probability_column] = maximum
    return differences


def run_dataset(dataset_key, gpu_id, output_root):
    args, summary_path = load_original_config(dataset_key)
    args.gpu_id = gpu_id
    device = require_cuda(gpu_id)
    set_seed(args.seed)
    print(f"[{dataset_key}] loading cached Initial EC and fold inputs on {device}")
    data, _ = load_pipeline_data(args, device)
    if "fmri_features" not in data:
        raise RuntimeError("FC features are required for neighborhood matching")
    subject_rows, fold_rows, neighbor_rows = [], [], []
    test_coverage = np.zeros(len(data["labels"]), dtype=np.int64)
    for fold, train_index, val_index, test_index in make_stratified_splits(
        data["labels"], args.n_splits, args.seed, args.validation_size
    ):
        print(f"[{dataset_key}] fold {fold}/{args.n_splits}")
        if set(train_index) & set(test_index) or set(val_index) & set(test_index):
            raise RuntimeError(f"Fold {fold}: train/validation/test overlap")
        test_coverage[test_index] += 1
        current = analyze_fold(
            args,
            data,
            fold,
            train_index,
            val_index,
            test_index,
            device,
            dataset_key,
        )
        subject_rows.extend(current[0])
        fold_rows.extend(current[1])
        neighbor_rows.extend(current[2])
    if not np.all(test_coverage == 1):
        raise RuntimeError("OOF coverage failure: every subject must be tested exactly once")

    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_frame = pd.DataFrame(subject_rows).sort_values(["fold", "subject_id"])
    fold_frame = pd.DataFrame(fold_rows).sort_values(["fold", "group"])
    neighbor_frame = pd.DataFrame(neighbor_rows).sort_values(
        ["fold", "split", "query_subject_id", "neighbor_rank"]
    )
    summary_frame = summarize_folds(fold_frame)
    subject_frame.to_csv(output_dir / "subject_level.csv", index=False)
    fold_frame.to_csv(output_dir / "fold_group_results.csv", index=False)
    neighbor_frame.to_csv(output_dir / "neighbor_audit.csv", index=False)
    summary_frame.to_csv(output_dir / "dataset_summary.csv", index=False)
    save_plot(summary_frame, output_dir / "group_auc_summary.png")
    auc_reproduction = validate_fold_auc_reproduction(subject_frame, summary_path)
    audit = {
        "dataset": dataset_key,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "source_summary": str(summary_path),
        "source_initial_ec": str(args.ec_path),
        "subjects": len(subject_frame),
        "folds": args.n_splits,
        "reference_k": args.reference_k,
        "seed": args.seed,
        "classifier_repeats": args.classifier_repeats,
        "mean_fd_field": "func_mean_fd",
        "mean_fd_missing_aligned_subjects": int(
            np.sum(~np.isfinite(data["qsr_qc"][:, tuple(args.qsr_qc_columns).index("func_mean_fd")]))
        ),
        "original_fold_auc_max_abs_difference": auc_reproduction,
        "leakage_checks": {
            "test_subject_once": True,
            "neighbors_train_only": True,
            "training_self_neighbors_excluded": True,
            "group_thresholds_train_only": True,
            "grouping_uses_diagnosis_or_predictions": False,
        },
    }
    with (output_dir / "run_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    return summary_frame


def main():
    cli = parse_args()
    datasets = DATASET_MODULES if cli.dataset == "all" else (cli.dataset,)
    summaries = [run_dataset(key, cli.gpu_id, cli.output_root) for key in datasets]
    combined = pd.concat(summaries, ignore_index=True)
    cli.output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cli.output_root / "all_datasets_summary.csv", index=False)


if __name__ == "__main__":
    main()
