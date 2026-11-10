"""Leakage-safe fold and cross-validation workflow."""
from __future__ import annotations

import numpy as np

from PR_EC_Inductive.downstream import train_classifier
from PR_EC_Inductive.model.mpr.population_reference import (
    ec_separability,
    edge_effect_sizes,
    normative_reference,
)
from PR_EC_Inductive.model.mpr import build_reference_graph, fused_graph, topk_graph
from PR_EC_Inductive.model.ablation.mpr_refinement import apply_pgr_refiner, train_pgr_refiner
from PR_EC_Inductive.model.qsr.qsr_refiner import apply_qsr_refiner, train_qsr_refiner
from PR_EC_Inductive.data.adhd200 import (
    apply_numeric_imputer,
    fit_numeric_imputer,
    prepare_adhd_fold_arrays,
)
from PR_EC_Inductive.model.individual_ec.ec_estimator import generate_fold_subject_ec
from PR_EC_Inductive.utils import make_stratified_splits, prepare_fold_arrays, set_seed


def build_fold_reference(
    args, arrays, fmri_arrays=None, splits=("train", "val", "test")
):
    common = dict(
        k=args.reference_k,
        bandwidth=args.reference_bandwidth,
        categorical_penalty=args.categorical_penalty,
        continuous_weights=args.continuous_weights,
        permute=args.permute_phenotype,
        seed=args.seed,
    )
    phenotype_weights = {}
    for split in splits:
        train_weights, query_weights = build_reference_graph(
            arrays["train_cont"], arrays["train_cat"],
            arrays[f"{split}_cont"], arrays[f"{split}_cat"], **common,
        )
        phenotype_weights[split] = train_weights if split == "train" else query_weights

    weights = phenotype_weights
    if args.graph_mode == "fusion":
        if fmri_arrays is None:
            raise ValueError("Fusion graph mode requires fold fMRI features")
        train_fmri = fmri_arrays["train"]
        mean, std = train_fmri.mean(axis=0), train_fmri.std(axis=0)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        standardized = {
            split: ((fmri_arrays[split] - mean) / std).astype(np.float32)
            for split in splits
        }
        fmri_weights = {}
        for split in splits:
            fmri_weights[split] = topk_graph(
                standardized["train"],
                standardized[split],
                args.reference_k,
                exclude_self=split == "train",
            )
        weights = {
            split: fused_graph(
                fmri_weights[split], phenotype_weights[split],
                args.fusion_beta, args.reference_k,
            )
            for split in splits
        }

    reference = {}
    for split in splits:
        reference[f"{split}_neighbor"], _ = normative_reference(
            arrays["train_ec"], weights[split]
        )
        reference[f"{split}_weights"] = weights[split]
    return reference


def run_fold(args, fold, data, train_index, val_index, test_index, device):
    """Fit refiners and classifiers using one fold-local data split."""
    active_splits = ("train", "val", "test")
    representations_to_run = tuple(args.representations)
    needs_pgr = "refined" in representations_to_run
    needs_qsr = "qc_refined" in representations_to_run
    fold_seed = args.seed + fold * 1000
    set_seed(fold_seed)
    fold_ec = generate_fold_subject_ec(
        args,
        fold,
        fold_seed,
        data["subjects"],
        train_index,
        val_index,
        test_index,
        device,
    )
    prepare_arrays = (
        prepare_adhd_fold_arrays
        if args.profile.name == "adhd200"
        else prepare_fold_arrays
    )
    arrays = prepare_arrays(
        fold_ec["train_ec"], fold_ec["val_ec"],
        fold_ec["test_ec"],
        data["continuous"][train_index], data["continuous"][val_index],
        data["continuous"][test_index],
        data["categorical_raw"][train_index], data["categorical_raw"][val_index],
        data["categorical_raw"][test_index],
    )
    fmri_arrays = None
    if args.graph_mode == "fusion" and (needs_pgr or needs_qsr):
        split_indices = {
            "train": train_index,
            "val": val_index,
            "test": test_index,
        }
        fmri_arrays = {
            split: data["fmri_features"][split_indices[split]]
            for split in active_splits
        }
    reference = (
        build_fold_reference(args, arrays, fmri_arrays, active_splits)
        if needs_pgr or needs_qsr
        else None
    )

    if needs_qsr and args.profile.name == "adhd200":
        confound_columns = tuple(args.profile.confound_columns)
        categorical_indices = tuple(
            index for index, column in enumerate(confound_columns)
            if column in ("Gender",)
        )
        confound_fills = fit_numeric_imputer(
            data["qsr_confound_values"][train_index], categorical_indices
        )
        qsr_train_confound = apply_numeric_imputer(
            data["qsr_confound_values"][train_index], confound_fills
        )
    elif needs_qsr:
        qsr_train_confound = data["qsr_confound_values"][train_index]

    pgr_metrics = {}
    test_pgr = None
    if needs_pgr:
        set_seed(fold_seed)
        pgr_model, train_pgr, pgr_metrics = train_pgr_refiner(
            args, arrays["train_ec"], reference["train_neighbor"], device
        )
        val_pgr = apply_pgr_refiner(
            pgr_model, arrays["val_ec"], reference["val_neighbor"], device
        )
        test_pgr = apply_pgr_refiner(
            pgr_model, arrays["test_ec"], reference["test_neighbor"], device
        )

    qsr_metrics = {}
    test_qsr = None
    if needs_qsr:
        set_seed(fold_seed + 1)
        qsr_refiner, train_qsr, sensitive_map, qsr_metrics = train_qsr_refiner(
            args, arrays["train_ec"], reference["train_neighbor"],
            data["qsr_qc"][train_index], qsr_train_confound,
            data["site_ids"][train_index], device, fold_seed + 1,
        )
        val_qsr = apply_qsr_refiner(
            qsr_refiner, arrays["val_ec"], reference["val_neighbor"], sensitive_map, device
        )
        test_qsr = apply_qsr_refiner(
            qsr_refiner, arrays["test_ec"], reference["test_neighbor"], sensitive_map, device
        )

    all_representations = {
        "original": {
            "train": arrays["train_ec"], "val": arrays["val_ec"],
            "test": arrays["test_ec"],
        },
    }
    if needs_pgr:
        all_representations["refined"] = {
            "train": train_pgr, "val": val_pgr, "test": test_pgr,
        }
    if needs_qsr:
        all_representations["qc_refined"] = {
            "train": train_qsr, "val": val_qsr, "test": test_qsr,
        }
    representations = {
        "original": all_representations["original"],
        **{
            name: all_representations[name]
            for name in representations_to_run
            if name != "original"
        },
    }
    labels = {
        "train": data["labels"][train_index],
        "val": data["labels"][val_index],
    }
    labels["test"] = data["labels"][test_index]
    classifier_args = dict(
        device=device,
        max_epochs=args.classifier_epochs,
        patience=args.classifier_patience,
        batch_size=args.batch_size,
        learning_rate=args.classifier_lr,
    )
    metric_runs = {name: [] for name in representations}
    probability_runs = {name: [] for name in representations}
    for repeat in range(args.classifier_repeats):
        classifier_seed = fold_seed + repeat + 1
        for name, representation in representations.items():
            metrics, _, predictions = train_classifier(
                representation["train"], labels["train"],
                representation["val"], labels["val"],
                representation["test"],
                labels["test"],
                seed=classifier_seed, **classifier_args,
            )
            metric_runs[name].append(metrics)
            probability_runs[name].append(predictions["test_probabilities"])

    result = {
        name: {
            key: float(np.mean([run[key] for run in runs]))
            for key in runs[0]
        }
        for name, runs in metric_runs.items()
    }
    original_test = all_representations["original"]["test"]
    _, original_effect = edge_effect_sizes(
        original_test, labels["test"], args.asd_label
    )
    for name, representation in representations.items():
        group = ec_separability(
            representation["test"], labels["test"], args.asd_label
        )
        edge, effect = edge_effect_sizes(
            representation["test"], labels["test"], args.asd_label
        )
        result.update({f"{name}_{key}": value for key, value in group.items()})
        result.update({f"{name}_{key}": value for key, value in edge.items()})
        result[f"{name}_variance_retention"] = float(
            np.var(representation["test"]) / max(np.var(original_test), 1e-12)
        )
        result[f"{name}_edge_abs_d_change"] = float(
            np.mean(np.abs(effect)) - np.mean(np.abs(original_effect))
        )
    for name in representations:
        result[f"{name}_paired_auc_delta_mean"] = float(
            result[name]["AUC"] - result["original"]["AUC"]
        )
    result["fold_id"] = int(fold)
    return {
        "metrics": result,
        "test_pgr": test_pgr,
        "test_qc": test_qsr,
        "test_original": fold_ec["test_ec"],
        "test_probabilities": {
            name: np.mean(np.stack(values), axis=0).astype(np.float32)
            for name, values in probability_runs.items()
        },
        "test_index": test_index,
        "ec_mean": arrays["ec_mean"],
        "ec_std": arrays["ec_std"],
        "refinement_metrics": {**pgr_metrics, **qsr_metrics},
        "encoder_training_metrics": fold_ec["training_metrics"],
        "encoder_manifest": fold_ec["manifest"],
    }


def run_cross_validation(args, data, device):
    """Run all folds and assemble test-only out-of-fold refined ECs."""
    ec_shape = (len(data["labels"]), args.profile.roi_count, args.profile.roi_count)
    oof_original = np.full(ec_shape, np.nan, dtype=np.float32)
    oof_pgr = np.full(ec_shape, np.nan, dtype=np.float32)
    oof_qc = np.full(ec_shape, np.nan, dtype=np.float32)
    fold_ids = np.full(len(data["labels"]), -1, dtype=np.int64)
    oof_probabilities = {
        name: np.full(len(data["labels"]), np.nan, dtype=np.float32)
        for name in ("original", "refined", "qc_refined")
    }
    selected_folds = set(args.folds or range(1, args.n_splits + 1))
    invalid_folds = selected_folds - set(range(1, args.n_splits + 1))
    if invalid_folds:
        raise ValueError(f"Invalid fold numbers: {sorted(invalid_folds)}")
    results, refinement_metrics, encoder_metrics = [], [], []
    for fold, train_index, val_index, test_index in make_stratified_splits(
        data["labels"], args.n_splits, args.seed, args.validation_size
    ):
        if fold not in selected_folds:
            continue
        fold_result = run_fold(
            args, fold, data, train_index, val_index, test_index, device
        )
        heldout = fold_result["test_index"]
        oof_original[heldout] = fold_result["test_original"]
        if fold_result["test_pgr"] is not None:
            restored_pgr = (
                fold_result["test_pgr"] * fold_result["ec_std"]
                + fold_result["ec_mean"]
            ).astype(np.float32)
            diagonal = np.arange(restored_pgr.shape[-1])
            restored_pgr[:, diagonal, diagonal] = 0.0
            oof_pgr[heldout] = restored_pgr
        if fold_result["test_qc"] is not None:
            restored_qc = (
                fold_result["test_qc"] * fold_result["ec_std"]
                + fold_result["ec_mean"]
            ).astype(np.float32)
            diagonal = np.arange(restored_qc.shape[-1])
            restored_qc[:, diagonal, diagonal] = 0.0
            oof_qc[heldout] = restored_qc
        fold_ids[heldout] = fold
        for name, probabilities in fold_result["test_probabilities"].items():
            oof_probabilities[name][heldout] = probabilities
        results.append(fold_result["metrics"])
        refinement_metrics.append({
            "fold": fold, **fold_result["refinement_metrics"]
        })
        encoder_metrics.append({
            "fold": fold,
            "training_metrics": fold_result["encoder_training_metrics"],
            "manifest": fold_result["encoder_manifest"],
        })
        report = " | ".join(
            f"{name}: " + ", ".join(
                f"{metric}={100 * value[metric]:.2f}%"
                for metric in ("ACC", "Recall", "AUC", "Precision", "F1")
            )
            for name, value in fold_result["metrics"].items()
            if isinstance(value, dict)
        )
        print(f"fold {fold} test: {report}")

    needs_pgr = "refined" in args.representations
    needs_qsr = "qc_refined" in args.representations
    completed = len(selected_folds) == args.n_splits
    if completed and (
        not np.isfinite(oof_original).all()
        or (needs_pgr and not np.isfinite(oof_pgr).all())
        or (needs_qsr and not np.isfinite(oof_qc).all())
        or np.any(fold_ids < 0)
        or any(
            not np.isfinite(oof_probabilities[name]).all()
            for name in ("original", *args.representations)
        )
    ):
        raise RuntimeError("PGR/QC refined OOF arrays are incomplete")
    return {
        "oof_original": oof_original,
        "oof_pgr": oof_pgr,
        "oof_qc": oof_qc,
        "oof_probabilities": oof_probabilities,
        "fold_ids": fold_ids,
        "fold_results": results,
        "refinement_metrics": refinement_metrics,
        "encoder_metrics": encoder_metrics,
        "completed": completed,
    }
