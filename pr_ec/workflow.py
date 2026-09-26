
from __future__ import annotations

import numpy as np

from pr_ec.downstream import train_classifier
from pr_ec.model.mpr.population_reference import (
    normative_reference,
)
from pr_ec.model.mpr import build_reference_graph, fused_graph, topk_graph
from pr_ec.model.qsr.qsr_refiner import apply_qsr_refiner, train_qsr_refiner
from pr_ec.data.adhd200 import (
    apply_numeric_imputer,
    fit_numeric_imputer,
    prepare_adhd_fold_arrays,
)
from pr_ec.utils import make_stratified_splits, prepare_fold_arrays, set_seed


def build_fold_reference(args, arrays, fmri_arrays=None):
    common = dict(
        k=args.reference_k,
        bandwidth=args.reference_bandwidth,
        categorical_penalty=args.categorical_penalty,
        continuous_weights=args.continuous_weights,
        permute=args.permute_phenotype,
        seed=args.seed,
    )
    phenotype_weights = {}
    for split in ("train", "val", "test"):
        train_weights, query_weights = build_reference_graph(
            arrays["train_cont"], arrays["train_cat"],
            arrays[f"{split}_cont"], arrays[f"{split}_cat"], **common,
        )
        phenotype_weights[split] = train_weights if split == "train" else query_weights

    weights = phenotype_weights
    if args.graph_mode == "fusion":
        if fmri_arrays is None:
            raise ValueError("Fusion graph mode requires fold fMRI features")
        train_fmri, val_fmri, test_fmri = fmri_arrays
        mean, std = train_fmri.mean(axis=0), train_fmri.std(axis=0)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        train_fmri, val_fmri, test_fmri = [
            ((values - mean) / std).astype(np.float32)
            for values in (train_fmri, val_fmri, test_fmri)
        ]
        fmri_weights = {
            "train": topk_graph(
                train_fmri, train_fmri, args.reference_k, exclude_self=True
            ),
            "val": topk_graph(train_fmri, val_fmri, args.reference_k),
            "test": topk_graph(train_fmri, test_fmri, args.reference_k),
        }
        weights = {
            split: fused_graph(
                fmri_weights[split], phenotype_weights[split],
                args.fusion_beta, args.reference_k,
            )
            for split in ("train", "val", "test")
        }

    reference = {}
    for split in ("train", "val", "test"):
        reference[f"{split}_neighbor"], _ = normative_reference(
            arrays["train_individual_ec"], weights[split]
        )
        reference[f"{split}_weights"] = weights[split]
    return reference


def run_fold(args, fold, data, train_index, val_index, test_index, device):
    
    fold_seed = args.seed + fold * 1000
    print(f"[Fold {fold}/{args.n_splits}] Starting")
    print(f"[Fold {fold}/{args.n_splits}] Preparing train/validation/test split")
    set_seed(fold_seed)
    prepare_arrays = (
        prepare_adhd_fold_arrays
        if args.profile.name == "adhd200"
        else prepare_fold_arrays
    )
    arrays = prepare_arrays(
        data["ec"][train_index], data["ec"][val_index], data["ec"][test_index],
        data["continuous"][train_index], data["continuous"][val_index], data["continuous"][test_index],
        data["categorical_raw"][train_index], data["categorical_raw"][val_index], data["categorical_raw"][test_index],
    )
    fmri_arrays = None
    if args.graph_mode == "fusion":
        fmri_arrays = tuple(
            data["fmri_features"][index]
            for index in (train_index, val_index, test_index)
        )
    reference = build_fold_reference(args, arrays, fmri_arrays)
    print(f"[Fold {fold}/{args.n_splits}] MPR reference graph completed")
    print(f"[Fold {fold}/{args.n_splits}] Building neighbor-reference EC")

    if args.profile.name == "adhd200":
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
    else:
        qsr_train_confound = data["qsr_confound_values"][train_index]

    qsr_metrics = {}
    set_seed(fold_seed + 1)
    print(f"[Fold {fold}/{args.n_splits}] Training QSR refiner")
    qsr_refiner, train_qsr, sensitive_map, qsr_metrics = train_qsr_refiner(
        args, arrays["train_individual_ec"], reference["train_neighbor"],
        data["qsr_qc"][train_index], qsr_train_confound,
        data["site_ids"][train_index], device, fold_seed + 1,
        fold=fold, total_folds=args.n_splits,
    )
    val_qsr = apply_qsr_refiner(
        qsr_refiner, arrays["val_individual_ec"], reference["val_neighbor"], sensitive_map, device
    )
    test_qsr = apply_qsr_refiner(
        qsr_refiner, arrays["test_individual_ec"], reference["test_neighbor"], sensitive_map, device
    )
    print(f"[Fold {fold}/{args.n_splits}] QSR refinement completed")
    print(f"[Fold {fold}/{args.n_splits}] Restored held-out QSR EC")

    representation = {
        "train": train_qsr,
        "val": val_qsr,
        "test": test_qsr,
    }
    labels = {
        "train": data["labels"][train_index],
        "val": data["labels"][val_index],
        "test": data["labels"][test_index],
    }
    classifier_args = dict(
        device=device,
        max_epochs=args.classifier_epochs,
        patience=args.classifier_patience,
        batch_size=args.batch_size,
        learning_rate=args.classifier_lr,
    )
    if fold == 1:
        print("[STAGE 4/4] Downstream classification using QSR Refinement")
    metric_runs = []
    for repeat in range(args.classifier_repeats):
        classifier_seed = fold_seed + repeat + 1
        metrics, _ = train_classifier(
            representation["train"], labels["train"],
            representation["val"], labels["val"],
            representation["test"], labels["test"],
            seed=classifier_seed, **classifier_args,
        )
        metric_runs.append(metrics)

    result = {
        "pr_ec": {
            key: float(np.mean([run[key] for run in metric_runs]))
            for key in metric_runs[0]
        }
    }
    metrics = result["pr_ec"]
    print(
        f"[Fold {fold}/{args.n_splits}] Classification | "
        f"ACC={100 * metrics['ACC']:.2f}% | "
        f"Recall={100 * metrics['Recall']:.2f}% | "
        f"AUC={100 * metrics['AUC']:.2f}% | "
        f"Precision={100 * metrics['Precision']:.2f}% | "
        f"F1={100 * metrics['F1']:.2f}%"
    )
    return {
        "metrics": result,
        "test_qsr": test_qsr,
        "test_index": test_index,
        "ec_mean": arrays["ec_mean"],
        "ec_std": arrays["ec_std"],
        "refinement_metrics": qsr_metrics,
    }


def run_cross_validation(args, data, device):
    
    oof_qc = np.full_like(data["ec"], np.nan, dtype=np.float32)
    fold_ids = np.full(len(data["ec"]), -1, dtype=np.int64)
    results, refinement_metrics = [], []
    for fold, train_index, val_index, test_index in make_stratified_splits(
        data["labels"], args.n_splits, args.seed, args.validation_size
    ):
        fold_result = run_fold(
            args, fold, data, train_index, val_index, test_index, device
        )
        heldout = fold_result["test_index"]
        if fold_result["test_qsr"] is not None:
            restored_qc = (
                fold_result["test_qsr"] * fold_result["ec_std"]
                + fold_result["ec_mean"]
            ).astype(np.float32)
            diagonal = np.arange(restored_qc.shape[-1])
            restored_qc[:, diagonal, diagonal] = 0.0
            oof_qc[heldout] = restored_qc
        fold_ids[heldout] = fold
        results.append(fold_result["metrics"])
        refinement_metrics.append({
            "fold": fold, **fold_result["refinement_metrics"]
        })

    if not np.isfinite(oof_qc).all() or np.any(fold_ids < 0):
        raise RuntimeError("QSR Refinement OOF arrays are incomplete")
    return {
        "oof_qc": oof_qc,
        "fold_ids": fold_ids,
        "fold_results": results,
        "refinement_metrics": refinement_metrics,
    }
