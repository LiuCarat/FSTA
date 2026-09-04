"""Leakage-safe fold and cross-validation workflow."""
from __future__ import annotations

import numpy as np

from Graph_BEC.downstream import train_classifier
from Graph_BEC.model.patient_graph.reference_bec import (
    bec_separability,
    edge_effect_sizes,
    normative_reference,
)
from Graph_BEC.model.patient_graph import build_reference_graph, fused_graph, topk_graph
from Graph_BEC.model.refinement import (
    apply_pgr_refiner,
    apply_qsr_refiner,
    train_pgr_refiner,
    train_qsr_refiner,
)
from Graph_BEC.data.adhd_utils import (
    apply_numeric_imputer,
    fit_numeric_imputer,
    prepare_adhd_fold_arrays,
)
from Graph_BEC.utils import make_stratified_splits, prepare_fold_arrays, set_seed


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
            arrays["train_bec"], weights[split]
        )
        reference[f"{split}_weights"] = weights[split]
    return reference


def run_fold(args, fold, data, train_index, val_index, test_index, device):
    """Fit refiners and classifiers using one fold-local data split."""
    representations_to_run = tuple(args.representations)
    needs_pgr = "refined" in representations_to_run
    needs_qsr = "qc_refined" in representations_to_run
    fold_seed = args.seed + fold * 1000
    set_seed(fold_seed)
    prepare_arrays = (
        prepare_adhd_fold_arrays
        if args.profile.name == "adhd200"
        else prepare_fold_arrays
    )
    arrays = prepare_arrays(
        data["bec"][train_index], data["bec"][val_index], data["bec"][test_index],
        data["continuous"][train_index], data["continuous"][val_index], data["continuous"][test_index],
        data["categorical_raw"][train_index], data["categorical_raw"][val_index], data["categorical_raw"][test_index],
    )
    fmri_arrays = None
    if args.graph_mode == "fusion" and (needs_pgr or needs_qsr):
        fmri_arrays = tuple(
            data["fmri_features"][index]
            for index in (train_index, val_index, test_index)
        )
    reference = (
        build_fold_reference(args, arrays, fmri_arrays)
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
            args, arrays["train_bec"], reference["train_neighbor"], device
        )
        val_pgr = apply_pgr_refiner(
            pgr_model, arrays["val_bec"], reference["val_neighbor"], device
        )
        test_pgr = apply_pgr_refiner(
            pgr_model, arrays["test_bec"], reference["test_neighbor"], device
        )

    qsr_metrics = {}
    test_qsr = None
    if needs_qsr:
        set_seed(fold_seed + 1)
        qsr_model, train_qsr, sensitive_map, qsr_metrics = train_qsr_refiner(
            args, arrays["train_bec"], reference["train_neighbor"],
            data["qsr_qc"][train_index], qsr_train_confound,
            data["site_ids"][train_index], device, fold_seed + 1,
        )
        val_qsr = apply_qsr_refiner(
            qsr_model, arrays["val_bec"], reference["val_neighbor"], sensitive_map, device
        )
        test_qsr = apply_qsr_refiner(
            qsr_model, arrays["test_bec"], reference["test_neighbor"], sensitive_map, device
        )

    all_representations = {
        "original": {
            "train": arrays["train_bec"], "val": arrays["val_bec"],
            "test": arrays["test_bec"],
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
        "test": data["labels"][test_index],
    }
    classifier_args = dict(
        device=device,
        max_epochs=args.classifier_epochs,
        patience=args.classifier_patience,
        batch_size=args.batch_size,
        learning_rate=args.classifier_lr,
    )
    metric_runs = {name: [] for name in representations}
    for repeat in range(args.classifier_repeats):
        classifier_seed = fold_seed + repeat + 1
        for name, representation in representations.items():
            metrics, _ = train_classifier(
                representation["train"], labels["train"],
                representation["val"], labels["val"],
                representation["test"], labels["test"],
                seed=classifier_seed, **classifier_args,
            )
            metric_runs[name].append(metrics)

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
        group = bec_separability(
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
        result[f"{name}_paired_auc_delta_mean"] = float(
            result[name]["AUC"] - result["original"]["AUC"]
        )
    return {
        "metrics": result,
        "test_pgr": test_pgr,
        "test_qc": test_qsr,
        "test_index": test_index,
        "bec_mean": arrays["bec_mean"],
        "bec_std": arrays["bec_std"],
        "refinement_metrics": {**pgr_metrics, **qsr_metrics},
    }


def run_cross_validation(args, data, device):
    """Run all folds and assemble test-only out-of-fold refined BECs."""
    oof_pgr = np.full_like(data["bec"], np.nan, dtype=np.float32)
    oof_qc = np.full_like(data["bec"], np.nan, dtype=np.float32)
    fold_ids = np.full(len(data["bec"]), -1, dtype=np.int64)
    results, refinement_metrics = [], []
    for fold, train_index, val_index, test_index in make_stratified_splits(
        data["labels"], args.n_splits, args.seed, args.validation_size
    ):
        fold_result = run_fold(
            args, fold, data, train_index, val_index, test_index, device
        )
        heldout = fold_result["test_index"]
        if fold_result["test_pgr"] is not None:
            restored_pgr = (
                fold_result["test_pgr"] * fold_result["bec_std"]
                + fold_result["bec_mean"]
            ).astype(np.float32)
            diagonal = np.arange(restored_pgr.shape[-1])
            restored_pgr[:, diagonal, diagonal] = 0.0
            oof_pgr[heldout] = restored_pgr
        if fold_result["test_qc"] is not None:
            restored_qc = (
                fold_result["test_qc"] * fold_result["bec_std"]
                + fold_result["bec_mean"]
            ).astype(np.float32)
            diagonal = np.arange(restored_qc.shape[-1])
            restored_qc[:, diagonal, diagonal] = 0.0
            oof_qc[heldout] = restored_qc
        fold_ids[heldout] = fold
        results.append(fold_result["metrics"])
        refinement_metrics.append({
            "fold": fold, **fold_result["refinement_metrics"]
        })
        report = " | ".join(
            f"{name} AUC={value['AUC']:.4f}"
            for name, value in fold_result["metrics"].items()
            if isinstance(value, dict)
        )
        print(f"fold {fold}: {report}")

    needs_pgr = "refined" in args.representations
    needs_qsr = "qc_refined" in args.representations
    if (
        (needs_pgr and not np.isfinite(oof_pgr).all())
        or (needs_qsr and not np.isfinite(oof_qc).all())
        or np.any(fold_ids < 0)
    ):
        raise RuntimeError("PGR/QC refined OOF arrays are incomplete")
    return {
        "oof_pgr": oof_pgr,
        "oof_qc": oof_qc,
        "fold_ids": fold_ids,
        "fold_results": results,
        "refinement_metrics": refinement_metrics,
    }
