"""Generate subject-level partial-correlation FC matrices and run the BEC classifier probe.

This baseline keeps the downstream evaluation identical to Graph-BEC. Only the
subject-level representation changes: FSTA/Graph-BEC BEC is replaced by a
Graphical-Lasso partial-correlation matrix of each subject's ROI time series.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import GraphicalLasso, LedoitWolf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_BEC.data import load_subject_dataset
from Graph_BEC.downstream import train_classifier
from Graph_BEC.dataset_configs import get_profile
from Graph_BEC.utils.folds import fit_bec_scaler, make_stratified_splits, transform_bec
from Graph_BEC.utils.runtime import set_seed


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--dataset", choices=["abide", "abide_ii", "adhd200"], default="abide"
    )
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)
    output_dir = Path(__file__).resolve().parent / "outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["abide", "abide_ii", "adhd200"],
        default=profile.name,
    )
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument(
        "--estimator",
        choices=["ledoit-wolf", "graphical-lasso"],
        default="ledoit-wolf",
    )
    parser.add_argument("--gl-alpha", type=float, default=0.1)
    parser.add_argument("--gl-max-iter", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--fc-path",
        type=Path,
        default=output_dir / f"subject_partial_correlation_fc_{profile.name}.npz",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-repeats", type=int, default=1)
    parser.add_argument("--patient-label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--control-label", type=int, default=0, choices=[0, 1])
    parser.add_argument("--regenerate-fc", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


def choose_device(gpu_id):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == "auto":
        return torch.device("cuda")
    return torch.device(f"cuda:{gpu_id}")


def fc_config(args):
    return {
        "method": "partial_correlation",
        "estimator": args.estimator,
        "diagonal": 0.0,
        "alpha": args.gl_alpha if args.estimator == "graphical-lasso" else None,
        "max_iter": args.gl_max_iter if args.estimator == "graphical-lasso" else None,
        "dataset": args.dataset,
        "pipeline": args.pipeline,
        "strategy": args.strategy,
        "derivative": args.derivative,
    }


def generate_partial_correlation_fc(series):
    """Return a partial-correlation matrix using default shrinkage settings."""
    return generate_partial_correlation_fc_with_config(
        series, estimator="ledoit-wolf", alpha=0.1, max_iter=100
    )


def generate_partial_correlation_fc_with_config(series, estimator, alpha, max_iter):
    """Estimate partial correlations from a regularized precision matrix."""
    if series.ndim != 2:
        raise ValueError(f"Expected [time, roi] time series, got {series.shape}")
    standard_deviation = np.std(series, axis=0)
    variable = standard_deviation >= 1e-6
    matrix = np.zeros((series.shape[1], series.shape[1]), dtype=np.float32)
    if np.count_nonzero(variable) >= 2:
        variable_series = series[:, variable]
        if estimator == "ledoit-wolf":
            covariance_estimator = LedoitWolf().fit(variable_series)
            precision = np.linalg.pinv(
                np.asarray(covariance_estimator.covariance_, dtype=np.float64)
            )
        else:
            covariance_estimator = GraphicalLasso(alpha=alpha, max_iter=max_iter)
            covariance_estimator.fit(variable_series)
            precision = np.asarray(covariance_estimator.precision_, dtype=np.float64)
        precision_diagonal = np.diag(precision)
        scale = np.sqrt(np.outer(precision_diagonal, precision_diagonal))
        partial = np.divide(
            -precision,
            scale,
            out=np.zeros_like(precision),
            where=scale > 1e-12,
        ).astype(np.float32)
        matrix[np.ix_(variable, variable)] = partial
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square FC matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Partial Correlation FC contains non-finite values")
    np.fill_diagonal(matrix, 0.0)
    return matrix


def save_fc_archive(path, fc, dataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(fc)
    np.savez_compressed(
        path,
        bec=np.asarray(fc, dtype=np.float32),
        labels=np.asarray(dataset["labels"][:count], dtype=np.int64),
        subject_ids=np.asarray(dataset["subject_ids"][:count]),
        site_ids=np.asarray(dataset["site_ids"][:count]),
        representation=np.asarray("partial_correlation_fc"),
        fc_config=np.asarray(json.dumps(fc_config(args), sort_keys=True)),
    )


def load_fc_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"bec", "labels", "subject_ids", "site_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing Partial Correlation FC arrays: {sorted(missing)}")
        return {key: archive[key] for key in archive.files}


def validate_archive(archive, dataset, args):
    count = len(archive["bec"])
    if count > len(dataset["subject_ids"]):
        raise ValueError("FC archive contains more subjects than the current dataset")
    expected_ids = np.asarray(dataset["subject_ids"][:count]).astype(str)
    archive_ids = np.asarray(archive["subject_ids"]).astype(str)
    if not np.array_equal(expected_ids, archive_ids):
        raise ValueError("FC archive subject order does not match the current dataset")
    if archive["bec"].ndim != 3 or archive["bec"].shape[1] != archive["bec"].shape[2]:
        raise ValueError(f"Expected FC shape [N, R, R], got {archive['bec'].shape}")
    if "fc_config" in archive:
        archived_config = json.loads(str(archive["fc_config"].item()))
        if archived_config != fc_config(args):
            raise ValueError("Existing FC archive uses a different dataset/configuration")


def generate_fc_archive(args, dataset):
    if args.fc_path.is_file() and not args.regenerate_fc:
        archive = load_fc_archive(args.fc_path)
        validate_archive(archive, dataset, args)
        if len(archive["bec"]) == len(dataset["time_series"]):
            print(f"using existing Partial Correlation FC archive: {args.fc_path}")
            return archive
        raise ValueError("Existing Partial Correlation FC archive is incomplete; use --regenerate-fc")

    matrices = []
    total = len(dataset["time_series"])
    for index, series in enumerate(dataset["time_series"], start=1):
        matrix = generate_partial_correlation_fc_with_config(
            series,
            estimator=args.estimator,
            alpha=args.gl_alpha,
            max_iter=args.gl_max_iter,
        )
        matrices.append(matrix)
        if index % 100 == 0 or index == total:
            print(f"Partial Correlation FC [{index}/{total}]")
    save_fc_archive(args.fc_path, matrices, dataset, args)
    return load_fc_archive(args.fc_path)


def classify_fc(args, archive, device):
    labels = np.asarray(archive["labels"], dtype=np.int64)
    rows = []
    for fold, train_index, val_index, test_index in make_stratified_splits(
        labels, args.n_splits, args.seed, args.validation_size
    ):
        train_mean, train_std = fit_bec_scaler(archive["bec"][train_index])
        train_fc = transform_bec(archive["bec"][train_index], train_mean, train_std)
        val_fc = transform_bec(archive["bec"][val_index], train_mean, train_std)
        test_fc = transform_bec(archive["bec"][test_index], train_mean, train_std)
        print(f"fold {fold}: train={len(train_index)}, val={len(val_index)}, test={len(test_index)}")
        for repeat in range(args.classifier_repeats):
            metrics, _ = train_classifier(
                train_fc,
                labels[train_index],
                val_fc,
                labels[val_index],
                test_fc,
                labels[test_index],
                device=device,
                seed=args.seed + fold * 1000 + repeat + 1,
                max_epochs=args.classifier_epochs,
                patience=args.classifier_patience,
                batch_size=32,
                learning_rate=args.classifier_lr,
            )
            rows.append({"fold": fold, "repeat": repeat + 1, **metrics})
            print("  " + ", ".join(f"{name}={value * 100:.2f}%" for name, value in metrics.items()))
    return rows


def save_results(args, rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.dataset
    with (args.output_dir / f"metrics_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)
    with (args.output_dir / f"metrics_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    names = [name for name in rows[0] if name not in {"fold", "repeat"}]
    summary = {
        name: {
            "mean": float(np.nanmean([row[name] for row in rows])),
            "std": float(np.nanstd([row[name] for row in rows])),
        }
        for name in names
    }
    with (args.output_dir / f"summary_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print("mean±std: " + ", ".join(
        f"{name}={value['mean'] * 100:.2f}±{value['std'] * 100:.2f}%"
        for name, value in summary.items()
    ))


def validate_args(args):
    if args.patient_label == args.control_label:
        raise ValueError("--patient-label and --control-label must be different")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.classifier_repeats < 1:
        raise ValueError("--classifier-repeats must be positive")
    if args.gl_alpha <= 0:
        raise ValueError("--gl-alpha must be positive")
    if args.gl_max_iter < 1:
        raise ValueError("--gl-max-iter must be positive")
    if args.generation_only and args.classification_only:
        raise ValueError("--generation-only and --classification-only cannot be used together")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    profile = get_profile(args.dataset)
    device = choose_device(args.gpu_id)
    print(f"dataset: {args.dataset}; device: {device}")

    if args.classification_only:
        archive = load_fc_archive(args.fc_path)
    else:
        dataset = load_subject_dataset(
            args.data_root,
            pipeline=args.pipeline,
            strategy=args.strategy,
            derivative=args.derivative,
            profile=profile,
            standardize=True,
            max_subjects=args.max_subjects,
            patient_label=args.patient_label,
            control_label=args.control_label,
        )
        archive = generate_fc_archive(args, dataset)
    print(f"Partial Correlation FC archive: {args.fc_path} shape={archive['bec'].shape}")
    if args.generation_only:
        return
    rows = classify_fc(args, archive, device)
    save_results(args, rows)
    print(f"classification results: {args.output_dir}")


if __name__ == "__main__":
    main()
