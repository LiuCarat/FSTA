"""Pearson upper-triangle FC + linear SVM with PR-EC's fold-local protocol."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PR_EC.data import FIXED_DATA_CONFIG, load_subject_dataset
from PR_EC.dataset_configs import get_profile
from PR_EC.utils.folds import make_stratified_splits


METRIC_NAMES = ("ACC", "Recall", "AUC", "Precision", "F1")


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=["abide", "abide_ii", "adhd200"], default="abide")
    selected, _ = selector.parse_known_args(argv)
    profile = get_profile(selected.dataset)
    output_root = Path(__file__).resolve().parent / "outputs" / "linear_svm" / profile.name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["abide", "abide_ii", "adhd200"], default=profile.name)
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--phenotype-csv", type=Path, default=profile.phenotype_path)
    parser.add_argument("--reference-ec-path", type=Path, help="PR-EC archive defining subject membership/order; auto-use the profile archive if present")
    parser.add_argument("--output-dir", type=Path, default=output_root)
    parser.add_argument("--fc-path", type=Path, help="FC cache; defaults to <output-dir>/subject_fc_<dataset>.npz")
    parser.add_argument("--max-subjects", type=int, help="Smoke tests only; applied after reference subject alignment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=1.0, help="Fixed C when --c-grid is absent")
    parser.add_argument("--c-grid", type=float, nargs="+", help="Optional C search using only PR-EC's fold-local train/validation split (validation AUC)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel C candidates; -1 uses all CPUs")
    parser.add_argument("--regenerate-fc", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    args = parser.parse_args(argv)
    if args.fc_path is None:
        args.fc_path = args.output_dir / f"subject_fc_{args.dataset}.npz"
    return args


def fc_config(args):
    profile = get_profile(args.dataset)
    return {
        "method": "pearson",
        "diagonal": 0.0,
        "dataset": args.dataset,
        "data_root": str(args.data_root.resolve()),
        "phenotype_csv": str(args.phenotype_csv.resolve()),
        "source_roi_count": profile.source_roi_count,
        "roi_count": profile.roi_count,
        **FIXED_DATA_CONFIG,
    }


def load_matched_dataset(args):
    profile = replace(get_profile(args.dataset), phenotype_path=args.phenotype_csv)
    dataset = load_subject_dataset(args.data_root, profile=profile, **FIXED_DATA_CONFIG)
    reference_path = args.reference_ec_path
    if reference_path is None and profile.ec_path.is_file():
        reference_path = profile.ec_path
    if reference_path is not None:
        with np.load(reference_path, allow_pickle=False) as reference:
            reference_ids = np.asarray(reference["subject_ids"]).astype(str)
            raw_ids = np.asarray(dataset["subject_ids"]).astype(str)
            if len(set(reference_ids)) != len(reference_ids) or len(set(raw_ids)) != len(raw_ids):
                raise ValueError("Subject IDs must be unique")
            by_subject = {subject: index for index, subject in enumerate(raw_ids)}
            missing = [subject for subject in reference_ids if subject not in by_subject]
            if missing:
                raise ValueError(f"PR-EC reference subjects lack matching ROI time series: {missing[:10]}")
            indices = np.asarray([by_subject[subject] for subject in reference_ids], dtype=np.int64)
            for key in ("time_series", "records"):
                dataset[key] = [dataset[key][index] for index in indices]
            for key in ("labels", "subject_ids", "site_ids"):
                dataset[key] = np.asarray(dataset[key])[indices]
            if not np.array_equal(dataset["site_ids"].astype(str), reference["site_ids"].astype(str)):
                raise ValueError("PR-EC reference sites differ from the current phenotype")
            if not np.array_equal(dataset["labels"], reference["labels"]):
                if args.dataset != "abide":
                    raise ValueError("PR-EC reference labels differ from the current phenotype")
                print("ABIDE-I: using phenotype-derived labels, matching PR-EC's archive-label correction")
            if "fold_ids" in reference:
                dataset["reference_fold_ids"] = reference["fold_ids"].copy()
        print(f"Aligned {len(indices)} subjects to PR-EC reference: {reference_path}")
    else:
        print("No PR-EC archive found; using the identical shared raw-data loader and subject order")
    dataset["reference_path"] = str(reference_path.resolve()) if reference_path is not None else None
    if args.max_subjects is not None:
        for key in ("time_series", "records", "labels", "subject_ids", "site_ids"):
            dataset[key] = dataset[key][:args.max_subjects]
        dataset.pop("reference_fold_ids", None)
        print("WARNING: --max-subjects is for smoke tests, not the full-cohort comparison")
    return dataset


def generate_pearson_fc(series):
    """Correlate each subject independently; constant ROIs have zero edges."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2 or series.shape[0] < 2 or series.shape[1] < 2:
        raise ValueError(f"Expected [time >= 2, roi >= 2], got {series.shape}")
    if not np.isfinite(series).all():
        raise ValueError("ROI time series contains non-finite values")
    variable = np.std(series, axis=0) >= 1e-6
    matrix = np.zeros((series.shape[1], series.shape[1]), dtype=np.float32)
    if np.count_nonzero(variable) >= 2:
        matrix[np.ix_(variable, variable)] = np.clip(
            np.corrcoef(series[:, variable], rowvar=False), -1.0, 1.0
        )
    np.fill_diagonal(matrix, 0.0)
    if not np.isfinite(matrix).all():
        raise ValueError("Pearson FC contains non-finite values")
    return matrix


def extract_fc_features(matrices):
    matrices = np.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or matrices.shape[1] < 2:
        raise ValueError(f"Expected FC shape [N, R, R], got {matrices.shape}")
    if not np.isfinite(matrices).all():
        raise ValueError("Pearson FC contains non-finite values")
    upper = np.triu_indices(matrices.shape[1], k=1)
    return matrices[:, upper[0], upper[1]].astype(np.float64)


def save_fc_archive(path, matrices, dataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ec=np.asarray(matrices, dtype=np.float32),
        labels=np.asarray(dataset["labels"], dtype=np.int64),
        subject_ids=np.asarray(dataset["subject_ids"]),
        site_ids=np.asarray(dataset["site_ids"]),
        representation=np.asarray("pearson_fc"),
        fc_config=np.asarray(json.dumps(fc_config(args), sort_keys=True)),
    )


def load_fc_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"ec", "labels", "subject_ids", "site_ids", "representation", "fc_config"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing Pearson FC arrays: {sorted(missing)}; use --regenerate-fc")
        return {key: archive[key] for key in archive.files}


def validate_archive(archive, dataset, args):
    expected_count = len(dataset["subject_ids"])
    roi_count = get_profile(args.dataset).roi_count
    if archive["ec"].shape != (expected_count, roi_count, roi_count):
        raise ValueError("FC archive subject/ROI count differs; use --regenerate-fc")
    for key in ("subject_ids", "labels", "site_ids"):
        if not np.array_equal(archive[key], np.asarray(dataset[key])):
            raise ValueError(f"FC archive {key} does not match PR-EC inputs; use --regenerate-fc")
    if str(archive["representation"].item()) != "pearson_fc":
        raise ValueError("Archive is not a Pearson FC representation")
    if json.loads(str(archive["fc_config"].item())) != fc_config(args):
        raise ValueError("FC archive preprocessing differs; use --regenerate-fc")
    matrices = archive["ec"]
    if not np.isfinite(matrices).all() or not np.allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-6):
        raise ValueError("FC archive must contain finite symmetric Pearson matrices")
    if np.any(np.abs(matrices) > 1.000001) or np.any(np.diagonal(matrices, axis1=1, axis2=2) != 0):
        raise ValueError("FC archive must have correlations in [-1, 1] and zero diagonals")


def generate_fc_archive(args, dataset):
    if args.classification_only or (args.fc_path.is_file() and not args.regenerate_fc):
        archive = load_fc_archive(args.fc_path)
        validate_archive(archive, dataset, args)
        print(f"Using validated Pearson FC archive: {args.fc_path}")
        return archive
    matrices = []
    total = len(dataset["time_series"])
    for index, series in enumerate(dataset["time_series"], start=1):
        matrices.append(generate_pearson_fc(series))
        if index % 100 == 0 or index == total:
            print(f"Pearson FC [{index}/{total}]")
    save_fc_archive(args.fc_path, matrices, dataset, args)
    archive = load_fc_archive(args.fc_path)
    validate_archive(archive, dataset, args)
    return archive


def make_classifier(c_value, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=c_value, probability=False, random_state=seed)),
    ])


def fit_fold_classifier(args, features, labels, train_index, val_index, fold):
    seed = args.seed + fold * 1000 + 1
    classifier = make_classifier(args.c, seed)
    selection = None
    if args.c_grid is not None:
        selection_index = np.concatenate([train_index, val_index])
        validation_fold = np.concatenate([
            np.full(len(train_index), -1, dtype=np.int64),
            np.zeros(len(val_index), dtype=np.int64),
        ])
        search = GridSearchCV(
            classifier,
            {"svm__C": sorted(set(args.c_grid))},
            scoring="roc_auc",
            cv=PredefinedSplit(validation_fold),
            refit=False,
            n_jobs=args.n_jobs,
            error_score="raise",
        )
        search.fit(features[selection_index], labels[selection_index])
        classifier.set_params(**search.best_params_)
        selection = {
            "best_validation_auc": float(search.best_score_),
            "candidates": [
                {"C": float(parameters["svm__C"]), "validation_auc": float(score)}
                for parameters, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
            ],
        }
    classifier.fit(features[train_index], labels[train_index])
    return classifier, selection


def classify_fc(args, archive, reference_fold_ids=None):
    features = extract_fc_features(archive["ec"])
    labels = np.asarray(archive["labels"], dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("The dataset must contain both canonical classes 0 and 1")
    if np.bincount(labels).min() < args.n_splits:
        raise ValueError("Each class must have at least --n-splits subjects")
    splits = list(make_stratified_splits(labels, args.n_splits, args.seed, args.validation_size))
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    for fold, _, _, test_index in splits:
        fold_ids[test_index] = fold
    if reference_fold_ids is not None and not np.array_equal(reference_fold_ids, fold_ids):
        raise ValueError("PR-EC reference fold_ids differ: check seed, fold count, labels, and subject order")
    rows, audit = [], []
    predictions = np.full(len(labels), -1, dtype=np.int64)
    decision_scores = np.full(len(labels), np.nan)
    for fold, train_index, val_index, test_index in splits:
        classifier, selection = fit_fold_classifier(args, features, labels, train_index, val_index, fold)
        scores = classifier.decision_function(features[test_index])
        predicted = classifier.predict(features[test_index])
        predictions[test_index], decision_scores[test_index] = predicted, scores
        metrics = {
            "ACC": float(accuracy_score(labels[test_index], predicted)),
            "Recall": float(recall_score(labels[test_index], predicted, zero_division=0)),
            "AUC": float(roc_auc_score(labels[test_index], scores)),
            "Precision": float(precision_score(labels[test_index], predicted, zero_division=0)),
            "F1": float(f1_score(labels[test_index], predicted, zero_division=0)),
        }
        c_value = float(classifier.named_steps["svm"].C)
        rows.append({"fold": fold, "C": c_value, **metrics})
        audit.append({
            "fold": fold, "classifier_seed": args.seed + fold * 1000 + 1,
            "train_index": train_index.tolist(), "val_index": val_index.tolist(),
            "test_index": test_index.tolist(), "C": c_value, "selection": selection,
        })
        print(f"fold {fold}: train={len(train_index)}, val={len(val_index)}, test={len(test_index)}, C={c_value:g}; "
              + ", ".join(f"{name}={value * 100:.2f}%" for name, value in metrics.items()))
    if not np.isfinite(decision_scores).all() or np.any(predictions < 0):
        raise RuntimeError("Incomplete out-of-fold predictions")
    return {
        "rows": rows, "splits": audit, "fold_ids": fold_ids,
        "predictions": predictions, "decision_scores": decision_scores,
        "feature_count": features.shape[1],
    }


def save_results(args, archive, dataset, result):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.dataset
    rows = result["rows"]
    with (args.output_dir / f"metrics_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        name: {"mean": float(np.mean([row[name] for row in rows])),
               "std": float(np.std([row[name] for row in rows]))}
        for name in METRIC_NAMES
    }
    metadata = {
        "representation": "pearson_fc_upper_triangle_k1", "classifier": "SVC(kernel=linear)",
        "preprocessing": fc_config(args), "reference_ec_path": dataset["reference_path"],
        "fc_path": str(args.fc_path.resolve()), "seed": args.seed, "n_splits": args.n_splits,
        "validation_size": args.validation_size, "subject_count": len(archive["labels"]),
        "feature_count": result["feature_count"], "subject_ids": archive["subject_ids"].astype(str).tolist(),
        "labels": archive["labels"].tolist(), "site_ids": archive["site_ids"].astype(str).tolist(),
        "fixed_C": args.c, "C_grid": args.c_grid, "selection_metric": "validation AUC" if args.c_grid else None,
        "protocol": "PR-EC train/val/test; scaler and final SVM fit on train only; optional C selection on val; no train+val refit; test used once",
        "decision_threshold": 0.0, "std_ddof": 0, "numpy_version": np.__version__,
        "sklearn_version": sklearn_version, "splits": result["splits"],
    }
    for name, payload in (("metrics", rows), ("summary", summary), ("protocol", metadata)):
        with (args.output_dir / f"{name}_{suffix}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(
        args.output_dir / f"oof_predictions_{suffix}.npz",
        subject_ids=archive["subject_ids"], site_ids=archive["site_ids"], labels=archive["labels"],
        fold_ids=result["fold_ids"], predictions=result["predictions"], decision_scores=result["decision_scores"],
    )
    print("mean±std: " + ", ".join(
        f"{name}={value['mean'] * 100:.2f}±{value['std'] * 100:.2f}%" for name, value in summary.items()
    ))


def validate_args(args):
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1")
    if args.max_subjects is not None and args.max_subjects < 1:
        raise ValueError("--max-subjects must be positive")
    if args.seed < 0 or args.seed > 2**32 - 1 - args.n_splits * 1000 - 1:
        raise ValueError("--seed must keep all fold-specific seeds within the uint32 range")
    if any(not np.isfinite(value) or value <= 0 for value in [args.c, *(args.c_grid or [])]):
        raise ValueError("All C values must be finite and positive")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")
    if args.generation_only and args.classification_only:
        raise ValueError("--generation-only and --classification-only cannot be used together")
    if args.classification_only and args.regenerate_fc:
        raise ValueError("--classification-only and --regenerate-fc cannot be used together")


def main():
    args = parse_args()
    validate_args(args)
    dataset = load_matched_dataset(args)
    archive = generate_fc_archive(args, dataset)
    print(f"Pearson FC: {archive['ec'].shape}; linear SVM (CPU); seed={args.seed}")
    if args.generation_only:
        return
    result = classify_fc(args, archive, dataset.get("reference_fold_ids"))
    save_results(args, archive, dataset, result)
    print(f"Classification results: {args.output_dir}")


if __name__ == "__main__":
    main()
