"""Generate Sparse VAR BECs and run the Graph-BEC classifier probe.

The estimator is a subject-level VAR(p) with elastic-net regularization.  Its
lagged directed coefficients are aggregated into a signed square BEC, while
the downstream split, scaling, model, and metrics remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = Path(__file__).resolve().parent
for path in (ROOT, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Graph_BEC.data import load_subject_dataset
from Graph_BEC.downstream import train_classifier
from Graph_BEC.profiles.configuration import get_profile
from Graph_BEC.utils.folds import fit_bec_scaler, make_stratified_splits, transform_bec
from Graph_BEC.utils.runtime import set_seed
from sparse_var import SparseVARConfig, generate_sparse_var_bec


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=["abide", "adhd200"], default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)
    output_dir = Path(__file__).resolve().parent / "outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["abide", "adhd200"], default=profile.name)
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--l1-ratio", type=float, default=1.0)
    parser.add_argument("--lag-decay", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--bec-path", type=Path, default=output_dir / f"subject_sparse_var_bec_{profile.name}.npz")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--classifier-epochs", type=int, default=profile.defaults.get("classifier_epochs", 100))
    parser.add_argument("--classifier-patience", type=int, default=profile.defaults.get("classifier_patience", 20))
    parser.add_argument("--classifier-lr", type=float, default=profile.defaults.get("classifier_lr", 1e-3))
    parser.add_argument("--classifier-repeats", type=int, default=profile.defaults.get("classifier_repeats", 1))
    parser.add_argument("--patient-label", type=int, default=profile.defaults.get("patient_label", 1), choices=[0, 1])
    parser.add_argument("--control-label", type=int, default=profile.defaults.get("control_label", 0), choices=[0, 1])
    parser.add_argument("--regenerate-bec", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


def choose_device(gpu_id):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id in {"auto", "cuda"}:
        return torch.device("cuda")
    if isinstance(gpu_id, str) and gpu_id.startswith("cuda:"):
        return torch.device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def var_config(args):
    return {
        "method": "sparse_var",
        "lags": args.lags,
        "alpha": args.alpha,
        "l1_ratio": args.l1_ratio,
        "lag_decay": args.lag_decay,
        "threshold": args.threshold,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "diagonal": 0.0,
    }


def load_bec_archive(path):
    archive = np.load(path, allow_pickle=False)
    required = {"bec", "labels", "subject_ids", "site_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"Missing BEC arrays: {sorted(missing)}")
    return {key: archive[key] for key in archive.files}


def save_bec_archive(path, bec, coefficients, dataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    bec = np.asarray(bec, dtype=np.float32)
    coefficients = np.asarray(coefficients, dtype=np.float32)
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC [subjects, roi, roi], got {bec.shape}")
    if coefficients.ndim != 4 or coefficients.shape[0] != bec.shape[0]:
        raise ValueError(
            "Expected VAR coefficients [subjects, lags, roi, roi] matching BEC; "
            f"got {coefficients.shape} for {bec.shape}"
        )
    np.savez_compressed(
        path,
        bec=bec,
        var_coefficients=coefficients,
        labels=np.asarray(dataset["labels"], dtype=np.int64),
        subject_ids=np.asarray(dataset["subject_ids"]),
        site_ids=np.asarray(dataset["site_ids"]),
        representation=np.asarray("sparse_var_bec"),
        var_config=np.asarray(json.dumps(var_config(args), sort_keys=True)),
        roi_names=np.asarray([f"ROI_{index + 1:03d}" for index in range(bec.shape[1])]),
    )


def generate_bec_archive(args, dataset):
    config = SparseVARConfig(
        lags=args.lags,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tolerance=args.tol,
        lag_decay=args.lag_decay,
        threshold=args.threshold,
    )
    if args.bec_path.is_file() and not args.regenerate_bec:
        archive = load_bec_archive(args.bec_path)
        if json.loads(str(archive["var_config"].item())) != var_config(args):
            raise ValueError("Existing archive uses a different Sparse VAR configuration")
        if len(archive["bec"]) == len(dataset["time_series"]):
            print(f"using existing Sparse VAR BEC archive: {args.bec_path}")
            return archive
        raise ValueError("Existing Sparse VAR archive is incomplete; use --regenerate-bec")

    bec, coefficients = [], []
    total = len(dataset["time_series"])
    for index, series in enumerate(dataset["time_series"], start=1):
        subject_bec, subject_coefficients = generate_sparse_var_bec(series, config)
        bec.append(subject_bec)
        coefficients.append(subject_coefficients)
        if index % 25 == 0 or index == total:
            print(f"Sparse VAR BEC [{index}/{total}]")
    save_bec_archive(args.bec_path, bec, coefficients, dataset, args)
    return load_bec_archive(args.bec_path)


def classify_bec(args, archive, device):
    labels = np.asarray(archive["labels"], dtype=np.int64)
    rows = []
    for fold, train_index, val_index, test_index in make_stratified_splits(
        labels, args.n_splits, args.seed, args.validation_size
    ):
        train_mean, train_std = fit_bec_scaler(archive["bec"][train_index])
        train_bec = transform_bec(archive["bec"][train_index], train_mean, train_std)
        val_bec = transform_bec(archive["bec"][val_index], train_mean, train_std)
        test_bec = transform_bec(archive["bec"][test_index], train_mean, train_std)
        print(f"fold {fold}: train={len(train_index)}, val={len(val_index)}, test={len(test_index)}")
        for repeat in range(args.classifier_repeats):
            metrics, _ = train_classifier(
                train_bec, labels[train_index], val_bec, labels[val_index],
                test_bec, labels[test_index], device=device,
                seed=args.seed + fold * 1000 + repeat + 1,
                max_epochs=args.classifier_epochs, patience=args.classifier_patience,
                batch_size=32, learning_rate=args.classifier_lr,
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
    print("mean±std: " + ", ".join(f"{name}={value['mean'] * 100:.2f}±{value['std'] * 100:.2f}%" for name, value in summary.items()))


def validate_args(args):
    if args.patient_label == args.control_label:
        raise ValueError("--patient-label and --control-label must be different")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.classifier_repeats < 1:
        raise ValueError("--classifier-repeats must be positive")
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
        archive = load_bec_archive(args.bec_path)
    else:
        dataset = load_subject_dataset(
            args.data_root, pipeline=args.pipeline, strategy=args.strategy,
            derivative=args.derivative, profile=profile, standardize=True,
            max_subjects=args.max_subjects, patient_label=args.patient_label,
            control_label=args.control_label,
        )
        archive = generate_bec_archive(args, dataset)
    print(f"Sparse VAR BEC archive: {args.bec_path} shape={archive['bec'].shape}")
    if args.generation_only:
        return
    rows = classify_bec(args, archive, device)
    save_results(args, rows)


if __name__ == "__main__":
    main()
