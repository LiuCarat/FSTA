#!/usr/bin/env python3
"""Individual NPI-MLP BEC generation with the Graph-BEC classifier protocol.

Each subject gets an independently fitted MLP surrogate brain. Its NPI EC
matrix is then evaluated with the same fold construction, BEC scaling,
DirectedBrainNetCNN, early stopping, threshold selection, and metrics used by
Graph-BEC.
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

import NPI
from Graph_BEC.data import load_subject_dataset
from Graph_BEC.downstream import train_classifier
from Graph_BEC.profiles.configuration import get_profile
from Graph_BEC.utils.folds import (
    fit_bec_scaler,
    make_stratified_splits,
    transform_bec,
)
from Graph_BEC.utils.runtime import select_device, set_seed


NPI_EC_AXIS = "rows=perturbed_source; columns=affected_target"


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=["abide", "adhd200"], default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)
    output_dir = BASELINE_DIR / "outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["abide", "adhd200"], default=profile.name)
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--bec-path",
        type=Path,
        default=output_dir / f"subject_npi_mlp_bec_{profile.name}.npz",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=profile.defaults.get("seed", 42))
    parser.add_argument("--n-splits", type=int, default=profile.defaults.get("n_splits", 10))
    parser.add_argument(
        "--validation-size",
        type=float,
        default=profile.defaults.get("validation_size", 0.2),
    )
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--train-proportion", type=float, default=0.8)
    parser.add_argument("--npi-batch-size", type=int, default=50)
    parser.add_argument("--npi-epochs", type=int, default=100)
    parser.add_argument("--npi-lr", type=float, default=1e-3)
    parser.add_argument("--npi-l2", type=float, default=5e-5)
    parser.add_argument("--pert-strength", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=profile.defaults.get("classifier_epochs", 100),
    )
    parser.add_argument(
        "--classifier-patience",
        type=int,
        default=profile.defaults.get("classifier_patience", 20),
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=profile.defaults.get("classifier_lr", 1e-3),
    )
    parser.add_argument(
        "--classifier-repeats",
        type=int,
        default=profile.defaults.get("classifier_repeats", 1),
    )
    parser.add_argument("--patient-label", type=int, default=profile.defaults.get("patient_label", 1), choices=[0, 1])
    parser.add_argument("--control-label", type=int, default=profile.defaults.get("control_label", 0), choices=[0, 1])
    parser.add_argument("--regenerate-bec", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


def npi_config(args, roi_count):
    return {
        "method": "individual_npi_mlp",
        "roi_count": roi_count,
        "steps": args.steps,
        "train_proportion": args.train_proportion,
        "batch_size": args.npi_batch_size,
        "epochs": args.npi_epochs,
        "learning_rate": args.npi_lr,
        "l2": args.npi_l2,
        "pert_strength": args.pert_strength,
        "hidden_dim": args.hidden_dim or 2 * roi_count,
        "latent_dim": args.latent_dim or int(0.8 * roi_count),
        "ec_axes": NPI_EC_AXIS,
        "uses_subject_labels": False,
        "fitting": "subject-wise surrogate fitting",
    }


def load_bec_archive(path):
    archive = np.load(path, allow_pickle=False)
    required = {"bec", "labels", "subject_ids", "site_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"Missing BEC arrays: {sorted(missing)}")
    return {key: archive[key] for key in archive.files}


def save_bec_archive(
    path,
    bec,
    test_reconstruction_mse,
    train_reconstruction_mse,
    dataset,
    config,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    bec = np.asarray(bec, dtype=np.float32)
    test_reconstruction_mse = np.asarray(test_reconstruction_mse, dtype=np.float32)
    train_reconstruction_mse = np.asarray(train_reconstruction_mse, dtype=np.float32)
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC [subjects, roi, roi], got {bec.shape}")
    if test_reconstruction_mse.shape != (len(bec),):
        raise ValueError(
            "Expected one reconstruction MSE per subject; "
            f"got {test_reconstruction_mse.shape} for {bec.shape}"
        )
    if train_reconstruction_mse.shape != (len(bec),):
        raise ValueError(
            "Expected one training reconstruction MSE per subject; "
            f"got {train_reconstruction_mse.shape} for {bec.shape}"
        )
    np.savez_compressed(
        path,
        bec=bec,
        npi_train_reconstruction_mse=train_reconstruction_mse,
        npi_test_reconstruction_mse=test_reconstruction_mse,
        labels=np.asarray(dataset["labels"], dtype=np.int64),
        subject_ids=np.asarray(dataset["subject_ids"]),
        site_ids=np.asarray(dataset["site_ids"]),
        npi_config=np.asarray(json.dumps(config, sort_keys=True)),
    )


def build_mlp(args, roi_count):
    return NPI.ANN_MLP(
        input_dim=args.steps * roi_count,
        hidden_dim=args.hidden_dim or 2 * roi_count,
        latent_dim=args.latent_dim or int(0.8 * roi_count),
        output_dim=roi_count,
    )


def generate_subject_npi_bec(time_series, args, subject_index):
    if time_series.ndim != 2:
        raise ValueError(f"Expected [time, roi] time series, got {time_series.shape}")
    if len(time_series) <= args.steps + 1:
        raise ValueError(
            f"Subject {subject_index} has too few time points for steps={args.steps}: "
            f"{time_series.shape}"
        )
    set_seed(args.seed + subject_index)
    input_x, target_y = NPI.multi2one(time_series, steps=args.steps)
    model = build_mlp(args, time_series.shape[1])
    model, train_loss, test_loss = NPI.train_NN(
        model,
        input_x,
        target_y,
        batch_size=args.npi_batch_size,
        train_set_proportion=args.train_proportion,
        num_epochs=args.npi_epochs,
        lr=args.npi_lr,
        l2=args.npi_l2,
    )
    ec = NPI.model_EC(model, input_x, target_y, pert_strength=args.pert_strength)
    ec = np.asarray(ec, dtype=np.float32)
    np.fill_diagonal(ec, 0.0)
    if not np.isfinite(ec).all():
        raise ValueError(f"Non-finite NPI-BEC for subject index {subject_index}")
    return ec, float(test_loss[-1]), float(train_loss[-1])


def generate_bec_archive(args, dataset, device):
    NPI.device = device
    config = npi_config(args, dataset["time_series"][0].shape[1])
    if args.bec_path.is_file() and not args.regenerate_bec:
        archive = load_bec_archive(args.bec_path)
        existing = json.loads(str(archive["npi_config"].item())) if "npi_config" in archive else None
        if existing != config:
            raise ValueError(
                "Existing NPI archive uses a different configuration; "
                "use --regenerate-bec or a different --bec-path"
            )
        if len(archive["bec"]) == len(dataset["time_series"]):
            print(f"using existing NPI-BEC archive: {args.bec_path}")
            return archive
        raise ValueError("Existing NPI archive is incomplete; use --regenerate-bec")

    bec, test_losses, train_losses = [], [], []
    total = len(dataset["time_series"])
    for index, series in enumerate(dataset["time_series"]):
        subject_bec, test_loss, train_loss = generate_subject_npi_bec(series, args, index)
        bec.append(subject_bec)
        test_losses.append(test_loss)
        train_losses.append(train_loss)
        if (index + 1) % 10 == 0 or index + 1 == total:
            print(f"Individual NPI-MLP BEC [{index + 1}/{total}]")
    save_bec_archive(
        args.bec_path,
        bec,
        test_losses,
        train_losses,
        dataset,
        config,
    )
    return load_bec_archive(args.bec_path)


def classify_bec(args, archive, device):
    labels = np.asarray(archive["labels"], dtype=np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("The dataset must contain exactly two patient/control labels")
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
                train_bec,
                labels[train_index],
                val_bec,
                labels[val_index],
                test_bec,
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
    with (args.output_dir / f"metrics_npi_mlp_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)
    with (args.output_dir / f"metrics_npi_mlp_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
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
    with (args.output_dir / f"summary_npi_mlp_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print("mean±std: " + ", ".join(
        f"{name}={value['mean'] * 100:.2f}±{value['std'] * 100:.2f}%"
        for name, value in summary.items()
    ))


def validate_args(args):
    if args.patient_label == args.control_label:
        raise ValueError("--patient-label and --control-label must be different")
    if args.steps < 1 or args.npi_epochs < 1 or args.npi_batch_size < 1:
        raise ValueError("NPI steps, epochs, and batch size must be positive")
    if not 0.0 < args.train_proportion < 1.0:
        raise ValueError("--train-proportion must be between 0 and 1")
    if args.generation_only and args.classification_only:
        raise ValueError("--generation-only and --classification-only cannot be combined")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    profile = get_profile(args.dataset)
    device = select_device(args.gpu_id)
    print(f"dataset: {args.dataset}; device: {device}; EC axes: {NPI_EC_AXIS}")
    if args.classification_only:
        archive = load_bec_archive(args.bec_path)
    else:
        dataset = load_subject_dataset(
            args.data_root,
            pipeline=args.pipeline,
            strategy=args.strategy,
            derivative=args.derivative,
            standardize=True,
            max_subjects=args.max_subjects,
            profile=profile,
            patient_label=args.patient_label,
            control_label=args.control_label,
        )
        archive = generate_bec_archive(args, dataset, device)
    if archive["bec"].ndim != 3 or archive["bec"].shape[1:] != (profile.roi_count, profile.roi_count):
        raise ValueError(
            f"Expected NPI-BEC shape [subjects, {profile.roi_count}, {profile.roi_count}], "
            f"got {archive['bec'].shape}"
        )
    print(f"NPI-BEC archive: {args.bec_path}; shape={archive['bec'].shape}")
    if args.generation_only:
        return
    rows = classify_bec(args, archive, device)
    save_results(args, rows)


if __name__ == "__main__":
    main()
