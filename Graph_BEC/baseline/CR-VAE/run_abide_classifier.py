"""Generate subject-level CR-VAE BECs, then run 10-fold classification.

CR-VAE exposes one Granger-causality matrix per fitted model. Consequently,
subject-level BEC evaluation requires fitting one CR-VAE model per subject.
The generated matrices are saved once and reused by the downstream folds.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_BEC.data import load_subject_dataset
from Graph_BEC.dataset_configs import get_profile
from Graph_BEC.downstream import train_classifier
from Graph_BEC.utils.folds import (
    fit_bec_scaler,
    make_stratified_splits,
    transform_bec,
)
from Graph_BEC.utils.runtime import set_seed
from models.cgru_error import CRVAE, train_phase1


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=["abide", "abide_ii"], default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)
    output_dir = Path(__file__).parent / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["abide", "abide_ii"], default=profile.name)
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--bec-path", type=Path, default=output_dir / f"subject_bec_{profile.name}.npz")
    parser.add_argument("--regenerate-bec", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use 100 CR-VAE iterations and batch size 64 for quick BEC generation",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="save the BEC archive every N subjects",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--crvae-context", type=int, default=20)
    parser.add_argument("--crvae-hidden", type=int, default=64)
    parser.add_argument("--crvae-max-iter", type=int, default=1000)
    # The original demo default is 2048, but that is too aggressive for
    # 90-ROI subject-level training on typical GPUs.
    parser.add_argument("--crvae-batch-size", type=int, default=256)
    parser.add_argument("--crvae-check-every", type=int, default=50)
    parser.add_argument("--crvae-lr", type=float, default=5e-2)
    parser.add_argument("--crvae-lambda", type=float, default=0.1)
    parser.add_argument("--crvae-ridge", type=float, default=0.0)
    parser.add_argument("--crvae-verbose", type=int, choices=[0, 1], default=0)

    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-repeats", type=int, default=1)
    parser.add_argument("--patient-label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--control-label", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def choose_device(gpu_id):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == "auto":
        return torch.device("cuda")
    return torch.device(f"cuda:{gpu_id}")


def crvae_config(args):
    return {
        "context": args.crvae_context,
        "hidden": args.crvae_hidden,
        "max_iter": args.crvae_max_iter,
        "batch_size": args.crvae_batch_size,
        "check_every": args.crvae_check_every,
        "lr": args.crvae_lr,
        "lambda": args.crvae_lambda,
        "ridge": args.crvae_ridge,
    }


def save_bec_archive(path, bec, dataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(bec)
    np.savez_compressed(
        path,
        bec=np.asarray(bec, dtype=np.float32),
        labels=np.asarray(dataset["labels"][:count], dtype=np.int64),
        subject_ids=np.asarray(dataset["subject_ids"][:count]),
        site_ids=np.asarray(dataset["site_ids"][:count]),
        crvae_config=np.asarray(json.dumps(crvae_config(args), sort_keys=True)),
    )


def load_bec_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"bec", "labels", "subject_ids", "site_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing CR-VAE BEC arrays: {sorted(missing)}")
        return {key: archive[key] for key in archive.files}


def validate_archive(archive, dataset, args):
    count = len(archive["bec"])
    if count > len(dataset["subject_ids"]):
        raise ValueError("BEC archive contains more subjects than the current dataset")
    expected = np.asarray(dataset["subject_ids"][:count]).astype(str)
    archived = np.asarray(archive["subject_ids"]).astype(str)
    if not np.array_equal(expected, archived):
        raise ValueError("BEC archive subject order does not match the current dataset")
    if archive["bec"].ndim != 3 or archive["bec"].shape[1] != archive["bec"].shape[2]:
        raise ValueError(f"Expected BEC shape [N, R, R], got {archive['bec'].shape}")
    if "crvae_config" in archive:
        archived_config = json.loads(str(archive["crvae_config"].item()))
        if archived_config != crvae_config(args):
            raise ValueError(
                "Existing BEC archive uses different CR-VAE parameters; "
                "use --regenerate-bec or restore the archived parameters"
            )


def train_subject_crvae(series, args, device, seed):
    if len(series) <= args.crvae_context:
        raise ValueError(
            f"Time series length {len(series)} must exceed context "
            f"{args.crvae_context}"
        )
    set_seed(seed)
    dimension = series.shape[1]
    full_connection = np.ones((dimension, dimension), dtype=np.float32)
    model = CRVAE(dimension, full_connection, hidden=args.crvae_hidden).to(device)
    inputs = torch.as_tensor(series[np.newaxis], dtype=torch.float32, device=device)
    train_phase1(
        model,
        inputs,
        context=args.crvae_context,
        lam=args.crvae_lambda,
        lam_ridge=args.crvae_ridge,
        lr=args.crvae_lr,
        max_iter=args.crvae_max_iter,
        check_every=args.crvae_check_every,
        verbose=args.crvae_verbose,
        batch_size=args.crvae_batch_size,
    )
    bec = model.GC(threshold=False).detach().cpu().numpy().astype(np.float32)
    diagonal = np.arange(dimension)
    bec[diagonal, diagonal] = 0.0
    return bec


def generate_subject_bec(args, dataset, device):
    existing = None
    if args.bec_path.is_file() and not args.regenerate_bec:
        existing = load_bec_archive(args.bec_path)
        validate_archive(existing, dataset, args)
        if len(existing["bec"]) == len(dataset["time_series"]):
            print(f"using complete BEC archive: {args.bec_path}")
            return existing
        print(
            f"resuming BEC generation at subject {len(existing['bec']) + 1} "
            f"from {args.bec_path}"
        )

    bec = [] if existing is None else list(existing["bec"])
    total = len(dataset["time_series"])
    for index in range(len(bec), total):
        subject_id = dataset["subject_ids"][index]
        print(f"CR-VAE BEC [{index + 1}/{total}] subject={subject_id}")
        subject_bec = train_subject_crvae(
            dataset["time_series"][index], args, device, args.seed + index
        )
        bec.append(subject_bec)
        if (index + 1) % args.checkpoint_every == 0 or index + 1 == total:
            save_bec_archive(args.bec_path, bec, dataset, args)
        del subject_bec

    archive = load_bec_archive(args.bec_path)
    validate_archive(archive, dataset, args)
    return archive


def classify_bec(args, archive, device):
    labels = np.where(archive["labels"] == 1, args.patient_label, args.control_label)
    rows = []
    for fold, train_index, val_index, test_index in make_stratified_splits(
        labels, args.n_splits, args.seed, args.validation_size
    ):
        mean, std = fit_bec_scaler(archive["bec"][train_index])
        train_bec = transform_bec(archive["bec"][train_index], mean, std)
        val_bec = transform_bec(archive["bec"][val_index], mean, std)
        test_bec = transform_bec(archive["bec"][test_index], mean, std)
        print(
            f"fold {fold}: train={len(train_index)}, "
            f"val={len(val_index)}, test={len(test_index)}"
        )
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
            report = ", ".join(
                f"{name}={value * 100:.2f}%" for name, value in metrics.items()
            )
            print(f"  repeat {repeat + 1}: {report}")
    return rows


def save_classification_results(args, rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)
    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [name for name in rows[0] if name not in {"fold", "repeat"}]
    summary = {
        name: {
            "mean": float(np.nanmean([row[name] for row in rows])),
            "std": float(np.nanstd([row[name] for row in rows])),
        }
        for name in metric_names
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    report = ", ".join(
        f"{name}={values['mean'] * 100:.2f}±{values['std'] * 100:.2f}%"
        for name, values in summary.items()
    )
    print(f"mean±std: {report}")


def validate_args(args):
    if args.patient_label == args.control_label:
        raise ValueError("--patient-label and --control-label must be different")
    if args.crvae_context < 11:
        raise ValueError("--crvae-context must be at least 11 for this CR-VAE code")
    if args.crvae_max_iter < 1:
        raise ValueError("--crvae-max-iter must be positive")
    if args.classifier_repeats < 1:
        raise ValueError("--classifier-repeats must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.generation_only and args.classification_only:
        raise ValueError(
            "--generation-only and --classification-only cannot be used together"
        )


def main():
    args = parse_args()
    if args.fast:
        args.crvae_max_iter = 100
        args.crvae_batch_size = 64
        args.crvae_check_every = 25
    validate_args(args)
    device = choose_device(args.gpu_id)
    print(f"device: {device}")
    if args.classification_only:
        archive = load_bec_archive(args.bec_path)
        print(f"subject BEC archive: {args.bec_path} shape={archive['bec'].shape}")
        rows = classify_bec(args, archive, device)
        save_classification_results(args, rows)
        print(f"classification results: {args.output_dir}")
        return
    dataset = load_subject_dataset(
        args.data_root,
        args.pipeline,
        args.strategy,
        args.derivative,
        standardize=True,
        max_subjects=args.max_subjects,
        profile=get_profile(args.dataset),
        patient_label=args.patient_label,
        control_label=args.control_label,
    )
    archive = generate_subject_bec(args, dataset, device)
    print(f"subject BEC archive: {args.bec_path} shape={archive['bec'].shape}")
    if args.generation_only:
        return
    rows = classify_bec(args, archive, device)
    save_classification_results(args, rows)
    print(f"classification results: {args.output_dir}")


if __name__ == "__main__":
    main()
