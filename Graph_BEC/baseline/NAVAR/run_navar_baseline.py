"""Generate subject-level NAVAR BEC matrices and classify them."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parent
for path in (ROOT, SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Graph_BEC.data import load_subject_dataset
from Graph_BEC.downstream import train_classifier
from Graph_BEC.profiles.configuration import get_profile
from Graph_BEC.utils.folds import fit_bec_scaler, make_stratified_splits, transform_bec
from Graph_BEC.utils.runtime import select_device, set_seed
from train_NAVAR import train_NAVAR


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
    parser.add_argument("--maxlags", type=int, default=5)
    parser.add_argument("--hidden-nodes", type=int, default=10)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sparsity-penalty", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--bec-path", type=Path, default=output_dir / f"subject_navar_bec_{profile.name}.npz")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
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


def navar_config(args):
    names = (
        "maxlags", "hidden_nodes", "hidden_layers", "epochs", "batch_size",
        "sparsity_penalty", "weight_decay", "dropout", "learning_rate", "lstm",
    )
    return {name: getattr(args, name) for name in names}


def load_archive(path):
    archive = np.load(path, allow_pickle=False)
    required = {"bec", "labels", "subject_ids", "site_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"Missing NAVAR arrays: {sorted(missing)}")
    return {key: archive[key] for key in archive.files}


def save_archive(path, matrices, dataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    bec = np.asarray(matrices, dtype=np.float32)
    np.savez_compressed(
        path,
        bec=bec,
        navar_scores=bec,
        labels=np.asarray(dataset["labels"], dtype=np.int64),
        subject_ids=np.asarray(dataset["subject_ids"]).astype(str),
        site_ids=np.asarray(dataset["site_ids"]).astype(str),
        representation=np.asarray("navar_contribution_std"),
        navar_config=np.asarray(json.dumps(navar_config(args), sort_keys=True)),
        roi_names=np.asarray([f"ROI_{index + 1:03d}" for index in range(bec.shape[1])]),
    )


def fit_subject(series, args, device):
    if series.shape[0] <= args.maxlags + 1:
        raise ValueError(
            f"NAVAR needs more than maxlags+1 time points; got T={series.shape[0]}"
        )
    scores, _, _ = train_NAVAR(
        np.asarray(series, dtype=np.float32),
        maxlags=args.maxlags,
        hidden_nodes=args.hidden_nodes,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        lambda1=args.sparsity_penalty,
        val_proportion=0.0,
        weight_decay=args.weight_decay,
        check_every=max(args.epochs // 10, 1),
        hidden_layers=args.hidden_layers,
        normalize=True,
        split_timeseries=False,
        lstm=args.lstm,
        device=device,
        verbose=False,
    )
    bec = np.asarray(scores, dtype=np.float32)
    if bec.shape != (series.shape[1], series.shape[1]):
        raise ValueError(f"Unexpected NAVAR score shape: {bec.shape}")
    if not np.isfinite(bec).all():
        raise ValueError("NAVAR produced non-finite BEC values")
    bec = bec.copy()
    np.fill_diagonal(bec, 0.0)
    return bec


def generate_archive(args, dataset, device):
    if args.bec_path.is_file() and not args.regenerate_bec:
        archive = load_archive(args.bec_path)
        stored = json.loads(str(archive["navar_config"].item()))
        if stored != navar_config(args):
            raise ValueError("Existing archive uses a different NAVAR configuration; use --regenerate-bec")
        if len(archive["bec"]) == len(dataset["time_series"]):
            print(f"using existing NAVAR archive: {args.bec_path}", flush=True)
            return archive
        raise ValueError("Existing NAVAR archive is incomplete; use --regenerate-bec")

    matrices = []
    total = len(dataset["time_series"])
    print(
        f"starting NAVAR fitting: {total} subjects; device={device}; "
        f"epochs={args.epochs}",
        flush=True,
    )
    start_time = time.perf_counter()
    for index, series in enumerate(dataset["time_series"], start=1):
        subject_start = time.perf_counter()
        set_seed(args.seed + index)
        matrices.append(fit_subject(series, args, device))
        if index == 1 or index == total or index % args.progress_every == 0:
            elapsed = time.perf_counter() - start_time
            per_subject = elapsed / index
            remaining = per_subject * (total - index)
            print(
                f"finished subject {index}/{total}; "
                f"current={time.perf_counter() - subject_start:.1f}s; "
                f"elapsed={elapsed / 60:.1f}min; "
                f"eta={remaining / 60:.1f}min",
                flush=True,
            )
    save_archive(args.bec_path, matrices, dataset, args)
    return load_archive(args.bec_path)


def classify_archive(args, archive, device):
    labels = np.asarray(archive["labels"], dtype=np.int64)
    rows = []
    for fold, train_index, val_index, test_index in make_stratified_splits(
        labels, args.n_splits, args.seed, args.validation_size
    ):
        mean, std = fit_bec_scaler(archive["bec"][train_index])
        train_bec = transform_bec(archive["bec"][train_index], mean, std)
        val_bec = transform_bec(archive["bec"][val_index], mean, std)
        test_bec = transform_bec(archive["bec"][test_index], mean, std)
        print(f"fold {fold}: train={len(train_index)}, val={len(val_index)}, test={len(test_index)}", flush=True)
        for repeat in range(args.classifier_repeats):
            metrics, _ = train_classifier(
                train_bec, labels[train_index], val_bec, labels[val_index],
                test_bec, labels[test_index], device=device,
                seed=args.seed + fold * 1000 + repeat + 1,
                max_epochs=args.classifier_epochs,
                patience=args.classifier_patience,
                batch_size=32,
                learning_rate=args.classifier_lr,
            )
            rows.append({"fold": fold, "repeat": repeat + 1, **metrics})
            print("  " + ", ".join(f"{name}={value * 100:.2f}%" for name, value in metrics.items()), flush=True)
    return rows


def save_results(args, rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.dataset
    with (args.output_dir / f"metrics_navar_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)
    with (args.output_dir / f"metrics_navar_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
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
    with (args.output_dir / f"summary_navar_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print(
        "mean±std: "
        + ", ".join(
            f"{name}={value['mean'] * 100:.2f}±{value['std'] * 100:.2f}%"
            for name, value in summary.items()
        ),
        flush=True,
    )


def validate_args(args):
    if args.patient_label == args.control_label:
        raise ValueError("patient-label and control-label must be different")
    if args.generation_only and args.classification_only:
        raise ValueError("generation-only and classification-only cannot be used together")
    if args.maxlags < 1 or args.hidden_nodes < 1 or args.hidden_layers < 1 or args.epochs < 1:
        raise ValueError("maxlags, hidden-nodes, hidden-layers, and epochs must be positive")
    if args.batch_size < 1 or not 0 <= args.dropout < 1:
        raise ValueError("batch-size must be positive and dropout must be in [0, 1)")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.gpu_id)
    if args.classification_only:
        archive = load_archive(args.bec_path)
    else:
        profile = get_profile(args.dataset)
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
        archive = generate_archive(args, dataset, device)
    print(
        f"NAVAR archive: {args.bec_path}; shape={archive['bec'].shape}; "
        f"classifier device={device}",
        flush=True,
    )
    if not args.generation_only:
        save_results(args, classify_archive(args, archive, device))


if __name__ == "__main__":
    main()
