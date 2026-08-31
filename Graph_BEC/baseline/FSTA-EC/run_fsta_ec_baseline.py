"""Generate subject-level FSTA-EC BECs and run downstream classification."""

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

from arguments import add_fsta_arguments
from Graph_BEC.data import load_subject_dataset
from Graph_BEC.downstream import train_classifier
from Graph_BEC.profiles.configuration import get_profile
from Graph_BEC.utils.folds import fit_bec_scaler, make_stratified_splits, transform_bec
from Graph_BEC.utils.runtime import select_device, set_seed
from bec_generation import generate_subject_bec


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
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--bec-path", type=Path, default=output_dir / f"subject_fsta_ec_bec_{profile.name}.npz")
    parser.add_argument("--max-subjects", type=int)
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
    add_fsta_arguments(parser)
    return parser.parse_args()


def fsta_config(args):
    names = (
        "window_length", "stride", "epochs", "alpha_sp",
        "batch_size", "d_model", "d_inner_hid", "d_k", "d_v",
        "n_head", "dropout", "n_warmup_steps", "lr_mul", "weight_decay",
        "adam_beta1", "adam_beta2", "num_hidden_layers", "num_attention_heads",
        "hidden_act", "attention_probs_dropout_prob", "hidden_dropout_prob",
        "initializer_range", "no_filters",
    )
    return {name: getattr(args, name) for name in names}


def load_archive(path):
    archive = np.load(path, allow_pickle=False)
    required = {"bec", "labels", "subject_ids", "site_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"Missing FSTA-EC arrays: {sorted(missing)}")
    return {key: archive[key] for key in archive.files}


def save_archive(path, generated, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bec=np.asarray(generated["bec"], dtype=np.float32),
        labels=np.asarray(generated["labels"], dtype=np.int64),
        subject_ids=np.asarray(generated["subject_ids"]).astype(str),
        site_ids=np.asarray(generated["site_ids"]).astype(str),
        reconstruction_mse=np.asarray(generated["reconstruction_mse"], dtype=np.float32),
        representation=np.asarray("fsta_ec_original"),
        fsta_config=np.asarray(json.dumps(fsta_config(args), sort_keys=True)),
        roi_names=np.asarray([f"ROI_{index + 1:03d}" for index in range(generated["bec"].shape[1])]),
    )


def generate_archive(args, dataset, device):
    if args.bec_path.is_file() and not args.regenerate_bec:
        archive = load_archive(args.bec_path)
        stored_config = json.loads(str(archive["fsta_config"].item()))
        current_config = fsta_config(args)
        if stored_config != current_config:
            differences = {
                name: {"archive": stored_config.get(name), "current": value}
                for name, value in current_config.items()
                if stored_config.get(name) != value
            }
            raise ValueError(
                "Existing archive uses a different FSTA-EC configuration: "
                f"{differences}. Use --regenerate-bec to rebuild it."
            )
        if len(archive["bec"]) == len(dataset["time_series"]):
            print(f"using existing FSTA-EC archive: {args.bec_path}", flush=True)
            return archive
        raise ValueError("Existing FSTA-EC archive is incomplete; use --regenerate-bec")

    generated, metrics = generate_subject_bec(args, dataset, device)
    generated["fsta_training"] = metrics
    save_archive(args.bec_path, generated, args)
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
        print(
            f"fold {fold}: train={len(train_index)}, val={len(val_index)}, "
            f"test={len(test_index)}",
            flush=True,
        )
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
            print(
                "  "
                + ", ".join(
                    f"{name}={value * 100:.2f}%"
                    for name, value in metrics.items()
                ),
                flush=True,
            )
    return rows


def save_results(args, rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.dataset
    with (args.output_dir / f"metrics_fsta_ec_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)
    with (args.output_dir / f"metrics_fsta_ec_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
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
    with (args.output_dir / f"summary_fsta_ec_{suffix}.json").open("w", encoding="utf-8") as handle:
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
    if args.window_length < 1 or args.stride < 1 or args.epochs < 1:
        raise ValueError("window-length, stride, and epochs must be positive")


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
            args.data_root, pipeline=args.pipeline, strategy=args.strategy,
            derivative=args.derivative, profile=profile, standardize=True,
            max_subjects=args.max_subjects, patient_label=args.patient_label,
            control_label=args.control_label,
        )
        archive = generate_archive(args, dataset, device)
    print(f"FSTA-EC archive: {args.bec_path}; shape={archive['bec'].shape}; device={device}", flush=True)
    if not args.generation_only:
        save_results(args, classify_archive(args, archive, device))


if __name__ == "__main__":
    main()
