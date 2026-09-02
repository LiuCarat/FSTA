"""Generate subject-level GVAR BECs and optionally run the Graph-BEC classifier."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from Graph_BEC.dataset_configs import get_profile
from Graph_BEC.utils.folds import fit_bec_scaler, make_stratified_splits, transform_bec
from Graph_BEC.utils.runtime import set_seed
from training import training_procedure


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=["abide", "abide_ii", "adhd200"], default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)
    output_dir = Path(__file__).resolve().parent / "outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["abide", "abide_ii", "adhd200"], default=profile.name)
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--hidden-layer-size", type=int, default=16)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda", dest="lmbd", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta-1", type=float, default=0.9)
    parser.add_argument("--beta-2", type=float, default=0.999)
    parser.add_argument("--lag-decay", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--bec-path", type=Path, default=output_dir / f"subject_gvar_bec_{profile.name}.npz")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
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
    parser.add_argument("--regenerate-bec", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


def choose_device(gpu_id, force_cpu=False):
    if force_cpu or gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id in {"auto", "cuda"}:
        return torch.device("cuda")
    if isinstance(gpu_id, str) and gpu_id.startswith("cuda:"):
        return torch.device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def gvar_config(args):
    return {
        "method": "GVAR",
        "order": args.order,
        "hidden_layer_size": args.hidden_layer_size,
        "num_hidden_layers": args.num_hidden_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lambda": args.lmbd,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "beta_1": args.beta_1,
        "beta_2": args.beta_2,
        "lag_decay": args.lag_decay,
        "workers": args.workers,
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
        raise ValueError("Expected GVAR coefficients [subjects, order, roi, roi]")
    np.savez_compressed(
        path,
        bec=bec,
        gvar_coefficients=coefficients,
        labels=np.asarray(dataset["labels"], dtype=np.int64),
        subject_ids=np.asarray(dataset["subject_ids"]),
        site_ids=np.asarray(dataset["site_ids"]),
        representation=np.asarray("gvar_bec"),
        gvar_config=np.asarray(json.dumps(gvar_config(args), sort_keys=True)),
        roi_names=np.asarray([f"ROI_{index + 1:03d}" for index in range(bec.shape[1])]),
    )


def _fit_subject(task):
    index, subject_id, series, settings = task
    started = time.perf_counter()
    causal_structure, coefficients = training_procedure(
        data=[series],
        order=settings["order"],
        hidden_layer_size=settings["hidden_layer_size"],
        end_epoch=settings["epochs"],
        batch_size=settings["batch_size"],
        lmbd=settings["lmbd"],
        gamma=settings["gamma"],
        seed=settings["seed"] + index,
        num_hidden_layers=settings["num_hidden_layers"],
        initial_learning_rate=settings["learning_rate"],
        beta_1=settings["beta_1"],
        beta_2=settings["beta_2"],
        use_cuda=settings["use_cuda"],
        verbose=False,
    )
    coefficients = np.asarray(coefficients, dtype=np.float32)
    if coefficients.ndim != 4:
        raise ValueError(f"GVAR returned unexpected coefficients: {coefficients.shape}")
    lagged = np.median(coefficients, axis=0)
    weights = np.power(settings["lag_decay"], np.arange(settings["order"], dtype=np.float32))
    bec = np.tensordot(weights, lagged, axes=(0, 0)) / weights.sum()
    np.fill_diagonal(bec, 0.0)
    return index, subject_id, bec, lagged, time.perf_counter() - started


def _subject_tasks(args, dataset):
    settings = {
        "order": args.order,
        "hidden_layer_size": args.hidden_layer_size,
        "num_hidden_layers": args.num_hidden_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lmbd": args.lmbd,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "beta_1": args.beta_1,
        "beta_2": args.beta_2,
        "lag_decay": args.lag_decay,
        "use_cuda": not args.cpu and torch.cuda.is_available(),
        "seed": args.seed,
    }
    return [
        (index, subject_id, series, settings)
        for index, (subject_id, series) in enumerate(
            zip(dataset["subject_ids"], dataset["time_series"]), start=1
        )
    ]


def generate_bec_archive(args, dataset):
    if args.bec_path.is_file() and not args.regenerate_bec:
        archive = load_bec_archive(args.bec_path)
        if json.loads(str(archive["gvar_config"].item())) != gvar_config(args):
            raise ValueError("Existing archive uses a different GVAR configuration")
        if len(archive["bec"]) == len(dataset["time_series"]):
            print(f"using existing GVAR BEC archive: {args.bec_path}", flush=True)
            return archive
        raise ValueError("Existing GVAR archive is incomplete; use --regenerate-bec")

    tasks = _subject_tasks(args, dataset)
    total = len(tasks)
    print(f"starting GVAR fitting: {total} subjects; workers={args.workers}", flush=True)
    results = [None] * total
    if args.workers == 1:
        completed = (_fit_subject(task) for task in tasks)
        for result in completed:
            index, subject_id, bec, coefficients, elapsed = result
            results[index - 1] = (bec, coefficients)
            print(f"finished subject {index}/{total}: {subject_id} in {elapsed / 60:.2f} min", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {executor.submit(_fit_subject, task): task[0] for task in tasks}
            for future in as_completed(futures):
                index, subject_id, bec, coefficients, elapsed = future.result()
                results[index - 1] = (bec, coefficients)
                print(f"finished subject {index}/{total}: {subject_id} in {elapsed / 60:.2f} min", flush=True)
    bec, coefficients = zip(*results)
    save_bec_archive(args.bec_path, bec, coefficients, dataset, args)
    return load_bec_archive(args.bec_path)


def classify_bec(args, archive, device):
    labels = np.asarray(archive["labels"], dtype=np.int64)
    rows = []
    for fold, train_index, val_index, test_index in make_stratified_splits(labels, args.n_splits, args.seed, args.validation_size):
        train_mean, train_std = fit_bec_scaler(archive["bec"][train_index])
        train_bec = transform_bec(archive["bec"][train_index], train_mean, train_std)
        val_bec = transform_bec(archive["bec"][val_index], train_mean, train_std)
        test_bec = transform_bec(archive["bec"][test_index], train_mean, train_std)
        print(f"fold {fold}: train={len(train_index)}, val={len(val_index)}, test={len(test_index)}", flush=True)
        for repeat in range(args.classifier_repeats):
            metrics, _ = train_classifier(train_bec, labels[train_index], val_bec, labels[val_index], test_bec, labels[test_index], device=device, seed=args.seed + fold * 1000 + repeat + 1, max_epochs=args.classifier_epochs, patience=args.classifier_patience, batch_size=32, learning_rate=args.classifier_lr)
            rows.append({"fold": fold, "repeat": repeat + 1, **metrics})
            print("  " + ", ".join(f"{name}={value * 100:.2f}%" for name, value in metrics.items()), flush=True)
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
    summary = {name: {"mean": float(np.nanmean([row[name] for row in rows])), "std": float(np.nanstd([row[name] for row in rows]))} for name in names}
    with (args.output_dir / f"summary_{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print("mean±std: " + ", ".join(f"{name}={value['mean'] * 100:.2f}±{value['std'] * 100:.2f}%" for name, value in summary.items()), flush=True)


def validate_args(args):
    if args.order < 1 or args.hidden_layer_size < 1 or args.num_hidden_layers < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("order, hidden-layer-size, num-hidden-layers, epochs, and batch-size must be positive")
    if args.lag_decay < 0 or args.workers < 1:
        raise ValueError("lag-decay must be non-negative and workers must be positive")
    if args.patient_label == args.control_label:
        raise ValueError("--patient-label and --control-label must be different")
    if args.n_splits < 2 or args.classifier_repeats < 1:
        raise ValueError("n-splits must be at least 2 and classifier-repeats must be positive")
    if args.generation_only and args.classification_only:
        raise ValueError("--generation-only and --classification-only cannot be used together")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    profile = get_profile(args.dataset)
    device = choose_device(args.gpu_id, args.cpu)
    print(f"dataset: {args.dataset}; classifier device: {device}; CUDA available: {torch.cuda.is_available()}", flush=True)
    if args.classification_only:
        archive = load_bec_archive(args.bec_path)
    else:
        print(f"loading fMRI data from {args.data_root}", flush=True)
        dataset = load_subject_dataset(args.data_root, pipeline=args.pipeline, strategy=args.strategy, derivative=args.derivative, profile=profile, standardize=True, max_subjects=args.max_subjects, patient_label=args.patient_label, control_label=args.control_label)
        print(f"loaded {len(dataset['time_series'])} subjects; starting subject-wise fitting", flush=True)
        archive = generate_bec_archive(args, dataset)
    print(f"GVAR BEC archive: {args.bec_path} shape={archive['bec'].shape}", flush=True)
    if args.generation_only:
        return
    save_results(args, classify_bec(args, archive, device))


if __name__ == "__main__":
    main()
