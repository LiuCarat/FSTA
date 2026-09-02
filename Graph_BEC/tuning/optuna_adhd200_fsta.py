"""Optuna search for ADHD200 FSTA parameters.

Each trial trains FSTA on all selected ADHD200 subjects, extracts one BEC per
subject, and evaluates Original-BEC classification with the same 10-fold
workflow used by the main pipeline. This module is intentionally separate
from the normal Graph-BEC workflow so other dataset pipelines are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Graph_BEC.main_adhd200 import parse_args
from Graph_BEC.data import load_subject_dataset
from Graph_BEC.dataset_configs import get_profile
from Graph_BEC.model.fusion_graph import load_phenotypes
from Graph_BEC.model.fsta_ec import generate_subject_bec, save_subject_bec
from Graph_BEC.utils import select_device, set_seed
from Graph_BEC.workflow import run_cross_validation

CLASSIFICATION_METRICS = ("ACC", "SPE", "AUC", "Precision", "Recall", "F1")
FIXED_WINDOW_LENGTH = 72
FIXED_STRIDE = 36


def parse_optuna_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--study-name", default="adhd200-fsta-original-allmetrics"
    )
    parser.add_argument("--storage", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def build_base_args(optuna_args):
    sys.argv = [
        sys.argv[0],
        "--input-mode", "raw",
        "--representations", "original",
        "--seed", str(optuna_args.seed),
        "--gpu-id", optuna_args.gpu_id,
    ]
    args = parse_args()
    args.input_mode = "raw"
    args.representations = ["original"]
    args.fsta_checkpoint = "best"
    return args


def suggest_fsta_args(trial, base_args):
    args = argparse.Namespace(**vars(base_args))
    args.epochs = trial.suggest_int("epochs", 50, 140, step=10)
    args.loss_mode = trial.suggest_categorical(
        "loss_mode", ["entropy"]
    )
    args.loss_alpha = trial.suggest_categorical(
        "loss_alpha", [0.001, 0.005, 0.01, 0.02, 0.05]
    )
    args.d_model = trial.suggest_categorical("d_model", [8, 16, 32])
    args.d_inner_hid = trial.suggest_categorical(
        "d_inner_hid", [32, 64, 128]
    )
    args.n_head = trial.suggest_categorical("n_head", [1, 2, 4])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.4)
    args.n_warmup_steps = trial.suggest_categorical(
        "n_warmup_steps", [500, 1000, 2000, 4000]
    )
    args.lr_mul = trial.suggest_float("lr_mul", 0.5, 2.0)
    args.weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    )
    return args


def prepare_subject_data(base_args):
    profile = get_profile("adhd200")
    subjects = load_subject_dataset(
        profile.data_root,
        profile=profile,
        patient_label=base_args.patient_label,
        control_label=base_args.control_label,
    )
    phenotype = load_phenotypes(
        profile.phenotype_path,
        subjects["subject_ids"],
        subjects["site_ids"],
        profile,
    )
    return subjects, phenotype


def save_trial_table(study, output_path):
    """Save every Optuna trial as one row for easy comparison."""
    parameter_names = sorted({
        name for trial in study.trials for name in trial.params
    })
    fieldnames = [
        "trial",
        "state",
        "value",
        "fsta_loss",
        "reconstruction_loss",
        "cv_objective",
        *[f"cv_{metric.lower()}_mean" for metric in CLASSIFICATION_METRICS],
        *[f"cv_{metric.lower()}_std" for metric in CLASSIFICATION_METRICS],
        *parameter_names,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "trial": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "fsta_loss": trial.user_attrs.get("fsta_loss"),
                "reconstruction_loss": trial.user_attrs.get("reconstruction_loss"),
                "cv_objective": trial.user_attrs.get("cv_objective"),
            }
            for metric in CLASSIFICATION_METRICS:
                metric_key = metric.lower()
                row[f"cv_{metric_key}_mean"] = trial.user_attrs.get(
                    f"cv_{metric_key}_mean"
                )
                row[f"cv_{metric_key}_std"] = trial.user_attrs.get(
                    f"cv_{metric_key}_std"
                )
            row.update(trial.params)
            writer.writerow(row)


def main():
    try:
        import optuna
    except ImportError as error:
        raise SystemExit(
            "Optuna is required. Install it with: pip install optuna"
        ) from error

    optuna_args = parse_optuna_args()
    if optuna_args.n_trials < 1:
        raise ValueError("--n-trials must be positive")

    base_args = build_base_args(optuna_args)
    device = select_device(base_args.gpu_id)
    subjects, phenotype = prepare_subject_data(base_args)
    labels = np.asarray(subjects["labels"], dtype=np.int64)
    time_lengths = np.asarray(
        [series.shape[0] for series in subjects["time_series"]],
        dtype=np.int64,
    )
    if int(time_lengths.min()) < FIXED_WINDOW_LENGTH:
        raise ValueError(
            f"ADHD200 contains a time series shorter than the fixed "
            f"window_length={FIXED_WINDOW_LENGTH}: min={time_lengths.min()}"
        )
    base_args.window_length = FIXED_WINDOW_LENGTH
    base_args.stride = FIXED_STRIDE
    print(
        f"ADHD200 time points: min={time_lengths.min()}, "
        f"max={time_lengths.max()}; "
        f"fixed maximum window_length={base_args.window_length}, "
        f"stride={base_args.stride}"
    )
    output_dir = optuna_args.output_dir or (
        base_args.profile.output_dir / "optuna_fsta_original"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        args = suggest_fsta_args(trial, base_args)
        set_seed(args.seed)
        bec_data, fsta_metrics = generate_subject_bec(args, subjects, device)
        data = {
            "bec": bec_data["bec"],
            "labels": labels,
            "subject_ids": subjects["subject_ids"],
            "site_ids": subjects["site_ids"],
            "continuous": phenotype["continuous"],
            "categorical_raw": phenotype["categorical_raw"],
        }
        experiment = run_cross_validation(args, data, device)
        metric_means = {}
        metric_stds = {}
        for metric in CLASSIFICATION_METRICS:
            values = np.asarray([
                result["original"][metric]
                for result in experiment["fold_results"]
            ], dtype=np.float64)
            if not np.isfinite(values).all():
                return -1.0
            metric_means[metric] = float(values.mean())
            metric_stds[metric] = float(values.std())

        objective_value = float(min(metric_means.values()))
        trial.set_user_attr("cv_objective", objective_value)
        trial.set_user_attr("n_splits", len(experiment["fold_results"]))
        for metric in CLASSIFICATION_METRICS:
            metric_key = metric.lower()
            trial.set_user_attr(
                f"cv_{metric_key}_mean", metric_means[metric]
            )
            trial.set_user_attr(
                f"cv_{metric_key}_std", metric_stds[metric]
            )
        print(
            f"trial {trial.number}: Original 10-fold "
            + " | ".join(
                f"{metric}={metric_means[metric]:.4f}±{metric_stds[metric]:.4f}"
                for metric in CLASSIFICATION_METRICS
            )
            + f" | objective(min)={objective_value:.4f}"
        )
        trial.set_user_attr("fsta_loss", fsta_metrics.get("loss"))
        trial.set_user_attr("reconstruction_loss", fsta_metrics.get("reconstruction_loss"))
        return objective_value

    study = optuna.create_study(
        study_name=optuna_args.study_name,
        storage=optuna_args.storage,
        load_if_exists=True,
        direction="maximize",
    )
    study.enqueue_trial({"loss_alpha": 0.01})
    study.optimize(
        objective,
        n_trials=optuna_args.n_trials,
        timeout=optuna_args.timeout,
    )
    save_trial_table(study, output_dir / "trials.csv")

    result = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "seed": base_args.seed,
        "n_splits": base_args.n_splits,
        "objective": "minimum_of_mean_original_10fold_metrics",
        "best_metric_means": {
            metric: study.best_trial.user_attrs.get(f"cv_{metric.lower()}_mean")
            for metric in CLASSIFICATION_METRICS
        },
    }
    (output_dir / "best_params.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    best_args = argparse.Namespace(**vars(base_args))
    for name, value in study.best_params.items():
        setattr(best_args, name, value)
    best_args.stride = best_args.window_length // 2
    best_args.fsta_checkpoint = "best"
    set_seed(best_args.seed)
    best_bec, best_metrics = generate_subject_bec(
        best_args, subjects, device
    )
    final_path = output_dir / "adhd200_original_bec_best.npz"
    save_subject_bec(final_path, best_bec)
    print(f"Saved best Original BEC: {final_path.resolve()}")
    print(f"Best FSTA metrics: {best_metrics}")


if __name__ == "__main__":
    main()
