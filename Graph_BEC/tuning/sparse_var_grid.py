"""Run the six predefined Sparse VAR configurations and rank the results.

This is a fixed grid experiment, not a stochastic hyperparameter search. Each
configuration generates its own subject-level BEC archive and is evaluated by
the same fold-safe BrainNetCNN classifier used by the Sparse VAR baseline.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPARSE_VAR_RUNNER = PROJECT_ROOT / "Graph_BEC/baseline/Sparse-VAR/run_sparse_var.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_sparse_var_runner():
    spec = importlib.util.spec_from_file_location("sparse_var_runner", SPARSE_VAR_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Sparse VAR runner from {SPARSE_VAR_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPARSE_VAR = load_sparse_var_runner()
from Graph_BEC.data import load_subject_dataset
from Graph_BEC.profiles.configuration import get_profile
from Graph_BEC.utils.runtime import set_seed

METRICS = ("ACC", "SPE", "AUC", "Precision", "Recall", "F1")
CONFIGURATIONS = (
    {"experiment": 1, "lags": 1, "alpha": 0.01, "l1_ratio": 1.0},
    {"experiment": 2, "lags": 1, "alpha": 0.03, "l1_ratio": 1.0},
    {"experiment": 3, "lags": 1, "alpha": 0.05, "l1_ratio": 1.0},
    {"experiment": 4, "lags": 1, "alpha": 0.05, "l1_ratio": 0.9},
    {"experiment": 5, "lags": 1, "alpha": 0.05, "l1_ratio": 0.7},
    {"experiment": 6, "lags": 2, "alpha": 0.05, "l1_ratio": 0.9},
)


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=["abide", "adhd200"], default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)
    default_output = PROJECT_ROOT / "Graph_BEC/baseline/Sparse-VAR/outputs/grid"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["abide", "adhd200"], default=profile.name)
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--output-dir", type=Path, default=default_output)
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
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--lag-decay", type=float, default=1.0)
    parser.add_argument("--regenerate-bec", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def make_runner_args(args, configuration, bec_path):
    values = vars(args).copy()
    values.update(configuration)
    values.update(
        {
            "bec_path": bec_path,
            "output_dir": args.output_dir,
            "regenerate_bec": args.regenerate_bec,
            "generation_only": False,
            "classification_only": False,
        }
    )
    return SimpleNamespace(**values)


def configuration_name(configuration):
    alpha = str(configuration["alpha"]).replace(".", "p")
    l1_ratio = str(configuration["l1_ratio"]).replace(".", "p")
    return f"exp{configuration['experiment']:02d}_lags{configuration['lags']}_alpha{alpha}_l1{l1_ratio}"


def summarize_rows(rows):
    summary = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values))
    return summary


def save_outputs(args, summaries, fold_rows):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"sparse_var_grid_{args.dataset}.csv"
    summary_fields = [
        "rank", "experiment", "configuration", "lags", "alpha", "l1_ratio",
        "bec_path", "nonzero_fraction", "mean_abs_bec", "AUC_mean", "AUC_std",
        "ACC_mean", "ACC_std", "SPE_mean", "SPE_std", "Precision_mean",
        "Precision_std", "Recall_mean", "Recall_std", "F1_mean", "F1_std",
    ]
    ranked = sorted(
        summaries,
        key=lambda row: (
            -np.nan_to_num(row["AUC_mean"], nan=-np.inf),
            -np.nan_to_num(row["F1_mean"], nan=-np.inf),
            -np.nan_to_num(row["ACC_mean"], nan=-np.inf),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(ranked)

    fold_path = args.output_dir / f"sparse_var_grid_folds_{args.dataset}.csv"
    fold_fields = sorted({key for row in fold_rows for key in row})
    with fold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(fold_rows)

    json_path = args.output_dir / f"sparse_var_grid_{args.dataset}.json"
    json_path.write_text(
        json.dumps({"ranked_results": ranked, "fold_results": fold_rows}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return ranked, summary_path, fold_path, json_path


def main():
    args = parse_args()
    if args.patient_label == args.control_label:
        raise ValueError("--patient-label and --control-label must be different")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")

    profile = get_profile(args.dataset)
    set_seed(args.seed)
    device = SPARSE_VAR.choose_device(args.gpu_id)
    print(f"dataset: {args.dataset}; device: {device}")
    print(f"loading subject time series from: {args.data_root}")
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
    print(f"subjects: {len(dataset['time_series'])}")

    summaries, fold_rows = [], []
    for configuration in CONFIGURATIONS:
        name = configuration_name(configuration)
        bec_path = args.output_dir / f"subject_sparse_var_bec_{args.dataset}_{name}.npz"
        if args.skip_existing and bec_path.is_file():
            archive = SPARSE_VAR.load_bec_archive(bec_path)
            print(f"\n[{name}] using existing archive: {bec_path}")
        else:
            runner_args = make_runner_args(args, configuration, bec_path)
            print(f"\n[{name}] generating BEC")
            archive = SPARSE_VAR.generate_bec_archive(runner_args, dataset)

        runner_args = make_runner_args(args, configuration, bec_path)
        print(f"[{name}] classifying BEC")
        rows = SPARSE_VAR.classify_bec(runner_args, archive, device)
        metrics = summarize_rows(rows)
        bec = np.asarray(archive["bec"], dtype=np.float32)
        nonzero_fraction = float(np.mean(np.abs(bec) > 1e-8))
        summary = {
            **configuration,
            "configuration": name,
            "bec_path": str(bec_path),
            "nonzero_fraction": nonzero_fraction,
            "mean_abs_bec": float(np.mean(np.abs(bec))),
            **metrics,
        }
        summaries.append(summary)
        for row in rows:
            fold_rows.append({**configuration, "configuration": name, **row})
        print(
            f"[{name}] AUC={summary['AUC_mean'] * 100:.2f}±{summary['AUC_std'] * 100:.2f}% | "
            f"F1={summary['F1_mean'] * 100:.2f}±{summary['F1_std'] * 100:.2f}% | "
            f"nonzero={nonzero_fraction * 100:.2f}%"
        )

    ranked, summary_path, fold_path, json_path = save_outputs(args, summaries, fold_rows)
    best = ranked[0]
    print("\nBest configuration")
    print(
        f"experiment {best['experiment']}: {best['configuration']} | "
        f"AUC={best['AUC_mean'] * 100:.2f}±{best['AUC_std'] * 100:.2f}% | "
        f"F1={best['F1_mean'] * 100:.2f}±{best['F1_std'] * 100:.2f}%"
    )
    print(f"summary: {summary_path}")
    print(f"folds: {fold_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
