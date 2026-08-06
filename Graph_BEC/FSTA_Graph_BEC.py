#!/usr/bin/env python3
"""End-to-end FSTA-Graph-BEC experiment runner.

Raw mode: ROI time series -> FSTA -> BEC -> label-free phenotype refinement.
BEC mode: existing subject_bec.npz -> label-free phenotype refinement.
Diagnosis labels are used only in final statistics and downstream classification.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from Graph_BEC.normative_bec import bec_separability, edge_effect_sizes
from Graph_BEC.data import (
    load_subject_dataset, load_bec_archive, make_stratified_splits,
    prepare_fold_arrays, select_device, set_seed,
)
from Graph_BEC.downstream import train_classifier
from Graph_BEC.model import (
    PGRBECStatic,
    train_fsta, extract_subject_bec,
)
from Graph_BEC.model.pgr_bec_static import static_refinement_loss
from Graph_BEC.phenotype import load_phenotypes, build_reference_bec

DEFAULT_BEC = ROOT / "downstream_abide_i/outputs/entropy/loss_alpha_0.01/seed_42/epochs_101/subject_bec.npz"
DEFAULT_DATA_ROOT = ROOT / "dataset/ABIDE-I"
DEFAULT_PHENOTYPE = ROOT / "dataset/ABIDE-I/Phenotypic_Processing_filled.csv"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/outputs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=["bec", "raw"], default="bec")
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--phenotype-csv", type=Path, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="auto")

    parser.add_argument("--window-length", type=int, default=78)
    parser.add_argument("--stride", type=int, default=39)
    parser.add_argument("--epochs", type=int, default=101)
    parser.add_argument("--fsta-checkpoint", choices=["final", "best"], default="final")
    parser.add_argument("--loss-mode", choices=["original", "entropy"], default="entropy")
    parser.add_argument("--loss-alpha", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--d-inner-hid", type=int, default=64)
    parser.add_argument("--d-k", type=int, default=8)
    parser.add_argument("--d-v", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-warmup-steps", type=int, default=4000)
    parser.add_argument("--lr-mul", type=float, default=1.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int, default=2)
    parser.add_argument("--hidden-act", default="gelu")
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0.5)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.5)
    parser.add_argument("--initializer-range", type=float, default=0.02)
    parser.add_argument("--no-filters", action="store_true")

    parser.add_argument("--refiner-epochs", type=int, default=80)
    parser.add_argument("--refiner-lr", type=float, default=1e-2)
    parser.add_argument("--gate-max", type=float, default=0.5) #0.5
    parser.add_argument("--gate-l1-weight", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--variance-retention", type=float, default=0.85)
    parser.add_argument("--reference-k", type=int, default=20)
    parser.add_argument("--reference-bandwidth", type=float, default=2.0) # 2
    parser.add_argument("--categorical-penalty", type=float, default=4.0) # 4
    # The two continuous weights correspond to FIQ and PIQ, respectively.
    parser.add_argument("--continuous-weights", type=float, nargs=2, default=[1.0, 0.3])
    parser.add_argument("--permute-phenotype", action="store_true")

    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-repeats", type=int, default=1)
    parser.add_argument("--tc-label", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def load_pipeline_data(args, device):
    fsta_metrics = None
    if args.input_mode == "raw":
        subjects = load_subject_dataset(args.data_root, args.pipeline, args.strategy, args.derivative, args.standardize, args.max_subjects)
        print(f"Training FSTA from {len(subjects['records'])} subject time series...")
        fsta_model, fsta_metrics = train_fsta(args, subjects["time_series"], device)
        extracted = extract_subject_bec(fsta_model, subjects["records"], subjects["time_series"], args.window_length, args.stride, device)
        data = {
            "bec": extracted["bec"], "labels": subjects["labels"],
            "subject_ids": subjects["subject_ids"], "site_ids": subjects["site_ids"],
            "reconstruction_mse": extracted["reconstruction_mse"],
        }
        if args.bec_path.is_file():
            archived = load_bec_archive(args.bec_path)
            if np.array_equal(data["subject_ids"].astype(str), archived["subject_ids"].astype(str)):
                difference = np.abs(data["bec"] - archived["bec"])
                print(
                    "raw-vs-archive BEC: "
                    f"max_abs={difference.max():.3e}, mean_abs={difference.mean():.3e}"
                )
    else:
        data = load_bec_archive(args.bec_path)
        if args.max_subjects is not None:
            data = {key: value[:args.max_subjects] if hasattr(value, "__len__") and len(value) == len(data["bec"]) else value for key, value in data.items()}
    phenotype = load_phenotypes(args.phenotype_csv, data["subject_ids"], data["site_ids"])
    data.update(phenotype)
    data["bec"] = np.asarray(data["bec"], dtype=np.float32)
    data["labels"] = np.asarray(data["labels"], dtype=np.int64)
    return data, fsta_metrics


def build_fold_reference(args, arrays):
    common = dict(
        k=args.reference_k, bandwidth=args.reference_bandwidth,
        categorical_penalty=args.categorical_penalty,
        continuous_weights=args.continuous_weights,
        permute=args.permute_phenotype, seed=args.seed,
    )
    train_neighbor, _, _ = build_reference_bec(
        arrays["train_bec"], arrays["train_cont"], arrays["train_cat"],
        arrays["train_cont"], arrays["train_cat"], **common
    )
    val_neighbor, _, _ = build_reference_bec(
        arrays["train_bec"], arrays["train_cont"], arrays["train_cat"],
        arrays["val_cont"], arrays["val_cat"], **common
    )
    test_neighbor, _, diagnostics = build_reference_bec(
        arrays["train_bec"], arrays["train_cont"], arrays["train_cat"],
        arrays["test_cont"], arrays["test_cat"], **common
    )
    return {
        "train_neighbor": train_neighbor,
        "val_neighbor": val_neighbor,
        "test_neighbor": test_neighbor,
        "diagnostics": diagnostics,
    }


def train_refiner(args, bec, neighbor, device):
    """Train the label-free Static phenotype gate."""
    model = PGRBECStatic(
        bec.shape[-1], hidden_channels=16, gate_max=args.gate_max
    ).to(device)
    original = torch.from_numpy(bec).float().to(device)
    neighbor_tensor = torch.from_numpy(neighbor).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.refiner_lr)
    best_state, best_loss, metrics = None, float("inf"), {}
    for _ in range(args.refiner_epochs):
        optimizer.zero_grad()
        refined, gate, _ = model(original, neighbor_tensor, return_parts=True)
        total, parts = static_refinement_loss(
            refined, original, gate, args.variance_retention,
            args.anchor_weight, args.gate_l1_weight, args.variance_weight
        )
        total.backward()
        optimizer.step()
        metrics = {
            "refiner_loss": float(total.item()),
            **{key: float(value.item()) for key, value in parts.items()},
        }
        if metrics["refiner_loss"] < best_loss:
            best_loss = metrics["refiner_loss"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        refined, gate, _ = model(original, neighbor_tensor, return_parts=True)
    metrics.update({
        "gate_mean": float(gate.mean()),
        "gate_max": float(gate.max()),
        "gate_fraction_above_0p01": float((gate > 0.01).float().mean()),
    })
    return model, refined.cpu().numpy(), metrics


def run_fold(args, fold, data, train_index, val_index, test_index, device):
    fold_seed = args.seed + fold * 1000
    set_seed(fold_seed)
    arrays = prepare_fold_arrays(
        data["bec"][train_index], data["bec"][val_index], data["bec"][test_index],
        data["continuous"][train_index], data["continuous"][val_index], data["continuous"][test_index],
        data["categorical_raw"][train_index], data["categorical_raw"][val_index], data["categorical_raw"][test_index],
    )
    reference = build_fold_reference(args, arrays)
    refiner, train_refined, refinement_metrics = train_refiner(
        args, arrays["train_bec"], reference["train_neighbor"], device
    )
    with torch.no_grad():
        val_refined = refiner(
            torch.from_numpy(arrays["val_bec"]).float().to(device),
            torch.from_numpy(reference["val_neighbor"]).float().to(device),
        ).cpu().numpy()
        test_refined = refiner(
            torch.from_numpy(arrays["test_bec"]).float().to(device),
            torch.from_numpy(reference["test_neighbor"]).float().to(device),
        ).cpu().numpy()
    train_labels, test_labels = data["labels"][train_index], data["labels"][test_index]
    classifier_args = dict(device=device, max_epochs=args.classifier_epochs, patience=args.classifier_patience, batch_size=args.batch_size, learning_rate=args.classifier_lr)
    original_runs, refined_runs = [], []
    for repeat in range(args.classifier_repeats):
        classifier_seed = fold_seed + repeat + 1
        original_metrics, _ = train_classifier(
            arrays["train_bec"], train_labels, arrays["val_bec"], data["labels"][val_index],
            arrays["test_bec"], test_labels, seed=classifier_seed, **classifier_args
        )
        refined_metrics, _ = train_classifier(
            train_refined, train_labels, val_refined, data["labels"][val_index],
            test_refined, test_labels, seed=classifier_seed, **classifier_args
        )
        original_runs.append(original_metrics)
        refined_runs.append(refined_metrics)
    metric_names = original_runs[0].keys()
    original_metrics = {key: float(np.mean([run[key] for run in original_runs])) for key in metric_names}
    refined_metrics = {key: float(np.mean([run[key] for run in refined_runs])) for key in metric_names}
    auc_deltas = np.asarray([
        refined["AUC"] - original["AUC"]
        for original, refined in zip(original_runs, refined_runs)
    ])
    original_group = bec_separability(arrays["test_bec"], test_labels, args.tc_label)
    refined_group = bec_separability(test_refined, test_labels, args.tc_label)
    original_edge, original_effect = edge_effect_sizes(arrays["test_bec"], test_labels, args.tc_label)
    refined_edge, refined_effect = edge_effect_sizes(test_refined, test_labels, args.tc_label)
    return {"original": original_metrics, "refined": refined_metrics,
            **{f"original_{key}": value for key, value in original_group.items()},
            **{f"refined_{key}": value for key, value in refined_group.items()},
            **{f"original_{key}": value for key, value in original_edge.items()},
            **{f"refined_{key}": value for key, value in refined_edge.items()},
            "edge_abs_d_change": float(np.mean(np.abs(refined_effect)) - np.mean(np.abs(original_effect))),
            "variance_retention": float(np.var(test_refined) / max(np.var(arrays["test_bec"]), 1e-12)),
            "paired_auc_delta_mean": float(auc_deltas.mean()),
            "paired_auc_delta_std": float(auc_deltas.std()),
            **refinement_metrics, **reference["diagnostics"]}


def save_results(args, fold_results, fsta_metrics):
    rows = []
    for fold, result in enumerate(fold_results, 1):
        row = {"fold": fold}
        for name in ("original", "refined"):
            row.update({f"{name}_{key}": value for key, value in result[name].items()})
        row.update({key: value for key, value in result.items() if key not in {"original", "refined"}}); rows.append(row)
    summary = {"config": vars(args), "fsta_training": fsta_metrics, "folds": rows}
    for name in ("original", "refined"):
        for metric in ("ACC", "SEN", "SPE", "AUC", "Precision", "Recall", "F1"):
            values = [row[f"{name}_{metric}"] for row in rows]
            summary[f"{name}_{metric}_mean"] = float(np.mean(values)); summary[f"{name}_{metric}_std"] = float(np.std(values))
            summary[f"{name}_{metric}_display"] = f"{100 * summary[f'{name}_{metric}_mean']:.2f}±{100 * summary[f'{name}_{metric}_std']:.2f}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.output_dir.iterdir():
        if path.is_file() and path.name not in {"experiment_summary.csv", "summary.json"}: path.unlink()
    with (args.output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main():
    args = parse_args(); set_seed(args.seed); device = select_device(args.gpu_id)
    print(f"Loading data from {args.data_root} with phenotype {args.phenotype_csv}...")
    data, fsta_metrics = load_pipeline_data(args, device)
    if np.unique(data["labels"]).size != 2:
        raise ValueError(
            "The selected subjects contain only one diagnosis class. "
            "Use the full dataset or choose a max-subjects subset containing both labels."
        )
    print(f"Input={args.input_mode}; subjects={len(data['bec'])}; BEC={data['bec'].shape}; labels={np.bincount(data['labels'])}; device={device}")
    results = []
    for fold, train_index, val_index, test_index in make_stratified_splits(data["labels"], args.n_splits, args.seed, args.validation_size):
        result = run_fold(args, fold, data, train_index, val_index, test_index, device); results.append(result)
        print(f"fold {fold}: original AUC={result['original']['AUC']:.4f}, refined AUC={result['refined']['AUC']:.4f}, original Fisher={result['original_bec_fisher_ratio']:.4f}, refined Fisher={result['refined_bec_fisher_ratio']:.4f}")
    summary = save_results(args, results, fsta_metrics)
    report_metrics = ("ACC", "SEN", "SPE", "AUC", "Precision", "Recall", "F1")
    print("\nmean±std (%)")
    print("representation | " + " | ".join(report_metrics))
    for name in ("original", "refined"):
        values = [summary[f"{name}_{metric}_display"] for metric in report_metrics]
        print(f"{name:13s} | " + " | ".join(values))


if __name__ == "__main__": main()
