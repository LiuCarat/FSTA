"""Configurable MemoryEC training/evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data.loader import load_splits, make_dataloader
from model.memory_ec import MemoryEC
from test import evaluate
from train import train as train_model
from utils import MODEL_TYPES, adaptive_threshold, directed_metrics, save_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
METRICS = ("precision", "recall", "directed_f1", "accuracy", "shd", "auprc", "auroc")
ABLATION_METRICS = (
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "directed_f1"),
    ("Accuracy", "accuracy"),
    ("SHD", "shd"),
)


def checkpoint_label(path: Path) -> str:
    return "best_val_auprc" if "best_val_auprc" in path.name else "best_val_loss"


def evaluate_checkpoints(args: argparse.Namespace) -> None:
    splits = load_splits()
    checkpoint_paths = [
        (model, OUT / f"{model}_best.pt") for model in MODEL_TYPES
    ] + [
        (model, OUT / f"{model}_best_val_auprc.pt") for model in MODEL_TYPES
    ]
    records = []
    for model_type, checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["model_config"]["model_type"] != model_type:
            raise ValueError(f"{checkpoint_path}: model type mismatch")
        model = MemoryEC(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        print(f"checkpoint={checkpoint_path.name} epoch={checkpoint['epoch']} loaded=true")
        tag = checkpoint_label(checkpoint_path)
        for split in ("val", "test"):
            evaluate(argparse.Namespace(
                seed=args.seed, threads=args.threads, device=args.device,
                checkpoint=str(checkpoint_path), split=split,
                batch_size=args.batch_size, time_shuffle=False,
                threshold=0.5, predictions_only=True, checkpoint_tag=tag,
            ))

        def scene_predictions(split: str) -> dict:
            result = {}
            for scene in splits[split]["scenes"]:
                file_path = OUT / f"{model_type}_{scene}_{tag}_predictions.npz"
                with np.load(file_path, allow_pickle=False) as data:
                    probs = data["probabilities"]
                    result[scene] = (probs.mean(axis=0), data["adj"])
            return result

        validation = scene_predictions("val")
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
        threshold_scores = []
        for threshold in grid:
            threshold_scores.append(np.mean([
                directed_metrics(adj, prob, float(threshold))["directed_f1"]
                for prob, adj in validation.values()
            ]))
        selected = float(grid[int(np.argmax(threshold_scores))])
        test_data = scene_predictions("test")
        per_method = {"fixed_0.5": [], "validation_selected": [], "fsta_adaptive": []}
        for scene, (probability, target) in test_data.items():
            thrs = {"fixed_0.5": 0.5, "validation_selected": selected,
                    "fsta_adaptive": adaptive_threshold(probability, 0.5)}
            for method, threshold in thrs.items():
                values = directed_metrics(target, probability, threshold)
                predicted = (probability >= threshold).astype(np.uint8)
                np.fill_diagonal(predicted, 0)
                mask = ~np.eye(target.shape[0], dtype=bool)
                labels = target[mask].astype(bool)
                pred_edges = predicted[mask].astype(bool)
                per_method[method].append({
                    "model": model_type, "checkpoint": tag,
                    "network": f"Network{int(scene.split('_')[0][7:])}",
                    "scene": scene, "method": method, "threshold": threshold,
                    "predicted_edges": int(pred_edges.sum()),
                    "tp": int((pred_edges & labels).sum()),
                    "fp": int((pred_edges & ~labels).sum()),
                    "fn": int((~pred_edges & labels).sum()),
                    **{key: values[key] for key in METRICS},
                })
        records.extend(row for rows in per_method.values() for row in rows)
        save_json({"model": model_type, "checkpoint": str(checkpoint_path),
                   "checkpoint_epoch": checkpoint["epoch"], "selection_split": "Network7",
                   "selected_threshold": selected,
                   "validation_mean_directed_f1": float(max(threshold_scores)),
                   "test_scenes": per_method},
                  OUT / f"{model_type}_{tag}_evaluation.json")

    # Scene-level values are macro-averaged within each network and overall.
    output_rows = []
    for model, checkpoint, method in sorted({(r["model"], r["checkpoint"], r["method"]) for r in records}):
        group = [r for r in records if (r["model"], r["checkpoint"], r["method"]) == (model, checkpoint, method)]
        for network in ("Network8", "Network9", "Overall"):
            part = group if network == "Overall" else [r for r in group if r["network"] == network]
            row = {"model": model, "checkpoint": checkpoint, "method": method, "network": network}
            for metric in METRICS:
                row[metric] = float(np.mean([r[metric] for r in part]))
            output_rows.append(row)
    out_csv = OUT / "checkpoint_threshold_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    save_json({"rows": output_rows, "scene_diagnostics": records}, OUT / "checkpoint_threshold_comparison.json")
    table_fields = ["checkpoint", "network", "method", "metric"] + list(MODEL_TYPES)
    table_rows = []
    for row in output_rows:
        for metric in METRICS:
            match = next((entry for entry in table_rows if entry["checkpoint"] == row["checkpoint"] and entry["network"] == row["network"] and entry["method"] == row["method"] and entry["metric"] == metric), None)
            if match is None:
                match = {"checkpoint": row["checkpoint"], "network": row["network"], "method": row["method"], "metric": metric}
                table_rows.append(match)
            match[row["model"]] = row[metric]
    with (OUT / "checkpoint_threshold_comparison_wide.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields)
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"saved={out_csv}")


def print_ablation_metrics(model_types: tuple[str, ...], args: argparse.Namespace) -> None:
    splits = load_splits()
    scene_records = []
    selected_thresholds = {}
    for model_type in model_types:
        checkpoint = OUT / f"{model_type}_best.pt"
        for split in ("val", "test"):
            evaluate(argparse.Namespace(
                seed=args.seed, threads=args.threads, device=args.device,
                checkpoint=str(checkpoint), split=split, batch_size=args.batch_size,
                time_shuffle=False, threshold=0.5, predictions_only=True,
                checkpoint_tag="ablation",
            ))

        def load_scenes(split: str) -> dict:
            scenes = {}
            for scene in splits[split]["scenes"]:
                path = OUT / f"{model_type}_{scene}_ablation_predictions.npz"
                with np.load(path, allow_pickle=False) as data:
                    scenes[scene] = (data["probabilities"].mean(axis=0), data["adj"])
            return scenes

        validation = load_scenes("val")
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
        val_f1 = [np.mean([
            directed_metrics(adj, probability, float(threshold))["directed_f1"]
            for probability, adj in validation.values()
        ]) for threshold in thresholds]
        selected = float(thresholds[int(np.argmax(val_f1))])
        selected_thresholds[model_type] = {
            "threshold": selected,
            "network7_mean_scene_f1": float(max(val_f1)),
        }
        methods = {"fixed_0.5": [], "validation_selected": [], "fsta_adaptive": []}
        for scene, (probability, target) in load_scenes("test").items():
            scene_thresholds = {
                "fixed_0.5": 0.5,
                "validation_selected": selected,
                "fsta_adaptive": adaptive_threshold(probability, 0.5),
            }
            for method, threshold in scene_thresholds.items():
                metrics = directed_metrics(target, probability, threshold)
                record = {
                    "model": model_type,
                    "scene": scene,
                    "network": f"Network{int(scene.split('_')[0][7:])}",
                    "method": method,
                    "threshold": float(threshold),
                    **{key: metrics[key] for _, key in ABLATION_METRICS},
                }
                methods[method].append(record)
                scene_records.append(record)
        print(f"\n{model_type}: Network7 validation-selected threshold = {selected:.2f}")
        print("FSTA adaptive thresholds by test scene: " + ", ".join(
            f"{record['scene']}={record['threshold']:.2f}"
            for record in methods["fsta_adaptive"]
        ))
        save_json({"model": model_type, "validation_threshold": selected,
                   "validation_mean_scene_f1": float(max(val_f1)), "test_scenes": methods},
                  OUT / f"{model_type}_ablation_threshold_metrics.json")

    labels = {
        "st_no_memory": "No Memory",
        "st_global_memory": "Global Memory",
        "st_pair_memory": "Pair Memory",
    }
    print("\nAblation test metrics: values are scene-level macro means; higher is better except SHD.")
    print("Overall = unweighted mean across all six Network8/9 test scenes (3 scenes per network).")
    print("AUPRC/AUROC omitted here as requested; these tables report the five threshold-based metrics.")
    for method, title in (("fixed_0.5", "Fixed threshold = 0.50"),
                          ("validation_selected", "Network7 validation-selected threshold"),
                          ("fsta_adaptive", "FSTA adaptive per-scene threshold (ratio = 0.50)")):
        print(f"\n=== {title} ===")
        for network in ("Network8", "Network9", "Overall"):
            print(f"\n{network}")
            print(f"{'Metric':<12}" + "".join(f"{labels[model]:>16}" for model in model_types))
            for display, key in ABLATION_METRICS:
                values = []
                for model in model_types:
                    rows = [r for r in scene_records if r["model"] == model and r["method"] == method]
                    if network != "Overall":
                        rows = [r for r in rows if r["network"] == network]
                    values.append(f"{np.mean([r[key] for r in rows]):.2f}")
                print(f"{display:<12}" + "".join(f"{value:>16}" for value in values))
    save_json({"overall_definition": "unweighted arithmetic mean of the six test scene-level metrics",
               "validation_thresholds": selected_thresholds, "scenes": scene_records},
              OUT / "ablation_threshold_comparison.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "ablation", "evaluate"), default="ablation")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="st_pair_memory")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-roi-permutation", action="store_false", dest="roi_permutation")
    parser.set_defaults(roi_permutation=True)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    if args.mode == "evaluate":
        evaluate_checkpoints(args)
        return
    model_types = MODEL_TYPES if args.mode == "ablation" else (args.model_type,)
    for model_type in model_types:
        print(f"\nTraining {model_type} with configurable main.py settings")
        train_model(argparse.Namespace(
            model_type=model_type,
            hidden_dim=args.hidden_dim,
            memory_slots=args.memory_slots,
            attention_heads=args.attention_heads,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed,
            device=args.device,
            threads=args.threads,
            roi_permutation=args.roi_permutation,
        ))
    if args.mode == "ablation":
        print_ablation_metrics(model_types, args)


if __name__ == "__main__":
    main()
