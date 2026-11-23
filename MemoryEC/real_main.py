"""Train MemoryEC on simulations and evaluate on the real MTLnet dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from MemoryEC.data.simulation.loader import PROCESSED_DIR, SPLITS_PATH, load_splits
from MemoryEC.data.real.loader import MTLNET_PROCESSED_DIR, load_mtlnet
from MemoryEC.sim_main import aggregate_predictions, load_model, threshold_from_validation
from train import resolve_device, train
from utils import (
    MODEL_TYPES,
    adaptive_threshold,
    configure_determinism,
    directed_metrics,
    save_json,
    zero_diagonal,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs" / "real" / "MTLnet"
THRESHOLD_METHODS = ("ratio", "validation")
TRAIN_KEYS = (
    "hidden_dim", "memory_slots", "memory_temperature", "attention_heads",
    "dropout", "learning_rate", "weight_decay", "batch_size", "epochs",
    "patience", "min_delta",
)
METRICS = (
    "precision", "recall", "directed_f1", "accuracy", "shd", "auprc",
    "auroc", "tp", "fp", "fn", "predicted_edges", "true_edges",
)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment_name(args: argparse.Namespace) -> str:
    name = f"simulation_to_mtlnet_{args.model_type}_r{args.repeats}"
    if args.memory_temperature != 1.0:
        name += f"_tau{args.memory_temperature:g}"
    return name


def selected_hemispheres(value: str) -> tuple[str, ...]:
    return ("left", "right") if value == "both" else (value,)


def validation_selection(args: argparse.Namespace, checkpoint_path: Path) -> dict:
    model, checkpoint = load_model(checkpoint_path, args.device)
    validation_scenes = load_splits(SPLITS_PATH)["val"]["scenes"]
    validation = aggregate_predictions(model, validation_scenes, args)
    threshold, score = threshold_from_validation(validation)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_selection": "val_loss",
        "threshold": threshold,
        "validation_family_macro_f1": score,
        "selection_criterion": "simulation validation family-macro F1",
        "threshold_tie_rule": "lowest threshold in 0.05..0.95 step 0.01",
        "test_used_for_selection": False,
    }


def train_repeat(args: argparse.Namespace, repeat_dir: Path) -> dict:
    options = vars(args).copy()
    for key in ("mode", "hemisphere", "threshold_method", "fsta_ratio", "repeats"):
        options.pop(key, None)
    options.update(output_dir=repeat_dir)
    train(argparse.Namespace(**options))
    checkpoint_path = repeat_dir / f"{args.model_type}_best.pt"
    selection = validation_selection(args, checkpoint_path)
    save_json(selection, repeat_dir / "validation_selection.json")
    return selection


def infer_mtlnet(model, side: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_mtlnet(side)
    bold = data["bold"]
    lengths = data["lengths"].astype(int)
    probabilities = []
    model.eval()
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        grouped_indices[int(length)].append(index)

    with torch.no_grad():
        for length in sorted(grouped_indices):
            indices = grouped_indices[length]
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                batch = torch.from_numpy(bold[batch_indices, :length]).float().to(args.device)
                probabilities.extend(model(batch).sigmoid().cpu().numpy())

    probability = zero_diagonal(np.mean(probabilities, axis=0))
    return probability, data["adj"], data["roi_names"]


def evaluate_side(
    model,
    side: str,
    selection: dict,
    args: argparse.Namespace,
    repeat_dir: Path,
) -> dict:
    probability, target, roi_names = infer_mtlnet(model, side, args)
    threshold = (
        adaptive_threshold(probability, args.fsta_ratio)
        if args.threshold_method == "ratio"
        else float(selection["threshold"])
    )
    mask = ~np.eye(target.shape[0], dtype=bool)
    labels = target[mask].astype(bool)
    prediction = probability[mask] >= threshold
    binary = (probability >= threshold).astype(np.uint8)
    np.fill_diagonal(binary, 0)
    np.savez_compressed(
        repeat_dir / f"{side}_prediction.npz",
        probability=probability,
        binary=binary,
        ground_truth=target,
        roi_names=roi_names,
        threshold=np.asarray(threshold),
    )
    return {
        "repeat": args.repeat,
        "seed": args.seed,
        "hemisphere": side,
        "model": args.model_type,
        "threshold_method": args.threshold_method,
        "threshold": threshold,
        "fsta_ratio": args.fsta_ratio,
        "validation_threshold": selection["threshold"],
        "checkpoint_epoch": selection["checkpoint_epoch"],
        **directed_metrics(target, probability, threshold),
        "tp": int((prediction & labels).sum()),
        "fp": int((prediction & ~labels).sum()),
        "fn": int((~prediction & labels).sum()),
        "predicted_edges": int(prediction.sum()),
        "true_edges": int(labels.sum()),
    }


def evaluate_repeat(
    args: argparse.Namespace,
    repeat_dir: Path,
    selection: dict,
) -> list[dict]:
    model, _ = load_model(Path(selection["checkpoint"]), args.device)
    rows = [
        evaluate_side(model, side, selection, args, repeat_dir)
        for side in selected_hemispheres(args.hemisphere)
    ]
    save_json(rows, repeat_dir / "results.json")
    return rows


def summarize(rows: list[dict], directory: Path, args: argparse.Namespace) -> None:
    write_csv(rows, directory / "results.csv")
    summary_rows = []
    for side in selected_hemispheres(args.hemisphere):
        group = [row for row in rows if row["hemisphere"] == side]
        if not group:
            continue
        summary_rows.append({
            "hemisphere": side,
            "model": args.model_type,
            "threshold_method": args.threshold_method,
            "fsta_ratio": args.fsta_ratio,
            **{f"{metric}_mean": float(np.nanmean([row[metric] for row in group])) for metric in METRICS},
            **{f"{metric}_std": float(np.nanstd([row[metric] for row in group])) for metric in METRICS},
        })
    write_csv(summary_rows, directory / "summary.csv")
    save_json(
        {
            "dataset": "MTLnet",
            "simulation_training_data": str(PROCESSED_DIR),
            "real_inference_data": str(MTLNET_PROCESSED_DIR),
            "repeats": args.repeats,
            "threshold_method": args.threshold_method,
            "fsta_ratio": args.fsta_ratio,
            "hemispheres": summary_rows,
        },
        directory / "summary.json",
    )
    for row in summary_rows:
        print(f"\n{row['hemisphere'].upper()} MTLnet ({args.threshold_method})")
        for metric in ("precision", "recall", "directed_f1", "accuracy", "shd", "auprc", "auroc"):
            print(f"  {metric}: {row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")


def config(args: argparse.Namespace) -> dict:
    return {
        "protocol": "simulation_train_mtlnet_inference_v1",
        "model": args.model_type,
        "repeats": args.repeats,
        "seeds": [args.seed + repeat for repeat in range(args.repeats)],
        "training": {key: getattr(args, key) for key in TRAIN_KEYS},
        "simulation_processed_dir": str(PROCESSED_DIR),
        "simulation_splits": str(SPLITS_PATH),
        "mtlnet_processed_dir": str(MTLNET_PROCESSED_DIR),
        "checkpoint_selection": "minimum simulation validation BCE",
        "threshold_options": list(THRESHOLD_METHODS),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--hemisphere", choices=("left", "right", "both"), default="both")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="st_pair_memory")
    parser.add_argument("--repeats", type=int, choices=(1, 5, 10), default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--memory-temperature", type=float, default=1.0)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument(
        "--threshold-method", "--threshold_method",
        choices=THRESHOLD_METHODS,
        default="ratio",
    )
    parser.add_argument("--fsta-ratio", "--fsta_ratio", type=float, default=0.4)
    args = parser.parse_args(argv)

    for key in ("hidden_dim", "memory_slots", "attention_heads", "batch_size", "epochs", "patience"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.hidden_dim % args.attention_heads:
        parser.error("hidden-dim must be divisible by attention-heads")
    if not math.isfinite(args.fsta_ratio) or not 0 <= args.fsta_ratio <= 1:
        parser.error("fsta-ratio must be finite and in [0,1]")
    if not math.isfinite(args.memory_temperature) or args.memory_temperature <= 0:
        parser.error("memory-temperature must be finite and positive")
    for key in ("learning_rate", "weight_decay", "min_delta", "dropout"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and non-negative")
    if args.dropout >= 1 or args.seed < 0 or args.seed + args.repeats - 1 >= 2**32:
        parser.error("dropout must be < 1; seeds must fit numpy's unsigned 32-bit range")
    return args


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    configure_determinism()
    args.device = str(resolve_device("cuda"))
    directory = OUTPUTS / experiment_name(args)
    config_path = directory / "config.json"
    expected_config = config(args)

    if args.mode == "train":
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        save_json(expected_config, config_path)
    elif not config_path.is_file():
        raise FileNotFoundError(f"No trained real-data experiment found: {directory}")
    elif json.loads(config_path.read_text(encoding="utf-8")) != expected_config:
        raise ValueError(f"Existing experiment configuration differs: {config_path}")

    print(f"Simulation training data: {PROCESSED_DIR}")
    print(f"MTLnet inference data: {MTLNET_PROCESSED_DIR}")
    print(f"Output: {directory}")
    print(f"Threshold: {args.threshold_method}; ratio={args.fsta_ratio:.2f}")
    all_rows = []
    for repeat, seed in enumerate(expected_config["seeds"], 1):
        repeat_dir = directory / f"repeat_{repeat:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        args.repeat = repeat
        args.seed = seed
        selection_path = repeat_dir / "validation_selection.json"
        if args.mode == "train":
            print(f"\nRepeat {repeat}/{args.repeats}, seed={seed}: training on simulations", flush=True)
            selection = train_repeat(args, repeat_dir)
        else:
            if not selection_path.is_file():
                raise FileNotFoundError(f"Missing validation selection: {selection_path}")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        print(
            f"Repeat {repeat}/{args.repeats}: MTLnet inference, "
            f"checkpoint_epoch={selection['checkpoint_epoch']}",
            flush=True,
        )
        all_rows.extend(evaluate_repeat(args, repeat_dir, selection))
    summarize(all_rows, directory, args)


if __name__ == "__main__":
    main()
