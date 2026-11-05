"""Evaluate a trained MemoryEC checkpoint on held-out network families."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.loader import load_splits, make_dataloader
from model.memory_ec import MemoryEC
from utils import (
    MODEL_TYPES,
    adaptive_threshold,
    base_network,
    directed_metrics,
    edge_diagnostics,
    save_json,
    select_directed_f1_threshold,
    set_seed,
    zero_diagonal,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
METRIC_NAMES = (
    "directed_f1",
    "precision",
    "recall",
    "accuracy",
    "shd",
)


def memory_statistics(
    weight_sum: np.ndarray | None,
    retrieval_count: int,
    gate_sum: float,
    gate_count: int,
) -> dict | None:
    if weight_sum is None or retrieval_count == 0:
        return None
    usage = weight_sum / retrieval_count
    entropy = float(-(usage * np.log(usage + 1e-12)).sum())
    return {
        "slot_usage": usage.tolist(),
        "normalized_entropy": entropy / np.log(len(usage)),
        "largest_slot_share": float(usage.max()),
        "mean_gate": gate_sum / gate_count if gate_count else None,
    }


def prediction_path(model_type: str, scene: str, output_dir: Path) -> Path:
    return output_dir / f"{model_type}_{scene}_predictions.npz"


def load_prediction(
    model_type: str,
    scene: str,
    output_dir: Path = OUTPUT_DIR,
    persist_zero_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved predictions and enforce zero self-connection probabilities."""
    path = prediction_path(model_type, scene, output_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prediction file: {path}. Restore existing predictions; "
            "this evaluation does not retrain or rerun the model."
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"probabilities", "mean_probability", "adj"}
        if set(data.files) != required:
            raise ValueError(f"{path} must contain exactly {sorted(required)}")
        original_probabilities = data["probabilities"]
        original_mean = data["mean_probability"]
        adj = zero_diagonal(data["adj"])

    probabilities = zero_diagonal(original_probabilities)
    mean_probability = zero_diagonal(original_mean)
    recomputed_mean = zero_diagonal(probabilities.mean(axis=0))
    if not np.allclose(mean_probability, recomputed_mean, atol=1e-7, rtol=1e-6):
        raise ValueError(f"{path}: mean_probability does not match subject mean")

    mask = ~np.eye(adj.shape[0], dtype=bool)
    if not np.array_equal(probabilities[..., mask], original_probabilities[..., mask]):
        raise RuntimeError(f"{path}: non-diagonal subject probabilities changed")
    if not np.array_equal(mean_probability[mask], original_mean[mask]):
        raise RuntimeError(f"{path}: non-diagonal mean probabilities changed")

    diagonal_changed = (
        np.any(np.diagonal(original_probabilities, axis1=-2, axis2=-1) != 0)
        or np.any(np.diag(original_mean) != 0)
        or np.any(np.diag(adj) != 0)
    )
    if persist_zero_diagonal and diagonal_changed:
        np.savez_compressed(
            path,
            probabilities=probabilities,
            mean_probability=mean_probability,
            adj=adj,
        )
    return probabilities, mean_probability, adj


def summarize_scenes(
    scene_data: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> dict:
    scene_results = {}
    diagnostics = {}
    for scene, (probability, adj) in sorted(scene_data.items()):
        metrics = directed_metrics(adj, probability, threshold)
        scene_results[scene] = {name: metrics[name] for name in METRIC_NAMES}
        diagnostics[scene] = edge_diagnostics(adj, probability, threshold)
    family_metrics: dict[int, list[dict[str, float]]] = defaultdict(list)
    for scene, metrics in scene_results.items():
        family_metrics[base_network(scene)].append(metrics)
    families = {
        str(family): {
            name: float(np.nanmean([metrics[name] for metrics in values]))
            for name in METRIC_NAMES
        }
        for family, values in sorted(family_metrics.items())
    }
    overall = {
        name: float(np.nanmean([metrics[name] for metrics in scene_results.values()]))
        for name in METRIC_NAMES
    }
    return {
        "threshold": threshold,
        "scenes": scene_results,
        "families": families,
        "overall_scene_mean": overall,
        "diagnostics": diagnostics,
    }


def select_validation_threshold(
    model_type: str,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Select and save one threshold using validation scenes only."""
    splits = load_splits()
    validation_data = {}
    for scene in splits["val"]["scenes"]:
        _, mean_probability, adj = load_prediction(model_type, scene, output_dir)
        validation_data[scene] = (mean_probability, adj)

    threshold_selection = select_directed_f1_threshold(
        [validation_data[scene][1] for scene in sorted(validation_data)],
        [validation_data[scene][0] for scene in sorted(validation_data)],
    )
    selected_threshold = threshold_selection["selected_threshold"]
    threshold_result = {
        "model_type": model_type,
        "selection_split": "val",
        "selection_scenes": sorted(validation_data),
        "criterion": "mean_scene_directed_f1",
        "range": {"minimum": 0.05, "maximum": 0.95, "step": 0.01},
        **threshold_selection,
        "validation_at_0.5": summarize_scenes(validation_data, 0.5),
        "validation_at_selected_threshold": summarize_scenes(
            validation_data, selected_threshold
        ),
    }
    threshold_path = output_dir / f"{model_type}_validation_threshold.json"
    save_json(threshold_result, threshold_path)
    print(
        f"model={model_type} selected_threshold={selected_threshold:.2f} "
        f"validation_mean_f1={threshold_selection['selected_mean_directed_f1']:.6f}"
    )
    print(f"saved={threshold_path}")
    return threshold_result


def load_validation_threshold(model_type: str, output_dir: Path = OUTPUT_DIR) -> tuple[float, Path]:
    """Read a previously saved validation-selected threshold."""
    threshold_path = output_dir / f"{model_type}_validation_threshold.json"
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"Missing validation threshold: {threshold_path}. Run --select-threshold "
            "using Network7 predictions before evaluating the test split."
        )
    with threshold_path.open(encoding="utf-8") as file:
        threshold_result = json.load(file)
    if threshold_result.get("model_type") != model_type:
        raise ValueError(f"{threshold_path}: model_type does not match {model_type}")
    if threshold_result.get("selection_split") != "val":
        raise ValueError(f"{threshold_path}: threshold was not selected on validation")
    return float(threshold_result["selected_threshold"]), threshold_path


def evaluate_existing_predictions(
    model_type: str,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Evaluate test predictions using only fixed and saved validation thresholds."""
    splits = load_splits()
    selected_threshold, threshold_path = load_validation_threshold(
        model_type, output_dir
    )

    test_data = {}
    for scene in splits["test"]["scenes"]:
        _, mean_probability, adj = load_prediction(model_type, scene, output_dir)
        test_data[scene] = (mean_probability, adj)
    comparison = {
        "model_type": model_type,
        "split": "test",
        "threshold_source": str(threshold_path),
        "test_ground_truth_used_for_threshold_selection": False,
        "fixed_0.5": summarize_scenes(test_data, 0.5),
        "validation_selected": summarize_scenes(test_data, selected_threshold),
    }
    comparison_path = output_dir / f"{model_type}_test_threshold_comparison.json"
    save_json(comparison, comparison_path)
    print(f"model={model_type} loaded_validation_threshold={selected_threshold:.2f}")
    print(f"saved={comparison_path}")
    return comparison


def summarize_existing_results(output_dir: Path = OUTPUT_DIR) -> list[dict]:
    """Write Network8/9 comparison tables for all three trained models."""
    model_labels = {
        "st_no_memory": "No Memory",
        "st_global_memory": "Global Memory",
        "st_pair_memory": "Pair Memory",
    }
    rows = []
    diagnostics = {}
    for model_type in MODEL_TYPES:
        path = output_dir / f"{model_type}_test_threshold_comparison.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing comparison file: {path}. Run --evaluate-existing first."
            )
        with path.open(encoding="utf-8") as file:
            result = json.load(file)
        for threshold_mode in ("fixed_0.5", "validation_selected"):
            threshold_result = result[threshold_mode]
            raw_diagnostics = threshold_result.get("diagnostics", {})
            diagnostics.setdefault(model_type, {})[threshold_mode] = {
                "scenes": raw_diagnostics,
                "families": {
                    f"Network{network}": {
                        "false_positive_count": sum(
                            raw_diagnostics.get(scene, {}).get(
                                "false_positive_count", 0
                            )
                            for scene in raw_diagnostics
                            if base_network(scene) == network
                        ),
                        "false_negative_count": sum(
                            raw_diagnostics.get(scene, {}).get(
                                "false_negative_count", 0
                            )
                            for scene in raw_diagnostics
                            if base_network(scene) == network
                        ),
                    }
                    for network in (8, 9)
                },
            }
            for network in (8, 9):
                metrics = threshold_result["families"][str(network)]
                rows.append(
                    {
                        "threshold_mode": threshold_mode,
                        "threshold": threshold_result["threshold"],
                        "network": f"Network{network}",
                        "model": model_labels[model_type],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["directed_f1"],
                        "accuracy": metrics["accuracy"],
                        "shd": metrics["shd"],
                    }
                )

    csv_path = output_dir / "network8_network9_model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        {"rows": rows, "diagnostics": diagnostics},
        output_dir / "network8_network9_model_comparison.json",
    )

    for threshold_mode in ("fixed_0.5", "validation_selected"):
        print(f"\n[{threshold_mode}]")
        print("network,model,precision,recall,f1,accuracy,shd")
        for row in rows:
            if row["threshold_mode"] == threshold_mode:
                print(
                    f"{row['network']},{row['model']},"
                    f"{row['precision']:.6f},{row['recall']:.6f},"
                    f"{row['f1']:.6f},{row['accuracy']:.6f},{row['shd']:.6f}"
                )
    print(f"\nsaved={csv_path}")
    return rows


def evaluate(arguments: argparse.Namespace) -> None:
    output_dir = Path(getattr(arguments, "output_dir", OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(arguments.seed)
    torch.set_num_threads(arguments.threads)
    device = torch.device(
        "cuda" if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto"
        else arguments.device
    )
    checkpoint_path = Path(arguments.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MemoryEC(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = make_dataloader(
        arguments.split,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        permute_rois=False,
        balanced=False,
    )
    predictions: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    targets: dict[str, np.ndarray] = {}
    subject_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    weight_sum = None
    retrieval_count = 0
    gate_sum = 0.0
    gate_count = 0

    with torch.no_grad():
        for batch in loader:
            bold = batch["bold"].to(device)
            if arguments.time_shuffle:
                shuffled = []
                for subject in bold:
                    shuffled.append(subject[torch.randperm(subject.shape[0], device=device)])
                bold = torch.stack(shuffled)
            logits, aux = model(bold, return_aux=True)
            probabilities = zero_diagonal(logits.sigmoid().cpu().numpy())
            adjacency = batch["adj"].numpy()
            for index, scene in enumerate(batch["scene"]):
                subject = int(batch["subject"][index])
                predictions[scene].append((subject, probabilities[index]))
                targets[scene] = adjacency[index]
                subject_metrics[scene].append(
                    directed_metrics(adjacency[index], probabilities[index], arguments.threshold)
                )

            weights = aux["memory_weights"]
            if weights is not None:
                if weights.ndim == 2:
                    selected = weights
                else:
                    nodes = weights.shape[1]
                    mask = ~torch.eye(nodes, dtype=torch.bool, device=device)
                    selected = weights[:, mask]
                batch_sum = selected.sum(dim=tuple(range(selected.ndim - 1))).cpu().numpy()
                weight_sum = batch_sum if weight_sum is None else weight_sum + batch_sum
                retrieval_count += int(np.prod(selected.shape[:-1]))
            gate = aux["gate"]
            if gate is not None:
                nodes = gate.shape[1]
                mask = ~torch.eye(nodes, dtype=torch.bool, device=device)
                selected_gate = gate[:, mask]
                gate_sum += float(selected_gate.sum())
                gate_count += selected_gate.numel()

    scene_results = {}
    family_results: dict[int, list[dict[str, float]]] = defaultdict(list)
    suffix = "_time_shuffled" if arguments.time_shuffle else ""
    model_type = checkpoint["model_config"]["model_type"]
    for scene in sorted(predictions):
        ordered = sorted(predictions[scene], key=lambda item: item[0])
        subject_probability = np.stack([item[1] for item in ordered])
        mean_probability = subject_probability.mean(axis=0)
        all_metrics = directed_metrics(
            targets[scene], mean_probability, arguments.threshold
        )
        metrics = {name: all_metrics[name] for name in METRIC_NAMES}
        mean_subject_metrics = {
            name: float(np.nanmean([item[name] for item in subject_metrics[scene]]))
            for name in METRIC_NAMES
        }
        scene_results[scene] = {
            "scene_level": metrics,
            "mean_subject_level": mean_subject_metrics,
            "subjects": len(ordered),
        }
        family_results[base_network(scene)].append(metrics)
        tag = getattr(arguments, "checkpoint_tag", "")
        tag_suffix = f"_{tag}" if tag else ""
        np.savez_compressed(
            output_dir / f"{model_type}_{scene}{tag_suffix}{suffix}_predictions.npz",
            probabilities=subject_probability,
            mean_probability=mean_probability,
            adj=targets[scene],
        )

    family_summary = {
        str(family): {
            name: float(np.nanmean([scene[name] for scene in scenes]))
            for name in METRIC_NAMES
        }
        for family, scenes in sorted(family_results.items())
    }
    overall = {
        name: float(
            np.nanmean(
                [result["scene_level"][name] for result in scene_results.values()]
            )
        )
        for name in METRIC_NAMES
    }
    results = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "parameter_count": checkpoint["parameter_count"],
        "split": arguments.split,
        "threshold": arguments.threshold,
        "time_shuffle": arguments.time_shuffle,
        "scenes": scene_results,
        "families": family_summary,
        "overall_scene_mean": overall,
        "memory": memory_statistics(
            weight_sum, retrieval_count, gate_sum, gate_count
        ),
    }
    if arguments.predictions_only:
        print(
            f"saved_predictions model={model_type} split={arguments.split} "
            f"scenes={len(scene_results)}"
        )
        return
    output_path = output_dir / f"{model_type}_{arguments.split}{suffix}_metrics.json"
    save_json(results, output_path)
    print(f"saved={output_path}")
    print(f"overall={overall}")
    if results["memory"] is not None:
        print(f"memory={results['memory']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-type", choices=MODEL_TYPES)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--time-shuffle", action="store_true")
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--checkpoint-tag", default="")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    modes = sum(
        (
            arguments.select_threshold,
            arguments.evaluate_existing,
            arguments.summarize_existing,
        )
    )
    if modes > 1:
        raise SystemExit(
            "Choose only one of --select-threshold, --evaluate-existing, "
            "and --summarize-existing"
        )
    if arguments.summarize_existing:
        summarize_existing_results()
    elif arguments.select_threshold:
        if arguments.model_type is None:
            raise SystemExit("--model-type is required with --select-threshold")
        select_validation_threshold(arguments.model_type)
    elif arguments.evaluate_existing:
        if arguments.model_type is None:
            raise SystemExit("--model-type is required with --evaluate-existing")
        evaluate_existing_predictions(arguments.model_type)
    else:
        if arguments.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --evaluate-existing is used")
        evaluate(arguments)
