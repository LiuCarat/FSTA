"""Train MemoryEC on all simulations, then infer frozen EC on MTLnet."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from MemoryEC.data.real.loader import MTLNET_PROCESSED_DIR, load_mtlnet
from MemoryEC.data.simulation.loader import PROCESSED_DIR, SPLITS_PATH, load_splits, make_dataloader
from MemoryEC.model.memory_ec import MemoryEC
from MemoryEC.train import resolve_device, run_epoch
from MemoryEC.utils import adaptive_threshold, configure_determinism, count_parameters, save_json, set_seed, zero_diagonal


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs" / "real" / "MTLnet"
TRAIN_KEYS = (
    "hidden_dim", "memory_slots", "memory_temperature", "attention_heads",
    "dropout", "learning_rate", "weight_decay", "batch_size", "epochs",
)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment_name(args: argparse.Namespace) -> str:
    name = f"all_simulations_to_mtlnet_{args.model_type}_r{args.repeats}"
    if args.memory_temperature != 1.0:
        name += f"_tau{args.memory_temperature:g}"
    return name


def selected_hemispheres(value: str) -> tuple[str, ...]:
    return ("left", "right") if value == "both" else (value,)


def load_checkpoint(path: Path, device: str) -> tuple[MemoryEC, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MemoryEC(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def all_simulation_scenes() -> list[str]:
    splits = load_splits(SPLITS_PATH)
    scenes = [scene for split in ("train", "val", "test") for scene in splits[split]["scenes"]]
    if len(scenes) != 18 or len(set(scenes)) != 18:
        raise ValueError(f"Expected 18 unique Sanchez-Romero scenes, found {len(set(scenes))}")
    networks = {int(scene.removeprefix("Network").split("_", maxsplit=1)[0]) for scene in scenes}
    if networks != set(range(1, 10)):
        raise ValueError(f"Expected Networks 1-9, found {sorted(networks)}")
    return scenes


def train_full_simulation(
    args: argparse.Namespace,
    repeat: int,
    seed: int,
    repeat_dir: Path,
    scenes: list[str],
) -> Path:
    set_seed(seed)
    loader = make_dataloader(
        "train",
        batch_size=args.batch_size,
        seed=seed,
        permute_rois=True,
        balanced=False,
        scenes=scenes,
    )
    model = MemoryEC(
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        memory_slots=args.memory_slots,
        memory_temperature=args.memory_temperature,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint_path = repeat_dir / f"{args.model_type}_final.pt"
    history_path = repeat_dir / f"{args.model_type}_history.json"
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        train_loss, _ = run_epoch(model, loader, torch.device(args.device), optimizer)
        history.append({"epoch": epoch, "train_loss": train_loss})
        save_json(history, history_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"repeat={repeat:02d} epoch={epoch:03d} train_loss={train_loss:.4f}", flush=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "epoch": args.epochs,
        "selection_metric": "predeclared_final_epoch",
        "parameter_count": count_parameters(model),
        "seed": seed,
        "training_scenes": scenes,
        "training_networks": list(range(1, 10)),
        "training_arguments": {key: getattr(args, key) for key in TRAIN_KEYS},
    }
    torch.save(checkpoint, checkpoint_path)
    save_json(
        {
            "repeat": repeat,
            "seed": seed,
            "epochs_ran": args.epochs,
            "checkpoint_selection": "final checkpoint at predeclared fixed epoch",
            "checkpoint": str(checkpoint_path),
            "training_scene_count": len(scenes),
            "training_scenes": scenes,
            "elapsed_seconds": time.time() - started,
        },
        repeat_dir / "train_summary.json",
    )
    return checkpoint_path


def selected_checkpoint(directory: Path, args: argparse.Namespace) -> Path:
    path = directory / f"repeat_{args.selected_repeat:02d}" / f"{args.model_type}_final.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {path}")
    return path


def infer_mtlnet_subjects(
    model: MemoryEC,
    side: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_mtlnet(side)
    bold = data["bold"]
    lengths = data["lengths"].astype(int)
    subject_probabilities = np.zeros((len(bold), bold.shape[-1], bold.shape[-1]), dtype=np.float32)
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for start in range(0, len(bold), args.batch_size):
            batch_indices = np.arange(start, min(start + args.batch_size, len(bold)))
            if len(set(lengths[batch_indices].tolist())) != 1:
                raise ValueError("MTLnet batch contains inconsistent timepoint lengths")
            length = int(lengths[batch_indices[0]])
            batch = torch.from_numpy(bold[batch_indices, :length]).float().to(args.device)
            subject_probabilities[batch_indices] = model(batch).sigmoid().cpu().numpy()
    subject_probabilities = zero_diagonal(subject_probabilities)
    group_probability = zero_diagonal(subject_probabilities.mean(axis=0))
    return subject_probabilities, group_probability, data["adj"], data["roi_names"]


def reference_edge_rows(
    binary: np.ndarray,
    reference: np.ndarray,
    probability: np.ndarray,
    roi_names: np.ndarray,
) -> list[dict]:
    rows = []
    for source_index, target_index in zip(*np.where((binary != 0) | (reference != 0))):
        predicted = bool(binary[source_index, target_index])
        expected = bool(reference[source_index, target_index])
        status = "matched" if predicted and expected else "extra" if predicted else "missed"
        rows.append({
            "source_roi": str(roi_names[source_index]),
            "target_roi": str(roi_names[target_index]),
            "edge": f"{roi_names[source_index]} -> {roi_names[target_index]}",
            "status": status,
            "figure_color": {"matched": "black", "extra": "green", "missed": "red"}[status],
            "probability": float(probability[source_index, target_index]),
            "predicted": int(predicted),
            "reference": int(expected),
        })
    return rows


def save_reference_comparison_figure(
    edge_rows: list[dict],
    roi_names: np.ndarray,
    side: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(roi_names), endpoint=False)
    positions = {
        str(name): np.array([np.cos(angle), np.sin(angle)])
        for name, angle in zip(roi_names, angles)
    }
    colors = {"matched": "black", "extra": "#16a34a", "missed": "#dc2626"}
    figure, axis = plt.subplots(figsize=(7, 7))
    for row in edge_rows:
        source = positions[row["source_roi"]]
        target = positions[row["target_roi"]]
        direction = target - source
        distance = np.linalg.norm(direction)
        start = source + direction / distance * 0.13
        end = target - direction / distance * 0.13
        bend = 0.14 if row["source_roi"] < row["target_roi"] else -0.14
        axis.add_patch(FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color=colors[row["status"]],
            connectionstyle=f"arc3,rad={bend}",
            alpha=0.9,
        ))
    for name, position in positions.items():
        axis.scatter(*position, s=950, color="white", edgecolor="#334155", linewidth=1.8, zorder=3)
        axis.text(*position, name, ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
    counts = {status: sum(row["status"] == status for row in edge_rows) for status in colors}
    axis.legend(
        handles=[
            Line2D([0], [0], color=colors[status], lw=2, label=f"{label}: {counts[status]}")
            for status, label in (
                ("matched", "Matched reference"),
                ("extra", "Extra predicted"),
                ("missed", "Missed reference"),
            )
        ],
        loc="lower center",
        frameon=False,
    )
    axis.set_title(f"MTLnet {side.capitalize()} Hemisphere", fontsize=14)
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.45, 1.25)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def infer_selected_model(directory: Path, args: argparse.Namespace) -> None:
    checkpoint_path = selected_checkpoint(directory, args)
    model, checkpoint = load_checkpoint(checkpoint_path, args.device)
    inference_dir = directory / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for side in selected_hemispheres(args.hemisphere):
        subject_probability, group_probability, reference, roi_names = infer_mtlnet_subjects(model, side, args)
        threshold = adaptive_threshold(group_probability, args.fsta_ratio)
        group_binary = (group_probability >= threshold).astype(np.uint8)
        np.fill_diagonal(group_binary, 0)
        edge_rows = reference_edge_rows(group_binary, reference, group_probability, roi_names)
        np.savez_compressed(
            inference_dir / f"{side}_ec_predictions.npz",
            subject_probability=subject_probability,
            group_probability=group_probability,
            group_binary=group_binary,
            reference=reference,
            roi_names=roi_names,
            subject_ids=load_mtlnet(side)["subject_ids"],
            threshold=np.asarray(threshold),
        )
        write_csv(edge_rows, inference_dir / f"{side}_reference_edge_comparison.csv")
        save_json(edge_rows, inference_dir / f"{side}_reference_edge_comparison.json")
        save_reference_comparison_figure(
            edge_rows,
            roi_names,
            side,
            inference_dir / f"{side}_reference_edge_comparison.png",
        )
        counts = {status: sum(row["status"] == status for row in edge_rows) for status in ("matched", "extra", "missed")}
        summaries.append({
            "hemisphere": side,
            "subjects": int(subject_probability.shape[0]),
            "subject_matrix_shape": list(subject_probability.shape[1:]),
            "threshold_method": "predeclared FSTA range ratio",
            "fsta_ratio": args.fsta_ratio,
            "threshold": threshold,
            **counts,
        })
        print(
            f"{side}: subjects={subject_probability.shape[0]} "
            f"matched={counts['matched']} extra={counts['extra']} missed={counts['missed']}",
            flush=True,
        )
    save_json(
        {
            "selected_checkpoint": str(checkpoint_path),
            "selected_repeat": args.selected_repeat,
            "selected_seed": checkpoint["seed"],
            "checkpoint_epoch": checkpoint["epoch"],
            "model_frozen": True,
            "reference_interpretation": "literature reference network, not directly measured ground-truth EC",
            "hemispheres": summaries,
        },
        inference_dir / "inference_summary.json",
    )


def config(args: argparse.Namespace) -> dict:
    scenes = all_simulation_scenes()
    return {
        "protocol": "all_18_simulations_train_then_frozen_mtlnet_inference_v1",
        "model": args.model_type,
        "repeats": args.repeats,
        "seeds": [args.seed + repeat for repeat in range(args.repeats)],
        "selected_repeat": args.selected_repeat,
        "checkpoint_selection": "predeclared repeat index and final checkpoint at fixed epoch",
        "real_reference_used_for_selection": False,
        "training": {key: getattr(args, key) for key in TRAIN_KEYS},
        "simulation_processed_dir": str(PROCESSED_DIR),
        "simulation_scene_count": len(scenes),
        "simulation_scenes": scenes,
        "simulation_networks": list(range(1, 10)),
        "mtlnet_processed_dir": str(MTLNET_PROCESSED_DIR),
        "mtlnet_subjects": 23,
        "mtlnet_numeric_timepoints": 420,
        "mtlnet_rois": ["CA1", "CA23DG", "SUB", "ERC", "BA35", "BA36", "PHC"],
        "inference_model_frozen": True,
        "threshold_method": "predeclared FSTA range ratio",
        "fsta_ratio": args.fsta_ratio,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "infer", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hemisphere", choices=("left", "right", "both"), default="both")
    parser.add_argument("--model-type", choices=("st_pair_memory",), default="st_pair_memory")
    parser.add_argument("--repeats", type=int, choices=(1, 3, 5, 10), default=1)
    parser.add_argument("--selected-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=128)
    parser.add_argument("--memory-temperature", type=float, default=1.0)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fsta-ratio", type=float, default=0.3)
    args = parser.parse_args(argv)

    for key in ("hidden_dim", "memory_slots", "attention_heads", "batch_size", "epochs"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if not 1 <= args.selected_repeat <= args.repeats:
        parser.error("selected-repeat must be between 1 and repeats")
    if args.hidden_dim % args.attention_heads:
        parser.error("hidden-dim must be divisible by attention-heads")
    if not math.isfinite(args.fsta_ratio) or not 0 <= args.fsta_ratio <= 1:
        parser.error("fsta-ratio must be finite and in [0,1]")
    if not math.isfinite(args.memory_temperature) or args.memory_temperature <= 0:
        parser.error("memory-temperature must be finite and positive")
    for key in ("learning_rate", "weight_decay", "dropout"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and non-negative")
    if args.dropout >= 1 or args.seed < 0 or args.seed + args.repeats - 1 >= 2**32:
        parser.error("dropout must be < 1; seeds must fit numpy's unsigned 32-bit range")
    return args


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    configure_determinism()
    args.device = str(resolve_device(args.device))
    directory = OUTPUTS / experiment_name(args)
    config_path = directory / "config.json"
    expected_config = config(args)

    if args.mode in {"train", "all"}:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        save_json(expected_config, config_path)
        scenes = expected_config["simulation_scenes"]
        for repeat, seed in enumerate(expected_config["seeds"], 1):
            repeat_dir = directory / f"repeat_{repeat:02d}"
            repeat_dir.mkdir(parents=True)
            print(
                f"\nRepeat {repeat}/{args.repeats}, seed={seed}: "
                f"training on all {len(scenes)} simulation scenes",
                flush=True,
            )
            train_full_simulation(args, repeat, seed, repeat_dir, scenes)
    else:
        if not config_path.is_file():
            raise FileNotFoundError(f"No trained MTLnet transfer experiment found: {directory}")
        if json.loads(config_path.read_text(encoding="utf-8")) != expected_config:
            raise ValueError(f"Existing experiment configuration differs: {config_path}")

    if args.mode in {"infer", "all"}:
        print(f"\nSelected repeat: {args.selected_repeat}; model frozen for MTLnet inference", flush=True)
        infer_selected_model(directory, args)


if __name__ == "__main__":
    main()
