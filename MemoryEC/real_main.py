"""Train MemoryEC on all simulations, then infer frozen EC on MTLnet."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from MemoryEC.data.real.loader import load_mtlnet
from MemoryEC.data.simulation.loader import SPLITS_PATH, load_splits, make_dataloader
from MemoryEC.model.memory_ec import MemoryEC
from MemoryEC.train import resolve_device, run_epoch
from MemoryEC.utils import adaptive_threshold, configure_determinism, save_json, set_seed, zero_diagonal


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs" / "real"
MODEL_TYPE = "st_pair_memory"
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
    name = f"all_simulations_to_mtlnet_{MODEL_TYPE}_r{len(args.seeds)}"
    if args.memory_temperature != 1.0:
        name += f"_tau{args.memory_temperature:g}"
    return name


def selected_hemispheres(value: str) -> tuple[str, ...]:
    return ("left", "right") if value == "both" else (value,)


def load_checkpoint(path: Path, device: str) -> MemoryEC:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MemoryEC(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def all_simulation_scenes() -> list[str]:
    splits = load_splits(SPLITS_PATH)
    return [scene for split in ("train", "val", "test") for scene in splits[split]["scenes"]]


def train_full_simulation(
    args: argparse.Namespace,
    repeat: int,
    seed: int,
    repeat_dir: Path,
    scenes: list[str],
) -> None:
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
        model_type=MODEL_TYPE,
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
    checkpoint_path = repeat_dir / f"{MODEL_TYPE}_final.pt"
    for epoch in range(1, args.epochs + 1):
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        train_loss, _ = run_epoch(model, loader, torch.device(args.device), optimizer)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"repeat={repeat:02d} epoch={epoch:03d} train_loss={train_loss:.4f}", flush=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config(),
        },
        checkpoint_path,
    )


def infer_mtlnet_subjects(
    model: MemoryEC,
    side: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_mtlnet(side)
    bold = data["bold"]
    subject_probabilities = np.zeros((len(bold), bold.shape[-1], bold.shape[-1]), dtype=np.float32)
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for start in range(0, len(bold), args.batch_size):
            batch_indices = np.arange(start, min(start + args.batch_size, len(bold)))
            batch = torch.from_numpy(bold[batch_indices]).float().to(args.device)
            subject_probabilities[batch_indices] = model(batch).sigmoid().cpu().numpy()
    subject_probabilities = zero_diagonal(subject_probabilities)
    group_probability = zero_diagonal(subject_probabilities.mean(axis=0))
    return (
        subject_probabilities,
        group_probability,
        data["adj"],
        data["roi_names"],
        data["subject_ids"],
    )


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
            "status": status,
            "probability": float(probability[source_index, target_index]),
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


def infer_repeat(
    directory: Path,
    args: argparse.Namespace,
    repeat: int,
    expected_seed: int,
) -> dict:
    checkpoint_path = directory / f"repeat_{repeat:02d}" / f"{MODEL_TYPE}_final.pt"
    model = load_checkpoint(checkpoint_path, args.device)
    inference_dir = directory / "inference" / f"repeat_{repeat:02d}"
    inference_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for side in selected_hemispheres(args.hemisphere):
        subject_probability, group_probability, reference, roi_names, subject_ids = infer_mtlnet_subjects(
            model, side, args
        )
        threshold = adaptive_threshold(group_probability, args.fsta_ratio)
        group_binary = (group_probability >= threshold).astype(np.uint8)
        np.fill_diagonal(group_binary, 0)
        edge_rows = reference_edge_rows(group_binary, reference, group_probability, roi_names)
        np.savez_compressed(
            inference_dir / f"{side}_ec_predictions.npz",
            subject_probability=subject_probability,
            group_probability=group_probability,
            group_binary=group_binary,
            roi_names=roi_names,
            subject_ids=subject_ids,
            threshold=np.asarray(threshold),
        )
        write_csv(edge_rows, inference_dir / f"{side}_reference_edge_comparison.csv")
        save_reference_comparison_figure(
            edge_rows,
            roi_names,
            side,
            inference_dir / f"{side}_reference_edge_comparison.png",
        )
        counts = {status: sum(row["status"] == status for row in edge_rows) for status in ("matched", "extra", "missed")}
        summaries.append({
            "hemisphere": side,
            "threshold": threshold,
            **counts,
        })
        print(
            f"{side}: subjects={subject_probability.shape[0]} "
            f"matched={counts['matched']} extra={counts['extra']} missed={counts['missed']}",
            flush=True,
        )
    return {
        "repeat": repeat,
        "seed": expected_seed,
        "hemispheres": summaries,
    }


def infer_all_repeats(directory: Path, args: argparse.Namespace) -> None:
    runs = []
    for repeat, seed in enumerate(args.seeds, 1):
        print(
            f"\nRepeat {repeat}/{len(args.seeds)}, seed={seed}: "
            "model frozen for MTLnet inference",
            flush=True,
        )
        runs.append(infer_repeat(directory, args, repeat, seed))
    save_json(
        {
            "seeds": args.seeds,
            "runs": runs,
        },
        directory / "inference" / "inference_summary.json",
    )


def config(args: argparse.Namespace) -> dict:
    scenes = all_simulation_scenes()
    return {
        "seeds": args.seeds,
        "training": {key: getattr(args, key) for key in TRAIN_KEYS},
        "simulation_scenes": scenes,
        "fsta_ratio": args.fsta_ratio,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "infer", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hemisphere", choices=("left", "right", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=128)
    parser.add_argument("--memory-temperature", type=float, default=1.0)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--fsta-ratio", type=float, default=0.3)
    args = parser.parse_args(argv)
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
                f"\nRepeat {repeat}/{len(args.seeds)}, seed={seed}: "
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
        infer_all_repeats(directory, args)


if __name__ == "__main__":
    main()
