"""Training implementation used by main.py; supports explicit family-local splits."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.loader import make_dataloader
from model.memory_ec import MemoryEC
from utils import (
    count_parameters,
    directed_metrics,
    masked_bce_loss,
    save_json,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run_epoch(
    model: MemoryEC,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_edges = 0
    scene_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    scene_targets: dict[str, np.ndarray] = {}

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            bold = batch["bold"].to(device)
            adj = batch["adj"].to(device)
            logits = model(bold)
            loss = masked_bce_loss(logits, adj)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            edge_count = bold.shape[0] * bold.shape[-1] * (bold.shape[-1] - 1)
            total_loss += float(loss.detach()) * edge_count
            total_edges += edge_count

            if not training:
                probabilities = logits.sigmoid().cpu().numpy()
                targets = adj.cpu().numpy()
                for index, scene in enumerate(batch["scene"]):
                    scene_probabilities[scene].append(probabilities[index])
                    scene_targets[scene] = targets[index]

    metrics: dict[str, float] = {}
    if not training:
        scene_metrics = [
            directed_metrics(scene_targets[scene], np.mean(predictions, axis=0))
            for scene, predictions in sorted(scene_probabilities.items())
        ]
        for name in ("directed_f1", "auprc", "auroc", "shd"):
            metrics[name] = float(np.nanmean([item[name] for item in scene_metrics]))
    return total_loss / total_edges, metrics


def save_loss_curve(history: list[dict], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train")
    plt.plot(epochs, [item["val_loss"] for item in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def train(arguments: argparse.Namespace) -> None:
    set_seed(arguments.seed)
    device = resolve_device("cuda")
    output_dir = Path(getattr(arguments, "output_dir", OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_scenes = getattr(arguments, "split_scenes", None)

    train_loader = make_dataloader(
        "train",
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        permute_rois=True,
        balanced=True,
        scenes=None if split_scenes is None else split_scenes["train"],
    )
    val_loader = make_dataloader(
        "val",
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        permute_rois=False,
        balanced=False,
        scenes=None if split_scenes is None else split_scenes["val"],
    )
    model = MemoryEC(
        model_type=arguments.model_type,
        hidden_dim=arguments.hidden_dim,
        memory_slots=arguments.memory_slots,
        attention_heads=arguments.attention_heads,
        dropout=arguments.dropout,
    ).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    checkpoint_path = output_dir / f"{arguments.model_type}_best.pt"
    auprc_checkpoint_path = output_dir / f"{arguments.model_type}_best_val_auprc.pt"
    history_path = output_dir / f"{arguments.model_type}_history.json"
    curve_path = output_dir / f"{arguments.model_type}_loss_curve.png"
    best_loss = float("inf")
    best_auprc = float("-inf")
    best_loss_epoch = None
    best_auprc_epoch = None
    stale_epochs = 0
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, arguments.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_loss, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        history.append(record)
        save_json(history, history_path)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_metrics['directed_f1']:.4f}",
                flush=True,
            )

        if val_metrics["auprc"] > best_auprc + arguments.min_delta:
            best_auprc = val_metrics["auprc"]
            best_auprc_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config(),
                    "epoch": epoch,
                    "selection_metric": "val_auprc",
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "parameter_count": parameter_count,
                    "seed": arguments.seed,
                    "training_arguments": vars(arguments),
                },
                auprc_checkpoint_path,
            )

        if val_loss < best_loss - arguments.min_delta:
            best_loss = val_loss
            best_loss_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model.config(),
                "epoch": epoch,
                "selection_metric": "val_loss",
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "parameter_count": parameter_count,
                "seed": arguments.seed,
                "training_arguments": vars(arguments),
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= arguments.patience:
                print(f"early_stop epoch={epoch}", flush=True)
                break

    curve_saved = save_loss_curve(history, curve_path)
    summary = {
        "model_type": arguments.model_type,
        "validation_checkpoint_criterion": "validation BCE",
        "parameter_count": parameter_count,
        "best_val_loss": best_loss,
        "best_val_loss_epoch": best_loss_epoch,
        "best_val_auprc": best_auprc,
        "best_val_auprc_epoch": best_auprc_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
        "auprc_checkpoint": str(auprc_checkpoint_path),
        "history": str(history_path),
        "loss_curve": str(curve_path) if curve_saved else None,
    }
    save_json(summary, output_dir / f"{arguments.model_type}_train_summary.json")
    print(
        f"done epochs={len(history)} best_epoch={best_loss_epoch} "
        f"val_loss={best_loss:.4f} time={summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
