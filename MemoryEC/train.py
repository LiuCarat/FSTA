"""Training implementation used by main.py; supports explicit family-local splits."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from MemoryEC.data.simulation.loader import make_dataloader
from MemoryEC.model.memory_ec import MemoryEC
from MemoryEC.utils import (
    count_parameters,
    directed_metrics,
    masked_bce_loss,
    save_json,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "simulation"


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
    lambda_memory: float = 0.1,
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_edges = 0
    loss_totals = defaultdict(float)
    total_pairs = 0
    slot_usage = None
    scene_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    scene_targets: dict[str, np.ndarray] = {}

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            bold = batch["bold"].to(device)
            adj = batch["adj"].to(device)
            if model.model_type == "st_pair_memory":
                logits, aux = model(bold, return_aux=True)
                nodes = adj.shape[-1]
                source, target = torch.triu_indices(nodes, nodes, offset=1, device=device)
                relation_target = (
                    adj[:, source, target] + 2 * adj[:, target, source]
                ).long()
                relation_logits = aux["relation_logits"][:, source, target]
                memory_logits = aux["memory_logits"][:, source, target]
                relation_loss = F.cross_entropy(
                    relation_logits.reshape(-1, 4),
                    relation_target.reshape(-1),
                )
                memory_loss = F.cross_entropy(
                    memory_logits.reshape(-1, 4),
                    relation_target.reshape(-1),
                )
                objective_loss = relation_loss + lambda_memory * memory_loss
                bce_loss = masked_bce_loss(logits, adj)
                loss = objective_loss if training else bce_loss
                pair_count = bold.shape[0] * source.numel()
                loss_totals["relation_loss"] += float(relation_loss.detach()) * pair_count
                loss_totals["memory_loss"] += float(memory_loss.detach()) * pair_count
                loss_totals["objective_loss"] += float(objective_loss.detach()) * pair_count
                loss_totals["bce_loss"] += float(bce_loss.detach()) * pair_count
                total_pairs += pair_count
                usage = aux["memory_weights"][:, source, target].mean(dim=(0, 1, 2))
                if slot_usage is None:
                    slot_usage = usage.detach() * pair_count
                else:
                    slot_usage += usage.detach() * pair_count
            else:
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

    metrics: dict = {}
    if model.model_type == "st_pair_memory":
        metrics.update({name: value / total_pairs for name, value in loss_totals.items()})
        metrics["slot_retrieval_utilization"] = (slot_usage / total_pairs).cpu().tolist()
        metrics.update(model.memory.assignment_statistics())
    else:
        metrics["bce_loss"] = total_loss / total_edges
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
    plt.ylabel("Loss")
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
        memory_temperature=arguments.memory_temperature,
        attention_heads=arguments.attention_heads,
        dropout=arguments.dropout,
    ).to(device)
    lambda_memory = getattr(arguments, "lambda_memory", 0.1)
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
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            lambda_memory,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            device,
            lambda_memory=lambda_memory,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        history.append(record)
        save_json(history, history_path)
        if epoch == 1 or epoch % 10 == 0:
            message = (
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_bce={val_loss:.4f} val_f1={val_metrics['directed_f1']:.4f}"
            )
            if arguments.model_type == "st_pair_memory":
                message += (
                    f" relation={train_metrics['relation_loss']:.4f}"
                    f" memory={train_metrics['memory_loss']:.4f}"
                )
            print(message, flush=True)

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
                    "memory_analysis": model.memory.assignment_statistics()
                    if model.memory is not None else None,
                    "training_metrics": train_metrics,
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
                "memory_analysis": model.memory.assignment_statistics()
                if model.memory is not None else None,
                "training_metrics": train_metrics,
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
