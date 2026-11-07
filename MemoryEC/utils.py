"""Shared training, normalization, and directed-graph evaluation utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


MODEL_TYPES = ("st_no_memory", "st_global_memory", "st_pair_memory")


def configure_determinism() -> None:
    """Configure deterministic kernels before initializing a CUDA context."""
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.mha.set_fastpath_enabled(False)


def set_seed(seed: int) -> None:
    configure_determinism()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_per_roi(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score each subject and ROI over time."""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def off_diagonal_mask(nodes: int, device: torch.device | None = None) -> torch.Tensor:
    return ~torch.eye(nodes, dtype=torch.bool, device=device)


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = off_diagonal_mask(logits.shape[-1], logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, mask], target[:, mask]
    )


def directed_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute directed metrics after excluding self-connections."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if target.shape != probability.shape or target.ndim != 2:
        raise ValueError("target and probability must be matching [N,N] arrays")
    mask = ~np.eye(target.shape[0], dtype=bool)
    labels = target[mask].astype(np.uint8)
    scores = probability[mask].astype(np.float64)
    prediction = (scores >= threshold).astype(np.uint8)
    auroc = float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else float("nan")
    return {
        "directed_f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": auroc,
        "shd": int(np.count_nonzero(labels != prediction)),
    }


def zero_diagonal(array: np.ndarray) -> np.ndarray:
    """Return a copy with a zero diagonal on the final two axes."""
    values = np.array(array, copy=True)
    if values.ndim < 2 or values.shape[-2] != values.shape[-1]:
        raise ValueError(f"Expected [...,N,N], got {values.shape}")
    diagonal = np.arange(values.shape[-1])
    values[..., diagonal, diagonal] = 0
    return values


def adaptive_threshold(probability: np.ndarray, ratio: float = 0.5) -> float:
    """FSTA-style scene threshold from off-diagonal sigmoid probabilities."""
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise ValueError("probability must be square [N,N]")
    values = probability[~np.eye(probability.shape[0], dtype=bool)]
    if not np.isfinite(values).all():
        raise ValueError("probability contains non-finite values")
    return float(values.min() + ratio * (values.max() - values.min()))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(data: dict | list, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n", encoding="utf-8")
