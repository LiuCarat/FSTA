"""Raw time-series to Original-EC generation for the STF-EC encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .training import train_stf_ec
from .utils import extract_subject_ec


def generate_subject_ec(args, subjects, device):
    """Train STF-EC and extract one directed Original EC per subject."""
    model, training_metrics = train_stf_ec(
        args,
        subjects["time_series"],
        device,
        subjects.get("window_ranges"),
    )
    checkpoint_path = getattr(args, "initial_ec_checkpoint_path", None)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "checkpoint_selection": str(args.stf_checkpoint),
                "training_metrics": training_metrics,
            },
            checkpoint_path,
        )
    config_path = getattr(args, "initial_ec_config_path", None)
    if config_path is not None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    extracted = extract_subject_ec(
        model,
        subjects["records"],
        subjects["time_series"],
        args.window_length,
        args.stride,
        device,
        subjects.get("window_ranges"),
    )
    return {
        "ec": extracted["ec"],
        "labels": subjects["labels"],
        "subject_ids": subjects["subject_ids"],
        "site_ids": subjects["site_ids"],
        "reconstruction_mse": extracted["reconstruction_mse"],
    }, training_metrics


def save_subject_ec(path, data):
    """Persist a reusable EC archive for Graph-EC's ``ec`` mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ec=data["ec"],
        labels=data["labels"],
        subject_ids=data["subject_ids"],
        site_ids=data["site_ids"],
        reconstruction_mse=data["reconstruction_mse"],
    )
