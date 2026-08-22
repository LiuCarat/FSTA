"""Raw time-series to Original-BEC generation for the FSTA-EC model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fsta_training import train_fsta
from .fsta_utils import extract_subject_bec


def generate_subject_bec(args, subjects, device):
    """Train FSTA and extract one directed Original BEC per subject."""
    model, training_metrics = train_fsta(args, subjects["time_series"], device)
    extracted = extract_subject_bec(
        model,
        subjects["records"],
        subjects["time_series"],
        args.window_length,
        args.stride,
        device,
    )
    return {
        "bec": extracted["bec"],
        "labels": subjects["labels"],
        "subject_ids": subjects["subject_ids"],
        "site_ids": subjects["site_ids"],
        "reconstruction_mse": extracted["reconstruction_mse"],
    }, training_metrics


def save_subject_bec(path, data):
    """Persist a reusable BEC archive for Graph-BEC's ``bec`` mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bec=data["bec"],
        labels=data["labels"],
        subject_ids=data["subject_ids"],
        site_ids=data["site_ids"],
        reconstruction_mse=data["reconstruction_mse"],
    )
