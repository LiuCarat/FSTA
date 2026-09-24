"""Raw time-series to subject-EC generation for the FSTA-EC baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fsta_training import train_fsta
from utils.utils import extract_subject_ec


def generate_subject_ec(args, subjects, device):
    """Train FSTA-EC and extract one directed EC matrix per subject."""
    model, training_metrics = train_fsta(args, subjects["time_series"], device)
    extracted = extract_subject_ec(
        model,
        subjects["records"],
        subjects["time_series"],
        args.window_length,
        args.stride,
        device,
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
