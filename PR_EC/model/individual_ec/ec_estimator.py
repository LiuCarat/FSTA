"""Raw time-series to Original-EC generation for the STF-EC encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
