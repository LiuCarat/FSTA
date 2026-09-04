"""Raw time-series to Original-BEC generation for the STF-BEC encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .training import train_stf_bec
from .utils import extract_subject_bec


def generate_subject_bec(args, subjects, device):
    """Train STF-BEC and extract one directed Original BEC per subject."""
    model, training_metrics = train_stf_bec(
        args,
        subjects["time_series"],
        device,
        subjects.get("window_ranges"),
    )
    extracted = extract_subject_bec(
        model,
        subjects["records"],
        subjects["time_series"],
        args.window_length,
        args.stride,
        device,
        subjects.get("window_ranges"),
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
