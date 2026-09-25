"""Raw time-series to Individual-EC generation."""

from __future__ import annotations

from .training import train_individual_ec
from .utils import extract_subject_ec


def generate_subject_ec(args, subjects, device):
    """Train Individual-EC and extract one directed EC per subject."""
    model, training_metrics = train_individual_ec(
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
