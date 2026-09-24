"""Shared data utilities for Graph-BEC dataset loaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SOURCE_ROI_COUNT = 116
ROI_COUNT = 90
ROI_INDICES = np.arange(ROI_COUNT, dtype=np.int64)


def standardize_time_series(time_series):
    time_series = np.asarray(time_series, dtype=np.float32)
    mean = time_series.mean(axis=0, keepdims=True)
    standard_deviation = time_series.std(axis=0, keepdims=True)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return ((time_series - mean) / standard_deviation).astype(np.float32, copy=False)


def validate_time_series(time_series, subject_id, roi_count=ROI_COUNT):
    if time_series.ndim != 2 or time_series.shape[1] != roi_count:
        raise ValueError(
            f"Every subject must provide a [T, {roi_count}] ROI time series; "
            f"{subject_id} has {time_series.shape}"
        )
    if not np.isfinite(time_series).all():
        raise ValueError(f"Non-finite values found for {subject_id}")
    return time_series.astype(np.float32, copy=False)


def load_bec_archive(path):
    archive = np.load(Path(path), allow_pickle=False)
    required = {"bec", "labels", "subject_ids", "site_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"Missing BEC arrays: {sorted(missing)}")
    return {key: archive[key] for key in archive.files}


def limit_archive_subjects(data, max_subjects):
    if max_subjects is None:
        return data
    subject_count = len(data["bec"])
    return {
        key: value[:max_subjects]
        if hasattr(value, "__len__") and len(value) == subject_count
        else value
        for key, value in data.items()
    }
