"""Shared data utilities for PR-EC dataset loaders."""
from __future__ import annotations

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
