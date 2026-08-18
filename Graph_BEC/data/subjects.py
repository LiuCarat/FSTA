"""Subject-level dataset and BEC archive loading."""

from __future__ import annotations

import numpy as np

from Graph_BEC.datasets.abide import load_abide_records, load_time_series


def load_subject_dataset(data_root, pipeline="cpac", strategy="filt_global",
                         derivative="rois_aal", standardize=True, max_subjects=None):
    records = load_abide_records(
        data_root=data_root,
        pipeline=pipeline,
        strategy=strategy,
        derivative=derivative,
    )
    if max_subjects is not None:
        records = records[:max_subjects]
    time_series = [load_time_series(record, standardize=standardize) for record in records]
    if any(series.ndim != 2 or series.shape[1] != 90 for series in time_series):
        raise ValueError("Every subject must provide a [T, 90] ROI time series")
    return {
        "records": records,
        "time_series": time_series,
        "labels": np.asarray([record.label for record in records], dtype=np.int64),
        "subject_ids": np.asarray([record.subject_id for record in records]),
        "site_ids": np.asarray([record.site_id for record in records]),
    }


def load_bec_archive(path):
    archive = np.load(path, allow_pickle=False)
    required = {"bec", "labels", "subject_ids", "site_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"Missing BEC arrays: {sorted(missing)}")
    return {key: archive[key] for key in archive.files}

