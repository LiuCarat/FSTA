"""Fold-safe QC confidence for fixed patient-graph neighborhoods."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DEFAULT_QC_COLUMNS = ("func_mean_fd", "func_dvars", "func_quality")


def _parse_numeric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def load_aligned_qc(csv_path, subject_ids, columns=DEFAULT_QC_COLUMNS):
    """Read QC columns and align them to the BEC subject order."""
    columns = tuple(columns)
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"FILE_ID", *columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing QC columns: {sorted(missing)}")
        rows = {
            row["FILE_ID"].strip(): row
            for row in reader
            if row.get("FILE_ID", "").strip()
            and row["FILE_ID"].strip() != "no_filename"
        }
    values = np.full((len(subject_ids), len(columns)), np.nan, dtype=np.float64)
    for row_index, subject_id in enumerate(subject_ids):
        row = rows.get(str(subject_id))
        if row is None:
            raise ValueError(f"Could not match QC subject: {subject_id}")
        for column_index, column in enumerate(columns):
            values[row_index, column_index] = _parse_numeric(row.get(column))
    return values.astype(np.float32)


def fit_qc_scaler(train_qc):
    """Fit log-IQR normalization using only one training fold."""
    values = np.asarray(train_qc, dtype=np.float64)
    raw_median = np.nanmedian(values, axis=0)
    raw_median[~np.isfinite(raw_median)] = 0.0
    filled = np.where(np.isfinite(values), values, raw_median)
    transformed = np.log1p(np.maximum(filled, 0.0))
    median = np.median(transformed, axis=0)
    q1, q3 = np.percentile(transformed, [25.0, 75.0], axis=0)
    iqr = q3 - q1
    iqr[~np.isfinite(iqr) | (iqr < 1e-6)] = 1.0
    return {
        "fill": raw_median.astype(np.float32),
        "median": median.astype(np.float32),
        "iqr": iqr.astype(np.float32),
    }


def transform_qc(qc, scaler):
    values = np.asarray(qc, dtype=np.float64)
    fill = np.asarray(scaler["fill"], dtype=np.float64)
    values = np.where(np.isfinite(values), values, fill)
    values = np.log1p(np.maximum(values, 0.0))
    return ((values - scaler["median"]) / scaler["iqr"]).astype(np.float32)


def prepare_qc_fold(train_qc, val_qc, test_qc):
    scaler = fit_qc_scaler(train_qc)
    return {
        "train": transform_qc(train_qc, scaler),
        "val": transform_qc(val_qc, scaler),
        "test": transform_qc(test_qc, scaler),
        "scaler": scaler,
    }


def _qc_pairwise_distance(query_qc, reference_qc):
    query_qc = np.asarray(query_qc, dtype=np.float32)
    reference_qc = np.asarray(reference_qc, dtype=np.float32)
    if query_qc.ndim != 2 or reference_qc.ndim != 2:
        raise ValueError("QC arrays must have shape [subjects, qc_features]")
    if query_qc.shape[1] != reference_qc.shape[1]:
        raise ValueError("query and reference QC feature counts must match")
    return np.abs(
        query_qc[:, None, :] - reference_qc[None, :, :]
    ).mean(axis=-1)


def fit_qc_mismatch_threshold(weights, train_qc, quantile=0.85):
    """Fit a dead-zone threshold from training-fold graph edges only."""
    weights = np.asarray(weights, dtype=np.float32)
    train_qc = np.asarray(train_qc, dtype=np.float32)
    if weights.ndim != 2 or weights.shape != (len(train_qc), len(train_qc)):
        raise ValueError("weights and train_qc must describe one training fold")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    distances = _qc_pairwise_distance(train_qc, train_qc)
    edge_distances = distances[weights > 0.0]
    if edge_distances.size == 0:
        edge_distances = distances[np.triu_indices(len(train_qc), k=1)]
    threshold = float(np.quantile(edge_distances, quantile))
    return max(threshold, 0.0)


def compute_qc_confidence(
    weights,
    query_qc,
    reference_qc,
    qc_lambda=0.5,
    min_confidence=0.9,
    mismatch_threshold=0.0,
):
    """Compute one confidence value per query from its fixed graph neighbors."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("weights must have shape [queries, references]")
    if weights.shape != (len(query_qc), len(reference_qc)):
        raise ValueError("weights and QC arrays have incompatible shapes")
    if qc_lambda < 0.0:
        raise ValueError("qc_lambda must be non-negative")
    if not 0.0 < min_confidence <= 1.0:
        raise ValueError("min_confidence must be in (0, 1]")
    if mismatch_threshold < 0.0:
        raise ValueError("mismatch_threshold must be non-negative")
    qc_distance = _qc_pairwise_distance(query_qc, reference_qc)
    excess = np.maximum(qc_distance - float(mismatch_threshold), 0.0)
    compatibility = min_confidence + (1.0 - min_confidence) * np.exp(
        -float(qc_lambda) * excess
    )
    compatibility = np.where(
        qc_distance <= mismatch_threshold, 1.0, compatibility
    )
    row_sum = np.maximum(weights.sum(axis=1), 1e-8)
    confidence = (weights * compatibility.astype(np.float32)).sum(axis=1) / row_sum
    audit = {
        "qc_confidence_mean": float(np.mean(confidence)),
        "qc_confidence_min": float(np.min(confidence)),
        "qc_confidence_std": float(np.std(confidence)),
        "qc_mean_neighbor_mismatch": float(
            np.mean((weights * qc_distance).sum(axis=1) / row_sum)
        ),
        "qc_penalized_subject_fraction": float(
            np.mean(confidence < 1.0 - 1e-8)
        ),
    }
    return confidence.astype(np.float32), audit


__all__ = [
    "DEFAULT_QC_COLUMNS",
    "load_aligned_qc",
    "fit_qc_scaler",
    "transform_qc",
    "prepare_qc_fold",
    "fit_qc_mismatch_threshold",
    "compute_qc_confidence",
]
