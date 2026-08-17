"""Fold-safe QC supervision utilities for QSR-BEC training."""
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
    """Load QC values in the same order as the BEC archive."""
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
    """Fit log-IQR QC normalization using only one training fold."""
    values = np.asarray(train_qc, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("train_qc must be a non-empty [subjects, features] array")
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


def transform_qc_badness(qc, scaler):
    """Return one-sided, fold-normalized QC badness scores."""
    values = np.asarray(qc, dtype=np.float64)
    fill = np.asarray(scaler["fill"], dtype=np.float64)
    values = np.where(np.isfinite(values), values, fill)
    normalized = (
        np.log1p(np.maximum(values, 0.0)) - scaler["median"]
    ) / scaler["iqr"]
    return np.maximum(normalized, 0.0).astype(np.float32)


def build_confound_design(site_ids, phenotype_values):
    """Encode non-diagnostic SITE/AGE/SEX/FIQ/PIQ confounds for regression."""
    sites = np.asarray(site_ids).astype(str)
    values = np.asarray(phenotype_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(sites):
        raise ValueError("phenotype_values must match site_ids by subject")
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    values = np.where(np.isfinite(values), values, medians)
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0)
    standard_deviations[standard_deviations < 1e-6] = 1.0
    continuous = (values - means) / standard_deviations
    categories = sorted(set(sites))
    if len(categories) > 1:
        reference = categories[0]
        site_columns = np.stack(
            [(sites == category).astype(np.float64) for category in categories if category != reference],
            axis=1,
        )
        return np.concatenate((continuous, site_columns), axis=1).astype(np.float32)
    return continuous.astype(np.float32)


def fit_qc_artifact_basis(train_bec, train_qc_badness, train_confounds, ridge=1e-3):
    """Estimate confound-controlled QC-associated directed BEC basis maps."""
    bec = np.asarray(train_bec, dtype=np.float64)
    qc_badness = np.asarray(train_qc_badness, dtype=np.float64)
    confounds = np.asarray(train_confounds, dtype=np.float64)
    if bec.ndim != 3 or len(bec) != len(qc_badness) or len(bec) != len(confounds):
        raise ValueError("BEC, QC, and confounds must have the same subject count")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    design = np.concatenate(
        (np.ones((len(bec), 1), dtype=np.float64), qc_badness, confounds), axis=1
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ bec.reshape(len(bec), -1),
    )
    basis = coefficients[1:1 + qc_badness.shape[1]].reshape(
        qc_badness.shape[1], bec.shape[1], bec.shape[2]
    )
    diagonal = np.arange(bec.shape[-1])
    basis[:, diagonal, diagonal] = 0.0
    return basis.astype(np.float32)


def build_qc_sensitive_map(qc_basis):
    """Build a [0, 1] fold-level QC sensitivity prior from basis magnitudes."""
    basis = np.asarray(qc_basis, dtype=np.float32)
    if basis.ndim != 3:
        raise ValueError("qc_basis must have shape [qc_features, nodes, nodes]")
    sensitivity = np.abs(basis).mean(axis=0)
    diagonal = np.arange(sensitivity.shape[-1])
    sensitivity[diagonal, diagonal] = 0.0
    maximum = float(sensitivity.max())
    if maximum > 1e-8:
        sensitivity /= maximum
    return sensitivity.astype(np.float32)


def _limit_relative_change(base, proposal, maximum_ratio):
    """Bound each subject's Frobenius-norm change without altering direction."""
    if maximum_ratio <= 0.0:
        return np.asarray(base, dtype=np.float32).copy()
    base = np.asarray(base, dtype=np.float32)
    delta = np.asarray(proposal, dtype=np.float32) - base
    base_norm = np.linalg.norm(base.reshape(len(base), -1), axis=1)
    delta_norm = np.linalg.norm(delta.reshape(len(delta), -1), axis=1)
    scale = np.minimum(1.0, maximum_ratio * np.maximum(base_norm, 1e-6) / np.maximum(delta_norm, 1e-6))
    bounded = base + delta * scale[:, None, None]
    diagonal = np.arange(base.shape[-1])
    bounded[:, diagonal, diagonal] = 0.0
    return bounded.astype(np.float32)


def build_pseudo_target(bec, qc_badness, qc_basis, eta, maximum_ratio):
    """Create a conservative QC-guided pseudo-target for training subjects."""
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    component = np.tensordot(
        np.asarray(qc_badness, dtype=np.float32),
        np.asarray(qc_basis, dtype=np.float32),
        axes=(1, 0),
    )
    proposal = np.asarray(bec, dtype=np.float32) - float(eta) * component
    return _limit_relative_change(bec, proposal, maximum_ratio)


def sample_joint_qc_delta(train_qc_badness, rng):
    """Sample correlated QC changes from two real training-fold subjects."""
    values = np.asarray(train_qc_badness, dtype=np.float32)
    if len(values) < 2:
        return np.zeros_like(values)
    first = rng.integers(0, len(values), size=len(values))
    second = rng.integers(0, len(values), size=len(values))
    return values[first] - values[second]


def qc_corrupt(pseudo_bec, qc_basis, qc_delta, scale, maximum_ratio):
    """Add bounded, joint-distribution QC-like perturbations to pseudo-targets."""
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    component = np.tensordot(
        np.asarray(qc_delta, dtype=np.float32),
        np.asarray(qc_basis, dtype=np.float32),
        axes=(1, 0),
    )
    proposal = np.asarray(pseudo_bec, dtype=np.float32) + float(scale) * component
    return _limit_relative_change(pseudo_bec, proposal, maximum_ratio)


def relative_change(reference, updated):
    """Return the mean per-subject relative Frobenius change for auditing."""
    reference = np.asarray(reference, dtype=np.float32)
    updated = np.asarray(updated, dtype=np.float32)
    numerator = np.linalg.norm((updated - reference).reshape(len(reference), -1), axis=1)
    denominator = np.maximum(np.linalg.norm(reference.reshape(len(reference), -1), axis=1), 1e-6)
    return float(np.mean(numerator / denominator))


__all__ = [
    "DEFAULT_QC_COLUMNS",
    "load_aligned_qc",
    "fit_qc_scaler",
    "transform_qc_badness",
    "build_confound_design",
    "fit_qc_artifact_basis",
    "build_qc_sensitive_map",
    "build_pseudo_target",
    "sample_joint_qc_delta",
    "qc_corrupt",
    "relative_change",
]
