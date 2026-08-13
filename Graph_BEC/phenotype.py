"""Phenotype loading, fMRI FC features, and multi-view patient graph construction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from Graph_BEC.normative_bec import (
    reference_weights,
)

MISSING_SENTINEL = -9000.0


def _parse_numeric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number) or number <= MISSING_SENTINEL:
        return np.nan
    return number


def _encode_sex(value):
    number = _parse_numeric(value)
    if number == 1:
        return 1.0
    if number == 2:
        return 0.0
    return np.nan


def load_aligned_phenotypes(csv_path, subject_ids, columns):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"FILE_ID", *columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing phenotype columns: {sorted(missing)}")
        rows = {
            row["FILE_ID"].strip(): row
            for row in reader
            if row.get("FILE_ID", "").strip()
            and row["FILE_ID"].strip() != "no_filename"
        }
    values = np.full((len(subject_ids), len(columns)), np.nan, dtype=np.float64)
    for subject_index, subject_id in enumerate(subject_ids):
        row = rows.get(str(subject_id))
        if row is None:
            raise ValueError(f"Could not match phenotype subject: {subject_id}")
        for column_index, column in enumerate(columns):
            parser = _encode_sex if column.upper() == "SEX" else _parse_numeric
            values[subject_index, column_index] = parser(row.get(column))
    return values


def load_phenotypes(phenotype_csv, subject_ids, site_ids):
    """Load SEX as categorical and FIQ/PIQ as continuous graph features."""
    values = load_aligned_phenotypes(
        phenotype_csv, subject_ids, ["SEX", "FIQ", "PIQ"]
    )
    sex = np.where(np.isfinite(values[:, 0]), values[:, 0], -1).astype(str)[:, None]
    continuous = values[:, [1, 2]].astype(np.float32)
    return {
        "continuous": continuous,
        "categorical_raw": sex,
        "site_ids": np.asarray(site_ids).astype(str),
    }


def build_reference_graph(train_cont, train_cat, query_cont, query_cat,
                           k=20, bandwidth=1.0, categorical_penalty=4.0,
                           continuous_weights=(1.0, 0.3), permute=False, seed=2026):
    """Return train/query phenotype graph weights for multi-view fusion."""
    train_cont = np.asarray(train_cont).copy()
    train_cat = np.asarray(train_cat).copy()
    if permute:
        order = np.random.default_rng(seed).permutation(len(train_cont))
        train_cont, train_cat = train_cont[order], train_cat[order]
    common = {
        "reference_continuous": train_cont,
        "reference_categorical": train_cat,
        "k": k,
        "bandwidth": bandwidth,
        "categorical_penalty": categorical_penalty,
        "continuous_weights": np.asarray(continuous_weights),
    }
    self_indices = np.arange(len(train_cont), dtype=np.int64)
    train_weights = reference_weights(
        train_cont, train_cat, self_indices=self_indices, **common
    )
    query_weights = reference_weights(query_cont, query_cat, **common)
    return train_weights, query_weights


# ---------------------------------------------------------------------------
#  fMRI 功能连接特征 & 多视图图融合  (was multiview_validation/validate_graph)
# ---------------------------------------------------------------------------

def subject_fc_features(time_series):
    """Convert each [T, 90] ROI series into its 4005-D FC upper triangle."""
    features = []
    upper = np.triu_indices(90, k=1)
    for index, series in enumerate(time_series, 1):
        values = np.asarray(series, dtype=np.float64)
        centered = values - values.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = standard_deviation[:, None] * standard_deviation[None, :]
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 1e-12,
        )
        np.fill_diagonal(correlation, 1.0)
        features.append(correlation[upper])
        if index == 1 or index == len(time_series) or index % 100 == 0:
            print(f"fMRI features [{index}/{len(time_series)}]")
    return np.asarray(features, dtype=np.float32)


def topk_graph(reference, query, neighbors, exclude_self=False):
    """Build row-normalized cosine top-k query-to-reference weights."""
    reference = np.asarray(reference, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    reference = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), 1e-8
    )
    query = query / np.maximum(
        np.linalg.norm(query, axis=1, keepdims=True), 1e-8
    )
    similarity = np.clip(query @ reference.T, 0.0, 1.0)
    if exclude_self and len(reference) == len(query):
        similarity[np.arange(len(query)), np.arange(len(reference))] = -np.inf
    return topk_row_normalize(similarity, neighbors, exclude_self=exclude_self)


def topk_row_normalize(scores, neighbors, exclude_self=False):
    """Select the largest scores in each row and row-normalize them."""
    scores = np.asarray(scores, dtype=np.float32).copy()
    if scores.ndim != 2:
        raise ValueError(f"scores must be [queries, references], got {scores.shape}")
    if exclude_self and len(scores) == scores.shape[1]:
        scores[np.arange(len(scores)), np.arange(scores.shape[1])] = -np.inf
    count = min(max(1, int(neighbors)), scores.shape[1] - int(exclude_self))
    output = np.zeros_like(scores, dtype=np.float32)
    for row in range(len(scores)):
        indices = np.argpartition(scores[row], -count)[-count:]
        values = np.maximum(scores[row, indices], 0.0)
        if values.sum() <= 1e-8:
            values = np.ones_like(values)
        output[row, indices] = values / values.sum()
    return output


def fused_graph(
    fmri_graph,
    phenotype_graph,
    beta,
    neighbors,
):
    """Fuse two row-normalized graphs and retain a common top-k size."""
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"fusion beta must be in [0, 1], got {beta}")
    fmri_graph = np.asarray(fmri_graph, dtype=np.float32)
    phenotype_graph = np.asarray(phenotype_graph, dtype=np.float32)
    if fmri_graph.shape != phenotype_graph.shape:
        raise ValueError(
            f"Graph shapes must match, got {fmri_graph.shape} and {phenotype_graph.shape}"
        )
    scores = float(beta) * fmri_graph + (1.0 - float(beta)) * phenotype_graph
    return topk_row_normalize(scores, neighbors)
