"""Phenotype loading and diagnosis-free patient graph construction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from Graph_BEC.normative_bec import (
    apply_continuous_scaler,
    fit_continuous_scaler,
    normative_reference,
    reference_diagnostics,
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
    values = load_aligned_phenotypes(
        phenotype_csv, subject_ids, ["AGE_AT_SCAN", "SEX", "FIQ"]
    )
    continuous = values[:, [0, 2]].astype(np.float32)
    sex = np.where(np.isfinite(values[:, 1]), values[:, 1], -1).astype(str)[:, None]
    return {
        "continuous": continuous,
        "categorical_raw": sex,
        "site_ids": np.asarray(site_ids).astype(str),
    }


def prepare_phenotype_fold(train_cont, val_cont, test_cont, train_cat,
                           val_cat, test_cat):
    scaler = fit_continuous_scaler(train_cont)
    categories = {
        value: index
        for index, value in enumerate(sorted(set(np.asarray(train_cat).astype(str))))
    }
    unknown = len(categories)
    encode = lambda values: np.asarray([
        categories.get(value, unknown) for value in np.asarray(values).astype(str)
    ])
    return {
        "train_cont": apply_continuous_scaler(train_cont, scaler),
        "val_cont": apply_continuous_scaler(val_cont, scaler),
        "test_cont": apply_continuous_scaler(test_cont, scaler),
        "train_cat": encode(train_cat)[:, None],
        "val_cat": encode(val_cat)[:, None],
        "test_cat": encode(test_cat)[:, None],
        "scaler": scaler,
    }


def build_reference_deviation(train_bec, train_cont, train_cat, query_cont,
                              query_cat, k=20, bandwidth=1.0,
                              categorical_penalty=4.0,
                              continuous_weights=(1.0, 0.3), permute=False,
                              seed=2026):
    """Build a train-only, diagnosis-free phenotype reference deviation."""
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
    _, global_mean = normative_reference(train_bec, train_weights)
    query_mean, _ = normative_reference(train_bec, query_weights)
    return query_mean - global_mean, reference_diagnostics(query_weights)


def build_reference_bec(train_bec, train_cont, train_cat, query_cont, query_cat,
                        k=20, bandwidth=1.0, categorical_penalty=4.0,
                        continuous_weights=(1.0, 0.3), permute=False, seed=2026):
    """Return the phenotype-neighbor BEC itself, without any diagnosis label."""
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
    _, global_mean = normative_reference(train_bec, train_weights)
    query_mean, _ = normative_reference(train_bec, query_weights)
    return query_mean, global_mean, reference_diagnostics(query_weights)
