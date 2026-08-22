"""ADHD200-only fold-local phenotype preprocessing."""
from __future__ import annotations

import numpy as np

from Graph_BEC.utils.folds import prepare_fold_arrays


CATEGORICAL_COLUMNS = ("Gender",)


def fit_category_imputer(train_values):
    values = np.asarray(train_values).astype(str)
    if values.ndim == 1:
        values = values[:, None]
    modes = []
    missing_values = {"", "-1", "nan", "None", "__MISSING__"}
    for column in range(values.shape[1]):
        observed = [value for value in values[:, column] if value not in missing_values]
        modes.append(max(set(observed), key=observed.count) if observed else "__MISSING__")
    return np.asarray(modes, dtype=object)


def apply_category_imputer(values, modes):
    values = np.asarray(values).astype(str)
    if values.ndim == 1:
        values = values[:, None]
    output = values.copy()
    missing = np.isin(output, ["", "-1", "nan", "None", "__MISSING__"])
    for column, mode in enumerate(np.asarray(modes).tolist()):
        output[missing[:, column], column] = mode
    return output


def fit_numeric_imputer(train_values, categorical_indices=()):
    values = np.asarray(train_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    fills = np.nanmedian(values, axis=0)
    for column in categorical_indices:
        observed = values[np.isfinite(values[:, column]), column]
        if len(observed):
            unique, counts = np.unique(observed, return_counts=True)
            fills[column] = unique[np.argmax(counts)]
    fills[~np.isfinite(fills)] = 0.0
    return fills.astype(np.float32)


def apply_numeric_imputer(values, fills):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    return np.where(np.isfinite(values), values, np.asarray(fills, dtype=np.float32))


def prepare_adhd_fold_arrays(
    train_bec, val_bec, test_bec,
    train_cont, val_cont, test_cont,
    train_cat, val_cat, test_cat,
):
    category_fills = fit_category_imputer(train_cat)
    train_cat, val_cat, test_cat = (
        apply_category_imputer(values, category_fills)
        for values in (train_cat, val_cat, test_cat)
    )
    continuous_fills = fit_numeric_imputer(train_cont)
    train_cont, val_cont, test_cont = (
        apply_numeric_imputer(values, continuous_fills)
        for values in (train_cont, val_cont, test_cont)
    )
    arrays = prepare_fold_arrays(
        train_bec, val_bec, test_bec,
        train_cont, val_cont, test_cont,
        train_cat, val_cat, test_cat,
    )
    arrays["adhd_continuous_imputer"] = continuous_fills
    arrays["adhd_category_imputer"] = category_fills
    return arrays
