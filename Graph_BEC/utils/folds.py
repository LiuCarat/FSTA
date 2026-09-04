"""Fold-safe preprocessing and stratified split helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from Graph_BEC.model.patient_graph.reference_bec import apply_continuous_scaler, fit_continuous_scaler


def fit_bec_scaler(train_bec):
    mean = np.asarray(train_bec).mean(axis=0)
    std = np.asarray(train_bec).std(axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def transform_bec(bec, mean, std):
    scaled = ((np.asarray(bec) - mean) / std).astype(np.float32)
    diagonal = np.arange(scaled.shape[-1])
    scaled[:, diagonal, diagonal] = 0.0
    return scaled


def _encode_category_splits(train_cat, val_cat, test_cat):
    train_cat = np.asarray(train_cat)
    val_cat = np.asarray(val_cat)
    test_cat = np.asarray(test_cat)
    if train_cat.ndim == 1:
        train_cat = train_cat[:, None]
        val_cat = val_cat[:, None]
        test_cat = test_cat[:, None]
    encoded_train, encoded_val, encoded_test = [], [], []
    for column in range(train_cat.shape[1]):
        train_values = train_cat[:, column].astype(str)
        categories = {
            value: index for index, value in enumerate(sorted(set(train_values)))
        }
        unknown = len(categories)

        def encode(values):
            return np.asarray(
                [categories.get(value, unknown) for value in values.astype(str)]
            )

        encoded_train.append(encode(train_cat[:, column]))
        encoded_val.append(encode(val_cat[:, column]))
        encoded_test.append(encode(test_cat[:, column]))
    return (
        np.stack(encoded_train, axis=1).astype(np.int64),
        np.stack(encoded_val, axis=1).astype(np.int64),
        np.stack(encoded_test, axis=1).astype(np.int64),
    )


def prepare_fold_arrays(
    train_bec,
    val_bec,
    test_bec,
    train_cont,
    val_cont,
    test_cont,
    train_cat,
    val_cat,
    test_cat,
):
    bec_mean, bec_std = fit_bec_scaler(train_bec)
    continuous_scaler = fit_continuous_scaler(train_cont)
    encoded_train, encoded_val, encoded_test = _encode_category_splits(
        train_cat, val_cat, test_cat
    )
    return {
        "train_bec": transform_bec(train_bec, bec_mean, bec_std),
        "val_bec": transform_bec(val_bec, bec_mean, bec_std),
        "test_bec": transform_bec(test_bec, bec_mean, bec_std),
        "train_cont": apply_continuous_scaler(train_cont, continuous_scaler),
        "val_cont": apply_continuous_scaler(val_cont, continuous_scaler),
        "test_cont": apply_continuous_scaler(test_cont, continuous_scaler),
        "train_cat": encoded_train,
        "val_cat": encoded_val,
        "test_cat": encoded_test,
        "bec_mean": bec_mean,
        "bec_std": bec_std,
        "continuous_scaler": continuous_scaler,
    }


def make_stratified_splits(labels, n_splits=5, seed=2026, validation_size=0.2):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_pool, test_index) in enumerate(
        splitter.split(labels, labels), 1
    ):
        train_index, val_index = train_test_split(
            train_pool,
            test_size=validation_size,
            stratify=labels[train_pool],
            random_state=seed + fold,
        )
        yield fold, train_index, val_index, test_index
