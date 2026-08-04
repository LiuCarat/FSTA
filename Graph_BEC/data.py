"""Dataset loading, fold preprocessing, deterministic seeds, and splits."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from downstream_abide_i.data import load_abide_records, load_time_series
from Graph_BEC.normative_bec import apply_continuous_scaler, fit_continuous_scaler


def set_seed(seed: int) -> None:
    """Reset every RNG used by one independently comparable stage."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def select_device(gpu_id):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == "auto":
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        gpu_id = int(np.argmax(free))
    return torch.device(f"cuda:{int(gpu_id)}")


def load_subject_dataset(data_root, pipeline="cpac", strategy="filt_global",
                         derivative="rois_aal", standardize=True, max_subjects=None):
    records = load_abide_records(data_root=data_root, pipeline=pipeline,
                                 strategy=strategy, derivative=derivative)
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


class RandomSubjectWindowDataset(Dataset):
    """Draw one deterministic window per subject for each FSTA epoch."""

    def __init__(self, time_series, window_length, seed):
        self.time_series = time_series
        self.window_length = window_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, index):
        series = self.time_series[index]
        maximum_start = series.shape[0] - self.window_length
        if maximum_start < 0:
            raise ValueError(f"Subject {index} is shorter than window_length")
        generator = np.random.default_rng(self.seed + self.epoch * len(self.time_series) + index)
        start = int(generator.integers(maximum_start + 1))
        return torch.from_numpy(series[start:start + self.window_length]).float()


def fixed_window_starts(time_points, window_length, stride):
    if time_points < window_length:
        raise ValueError("time series is shorter than window_length")
    starts = list(range(0, time_points - window_length + 1, stride))
    final_start = time_points - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


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
        categories = {value: index for index, value in enumerate(sorted(set(train_values)))}
        unknown = len(categories)
        encode = lambda values: np.asarray([
            categories.get(value, unknown) for value in values.astype(str)
        ])
        encoded_train.append(encode(train_cat[:, column]))
        encoded_val.append(encode(val_cat[:, column]))
        encoded_test.append(encode(test_cat[:, column]))
    return (
        np.stack(encoded_train, axis=1).astype(np.int64),
        np.stack(encoded_val, axis=1).astype(np.int64),
        np.stack(encoded_test, axis=1).astype(np.int64),
    )


def prepare_fold_arrays(train_bec, val_bec, test_bec, train_cont, val_cont,
                        test_cont, train_cat, val_cat, test_cat):
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
    for fold, (train_pool, test_index) in enumerate(splitter.split(labels, labels), 1):
        train_index, val_index = train_test_split(
            train_pool,
            test_size=validation_size,
            stratify=labels[train_pool],
            random_state=seed + fold,
        )
        yield fold, train_index, val_index, test_index
