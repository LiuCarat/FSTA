"""Data loading, windowing, and runtime helpers for DTS-EC."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROI_COUNT = 90
SOURCE_ROI_COUNT = 116
DX_TO_LABEL = {1: 0, 2: 1}


@dataclass(frozen=True)
class ABIDERecord:
    subject_id: str
    site_id: str
    label: int
    time_series_path: Path


def load_subject_dataset(
    data_root: Path,
    pipeline: str = "cpac",
    strategy: str = "filt_noglobal",
    derivative: str = "rois_aal",
) -> dict:
    """Load standardized first-90 AAL ROI series for matched ABIDE-I subjects."""
    data_root = Path(data_root)
    phenotype_path = _phenotype_path(data_root)
    phenotype_rows = _read_phenotypes(phenotype_path)
    time_series_dir = data_root / pipeline / strategy
    suffix = f"_{derivative}.1D"
    records = [
        _record_from_path(path, suffix, phenotype_rows)
        for path in sorted(time_series_dir.glob(f"*{suffix}"))
    ]
    records = [record for record in records if record is not None]
    if not records:
        raise FileNotFoundError(f"No matched ABIDE ROI files found in {time_series_dir}")

    return {
        "time_series": [load_time_series(record) for record in records],
        "labels": np.asarray([record.label for record in records], dtype=np.int64),
        "subject_ids": np.asarray([record.subject_id for record in records]),
        "site_ids": np.asarray([record.site_id for record in records]),
    }


def _phenotype_path(data_root: Path) -> Path:
    for filename in (
        "Phenotypic_V1_0b_preprocessed1.csv",
        "Phenotypic_Processing_filled.csv",
    ):
        path = data_root / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"ABIDE phenotype file not found in {data_root}")


def _read_phenotypes(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["FILE_ID"].strip(): row
            for row in csv.DictReader(handle)
            if row.get("FILE_ID", "").strip() != "no_filename"
        }


def _record_from_path(path: Path, suffix: str, phenotype_rows: dict) -> ABIDERecord | None:
    subject_id = path.name[: -len(suffix)]
    phenotype = phenotype_rows.get(subject_id)
    if phenotype is None:
        return None
    diagnosis_code = int(float(phenotype["DX_GROUP"]))
    if diagnosis_code not in DX_TO_LABEL:
        return None
    return ABIDERecord(
        subject_id=subject_id,
        site_id=phenotype["SITE_ID"].strip(),
        label=DX_TO_LABEL[diagnosis_code],
        time_series_path=path,
    )


def load_time_series(record: ABIDERecord) -> np.ndarray:
    """Load one ROI series, retain 90 regions, and standardize each region."""
    series = np.loadtxt(record.time_series_path, dtype=np.float32)
    if series.ndim != 2 or series.shape[1] != SOURCE_ROI_COUNT:
        raise ValueError(
            f"Expected [time, {SOURCE_ROI_COUNT}] data for {record.subject_id}, "
            f"got {series.shape}"
        )
    if not np.isfinite(series).all():
        raise ValueError(f"Non-finite values found for {record.subject_id}")
    series = series[:, :ROI_COUNT]
    standard_deviation = series.std(axis=0, keepdims=True)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return ((series - series.mean(axis=0, keepdims=True)) / standard_deviation).astype(
        np.float32, copy=False
    )


class RandomSubjectWindowDataset(Dataset):
    """Draw one reproducible random window per subject and epoch."""

    def __init__(self, series: list[np.ndarray], window_length: int, seed: int):
        self.series = series
        self.window_length = window_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> torch.Tensor:
        series = self.series[index]
        start = np.random.default_rng(
            self.seed + self.epoch * len(self.series) + index
        ).integers(series.shape[0] - self.window_length + 1)
        return torch.from_numpy(series[start : start + self.window_length]).float()


def pad_series(series: np.ndarray, window_length: int) -> np.ndarray:
    """Pad a short subject by repeating its last time point."""
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 2 or series.shape[0] == 0:
        raise ValueError("Each subject must have a non-empty [time, roi] array")
    if series.shape[0] >= window_length:
        return series
    padding = np.repeat(series[-1:], window_length - series.shape[0], axis=0)
    return np.concatenate((series, padding), axis=0)


def fixed_window_starts(time_points: int, window_length: int, stride: int) -> list[int]:
    starts = list(range(0, time_points - window_length + 1, stride))
    final_start = time_points - window_length
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_window_loader(
    series: list[np.ndarray], window_length: int, batch_size: int, seed: int, epoch: int
) -> DataLoader:
    dataset = RandomSubjectWindowDataset(
        [pad_series(item, window_length) for item in series], window_length, seed
    )
    dataset.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + epoch),
        num_workers=0,
    )


def split_series(
    series: list[np.ndarray], validation_size: float, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(series) < 2:
        raise ValueError("At least two subjects are required for a train/validation split")
    validation_count = min(
        max(1, round(len(series) * validation_size)), len(series) - 1
    )
    validation_indices = set(
        np.random.default_rng(seed).choice(
            len(series), size=validation_count, replace=False
        )
    )
    return (
        [item for index, item in enumerate(series) if index not in validation_indices],
        [item for index, item in enumerate(series) if index in validation_indices],
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(gpu_id: str) -> torch.device:
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == "auto":
        gpu_id = str(
            max(
                range(torch.cuda.device_count()),
                key=lambda index: torch.cuda.mem_get_info(index)[0],
            )
        )
    return torch.device(f"cuda:{int(gpu_id)}")
