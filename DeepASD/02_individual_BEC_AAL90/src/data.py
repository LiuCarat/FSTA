from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def parse_bool_int(value) -> bool:
    return bool(int(value))


def parse_columns(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def build_manifest(
    roi_root: str | Path,
    phenotypic_csv: str | Path,
    atlas: str = "aal",
) -> tuple[pd.DataFrame, dict]:
    """Match FILE_ID with *_rois_aal.1D files."""
    roi_root = Path(roi_root)
    phenotype = pd.read_csv(phenotypic_csv, low_memory=False)
    phenotype = phenotype.loc[
        :,
        ~phenotype.columns.astype(str).str.startswith("Unnamed:"),
    ]

    for column in ["FILE_ID", "DX_GROUP"]:
        if column not in phenotype.columns:
            raise ValueError(f"Missing column: {column}")

    suffix = f"_rois_{atlas.lower()}.1D"
    roi_map = {}
    for path in roi_root.rglob(f"*{suffix}"):
        subject_id = path.name[: -len(suffix)]
        roi_map.setdefault(subject_id.lower(), str(path.resolve()))

    rows = []
    unmatched = []
    for _, row in phenotype.iterrows():
        file_id = str(row.get("FILE_ID", "")).strip()
        if not file_id or file_id.lower() == "no_filename":
            continue

        roi_path = roi_map.get(file_id.lower())
        if roi_path is None:
            unmatched.append(file_id)
            continue

        dx = pd.to_numeric(row.get("DX_GROUP"), errors="coerce")
        if dx == 1:
            label = 1
        elif dx == 2:
            label = 0
        else:
            continue

        item = row.to_dict()
        item.update(
            {
                "subject_id": file_id,
                "roi_path": roi_path,
                "label": label,
                "site_id": str(row.get("SITE_ID", "")).strip(),
            }
        )
        rows.append(item)

    manifest = pd.DataFrame(rows).drop_duplicates("subject_id")
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No subject was matched.")

    report = {
        "roi_files_found": len(roi_map),
        "phenotypic_rows": len(phenotype),
        "matched_subjects": len(manifest),
        "unmatched_file_ids": unmatched,
    }
    return manifest, report


def _looks_like_index(values: np.ndarray) -> bool:
    if values.ndim != 1 or len(values) < 3:
        return False
    if not np.isfinite(values).all():
        return False
    diff = np.diff(values)
    return bool(np.all(diff > 0) and np.std(diff) < 1e-5)


def read_roi_1d(
    path: str | Path,
    num_rois: int = 90,
    allow_aal116_to_aal90: bool = True,
) -> np.ndarray:
    """Return one subject as [ROI, time]."""
    data = np.loadtxt(path, comments="#", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    candidates = {num_rois}
    if num_rois == 90 and allow_aal116_to_aal90:
        candidates.add(116)

    # Remove an optional time/index column.
    if data.shape[1] - 1 in candidates and _looks_like_index(data[:, 0]):
        data = data[:, 1:]
    elif data.shape[0] - 1 in candidates and _looks_like_index(data[0, :]):
        data = data[1:, :]

    if data.shape[1] in candidates:
        roi_by_time = data.T
    elif data.shape[0] in candidates:
        roi_by_time = data
    else:
        raise ValueError(
            f"{path}: expected 90 or 116 ROI columns, got {data.shape}"
        )

    # Standard AAL116: first 90 cerebral regions, last 26 cerebellar/vermis.
    if roi_by_time.shape[0] == 116 and num_rois == 90:
        if not allow_aal116_to_aal90:
            raise ValueError("AAL116 to AAL90 conversion is disabled.")
        roi_by_time = roi_by_time[:90]

    if roi_by_time.shape[0] != num_rois:
        raise ValueError(
            f"{path}: expected {num_rois} ROIs, got {roi_by_time.shape[0]}"
        )

    # Per-subject, per-ROI z-score.
    roi_by_time = np.nan_to_num(roi_by_time, nan=0.0)
    mean = roi_by_time.mean(axis=1, keepdims=True)
    std = roi_by_time.std(axis=1, keepdims=True)
    return ((roi_by_time - mean) / np.maximum(std, 1e-6)).astype(
        np.float32
    )


@dataclass
class ColumnStats:
    median: float
    mean: float
    std: float
    missing_fraction: float


class NumericPreprocessor:
    """Fold-safe median imputation and standardization."""

    def __init__(self, columns: Sequence[str], name: str):
        self.columns = list(columns)
        self.name = name
        self.stats: dict[str, ColumnStats] = {}

    @staticmethod
    def _column(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            raise ValueError(f"Column not found: {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        return values.mask(values <= -9000).astype(float)

    def fit(self, frame: pd.DataFrame) -> "NumericPreprocessor":
        for column in self.columns:
            values = self._column(frame, column)
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            std = float(filled.std(ddof=0))
            self.stats[column] = ColumnStats(
                median=median,
                mean=float(filled.mean()),
                std=std if np.isfinite(std) and std >= 1e-6 else 1.0,
                missing_fraction=float(values.isna().mean()),
            )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        output = []
        for column in self.columns:
            stats = self.stats[column]
            values = self._column(frame, column).fillna(stats.median)
            values = (values.to_numpy() - stats.mean) / stats.std
            output.append(values[:, None])
        return np.concatenate(output, axis=1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        content = {
            "name": self.name,
            "columns": self.columns,
            "stats": {
                key: vars(value)
                for key, value in self.stats.items()
            },
        }
        Path(path).write_text(
            json.dumps(content, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def crop_or_pad(
    timeseries: np.ndarray,
    window_length: int,
    start: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    num_rois, total_time = timeseries.shape

    if total_time >= window_length:
        if start is None:
            start = (total_time - window_length) // 2
        start = int(np.clip(start, 0, total_time - window_length))
        return (
            timeseries[:, start : start + window_length],
            np.ones(window_length, dtype=np.float32),
        )

    window = np.zeros((num_rois, window_length), dtype=np.float32)
    mask = np.zeros(window_length, dtype=np.float32)
    window[:, :total_time] = timeseries
    mask[:total_time] = 1.0
    return window, mask


class ABIDEWindowDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: Sequence[int],
        phenotype_array: np.ndarray,
        qc_array: np.ndarray,
        num_rois: int,
        window_length: int,
        training: bool,
        allow_aal116_to_aal90: bool,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.phenotype_array = phenotype_array
        self.qc_array = qc_array
        self.num_rois = num_rois
        self.window_length = window_length
        self.training = training
        self.allow_aal116_to_aal90 = allow_aal116_to_aal90

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = int(self.indices[position])
        row = self.manifest.iloc[index]
        timeseries = read_roi_1d(
            row["roi_path"],
            self.num_rois,
            self.allow_aal116_to_aal90,
        )

        start = None
        if self.training and timeseries.shape[1] > self.window_length:
            start = np.random.randint(
                0,
                timeseries.shape[1] - self.window_length + 1,
            )

        window, mask = crop_or_pad(
            timeseries,
            self.window_length,
            start,
        )
        return {
            "timeseries": torch.from_numpy(window),
            "time_mask": torch.from_numpy(mask),
            "phenotype": torch.from_numpy(
                self.phenotype_array[index]
            ),
            "qc": torch.from_numpy(self.qc_array[index]),
        }


def evaluation_windows(
    timeseries: np.ndarray,
    window_length: int,
    num_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if timeseries.shape[1] <= window_length or num_windows <= 1:
        window, mask = crop_or_pad(timeseries, window_length)
        return window[None], mask[None]

    starts = np.linspace(
        0,
        timeseries.shape[1] - window_length,
        num_windows,
    ).round().astype(int)

    windows, masks = zip(
        *(crop_or_pad(timeseries, window_length, int(start))
          for start in starts)
    )
    return (
        np.stack(windows).astype(np.float32),
        np.stack(masks).astype(np.float32),
    )
