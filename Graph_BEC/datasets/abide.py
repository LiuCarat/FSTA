"""Local ABIDE-I data loader used by Graph_BEC."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DX_TO_LABEL = {1: 0, 2: 1}
LABEL_TO_GROUP = {0: "HC", 1: "ASD"}
SOURCE_ROI_COUNT = 116
ROI_COUNT = 90
ROI_INDICES = np.arange(ROI_COUNT, dtype=np.int64)


@dataclass(frozen=True)
class ABIDERecord:
    subject_id: str
    site_id: str
    label: int
    diagnosis: str
    time_series_path: Path


def load_abide_records(
    data_root,
    pipeline="cpac",
    strategy="filt_noglobal",
    derivative="rois_aal",
):
    """Match ABIDE ROI files with the local phenotype table."""
    data_root = Path(data_root)
    phenotype_path = data_root / "Phenotypic_V1_0b_preprocessed1.csv"
    if not phenotype_path.is_file():
        phenotype_path = data_root / "Phenotypic_Processing_filled.csv"
    time_series_dir = data_root / pipeline / strategy
    suffix = f"_{derivative}.1D"

    with phenotype_path.open(newline="", encoding="utf-8-sig") as handle:
        phenotype_rows = {
            row["FILE_ID"].strip(): row
            for row in csv.DictReader(handle)
            if row.get("FILE_ID", "").strip() != "no_filename"
        }

    records = []
    for time_series_path in sorted(time_series_dir.glob(f"*{suffix}")):
        subject_id = time_series_path.name[: -len(suffix)]
        row = phenotype_rows.get(subject_id)
        if row is None:
            continue
        diagnosis_code = int(float(row["DX_GROUP"]))
        if diagnosis_code not in DX_TO_LABEL:
            continue
        label = DX_TO_LABEL[diagnosis_code]
        records.append(
            ABIDERecord(
                subject_id=subject_id,
                site_id=row["SITE_ID"].strip(),
                label=label,
                diagnosis=LABEL_TO_GROUP[label],
                time_series_path=time_series_path,
            )
        )

    if not records:
        raise FileNotFoundError(
            f"No ABIDE ROI files matched phenotype records in {time_series_dir}"
        )
    return records


def load_time_series(record, standardize=True):
    """Load 116 AAL columns and retain the first 90 ROIs."""
    time_series = np.loadtxt(record.time_series_path, dtype=np.float32)
    if time_series.ndim != 2:
        raise ValueError(
            f"Expected a 2D time series for {record.subject_id}, got {time_series.shape}"
        )
    if time_series.shape[1] != SOURCE_ROI_COUNT:
        raise ValueError(
            f"Expected {SOURCE_ROI_COUNT} source columns for {record.subject_id}, "
            f"got {time_series.shape[1]}"
        )
    if not np.isfinite(time_series).all():
        raise ValueError(f"Non-finite values found for {record.subject_id}")

    time_series = time_series[:, ROI_INDICES]
    if standardize:
        mean = time_series.mean(axis=0, keepdims=True)
        standard_deviation = time_series.std(axis=0, keepdims=True)
        standard_deviation[standard_deviation < 1e-6] = 1.0
        time_series = (time_series - mean) / standard_deviation
    return time_series.astype(np.float32, copy=False)

