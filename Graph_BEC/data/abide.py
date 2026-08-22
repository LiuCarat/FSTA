"""ABIDE-I subject and time-series loading."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Graph_BEC.data.common import (
    ROI_COUNT,
    ROI_INDICES,
    SOURCE_ROI_COUNT,
    standardize_time_series,
    validate_time_series,
)

TC_LABEL = 0
ASD_LABEL = 1
DX_TO_LABEL = {1: TC_LABEL, 2: ASD_LABEL}
LABEL_TO_GROUP = {TC_LABEL: "TC", ASD_LABEL: "ASD"}


@dataclass(frozen=True)
class ABIDERecord:
    subject_id: str
    site_id: str
    label: int
    diagnosis: str
    time_series_path: Path


def load_abide_records(data_root, pipeline="cpac", strategy="filt_noglobal", derivative="rois_aal"):
    data_root = Path(data_root)
    phenotype_candidates = (
        data_root / "ABIDEII_phenotype_graphbec.csv",
        data_root / "Phenotypic_V1_0b_preprocessed1.csv",
        data_root / "Phenotypic_Processing_filled.csv",
    )
    phenotype_path = next((path for path in phenotype_candidates if path.is_file()), None)
    if phenotype_path is None:
        raise FileNotFoundError(
            "No ABIDE phenotype CSV found; tried "
            + ", ".join(str(path) for path in phenotype_candidates)
        )
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
        records.append(ABIDERecord(
            subject_id=subject_id,
            site_id=row["SITE_ID"].strip(),
            label=label,
            diagnosis=LABEL_TO_GROUP[label],
            time_series_path=time_series_path,
        ))
    if not records:
        raise FileNotFoundError(
            f"No ABIDE ROI files matched phenotype records in {time_series_dir}"
        )
    return records


def load_abide_time_series(record, standardize=True):
    time_series = np.loadtxt(record.time_series_path, dtype=np.float32)
    if time_series.ndim != 2 or time_series.shape[1] not in {ROI_COUNT, SOURCE_ROI_COUNT}:
        raise ValueError(
            f"Expected {ROI_COUNT} or {SOURCE_ROI_COUNT} ROI columns for {record.subject_id}, "
            f"got {time_series.shape}"
        )
    if not np.isfinite(time_series).all():
        raise ValueError(f"Non-finite values found for {record.subject_id}")
    if time_series.shape[1] == SOURCE_ROI_COUNT:
        time_series = time_series[:, ROI_INDICES]
    if standardize:
        time_series = standardize_time_series(time_series)
    return validate_time_series(time_series, record.subject_id, ROI_COUNT)
