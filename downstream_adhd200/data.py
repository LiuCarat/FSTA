"""ADHD200 phenotype and ROI time-series loading utilities."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROI_COUNT = 116
ROI_COUNT = 90
ROI_INDICES = np.arange(ROI_COUNT, dtype=np.int64)
DX_TO_LABEL = {0: 0, 1: 1, 2: 1, 3: 1}
LABEL_TO_GROUP = {0: "HC", 1: "ADHD"}
TIME_SERIES_PATTERN = re.compile(
    r"^(?P<prefix>sf?nwmrda)(?P<subject_id>\d+)_session_(?P<session>\d+)_rest_(?P<rest>\d+)_aal_TCs\.1D$"
)


def _normalize_subject_id(subject_id):
    subject_id = str(subject_id).strip()
    if subject_id.isdigit():
        return str(int(subject_id))
    return subject_id


@dataclass(frozen=True)
class ADHDRecord:
    subject_id: str
    site_id: str
    label: int
    diagnosis: str
    dx_code: int
    time_series_path: Path


def _read_phenotype_rows(data_root):
    phenotype_root = Path(data_root) / "training_pheno"
    rows = {}
    for phenotype_path in sorted(phenotype_root.glob("*/*_phenotypic.csv")):
        site_id = phenotype_path.parent.name
        with phenotype_path.open(newline="", encoding="utf-8-sig") as phenotype_file:
            for row in csv.DictReader(phenotype_file):
                subject_column = "ScanDir ID" if "ScanDir ID" in row else "ScanDirID"
                subject_id = _normalize_subject_id(row[subject_column])
                if subject_id:
                    rows[subject_id] = (site_id, row)
    return rows


def load_adhd_records(
    data_root=REPO_ROOT / "dataset/ADHD200",
    file_prefix="sfnwmrda",
    session=1,
    rest=1,
):
    """Match ADHD200 preprocessed ROI files with phenotype rows."""
    data_root = Path(data_root)
    phenotype_rows = _read_phenotype_rows(data_root)
    time_series_root = data_root / "AAL_TCs_filtfix"
    records = []
    unmatched = []
    seen_subjects = set()

    for time_series_path in sorted(time_series_root.glob("*/*/*.1D")):
        match = TIME_SERIES_PATTERN.match(time_series_path.name)
        if match is None or match.group("prefix") != file_prefix:
            continue
        if int(match.group("session")) != session or int(match.group("rest")) != rest:
            continue
        subject_id = _normalize_subject_id(match.group("subject_id"))
        if subject_id in seen_subjects:
            continue
        phenotype = phenotype_rows.get(subject_id)
        if phenotype is None:
            unmatched.append(subject_id)
            continue
        site_id, row = phenotype
        raw_dx = row.get("DX", "").strip()
        try:
            dx_code = int(float(raw_dx))
        except ValueError as error:
            raise ValueError(f"Invalid ADHD200 DX={raw_dx!r} for subject {subject_id}") from error
        if dx_code not in DX_TO_LABEL:
            raise ValueError(f"Unsupported ADHD200 DX={dx_code} for subject {subject_id}")
        label = DX_TO_LABEL[dx_code]
        records.append(
            ADHDRecord(
                subject_id=subject_id,
                site_id=site_id,
                label=label,
                diagnosis=LABEL_TO_GROUP[label],
                dx_code=dx_code,
                time_series_path=time_series_path,
            )
        )
        seen_subjects.add(subject_id)

    if not records:
        raise FileNotFoundError(
            f"No ADHD200 time series matched under {time_series_root} "
            f"with prefix={file_prefix!r}, session={session}, rest={rest}"
        )
    return records


def load_time_series(record, standardize=True):
    """Load tab-separated 116-ROI AAL time series and retain 90 ROIs."""
    time_series = np.loadtxt(
        record.time_series_path,
        dtype=np.float32,
        delimiter="\t",
        skiprows=1,
        usecols=range(2, 2 + SOURCE_ROI_COUNT),
    )
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
        std = time_series.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        time_series = (time_series - mean) / std
    return time_series.astype(np.float32, copy=False)


def write_manifest(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "subject_id", "site_id", "diagnosis", "label", "dx_code",
                "time_points", "roi_count", "time_series_path",
            ]
        )
        for record in records:
            time_series = load_time_series(record, standardize=False)
            writer.writerow(
                [
                    record.subject_id,
                    record.site_id,
                    record.diagnosis,
                    record.label,
                    record.dx_code,
                    time_series.shape[0],
                    time_series.shape[1],
                    str(record.time_series_path),
                ]
            )


__all__ = [
    "ADHDRecord",
    "DX_TO_LABEL",
    "LABEL_TO_GROUP",
    "ROI_COUNT",
    "SOURCE_ROI_COUNT",
    "load_adhd_records",
    "load_time_series",
    "write_manifest",
]
