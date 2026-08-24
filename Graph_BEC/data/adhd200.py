"""ADHD200 subject and time-series loading."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Graph_BEC.data.common import standardize_time_series, validate_time_series


@dataclass(frozen=True)
class ADHD200Record:
    subject_id: str
    site_id: str
    label: int
    diagnosis: str
    time_series_path: Path
    time_series_paths: tuple[Path, ...]


def load_adhd200_records(data_root, profile, patient_label=1, control_label=0):
    data_root = Path(data_root)
    with Path(profile.phenotype_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = {
            str(row[profile.phenotype_id_column]).strip(): row
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get(profile.phenotype_id_column, "")).strip()
        }
    records = []
    cleaned_root = data_root / "cleaned" / "AAL_TCs_filtfix"
    series_root = cleaned_root if cleaned_root.is_dir() else data_root / "AAL_TCs_filtfix"
    excluded = {
        f"{subject_id.split('/', 1)[0]}/{str(int(subject_id.split('/', 1)[1]))}"
        if "/" in subject_id and subject_id.split("/", 1)[1].isdigit()
        else subject_id
        for subject_id in profile.exclude_subjects
    }
    for site_dir in sorted(path for path in series_root.iterdir() if path.is_dir()):
        for subject_dir in sorted(path for path in site_dir.iterdir() if path.is_dir()):
            subject_id = f"{site_dir.name}/{subject_dir.name}"
            if subject_id in excluded:
                continue
            normalized_id = subject_dir.name
            if normalized_id.isdigit():
                normalized_id = str(int(normalized_id))
            row = rows.get(normalized_id)
            if row is None:
                continue
            diagnosis = str(row.get(profile.patient_column, "")).strip()
            if diagnosis in profile.patient_values:
                label = patient_label
            elif diagnosis in profile.control_values:
                label = control_label
            else:
                continue
            candidates = sorted(
                path for path in subject_dir.glob("*_aal_TCs.1D")
                if "*" not in path.name and path.stat().st_size > 0
            )
            if not candidates:
                continue
            preferred = [path for path in candidates if path.name.startswith("sfnwmrda")]
            selected_paths = tuple(preferred or candidates)
            records.append(ADHD200Record(
                subject_id=subject_id,
                site_id=site_dir.name,
                label=label,
                diagnosis=diagnosis,
                time_series_path=selected_paths[0],
                time_series_paths=selected_paths,
            ))
    if not records:
        raise FileNotFoundError(
            f"No ADHD200 ROI files matched phenotype records in {series_root}"
        )
    return records


def load_adhd200_time_series(
    record,
    source_roi_count=116,
    roi_count=90,
    standardize=True,
    return_run_ranges=False,
):
    runs = []
    run_ranges = []
    offset = 0
    for path in record.time_series_paths:
        time_series = np.loadtxt(
            path,
            dtype=np.float32,
            skiprows=1,
            usecols=np.arange(2, 2 + source_roi_count),
        )
        if time_series.ndim != 2 or time_series.shape[1] != source_roi_count:
            raise ValueError(
                f"Expected [{record.subject_id}] data with "
                f"{source_roi_count} ROI columns in {path.name}, got {time_series.shape}"
            )
        if not np.isfinite(time_series).all():
            raise ValueError(f"Non-finite values found for {record.subject_id}: {path.name}")
        time_series = time_series[:, :roi_count]
        if standardize:
            time_series = standardize_time_series(time_series)
        time_series = validate_time_series(time_series, record.subject_id, roi_count)
        runs.append(time_series)
        run_ranges.append((offset, offset + len(time_series)))
        offset += len(time_series)

    combined = validate_time_series(
        np.concatenate(runs, axis=0), record.subject_id, roi_count
    )
    if return_run_ranges:
        return combined, tuple(run_ranges)
    return combined
