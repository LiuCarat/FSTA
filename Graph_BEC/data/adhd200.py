"""ADHD200 subject and time-series loading."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Graph_BEC.data.common import standardize_time_series, validate_time_series
from Graph_BEC.utils.folds import prepare_fold_arrays


def normalize_value(value):
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else str(number)


def normalize_row(row):
    return {
        str(key).strip(): normalize_value(value)
        for key, value in row.items()
        if key is not None
    }


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


@dataclass(frozen=True)
class ADHD200Record:
    subject_id: str
    site_id: str
    label: int
    diagnosis: str
    time_series_path: Path


def load_adhd200_records(data_root, profile, patient_label=1, control_label=0):
    data_root = Path(data_root)
    delimiter = "\t" if profile.phenotype_format == "tsv" else ","
    with Path(profile.phenotype_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = {}
        for raw_row in csv.DictReader(handle, delimiter=delimiter):
            row = normalize_row(raw_row)
            subject = normalize_value(row.get(profile.phenotype_id_column))
            if subject:
                rows[subject] = row

    flat_root = data_root / "cpac" / "filt_noglobal"
    flat_paths = sorted(flat_root.glob("*_rois_aal.1D")) if flat_root.is_dir() else []
    excluded = {str(subject_id).strip() for subject_id in profile.exclude_subjects}
    records = []
    for time_series_path in flat_paths:
        subject_id = time_series_path.name.removesuffix("_rois_aal.1D")
        subject_id = subject_id.removeprefix("sub-")
        subject_id = str(int(subject_id)) if subject_id.isdigit() else subject_id
        if subject_id in excluded:
            continue
        row = rows.get(subject_id)
        if row is None:
            continue
        diagnosis = normalize_value(row.get(profile.patient_column))
        if diagnosis in profile.patient_values:
            label = patient_label
        elif diagnosis in profile.control_values:
            label = control_label
        else:
            continue
        records.append(ADHD200Record(
            subject_id=subject_id,
            site_id=str(row.get(profile.site_column, "")).strip() or "unknown",
            label=label,
            diagnosis=diagnosis,
            time_series_path=time_series_path,
        ))
    if not records:
        diagnosis_values = sorted({row.get(profile.patient_column, "") for row in rows.values()})
        if diagnosis_values == [""]:
            raise ValueError(
                f"ADHD phenotype column {profile.patient_column!r} is empty in "
                f"{profile.phenotype_path}; restore the original ADHD200 diagnosis labels "
                "before running Graph_BEC"
            )
        raise FileNotFoundError(
            f"No ADHD200 ROI files matched phenotype records in {flat_root}"
        )
    return records


def load_adhd200_time_series(
    record,
    source_roi_count=116,
    roi_count=90,
    standardize=True,
):
    # ADHD200 is now represented by one selected run per subject, matching
    # the ABIDE loader. Do not concatenate multiple runs into one sequence.
    path = record.time_series_path
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
    combined = time_series[:, :roi_count]
    if standardize:
        combined = standardize_time_series(combined)
    combined = validate_time_series(combined, record.subject_id, roi_count)
    return combined
