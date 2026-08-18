"""Dataset loading and experiment-input assembly for Graph_BEC."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Graph_BEC.phenotype import (
    load_aligned_phenotypes,
    load_phenotypes,
    subject_fc_features,
)
from Graph_BEC.qc import load_aligned_qc


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


def load_subject_dataset(
    data_root,
    pipeline="cpac",
    strategy="filt_global",
    derivative="rois_aal",
    standardize=True,
    max_subjects=None,
):
    records = load_abide_records(
        data_root=data_root,
        pipeline=pipeline,
        strategy=strategy,
        derivative=derivative,
    )
    if max_subjects is not None:
        records = records[:max_subjects]
    time_series = [load_time_series(record, standardize=standardize) for record in records]
    if any(series.ndim != 2 or series.shape[1] != ROI_COUNT for series in time_series):
        raise ValueError(f"Every subject must provide a [T, {ROI_COUNT}] ROI time series")
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


FIXED_DATA_CONFIG = {
    "pipeline": "cpac",
    "strategy": "filt_noglobal",
    "derivative": "rois_aal",
    "standardize": True,
    "max_subjects": None,
}


def load_pipeline_data(args, device):
    """Load or generate BEC matrices and attach graph/QC covariates."""
    from Graph_BEC.baseline.FSTA_EC import generate_subject_bec, save_subject_bec

    fsta_metrics = None
    subjects = None
    if args.input_mode == "raw":
        subjects = load_subject_dataset(
            args.data_root,
            FIXED_DATA_CONFIG["pipeline"],
            FIXED_DATA_CONFIG["strategy"],
            FIXED_DATA_CONFIG["derivative"],
            FIXED_DATA_CONFIG["standardize"],
            FIXED_DATA_CONFIG["max_subjects"],
        )
        print(f"Training FSTA from {len(subjects['records'])} subject time series...")
        data, fsta_metrics = generate_subject_bec(args, subjects, device)
        _report_existing_archive_difference(args.bec_path, data)
        save_subject_bec(args.bec_path, data)
        print(f"Saved FSTA-EC BEC archive: {args.bec_path.resolve()}")
    else:
        data = load_bec_archive(args.bec_path)
        data = _limit_archive_subjects(data, FIXED_DATA_CONFIG["max_subjects"])

    if args.graph_mode == "fusion":
        data["fmri_features"] = subject_fc_features(
            _aligned_graph_time_series(args, data, subjects)
        )
    data.update(load_phenotypes(args.phenotype_csv, data["subject_ids"], data["site_ids"]))
    data["qsr_qc"] = load_aligned_qc(
        args.phenotype_csv, data["subject_ids"], args.qsr_qc_columns
    )
    data["qsr_confound_values"] = load_aligned_phenotypes(
        args.phenotype_csv,
        data["subject_ids"],
        ["AGE_AT_SCAN", "SEX", "FIQ", "PIQ"],
    ).astype(np.float32)
    data["bec"] = np.asarray(data["bec"], dtype=np.float32)
    data["labels"] = np.asarray(data["labels"], dtype=np.int64)
    return data, fsta_metrics


def _report_existing_archive_difference(path, generated):
    if not path.is_file():
        return
    archived = load_bec_archive(path)
    if not np.array_equal(
        generated["subject_ids"].astype(str), archived["subject_ids"].astype(str)
    ):
        return
    difference = np.abs(generated["bec"] - archived["bec"])
    print(
        "raw-vs-archive BEC: "
        f"max_abs={difference.max():.3e}, mean_abs={difference.mean():.3e}"
    )


def _limit_archive_subjects(data, max_subjects):
    if max_subjects is None:
        return data
    subject_count = len(data["bec"])
    return {
        key: value[:max_subjects]
        if hasattr(value, "__len__") and len(value) == subject_count
        else value
        for key, value in data.items()
    }


def _aligned_graph_time_series(args, data, raw_subjects):
    if raw_subjects is not None:
        return raw_subjects["time_series"]
    graph_subjects = load_subject_dataset(
        args.data_root,
        FIXED_DATA_CONFIG["pipeline"],
        FIXED_DATA_CONFIG["strategy"],
        FIXED_DATA_CONFIG["derivative"],
        FIXED_DATA_CONFIG["standardize"],
        FIXED_DATA_CONFIG["max_subjects"],
    )
    by_subject = {
        str(subject_id): series
        for subject_id, series in zip(
            graph_subjects["subject_ids"], graph_subjects["time_series"]
        )
    }
    try:
        return [by_subject[str(subject_id)] for subject_id in data["subject_ids"]]
    except KeyError as error:
        raise ValueError(
            "Fusion mode requires raw ROI time series for every BEC subject"
        ) from error


__all__ = [
    "ABIDERecord",
    "FIXED_DATA_CONFIG",
    "load_abide_records",
    "load_bec_archive",
    "load_pipeline_data",
    "load_subject_dataset",
    "load_time_series",
]
