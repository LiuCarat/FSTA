"""Dataset loading and experiment-input assembly for Graph-BEC."""
from __future__ import annotations

import numpy as np

from Graph_BEC.data.abide import (
    ABIDERecord,
    ASD_LABEL,
    DX_TO_LABEL,
    LABEL_TO_GROUP,
    TC_LABEL,
    load_abide_records,
    load_abide_time_series,
)
from Graph_BEC.data.adhd200 import (
    ADHD200Record,
    load_adhd200_records,
    load_adhd200_time_series,
)
from Graph_BEC.data.common import (
    ROI_COUNT,
    SOURCE_ROI_COUNT,
    limit_archive_subjects,
    load_bec_archive,
)
from Graph_BEC.model.phenotype import (
    load_aligned_phenotypes,
    load_phenotypes,
    subject_fc_features,
)
from Graph_BEC.model.qc import load_aligned_qc

FIXED_DATA_CONFIG = {
    "pipeline": "cpac",
    "strategy": "filt_noglobal",
    "derivative": "rois_aal",
    "standardize": True,
    "max_subjects": None,
}


def load_subject_dataset(
    data_root,
    pipeline="cpac",
    strategy="filt_global",
    derivative="rois_aal",
    standardize=True,
    max_subjects=None,
    profile=None,
    patient_label=1,
    control_label=0,
):
    if profile is not None and profile.name == "adhd200":
        records = load_adhd200_records(
            data_root, profile, patient_label=patient_label, control_label=control_label
        )
        series_loader = lambda record: load_adhd200_time_series(
            record, profile.source_roi_count, profile.roi_count, standardize
        )
        roi_count = profile.roi_count
    else:
        records = load_abide_records(data_root, pipeline, strategy, derivative)
        series_loader = lambda record: load_abide_time_series(record, standardize)
        roi_count = ROI_COUNT

    time_series = [series_loader(record) for record in records]
    if max_subjects is not None:
        records = records[:max_subjects]
        time_series = time_series[:max_subjects]
    if not records:
        raise ValueError("No subjects remain after data loading")
    if any(series.ndim != 2 or series.shape[1] != roi_count for series in time_series):
        raise ValueError(f"Every subject must provide a [T, {roi_count}] ROI time series")
    return {
        "records": records,
        "time_series": time_series,
        "labels": np.asarray([record.label for record in records], dtype=np.int64),
        "subject_ids": np.asarray([record.subject_id for record in records]),
        "site_ids": np.asarray([record.site_id for record in records]),
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
            profile=args.profile,
            patient_label=args.patient_label,
            control_label=args.control_label,
        )
        print(f"Training FSTA from {len(subjects['records'])} subject time series...")
        data, fsta_metrics = generate_subject_bec(args, subjects, device)
        _report_existing_archive_difference(args.bec_path, data)
        save_subject_bec(args.bec_path, data)
        print(f"Saved FSTA-EC BEC archive: {args.bec_path.resolve()}")
    else:
        data = limit_archive_subjects(
            load_bec_archive(args.bec_path), FIXED_DATA_CONFIG["max_subjects"]
        )

    if args.graph_mode == "fusion":
        data["fmri_features"] = subject_fc_features(
            _aligned_graph_time_series(args, data, subjects)
        )
    data.update(
        load_phenotypes(
            args.phenotype_csv, data["subject_ids"], data["site_ids"], args.profile
        )
    )
    data["qsr_qc"] = load_aligned_qc(
        args.phenotype_csv, data["subject_ids"], args.qsr_qc_columns, args.profile
    )
    data["qsr_confound_values"] = load_aligned_phenotypes(
        args.phenotype_csv,
        data["subject_ids"],
        args.profile.confound_columns,
        args.profile,
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
        profile=args.profile,
        patient_label=args.patient_label,
        control_label=args.control_label,
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
    "ABIDERecord", "ADHD200Record", "ASD_LABEL", "DX_TO_LABEL",
    "FIXED_DATA_CONFIG", "LABEL_TO_GROUP", "ROI_COUNT", "SOURCE_ROI_COUNT",
    "TC_LABEL", "load_abide_records", "load_abide_time_series",
    "load_adhd200_records", "load_adhd200_time_series", "load_bec_archive",
    "load_pipeline_data", "load_subject_dataset", "load_aligned_phenotypes",
]
