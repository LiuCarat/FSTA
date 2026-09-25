from __future__ import annotations

import numpy as np

from PR_EC.data.abide import (
    ABIDERecord,
    ASD_LABEL,
    DX_TO_LABEL,
    LABEL_TO_GROUP,
    TC_LABEL,
    load_abide_records,
    load_abide_time_series,
)
from PR_EC.data.adhd200 import (
    ADHD200Record,
    apply_category_imputer,
    apply_numeric_imputer,
    fit_category_imputer,
    fit_numeric_imputer,
    load_adhd200_records,
    load_adhd200_time_series,
    prepare_adhd_fold_arrays,
)
from PR_EC.data.common import (
    ROI_COUNT,
    SOURCE_ROI_COUNT,
)
from PR_EC.model.mpr import (
    load_aligned_phenotypes,
    load_phenotypes,
    subject_fc_features,
)
from PR_EC.model.qsr.qc_target import load_aligned_qc

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
        time_series = []
        window_ranges = []
        for record in records:
            series = load_adhd200_time_series(
                record,
                profile.source_roi_count,
                profile.roi_count,
                standardize,
            )
            time_series.append(series)
        roi_count = profile.roi_count

        window_ranges = None
    else:
        records = load_abide_records(data_root, pipeline, strategy, derivative, profile=profile)
        series_loader = lambda record: load_abide_time_series(record, standardize)
        roi_count = ROI_COUNT
        time_series = [series_loader(record) for record in records]
        window_ranges = None
    if max_subjects is not None:
        records = records[:max_subjects]
        time_series = time_series[:max_subjects]
        if window_ranges is not None:
            window_ranges = window_ranges[:max_subjects]
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
        "window_ranges": window_ranges,
    }


def load_pipeline_data(args, device):
    """Generate Individual-EC matrices and attach graph/QC covariates."""
    from PR_EC.model.individual_ec import generate_subject_ec

    subjects = None
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
    print("[STAGE 1/4] Training Individual-EC encoder")
    print(f"[INFO] Subjects loaded: {len(subjects['records'])}")
    print(f"[INFO] ROI shape: [T, {subjects['time_series'][0].shape[1]}]")
    data = generate_subject_ec(args, subjects, device)

    if args.profile.name == "abide_i":
        data["labels"] = _labels_from_abide_phenotype(
            args.phenotype_csv,
            data["subject_ids"],
            args.profile,
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
    data["individual_ec"] = np.asarray(data["individual_ec"], dtype=np.float32)
    data["labels"] = np.asarray(data["labels"], dtype=np.int64)
    return data


def _labels_from_abide_phenotype(phenotype_csv, subject_ids, profile):
    """Build canonical TC=0/ASD=1 labels from the phenotype source.

    The phenotype source is used to align canonical binary labels by subject ID.
    """
    diagnosis = load_aligned_phenotypes(
        phenotype_csv,
        subject_ids,
        (profile.patient_column,),
        profile,
    )[:, 0]
    patient_values = {float(value) for value in profile.patient_values}
    control_values = {float(value) for value in profile.control_values}
    labels = np.full(len(diagnosis), -1, dtype=np.int64)
    labels[np.isin(diagnosis, list(patient_values))] = 1
    labels[np.isin(diagnosis, list(control_values))] = 0
    if np.any(labels < 0):
        unknown = np.unique(diagnosis[labels < 0]).tolist()
        raise ValueError(
            f"Unknown ABIDE diagnosis values in phenotype: {unknown}"
        )
    return labels


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
            "Fusion mode requires raw ROI time series for every EC subject"
        ) from error


__all__ = [
    "ABIDERecord", "ADHD200Record", "ASD_LABEL", "DX_TO_LABEL",
    "FIXED_DATA_CONFIG", "LABEL_TO_GROUP", "ROI_COUNT", "SOURCE_ROI_COUNT",
    "TC_LABEL", "load_abide_records", "load_abide_time_series",
    "load_adhd200_records", "load_adhd200_time_series",
    "load_pipeline_data", "load_subject_dataset", "load_aligned_phenotypes",
]
