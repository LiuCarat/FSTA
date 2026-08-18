"""Assemble the inputs required by the Graph-BEC experiment pipeline."""

from __future__ import annotations

import numpy as np

from Graph_BEC.baseline.FSTA_EC import generate_subject_bec, save_subject_bec
from Graph_BEC.data.subjects import load_bec_archive, load_subject_dataset
from Graph_BEC.phenotype import (
    load_aligned_phenotypes,
    load_phenotypes,
    subject_fc_features,
)
from Graph_BEC.qc import load_aligned_qc


FIXED_DATA_CONFIG = {
    "pipeline": "cpac",
    "strategy": "filt_noglobal",
    "derivative": "rois_aal",
    "standardize": True,
    "max_subjects": None,
}


def load_pipeline_data(args, device):
    """Load or generate BEC matrices and attach graph/QC covariates."""
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

    data.update(
        load_phenotypes(args.phenotype_csv, data["subject_ids"], data["site_ids"])
    )
    data["qsr_qc"] = load_aligned_qc(
        args.phenotype_csv,
        data["subject_ids"],
        args.qsr_qc_columns,
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
        generated["subject_ids"].astype(str),
        archived["subject_ids"].astype(str),
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
            graph_subjects["subject_ids"],
            graph_subjects["time_series"],
        )
    }
    try:
        return [by_subject[str(subject_id)] for subject_id in data["subject_ids"]]
    except KeyError as error:
        raise ValueError(
            "Fusion mode requires raw ROI time series for every BEC subject"
        ) from error
