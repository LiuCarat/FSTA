"""Compatibility registry for non-main utilities and baseline scripts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    data_root: Path
    phenotype_path: Path
    output_dir: Path
    bec_path: Path
    refined_bec_path: Path
    qsr_refined_bec_path: Path
    phenotype_format: str
    phenotype_id_column: str
    patient_column: str
    control_column: str
    patient_values: tuple[str, ...]
    control_values: tuple[str, ...]
    site_column: str
    sex_column: str
    continuous_columns: tuple[str, ...]
    confound_columns: tuple[str, ...]
    qc_columns: tuple[str, ...]
    source_roi_count: int
    roi_count: int
    exclude_subjects: tuple[str, ...]

def get_profile(name):
    modules = {
        "abide": "Graph_BEC.main_abide_i",
        "abide_ii": "Graph_BEC.main_abide_ii",
        "adhd200": "Graph_BEC.main_adhd200",
    }
    try:
        module_name = modules[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset profile: {name}") from error
    import importlib
    return importlib.import_module(module_name).DATASET_CONFIG

DATASET_NAMES = ("abide", "abide_ii", "adhd200")
