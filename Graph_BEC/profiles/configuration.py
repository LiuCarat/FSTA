"""Registry for dataset-specific Graph-BEC experiment configurations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    defaults: dict[str, Any]


def get_profile(name: str) -> ExperimentProfile:
    if name == "abide":
        from Graph_BEC.profiles.abide_i import PROFILE
    elif name == "adhd200":
        from Graph_BEC.profiles.adhd200 import PROFILE
    else:
        raise ValueError(f"Unknown dataset profile: {name}")
    return PROFILE


PROFILES = {name: get_profile(name) for name in ("abide", "adhd200")}
