"""Dataset configuration definitions for the public PR-EC entry points."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    data_root: Path
    phenotype_path: Path
    PR_EC_PATH: Path
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
