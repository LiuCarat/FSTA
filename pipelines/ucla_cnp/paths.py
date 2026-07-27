"""Shared filesystem locations for the UCLA CNP pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset" / "ucla_cnp"
FMRIPREP_DIR = DATASET_DIR / "derivatives" / "fmriprep"
WORK_DIR = DATASET_DIR / "derivatives" / "work"
LICENSE_FILE = DATASET_DIR / "license.txt"
SUBJECT_LISTS_DIR = DATASET_DIR / "subject_lists"
BDCORE20_DIR = REPO_ROOT / "dataset" / "BDCore20"

SPACE = "MNI152NLin6Asym"
ATLAS_IMAGE_NAME = "BD_Core20_dseg.nii.gz"
ATLAS_LABELS_NAME = "BD_Core20_labels.tsv"
