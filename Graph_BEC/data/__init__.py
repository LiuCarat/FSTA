"""Data loading, fold preprocessing, runtime, and FSTA window utilities."""

from Graph_BEC.data.folds import make_stratified_splits, prepare_fold_arrays
from Graph_BEC.data.runtime import select_device, set_seed
from Graph_BEC.data.subjects import load_bec_archive, load_subject_dataset
from Graph_BEC.data.windows import RandomSubjectWindowDataset, fixed_window_starts

__all__ = [
    "RandomSubjectWindowDataset",
    "fixed_window_starts",
    "load_bec_archive",
    "load_subject_dataset",
    "make_stratified_splits",
    "prepare_fold_arrays",
    "select_device",
    "set_seed",
]

