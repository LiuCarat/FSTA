"""General-purpose runtime, split, and window utilities."""

from Graph_BEC.utils.folds import make_stratified_splits, prepare_fold_arrays
from Graph_BEC.utils.runtime import select_device, set_seed
from Graph_BEC.utils.windows import RandomSubjectWindowDataset, fixed_window_starts

__all__ = [
    "RandomSubjectWindowDataset",
    "fixed_window_starts",
    "make_stratified_splits",
    "prepare_fold_arrays",
    "select_device",
    "set_seed",
]
