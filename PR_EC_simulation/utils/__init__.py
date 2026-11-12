"""General-purpose runtime, split, and window utilities."""

from .runtime import select_device, set_seed
from .windows import RandomSubjectWindowDataset, fixed_window_starts


def __getattr__(name):
    if name in {"make_stratified_splits", "prepare_fold_arrays"}:
        from . import folds
        return getattr(folds, name)
    raise AttributeError(name)

__all__ = [
    "RandomSubjectWindowDataset",
    "fixed_window_starts",
    "make_stratified_splits",
    "prepare_fold_arrays",
    "select_device",
    "set_seed",
]
