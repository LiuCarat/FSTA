"""FSTA-EC baseline: FSTA model, training loop, and BEC extraction."""

from .fsta_components import FSTA, ScheduledOptim
from .arguments import add_fsta_arguments
from .bec_generation import generate_subject_bec, save_subject_bec
from .fsta_training import build_fsta, train_fsta
from .fsta_utils import FSTAWindowLoss, extract_subject_bec

__all__ = [
    "FSTA",
    "ScheduledOptim",
    "add_fsta_arguments",
    "generate_subject_bec",
    "save_subject_bec",
    "FSTAWindowLoss",
    "build_fsta",
    "train_fsta",
    "extract_subject_bec",
]
