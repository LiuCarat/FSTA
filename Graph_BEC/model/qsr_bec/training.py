"""Training and inference entry points for the final QSR-BEC module."""

from Graph_BEC.model.refinement import apply_qsr_refiner, train_qsr_refiner

__all__ = ["apply_qsr_refiner", "train_qsr_refiner"]
