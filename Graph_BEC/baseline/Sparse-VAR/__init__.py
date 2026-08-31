"""Sparse VAR baseline for Graph-BEC."""
from .sparse_var import (
    SparseVARConfig,
    coefficients_to_bec,
    fit_sparse_var,
    generate_sparse_var_bec,
    make_var_design,
)

__all__ = [
    "SparseVARConfig",
    "coefficients_to_bec",
    "fit_sparse_var",
    "generate_sparse_var_bec",
    "make_var_design",
]
