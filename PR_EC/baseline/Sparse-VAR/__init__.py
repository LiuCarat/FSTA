"""Sparse VAR baseline for Graph-EC."""
from .sparse_var import (
    SparseVARConfig,
    coefficients_to_ec,
    fit_sparse_var,
    generate_sparse_var_ec,
    make_var_design,
)

__all__ = [
    "SparseVARConfig",
    "coefficients_to_ec",
    "fit_sparse_var",
    "generate_sparse_var_ec",
    "make_var_design",
]
