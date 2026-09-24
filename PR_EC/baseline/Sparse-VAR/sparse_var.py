"""Sparse vector autoregressive BEC generation.

This module implements the subject-wise penalized VAR estimator used by the
Sparse VAR baseline.  Each response ROI is regressed on lagged ROI values with
an elastic-net penalty.  The resulting directed coefficient matrices are
aggregated across lags into a square BEC representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet


@dataclass(frozen=True)
class SparseVARConfig:
    """Configuration for a penalized VAR(p) model."""

    lags: int = 1
    alpha: float = 0.05
    l1_ratio: float = 1.0
    max_iter: int = 10000
    tolerance: float = 1e-4
    lag_decay: float = 1.0
    threshold: float = 0.0

    def validate(self) -> None:
        if self.lags < 1:
            raise ValueError("lags must be at least 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 < self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in (0, 1]")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.lag_decay <= 0:
            raise ValueError("lag_decay must be positive")
        if self.threshold < 0:
            raise ValueError("threshold cannot be negative")


def make_var_design(time_series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Create ``X_t=[y_{t-1},...,y_{t-p}]`` and ``Y_t=y_t`` arrays."""
    series = np.asarray(time_series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(f"Expected [time, roi] time series, got {series.shape}")
    observations, roi_count = series.shape
    if observations <= lags:
        raise ValueError(
            f"Need more than {lags} time points for VAR({lags}), got {observations}"
        )
    if not np.isfinite(series).all():
        raise ValueError("Sparse VAR input contains non-finite values")
    x = np.concatenate(
        [series[lags - lag : observations - lag] for lag in range(1, lags + 1)],
        axis=1,
    )
    y = series[lags:]
    if x.shape != (observations - lags, roi_count * lags):
        raise RuntimeError(f"Unexpected VAR design shape: {x.shape}")
    return x, y


def fit_sparse_var(time_series: np.ndarray, config: SparseVARConfig | None = None) -> np.ndarray:
    """Fit a sparse VAR and return coefficients shaped ``[lags, roi, roi]``.

    Matrix convention is ``coefficients[lag, source, target]``: a non-zero
    value means the source ROI at the selected lag predicts the target ROI.
    This convention matches the directed-channel handling in the downstream
    classifier after the matrix is transposed into its two edge channels.
    """
    config = config or SparseVARConfig()
    config.validate()
    x, y = make_var_design(time_series, config.lags)
    roi_count = y.shape[1]
    coefficients = np.zeros((config.lags, roi_count, roi_count), dtype=np.float32)

    for target in range(roi_count):
        estimator = ElasticNet(
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            fit_intercept=False,
            max_iter=config.max_iter,
            tol=config.tolerance,
            selection="cyclic",
        )
        estimator.fit(x, y[:, target])
        coefficients[:, :, target] = estimator.coef_.reshape(config.lags, roi_count)

    if config.threshold > 0:
        coefficients[np.abs(coefficients) < config.threshold] = 0.0
    if not np.isfinite(coefficients).all():
        raise ValueError("Sparse VAR coefficients contain non-finite values")
    return coefficients


def coefficients_to_bec(coefficients: np.ndarray, lag_decay: float = 1.0) -> np.ndarray:
    """Aggregate lagged directed coefficients into one signed BEC matrix."""
    values = np.asarray(coefficients, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"Expected [lags, roi, roi] coefficients, got {values.shape}")
    weights = np.power(float(lag_decay), np.arange(values.shape[0], dtype=np.float32))
    bec = np.tensordot(weights, values, axes=(0, 0)) / np.sum(weights)
    bec = np.asarray(bec, dtype=np.float32)
    np.fill_diagonal(bec, 0.0)
    if not np.isfinite(bec).all():
        raise ValueError("Sparse VAR BEC contains non-finite values")
    return bec


def generate_sparse_var_bec(
    time_series: np.ndarray, config: SparseVARConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one subject and return ``(BEC, lagged_coefficients)``."""
    config = config or SparseVARConfig()
    coefficients = fit_sparse_var(time_series, config)
    return coefficients_to_bec(coefficients, config.lag_decay), coefficients
