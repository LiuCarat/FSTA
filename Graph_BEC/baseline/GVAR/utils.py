"""Utilities for constructing lagged GVAR training samples."""

import numpy as np


def construct_training_dataset(data, order):
    """Build lagged predictors, responses, and time indices for GVAR."""
    if not isinstance(data, list):
        data = [data]

    predictors = []
    responses = []
    time_indices = []
    offset = 0
    for series in data:
        if series.ndim != 2:
            raise ValueError("Each time series must have shape [time, variables]")
        time_points, num_vars = series.shape
        if time_points <= order:
            raise ValueError("Each time series must be longer than the GVAR order")

        predictors.append(np.stack([series[index - order:index] for index in range(order, time_points)]))
        responses.append(series[order:])
        time_indices.append(np.arange(order, time_points) + offset)
        offset += time_points

    return np.concatenate(predictors), np.concatenate(responses), np.concatenate(time_indices)
