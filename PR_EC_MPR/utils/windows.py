"""Time-series window sampling utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RandomSubjectWindowDataset(Dataset):
    """Draw one deterministic window per subject for each training epoch."""

    def __init__(self, time_series, window_length, seed, window_ranges=None):
        self.time_series = time_series
        self.window_length = window_length
        self.seed = seed
        self.epoch = 0
        self.window_ranges = window_ranges

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, index):
        series = self.time_series[index]
        generator = np.random.default_rng(
            self.seed + self.epoch * len(self.time_series) + index
        )
        ranges = (
            self.window_ranges[index]
            if self.window_ranges is not None
            else ((0, series.shape[0]),)
        )
        valid_ranges = [
            (range_start, range_end)
            for range_start, range_end in ranges
            if range_end - range_start >= self.window_length
        ]
        if not valid_ranges:
            raise ValueError(
                f"Subject {index} has no run long enough for window_length"
            )
        range_start, range_end = valid_ranges[
            int(generator.integers(len(valid_ranges)))
        ]
        maximum_start = range_end - range_start - self.window_length
        start = range_start + int(generator.integers(maximum_start + 1))
        return torch.from_numpy(
            series[start : start + self.window_length]
        ).float()


def fixed_window_starts(time_points, window_length, stride):
    if time_points < window_length:
        raise ValueError("time series is shorter than window_length")
    starts = list(range(0, time_points - window_length + 1, stride))
    final_start = time_points - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts
