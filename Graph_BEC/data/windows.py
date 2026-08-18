"""Window sampling utilities used by the FSTA-EC baseline."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RandomSubjectWindowDataset(Dataset):
    """Draw one deterministic window per subject for each FSTA epoch."""

    def __init__(self, time_series, window_length, seed):
        self.time_series = time_series
        self.window_length = window_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, index):
        series = self.time_series[index]
        maximum_start = series.shape[0] - self.window_length
        if maximum_start < 0:
            raise ValueError(f"Subject {index} is shorter than window_length")
        generator = np.random.default_rng(self.seed + self.epoch * len(self.time_series) + index)
        start = int(generator.integers(maximum_start + 1))
        return torch.from_numpy(series[start:start + self.window_length]).float()


def fixed_window_starts(time_points, window_length, stride):
    if time_points < window_length:
        raise ValueError("time series is shorter than window_length")
    starts = list(range(0, time_points - window_length + 1, stride))
    final_start = time_points - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts

