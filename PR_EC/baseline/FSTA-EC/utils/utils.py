"""Original FSTA-EC loss and subject-level BEC extraction utilities."""

import copy

import numpy as np
import torch
import torch.nn as nn


def sliding_window_cutting(data, window_size, overlap):
    """Cut [subjects, time, nodes] into fixed-length overlapping windows."""
    step = window_size - overlap
    subjects, time_points, nodes = data.shape
    window_count = (time_points - window_size) // step + 1
    if (time_points - window_size) % step != 0:
        window_count += 1
    windows = torch.empty(subjects * window_count, window_size, nodes)
    for subject in range(subjects):
        for index in range(window_count - 1):
            start = step * index
            windows[subject * window_count + index] = data[subject, start:start + window_size]
        windows[subject * window_count + window_count - 1] = data[subject, -window_size:]
    return windows


class loss_func(nn.Module):
    def __init__(self, alpha_sp):
        super().__init__()
        self.alpha_sp = alpha_sp

    def forward(self, output, adj, label):
        prediction_loss = nn.functional.mse_loss(output, label)
        sparsity_loss = torch.sum(adj)
        return prediction_loss + self.alpha_sp * sparsity_loss


def change01(adj, threshold):
    binary = copy.deepcopy(adj)
    binary = np.where(binary >= threshold, 1, 0)
    np.fill_diagonal(binary, 0)
    return binary


@torch.no_grad()
def extract_subject_bec(model, records, time_series, window_length, stride, device):
    model.eval()
    all_bec, all_mse = [], []
    for index, (record, series) in enumerate(zip(records, time_series), 1):
        windows = sliding_window_cutting(
            torch.from_numpy(np.asarray(series, dtype=np.float32))[None, ...],
            window_length,
            window_length - stride,
        ).to(device)
        reconstruction, attention = model(windows)
        bec = attention.cpu().numpy().T.astype(np.float32)
        np.fill_diagonal(bec, 0.0)
        all_bec.append(bec)
        all_mse.append(float((reconstruction - windows).pow(2).mean().item()))
        if index == 1 or index == len(records) or index % 100 == 0:
            print(f"BEC [{index}/{len(records)}] subject={record.subject_id} mse={all_mse[-1]:.6f}")
    return {
        "bec": np.stack(all_bec),
        "reconstruction_mse": np.asarray(all_mse, dtype=np.float32),
    }
