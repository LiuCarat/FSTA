
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pr_ec.utils import fixed_window_starts






class IndividualECWindowLoss(nn.Module):
    MODES = ("original", "entropy")

    def __init__(self, mode="entropy", alpha=0.01, node_count=90, eps=1e-8):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown FSTA loss mode: {mode}")
        self.mode = mode
        self.alpha = alpha
        self.node_count = node_count
        self.eps = eps
        self.reconstruction = nn.MSELoss()

    def forward(self, reconstruction, attention, target):
        prediction = self.reconstruction(reconstruction, target)
        if self.mode == "original":
            regularizer = attention.sum()
        else:
            probability = attention.clamp_min(self.eps)
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            regularizer = (
                (-(probability * probability.log()).sum(dim=-1)).mean()
                / torch.log(
                    torch.tensor(float(self.node_count), device=attention.device)
                )
            )
        return prediction + self.alpha * regularizer, prediction, regularizer






@torch.no_grad()
def extract_subject_ec(
    model, records, time_series, window_length, stride, device, window_ranges=None
):
    model.eval()
    print("[STAGE 2/4] Generating subject-level EC")
    print("[Individual-EC] Inference started")
    all_ec, all_mse = [], []
    for index, (record, series) in enumerate(zip(records, time_series), 1):
        attentions, errors = [], []
        ranges = (
            window_ranges[index - 1]
            if window_ranges is not None
            else ((0, series.shape[0]),)
        )
        starts = []
        for range_start, range_end in ranges:
            starts.extend(
                range_start + start
                for start in fixed_window_starts(
                    range_end - range_start, window_length, stride
                )
            )
        for start in starts:
            window = (
                torch.from_numpy(series[start : start + window_length])
                .float()
                .unsqueeze(0)
                .to(device)
            )
            reconstruction, attention = model(window)
            attentions.append(attention.squeeze(0).cpu().numpy())
            errors.append(
                float((reconstruction - window).pow(2).mean().item())
            )
        ec = np.mean(np.stack(attentions), axis=0).T.astype(np.float32)
        np.fill_diagonal(ec, 0.0)
        all_ec.append(ec)
        all_mse.append(np.mean(errors))
        if index == 1 or index == len(records) or index % 100 == 0:
            print(
                f"[Individual-EC] Generated EC for {index}/{len(records)} subjects | "
                f"mse={np.mean(errors):.6f}"
            )
    ec_array = np.stack(all_ec)
    print("[Individual-EC] EC generation completed")
    return {
        "ec": ec_array,
        "reconstruction_mse": np.asarray(all_mse, dtype=np.float32),
    }
