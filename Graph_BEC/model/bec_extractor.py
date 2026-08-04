"""Extract subject-level BEC matrices from a trained FSTA."""
from __future__ import annotations
import numpy as np
import torch
from Graph_BEC.data import fixed_window_starts


@torch.no_grad()
def extract_subject_bec(model, records, time_series, window_length, stride, device):
    model.eval(); all_bec, all_mse = [], []
    for index, (record, series) in enumerate(zip(records, time_series), 1):
        attentions, errors = [], []
        for start in fixed_window_starts(series.shape[0], window_length, stride):
            window = torch.from_numpy(series[start:start + window_length]).float().unsqueeze(0).to(device)
            reconstruction, attention = model(window)
            attentions.append(attention.squeeze(0).cpu().numpy())
            errors.append(float((reconstruction - window).pow(2).mean().item()))
        bec = np.mean(np.stack(attentions), axis=0).T.astype(np.float32)
        np.fill_diagonal(bec, 0.0); all_bec.append(bec); all_mse.append(np.mean(errors))
        if index == 1 or index == len(records) or index % 100 == 0:
            print(f"BEC [{index}/{len(records)}] subject={record.subject_id} mse={np.mean(errors):.6f}")
    return {"bec": np.stack(all_bec), "reconstruction_mse": np.asarray(all_mse, dtype=np.float32)}
