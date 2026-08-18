"""Train lagged FSTC-EC and generate subject-level EC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_BEC.data import load_subject_dataset
from Graph_BEC.utils import (
    RandomSubjectWindowDataset,
    fixed_window_starts,
    select_device,
    set_seed,
)
from Graph_BEC.model.fstc_ec import FSTCEC, fstc_ec_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/ABIDE-I")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "Graph_BEC/outputs/fstc_ec_v3.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "Graph_BEC/outputs/fstc_ec_v3_subject_ec.npz")
    parser.add_argument("--window-length", type=int, default=78)
    parser.add_argument("--stride", type=int, default=39)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-sparse", type=float, default=1e-6) #1e-6
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="auto")
    return parser.parse_args()


@torch.no_grad()
def generate_subject_ec(model, subjects, window_length, stride, device):
    all_ec, all_prediction_mse, all_baseline_mse = [], [], []
    for index, series in enumerate(subjects["time_series"], 1):
        windows_logits, errors, baseline_errors = [], [], []
        for start in fixed_window_starts(series.shape[0], window_length, stride):
            window = torch.from_numpy(
                series[start : start + window_length]
            ).float().unsqueeze(0).to(device)
            output = model(window)
            prediction = output["future_prediction"]
            windows_logits.append(output["bec_logits"].squeeze(0).cpu().numpy())
            errors.append(float((prediction - window[:, 1:]).pow(2).mean().item()))
            baseline_errors.append(
                float((window[:, :-1] - window[:, 1:]).pow(2).mean().item())
            )
        ec = np.tanh(np.mean(windows_logits, axis=0)).astype(np.float32)
        np.fill_diagonal(ec, 0.0)
        all_ec.append(ec)
        all_prediction_mse.append(np.mean(errors))
        all_baseline_mse.append(np.mean(baseline_errors))
        if index == 1 or index == len(subjects["time_series"]) or index % 100 == 0:
            print(f"EC [{index}/{len(subjects['time_series'])}] mse={np.mean(errors):.6f}")
    return (
        np.stack(all_ec),
        np.asarray(all_prediction_mse, dtype=np.float32),
        np.asarray(all_baseline_mse, dtype=np.float32),
    )


def save_ec_archive(path, subjects, ec, prediction_mse, baseline_mse):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bec=ec,
        ec=ec,
        labels=subjects["labels"],
        subject_ids=subjects["subject_ids"],
        site_ids=subjects["site_ids"],
        prediction_mse=prediction_mse,
        baseline_prediction_mse=baseline_mse,
    )


def make_loader(time_series, args, epoch):
    dataset = RandomSubjectWindowDataset(time_series, args.window_length, args.seed)
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(args.seed + epoch)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    return loader


def train_model(model, time_series, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        records = []
        for windows in make_loader(time_series, args, epoch):
            windows = windows.to(device)
            output = model(windows)
            delta_target = windows[:, 1:] - windows[:, :-1]
            loss, parts = fstc_ec_loss(
                output, delta_target, lambda_sparse=args.lambda_sparse
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                target = windows[:, 1:]
                baseline_mse = torch.mean((windows[:, :-1] - target) ** 2)
                full_mse = torch.mean((output["future_prediction"] - target) ** 2)
                ec = output["bec"]
                edge_mask = ~torch.eye(
                    ec.shape[-1], dtype=torch.bool, device=ec.device
                )
                edge_values = ec[:, edge_mask]
                records.append(
                    [
                        loss.item(),
                        baseline_mse.item(),
                        parts["delta"].item(),
                        full_mse.item(),
                        edge_values.abs().mean().item(),
                        edge_values.std(dim=0).mean().item(),
                        output["delta_prediction"].std().item(),
                        delta_target.std().item(),
                    ]
                )

        values = np.mean(records, axis=0)
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            baseline_mse, full_mse = values[1], values[3]
            gain = (baseline_mse - full_mse) / max(baseline_mse, 1e-12)
            print(
                f"epoch={epoch}/{args.epochs} "
                f"loss={values[0]:.6f} baseline={baseline_mse:.6f} "
                f"delta={values[2]:.6f} full={full_mse:.6f} "
                f"gain={gain * 100:.2f}% ec_abs={values[4]:.6f} "
                f"between_edge_std={values[5]:.6f} "
                f"delta_std={values[6]:.6f} target_delta_std={values[7]:.6f}"
            )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.gpu_id)
    subjects = load_subject_dataset(
        args.data_root,
        pipeline="cpac",
        strategy="filt_noglobal",
        derivative="rois_aal",
        standardize=True,
        max_subjects=None,
    )
    model = FSTCEC(window_length=args.window_length).to(device)
    print(f"FSTC-EC V3 subjects={len(subjects['time_series'])}; device={device}")
    train_model(model, subjects["time_series"], args, device)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "window_length": args.window_length,
                "roi_count": 90,
                "hidden_dim": 32,
                "edge_dim": 16,
                "tcn_layers": 2,
                "kernel_size": 3,
                "dropout": 0.1,
            },
            "seed": args.seed,
            "lambda_sparse": args.lambda_sparse,
        },
        args.checkpoint,
    )
    ec, prediction_mse, baseline_mse = generate_subject_ec(
        model, subjects, args.window_length, args.stride, device
    )
    save_ec_archive(args.output, subjects, ec, prediction_mse, baseline_mse)
    gain = (baseline_mse - prediction_mse) / np.maximum(baseline_mse, 1e-12)
    print(f"Saved checkpoint: {args.checkpoint.resolve()}")
    print(f"Saved EC archive: {args.output.resolve()} | shape={ec.shape}")
    print(f"Prediction MSE: {prediction_mse.mean():.6f}; baseline MSE: {baseline_mse.mean():.6f}; gain: {gain.mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
