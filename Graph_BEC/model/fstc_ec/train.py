"""Train factorized FSTA-EC and export subject-level BEC matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(PACKAGE_DIR.parent))
    from fstc_ec.fstc_ec import FSTCECReconstruction
    from fstc_ec.losses import reconstruction_stage_loss
    from fstc_ec.utils import (
        fixed_window_starts,
        load_subject_dataset,
        make_window_loader,
        pad_series,
        select_device,
        set_seed,
        split_series,
    )
else:
    from .fstc_ec import FSTCECReconstruction
    from .losses import reconstruction_stage_loss
    from .utils import (
        fixed_window_starts,
        load_subject_dataset,
        make_window_loader,
        pad_series,
        select_device,
        set_seed,
        split_series,
    )

ROOT = (
    PACKAGE_DIR.parents[2]
    if PACKAGE_DIR.parents[1].name == "Graph_BEC"
    else Path.cwd()
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/ABIDE-I")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "Graph_BEC/outputs/factorized_fsta_ec.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Graph_BEC/outputs/factorized_fsta_ec_subject_ec.npz",
    )
    parser.add_argument("--window-length", type=int, default=78)
    parser.add_argument("--stride", type=int, default=39)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ec-dim", type=int, default=16)
    parser.add_argument("--ec-temperature", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--checkpoint-selection", choices=["best", "final"], default="best")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="auto")
    return parser.parse_args()


def make_model(args, device):
    return FSTCECReconstruction(
        window_length=args.window_length,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        ec_dim=args.ec_dim,
        ec_temperature=args.ec_temperature,
        dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def evaluate(model, series, args, device):
    model.eval()
    losses = []
    for windows in make_window_loader(series, args.window_length, args.batch_size, args.seed, 0):
        windows = windows.to(device)
        losses.append(torch.mean((model(windows)["reconstruction"] - windows) ** 2).item())
    return float(np.mean(losses))


def train_model(model, series, args, device):
    train_series, validation_series = split_series(series, args.validation_size, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_validation = float("inf")
    final_validation = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        records = []
        loader = make_window_loader(
            train_series, args.window_length, args.batch_size, args.seed, epoch
        )
        for windows in loader:
            windows = windows.to(device)
            output = model(windows)
            loss, parts = reconstruction_stage_loss(
                output, windows, entropy_weight=args.entropy_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            records.append(
                (
                    parts["reconstruction"].item(),
                    parts["entropy"].item(),
                    output["bec"].amax(dim=-1).mean().item(),
                )
            )

        final_validation = evaluate(model, validation_series, args, device)
        if final_validation < best_validation:
            best_validation = final_validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch == args.epochs or epoch % 20 == 0:
            reconstruction, entropy, peak = np.mean(records, axis=0)
            print(
                f"stage=reconstruction epoch={epoch}/{args.epochs} "
                f"recon={reconstruction:.6f} entropy={entropy:.6f} "
                f"peak={peak:.4f} val_recon={final_validation:.6f}"
            )

    if args.checkpoint_selection == "best" and best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_subjects": len(train_series),
        "validation_subjects": len(validation_series),
        "checkpoint_selection": args.checkpoint_selection,
        "best_validation_mse": float(best_validation),
        "final_validation_mse": float(final_validation),
        "selected_validation_mse": float(
            best_validation if args.checkpoint_selection == "best" else final_validation
        ),
    }


@torch.no_grad()
def generate_subject_ec(model, subjects, args, device):
    model.eval()
    all_ec, all_errors = [], []
    for series in subjects["time_series"]:
        series = pad_series(series, args.window_length)
        window_ecs, errors = [], []
        for start in fixed_window_starts(series.shape[0], args.window_length, args.stride):
            window = torch.from_numpy(series[start : start + args.window_length]).float()
            window = window.unsqueeze(0).to(device)
            output = model(window)
            window_ecs.append(output["bec"].squeeze(0).cpu().numpy())
            errors.append((output["reconstruction"] - window).pow(2).mean().item())
        ec = np.mean(window_ecs, axis=0).astype(np.float32)
        np.fill_diagonal(ec, 0.0)
        all_ec.append(ec)
        all_errors.append(np.mean(errors))
    return np.stack(all_ec), np.asarray(all_errors, dtype=np.float32)


def save_ec_archive(path, subjects, ec, reconstruction_mse):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bec=ec,
        ec=ec,
        labels=subjects["labels"],
        subject_ids=subjects["subject_ids"],
        site_ids=subjects["site_ids"],
        reconstruction_mse=reconstruction_mse,
        model_version=np.asarray("factorized_fsta_ec_v1"),
        bec_direction=np.asarray("source_to_target_rows_to_columns"),
    )


def summarize_ec(ec):
    incoming = ec.transpose(0, 2, 1)
    probability = np.clip(incoming, 1e-12, None)
    entropy = -(probability * np.log(probability)).sum(axis=-1)
    entropy /= np.log(float(incoming.shape[-1] - 1))
    return {
        "incoming_sum_mean": float(incoming.sum(axis=-1).mean()),
        "incoming_sum_std": float(incoming.sum(axis=-1).std()),
        "entropy_mean": float(entropy.mean()),
        "peak_mean": float(incoming.max(axis=-1).mean()),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.gpu_id)
    subjects = load_subject_dataset(args.data_root)
    model = make_model(args, device)
    print(f"Factorized FSTA-EC subjects={len(subjects['time_series'])}; device={device}")
    metrics = train_model(model, subjects["time_series"], args, device)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "training_metrics": metrics,
            "model_state": model.state_dict(),
            "model_config": vars(args),
            "model_type": "Factorized FSTA temporal encoder + directed EC adapter + signal flow",
        },
        args.checkpoint,
    )
    ec, reconstruction_mse = generate_subject_ec(model, subjects, args, device)
    save_ec_archive(args.output, subjects, ec, reconstruction_mse)
    summary = summarize_ec(ec)
    print(f"Saved checkpoint: {args.checkpoint.resolve()}")
    print(f"Saved EC archive: {args.output.resolve()} | shape={ec.shape}")
    print(f"Selected validation reconstruction MSE: {metrics['selected_validation_mse']:.6f}")
    print(
        f"Exported EC: incoming_sum={summary['incoming_sum_mean']:.4f}±"
        f"{summary['incoming_sum_std']:.4f}; entropy={summary['entropy_mean']:.4f}; "
        f"peak={summary['peak_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
