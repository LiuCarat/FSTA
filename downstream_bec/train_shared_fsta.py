import argparse
import csv
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.Optim import ScheduledOptim
from downstream_bec.data.bdcore20_dataset import (
    BDCore20Dataset,
    load_roi_names,
)
from downstream_bec.models.subject_fsta import SubjectFSTA, SubjectFSTALoss


def set_deterministic_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(gpu_id):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == "auto":
        free_memory = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
        return torch.device(f"cuda:{int(np.argmax(free_memory))}")
    return torch.device(f"cuda:{int(gpu_id)}")


def build_model_options(args, dataset):
    return SimpleNamespace(
        time_num=dataset.time_num,
        nodes_num=dataset.nodes_num,
        d_model=args.d_model,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_act=args.hidden_act,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        hidden_dropout_prob=args.hidden_dropout_prob,
        initializer_range=args.initializer_range,
        no_filters=args.no_filters,
    )


def build_model(args, dataset, device):
    options = build_model_options(args, dataset)
    return SubjectFSTA(
        opt=options,
        time_num=dataset.time_num,
        d_model=args.d_model,
        d_inner=args.d_inner_hid,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
    ).to(device)


def save_history(path, rows):
    with path.open("w", newline="") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["epoch", "loss", "reconstruction_loss", "regularizer"],
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def extract_subject_bec(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    raw_attention_batches = []
    labels = []
    indices = []

    for time_series, batch_labels, batch_indices in loader:
        time_series = time_series.to(device)
        _, raw_attention = model(time_series)
        raw_attention_batches.append(raw_attention.cpu())
        labels.append(batch_labels)
        indices.append(batch_indices)

    raw_attention = torch.cat(raw_attention_batches, dim=0)
    labels = torch.cat(labels, dim=0)
    indices = torch.cat(indices, dim=0)

    order = torch.argsort(indices)
    raw_attention = raw_attention[order]
    labels = labels[order]

    bec = raw_attention.transpose(-1, -2).clone()
    diagonal = torch.arange(bec.size(-1))
    bec[:, diagonal, diagonal] = 0.0
    return raw_attention.numpy(), bec.numpy(), labels.numpy()


def train_one_seed(args, dataset, roi_names, device, seed):
    set_deterministic_seed(seed)
    seed_dir = Path(args.output_root) / args.loss_mode / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = build_model(args, dataset, device)
    base_optimizer = optim.Adam(
        model.parameters(),
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )
    optimizer = ScheduledOptim(
        base_optimizer,
        args.lr_mul,
        args.d_model,
        args.n_warmup_steps,
    )
    criterion = SubjectFSTALoss(
        mode=args.loss_mode,
        alpha=args.loss_alpha,
    ).to(device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_values = []
        for time_series, _, _ in train_loader:
            time_series = time_series.to(device)
            optimizer.zero_grad()
            reconstruction, subject_attention = model(time_series)
            loss, prediction_loss, regularizer = criterion(
                reconstruction,
                time_series,
                subject_attention,
            )
            loss.backward()
            optimizer.step_and_update_lr()
            epoch_values.append(
                (loss.item(), prediction_loss.item(), regularizer.item())
            )

        averages = np.mean(epoch_values, axis=0)
        history.append(
            {
                "epoch": epoch,
                "loss": float(averages[0]),
                "reconstruction_loss": float(averages[1]),
                "regularizer": float(averages[2]),
            }
        )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"seed={seed} mode={args.loss_mode} epoch={epoch}/{args.epochs} "
                f"loss={averages[0]:.6f} reconstruction={averages[1]:.6f} "
                f"regularizer={averages[2]:.6f}"
            )

    model_config = vars(args).copy()
    model_config.update(
        {
            "seed": seed,
            "time_num": dataset.time_num,
            "nodes_num": dataset.nodes_num,
        }
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "config": model_config},
        seed_dir / "model.pt",
    )
    save_history(seed_dir / "training_history.csv", history)

    raw_attention, bec, labels = extract_subject_bec(
        model,
        dataset,
        args.batch_size,
        device,
    )
    subject_ids = np.asarray(dataset.subject_ids)
    roi_names = np.asarray(roi_names)
    np.savez_compressed(
        seed_dir / "subject_bec.npz",
        raw_attention=raw_attention,
        bec=bec,
        labels=labels,
        subject_ids=subject_ids,
        roi_names=roi_names,
    )

    individual_dir = seed_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    for subject_id, label, subject_bec in zip(subject_ids, labels, bec):
        group = "BD" if int(label) == 1 else "HC"
        np.save(individual_dir / f"{subject_id}_{group}.npy", subject_bec)

    asymmetry = float(np.mean(np.abs(bec - bec.transpose(0, 2, 1))))
    between_subject_std = float(bec.std(axis=0).mean())
    summary = {
        "seed": seed,
        "loss_mode": args.loss_mode,
        "shape": list(bec.shape),
        "mean_directed_asymmetry": asymmetry,
        "mean_between_subject_std": between_subject_std,
    }
    with (seed_dir / "bec_summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Saved {bec.shape[0]} subject BEC matrices to {seed_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset/BDCore20")
    parser.add_argument("--roi_labels", default="dataset/BDCore20/atlas/roi_labels.tsv")
    parser.add_argument("--output_root", default="downstream_bec/outputs/shared_fsta")
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--loss_mode", choices=SubjectFSTALoss.MODES, default="original_sum")
    parser.add_argument("--loss_alpha", type=float, default=0.8)
    parser.add_argument("--skiprows", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=301)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--gpu_id", default="auto")
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_inner_hid", type=int, default=64)
    parser.add_argument("--d_k", type=int, default=8)
    parser.add_argument("--d_v", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_warmup_steps", type=int, default=4000)
    parser.add_argument("--lr_mul", type=float, default=1.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--hidden_act", default="gelu")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--no_filters", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = BDCore20Dataset(args.data_root, skiprows=args.skiprows)
    roi_names = load_roi_names(args.roi_labels)
    if len(roi_names) != dataset.nodes_num:
        raise ValueError(
            f"ROI count {len(roi_names)} does not match node count {dataset.nodes_num}"
        )
    device = select_device(args.gpu_id)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    print(
        f"Loaded {len(dataset)} subjects with shape "
        f"[{dataset.time_num}, {dataset.nodes_num}] on {device}"
    )
    for seed in seeds:
        train_one_seed(args, dataset, roi_names, device, seed)


if __name__ == "__main__":
    main()
