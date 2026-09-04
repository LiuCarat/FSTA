"""Run the VarCoNet representation baseline on ABIDE-I or ABIDE-II.

The original VarCoNet paper repository expects pre-packed ``ABIDE*_nilearn``
files and contains several unrelated experiments.  This entry point uses the
Graph-BEC data loader and downstream classifier so that the only changed part
is the subject representation: a VarCoNet encoder is trained on each training
fold with the original two-view InfoNCE objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_BEC.data import load_subject_dataset
from Graph_BEC.dataset_configs import get_profile
from Graph_BEC.downstream import train_classifier
from Graph_BEC.utils.folds import fit_bec_scaler, make_stratified_splits, transform_bec
from Graph_BEC.utils.runtime import set_seed
from Graph_BEC.baseline.VarCoNet.model_scripts.VarCoNet import VarCoNet


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=("abide", "abide_ii"), default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)
    output_dir = Path(__file__).resolve().parent / "outputs" / profile.name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("abide", "abide_ii"), default=profile.name)
    parser.add_argument("--data-root", type=Path, default=profile.data_root)
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--representation-path", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--encoder-epochs", type=int, default=100)
    parser.add_argument("--encoder-patience", type=int, default=20)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-repeats", type=int, default=1)
    parser.add_argument("--regenerate-representation", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


def device_for(gpu_id):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda" if gpu_id == "auto" else f"cuda:{gpu_id}")


def defaults(args):
    if args.dataset == "abide":
        return (1, 2048, 0.0537, 2)
    return (2, 512, 0.0541, 2)


def pad_series(series, length):
    result = np.zeros((length, series.shape[1]), dtype=np.float32)
    result[: min(length, len(series))] = series[:length]
    return result


def make_view(batch, min_length, generator):
    views = []
    for series in batch:
        valid = len(series)
        if valid <= min_length:
            start, end = 0, valid
        else:
            span = int(torch.randint(min_length, valid + 1, (1,), generator=generator).item())
            start = int(torch.randint(0, valid - span + 1, (1,), generator=generator).item())
            end = start + span
        view = series[start:end].clone()
        view = view + torch.randn(view.shape, generator=generator, dtype=view.dtype) * 0.01
        views.append(view)
    return views


def edge_features_to_matrix(encoded, roi_count):
    matrix = encoded.new_zeros((encoded.shape[0], roi_count, roi_count))
    upper = torch.triu_indices(roi_count, roi_count, offset=1, device=encoded.device)
    matrix[:, upper[0], upper[1]] = encoded
    matrix[:, upper[1], upper[0]] = encoded
    return matrix


def train_encoder(train_series, roi_count, args, device, max_length, seed):
    default_heads, default_ff, default_tau, default_batch = defaults(args)
    heads = args.n_heads or default_heads
    feedforward = args.dim_feedforward or default_ff
    tau = args.tau or default_tau
    lr = args.encoder_lr or (0.00024 if args.dataset == "abide" else 0.00014)
    if roi_count % heads:
        raise ValueError(f"ROI count {roi_count} must be divisible by --n-heads {heads}")
    config = {"layers": args.layers, "n_heads": heads, "dim_feedforward": feedforward, "max_length": max_length}
    model = VarCoNet(config, roi_count).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state, best_loss = None, float("inf")
    waiting = 0
    for epoch in range(args.encoder_epochs):
        model.train()
        order = torch.randperm(len(train_series), generator=generator).tolist()
        losses = []
        for offset in range(0, len(order), args.batch_size or default_batch):
            indices = order[offset : offset + (args.batch_size or default_batch)]
            batch = [torch.from_numpy(train_series[index]) for index in indices]
            view1 = torch.stack([torch.from_numpy(pad_series(x.numpy(), max_length)) for x in make_view(batch, args.min_length, generator)]).to(device)
            view2 = torch.stack([torch.from_numpy(pad_series(x.numpy(), max_length)) for x in make_view(batch, args.min_length, generator)]).to(device)
            z1, z2 = model(view1), model(view2)
            logits = (F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T) / tau
            targets = torch.arange(len(indices), device=device)
            loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
        current = float(np.mean(losses)) if losses else float("inf")
        if current < best_loss:
            best_loss, best_state, waiting = current, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            waiting += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  encoder epoch {epoch + 1}: loss={current:.4f}")
        if waiting >= args.encoder_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def encode(model, series, device, max_length, roi_count):
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(np.stack([pad_series(x, max_length) for x in series])).to(device)
        return edge_features_to_matrix(model(batch), roi_count).cpu().numpy().astype(np.float32)


def save_archive(path, representations, dataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bec=representations, labels=dataset["labels"], subject_ids=dataset["subject_ids"], site_ids=dataset["site_ids"], representation="varconet", dataset=args.dataset)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def classify_fold(args, train_fc, train_labels, val_fc, val_labels, test_fc, test_labels, device, fold):
    rows = []
    print(f"fold {fold}: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}")
    mean, std = fit_bec_scaler(train_fc)
    train_fc = transform_bec(train_fc, mean, std)
    val_fc = transform_bec(val_fc, mean, std)
    test_fc = transform_bec(test_fc, mean, std)
    for repeat in range(args.classifier_repeats):
        metrics, _ = train_classifier(train_fc, train_labels, val_fc, val_labels, test_fc, test_labels, device=device, seed=args.seed + fold * 1000 + repeat + 1, max_epochs=args.classifier_epochs, patience=args.classifier_patience, batch_size=32, learning_rate=args.classifier_lr)
        rows.append({"fold": fold, "repeat": repeat + 1, **metrics})
        print(
            "  fold result | "
            + " | ".join(
                f"{name}={metrics[name] * 100:.2f}%"
                for name in ("ACC", "SPE", "AUC", "Precision", "Recall", "F1")
            ),
            flush=True,
        )
    return rows


def save_results(path, rows):
    if not rows:
        raise ValueError("No classification results were produced")
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics.json").open("w") as handle: json.dump(rows, handle, indent=2)
    with (path / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    names = [name for name in rows[0] if name not in {"fold", "repeat"}]
    summary = {name: {"mean": float(np.nanmean([row[name] for row in rows])), "std": float(np.nanstd([row[name] for row in rows]))} for name in names}
    with (path / "summary.json").open("w") as handle: json.dump(summary, handle, indent=2)
    metrics = ("ACC", "SPE", "AUC", "Precision", "Recall", "F1")
    print(f"classification results saved to: {path}", flush=True)
    print("\nfold-local mean±std (%)", flush=True)
    print("fold | " + " | ".join(metrics), flush=True)
    for row in rows:
        print(
            f"{int(row['fold']):>4d} | "
            + " | ".join(f"{row[name] * 100:>9.2f}" for name in metrics),
            flush=True,
        )
    print("mean | " + " | ".join(f"{summary[name]['mean'] * 100:>9.2f}" for name in metrics), flush=True)
    print(" std | " + " | ".join(f"{summary[name]['std'] * 100:>9.2f}" for name in metrics), flush=True)
    print(
        "mean±std | "
        + " | ".join(
            f"{summary[name]['mean'] * 100:.2f}±{summary[name]['std'] * 100:.2f}"
            for name in metrics
        ),
        flush=True,
    )


def main():
    args = parse_args()
    if args.generation_only and args.classification_only: raise ValueError("generation-only and classification-only are mutually exclusive")
    set_seed(args.seed)
    profile = get_profile(args.dataset)
    output_dir = args.output_dir
    representation_path = args.representation_path or output_dir / "subject_varconet.npz"
    device = device_for(args.gpu_id)
    if args.classification_only:
        archive = load_archive(representation_path)
    else:
        dataset = load_subject_dataset(args.data_root, pipeline=args.pipeline, strategy=args.strategy, derivative=args.derivative, profile=profile, standardize=True, max_subjects=args.max_subjects)
        max_length = max(len(series) for series in dataset["time_series"])
        matrices = []
        rows = []
        splits = list(make_stratified_splits(dataset["labels"], args.n_splits, args.seed, args.validation_size))
        for fold, train_idx, val_idx, test_idx in splits:
            print(f"training VarCoNet representation for fold {fold}")
            model = train_encoder([dataset["time_series"][i] for i in train_idx], dataset["time_series"][0].shape[1], args, device, max_length, args.seed + fold)
            train_fc = encode(model, [dataset["time_series"][i] for i in train_idx], device, max_length, dataset["time_series"][0].shape[1])
            val_fc = encode(model, [dataset["time_series"][i] for i in val_idx], device, max_length, dataset["time_series"][0].shape[1])
            test_fc = encode(model, [dataset["time_series"][i] for i in test_idx], device, max_length, dataset["time_series"][0].shape[1])
            rows.extend(classify_fold(args, train_fc, dataset["labels"][train_idx], val_fc, dataset["labels"][val_idx], test_fc, dataset["labels"][test_idx], device, fold))
            save_results(output_dir, rows)
            if fold == 1:
                matrices.append(encode(model, dataset["time_series"], device, max_length, dataset["time_series"][0].shape[1]))
        archive = {"bec": matrices[0], "labels": dataset["labels"], "subject_ids": dataset["subject_ids"], "site_ids": dataset["site_ids"]}
        save_archive(representation_path, archive["bec"], dataset, args)
    print(f"VarCoNet representation: {representation_path} shape={archive['bec'].shape}")
    if args.generation_only: return
    if args.classification_only:
        labels = archive["labels"].astype(np.int64)
        rows = []
        for fold, train_idx, val_idx, test_idx in make_stratified_splits(labels, args.n_splits, args.seed, args.validation_size):
            rows.extend(classify_fold(args, archive["bec"][train_idx], labels[train_idx], archive["bec"][val_idx], labels[val_idx], archive["bec"][test_idx], labels[test_idx], device, fold))
    if args.classification_only:
        save_results(output_dir, rows)


if __name__ == "__main__": main()
