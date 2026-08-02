import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from downstream_bec.evaluation.metrics import (
    classification_metrics,
    classification_metrics_from_predictions,
    select_youden_threshold,
)
from downstream_bec.models.directed_brainnetcnn import DirectedBrainNetCNN


def set_seed(seed):
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


def fit_edge_scaler(train_bec):
    mean = train_bec.mean(axis=0)
    std = train_bec.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def transform_bec(bec, mean, std):
    scaled = (bec - mean) / std
    diagonal = np.arange(scaled.shape[-1])
    scaled[:, diagonal, diagonal] = 0.0
    directed_channels = np.stack(
        [scaled, scaled.transpose(0, 2, 1)],
        axis=1,
    )
    return directed_channels.astype(np.float32)


def make_loader(features, labels, indices, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(labels[indices].astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    for features, batch_labels in loader:
        logits = model(features.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def train_fold(
    args,
    bec,
    labels,
    subject_ids,
    train_indices,
    val_indices,
    test_indices,
    fold_dir,
    device,
    fold_seed,
):
    mean, std = fit_edge_scaler(bec[train_indices])
    features = transform_bec(bec, mean, std)
    np.savez_compressed(fold_dir / "edge_scaler.npz", mean=mean, std=std)

    train_loader = make_loader(
        features,
        labels,
        train_indices,
        args.batch_size,
        True,
        fold_seed,
    )
    val_loader = make_loader(
        features,
        labels,
        val_indices,
        args.batch_size,
        False,
        fold_seed,
    )
    test_loader = make_loader(
        features,
        labels,
        test_indices,
        args.batch_size,
        False,
        fold_seed,
    )

    model = DirectedBrainNetCNN(
        nodes_num=bec.shape[-1],
        e2e_channels=(args.e2e1_channels, args.e2e2_channels),
        e2n_channels=args.e2n_channels,
        n2g_channels=args.n2g_channels,
        fc_channels=args.fc_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_labels = labels[train_indices]
    negative_count = max(int((train_labels == 0).sum()), 1)
    positive_count = max(int((train_labels == 1).sum()), 1)
    pos_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -np.inf
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for features_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            logits = model(features_batch.to(device))
            loss = criterion(logits, labels_batch.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_labels, val_probabilities = predict(model, val_loader, device)
        val_metrics = classification_metrics(val_labels, val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **val_metrics,
            }
        )
        if val_metrics["AUC"] > best_auc + args.min_delta:
            best_auc = val_metrics["AUC"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), fold_dir / "brainnetcnn_best.pt")

    with (fold_dir / "history.csv").open("w", newline="") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["epoch", "train_loss", "ACC", "SEN", "SPE", "AUC"],
        )
        writer.writeheader()
        writer.writerows(history)

    val_labels, val_probabilities = predict(model, val_loader, device)
    threshold = select_youden_threshold(val_labels, val_probabilities)
    test_labels, test_probabilities = predict(model, test_loader, device)
    test_metrics = classification_metrics(
        test_labels,
        test_probabilities,
        threshold=threshold,
    )
    test_metrics["threshold"] = threshold
    test_predictions = (test_probabilities >= threshold).astype(np.int64)

    with (fold_dir / "predictions.csv").open("w", newline="") as prediction_file:
        writer = csv.writer(prediction_file)
        writer.writerow(
            ["subject_id", "label", "probability", "threshold", "prediction"]
        )
        for index, label, probability, prediction in zip(
            test_indices,
            test_labels.astype(np.int64),
            test_probabilities,
            test_predictions,
        ):
            writer.writerow(
                [
                    subject_ids[index],
                    label,
                    float(probability),
                    threshold,
                    int(prediction),
                ]
            )

    return test_metrics, test_probabilities, test_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bec_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_path", default="downstream_bec/splits/brainnetcnn_5fold_seed42.json")
    parser.add_argument("--cv_seed", type=int, default=42)
    parser.add_argument("--classifier_seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--e2e1_channels", type=int, default=4)
    parser.add_argument("--e2e2_channels", type=int, default=8)
    parser.add_argument("--e2n_channels", type=int, default=16)
    parser.add_argument("--n2g_channels", type=int, default=16)
    parser.add_argument("--fc_channels", type=int, default=8)
    parser.add_argument("--gpu_id", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    device = select_device(args.gpu_id)

    data = np.load(args.bec_path, allow_pickle=False)
    bec = data["bec"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    subject_ids = data["subject_ids"].astype(str)

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.cv_seed,
    )
    split_records = []
    outer_splits = list(outer_cv.split(bec, labels))
    for fold, (development_indices, test_indices) in enumerate(outer_splits, start=1):
        train_indices, val_indices = train_test_split(
            development_indices,
            test_size=args.val_fraction,
            stratify=labels[development_indices],
            random_state=args.cv_seed + fold,
        )
        split_records.append(
            {
                "fold": fold,
                "train_subject_ids": subject_ids[train_indices].tolist(),
                "val_subject_ids": subject_ids[val_indices].tolist(),
                "test_subject_ids": subject_ids[test_indices].tolist(),
            }
        )

    if split_path.exists():
        with split_path.open() as split_file:
            saved_splits = json.load(split_file)
        if saved_splits != split_records:
            raise ValueError(
                f"Existing split file {split_path} does not match current subjects/settings"
            )
    else:
        with split_path.open("w") as split_file:
            json.dump(split_records, split_file, indent=2)

    all_fold_metrics = []
    oof_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    oof_predictions = np.full(len(labels), -1, dtype=np.int64)
    for fold, (development_indices, test_indices) in enumerate(outer_splits, start=1):
        train_indices, val_indices = train_test_split(
            development_indices,
            test_size=args.val_fraction,
            stratify=labels[development_indices],
            random_state=args.cv_seed + fold,
        )
        fold_seed = args.classifier_seed + fold
        set_seed(fold_seed)
        fold_dir = output_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=True)
        metrics, probabilities, predictions = train_fold(
            args,
            bec,
            labels,
            subject_ids,
            train_indices,
            val_indices,
            test_indices,
            fold_dir,
            device,
            fold_seed,
        )
        metrics["fold"] = fold
        all_fold_metrics.append(metrics)
        oof_probabilities[test_indices] = probabilities
        oof_predictions[test_indices] = predictions
        print(f"fold={fold} {metrics}")

    with (output_dir / "fold_metrics.csv").open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["fold", "ACC", "SEN", "SPE", "AUC", "threshold"],
        )
        writer.writeheader()
        writer.writerows(all_fold_metrics)
        values = np.asarray(
            [[row[name] for name in ("ACC", "SEN", "SPE", "AUC")] for row in all_fold_metrics]
        )
        writer.writerow(
            {
                "fold": "mean",
                **dict(zip(("ACC", "SEN", "SPE", "AUC"), values.mean(axis=0))),
                "threshold": np.mean([row["threshold"] for row in all_fold_metrics]),
            }
        )
        writer.writerow(
            {
                "fold": "std",
                **dict(zip(("ACC", "SEN", "SPE", "AUC"), values.std(axis=0))),
                "threshold": np.std([row["threshold"] for row in all_fold_metrics]),
            }
        )

    oof_metrics = classification_metrics_from_predictions(
        labels,
        oof_probabilities,
        oof_predictions,
    )
    with (output_dir / "oof_predictions.csv").open("w", newline="") as oof_file:
        writer = csv.writer(oof_file)
        writer.writerow(["subject_id", "label", "probability", "prediction"])
        for subject_id, label, probability, prediction in zip(
            subject_ids,
            labels,
            oof_probabilities,
            oof_predictions,
        ):
            writer.writerow(
                [subject_id, int(label), float(probability), int(prediction)]
            )
    with (output_dir / "oof_metrics.json").open("w") as metrics_file:
        json.dump(oof_metrics, metrics_file, indent=2)
    print(f"OOF metrics: {oof_metrics}")


if __name__ == "__main__":
    main()
