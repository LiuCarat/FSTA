import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from downstream_bec.evaluation.metrics import (
    classification_metrics,
    classification_metrics_from_predictions,
    select_youden_threshold,
)


def load_split_indices(split_path, subject_ids):
    with Path(split_path).open() as split_file:
        split_records = json.load(split_file)
    id_to_index = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    splits = []
    for record in split_records:
        splits.append(
            {
                "fold": record["fold"],
                "train": np.asarray(
                    [id_to_index[subject_id] for subject_id in record["train_subject_ids"]]
                ),
                "val": np.asarray(
                    [id_to_index[subject_id] for subject_id in record["val_subject_ids"]]
                ),
                "test": np.asarray(
                    [id_to_index[subject_id] for subject_id in record["test_subject_ids"]]
                ),
            }
        )
    return splits


def build_model(model_name, seed):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=0.01,
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def prepare_features(model_name, train_features, val_features, test_features):
    if model_name != "logistic_regression":
        return train_features, val_features, test_features
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_features),
        scaler.transform(val_features),
        scaler.transform(test_features),
    )


def run_model(model_name, features, labels, subject_ids, splits, output_dir, seed):
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    oof_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    oof_predictions = np.full(len(labels), -1, dtype=np.int64)

    for split in splits:
        train_indices = split["train"]
        val_indices = split["val"]
        test_indices = split["test"]
        train_features, val_features, test_features = prepare_features(
            model_name,
            features[train_indices],
            features[val_indices],
            features[test_indices],
        )
        model = build_model(model_name, seed + split["fold"])
        model.fit(train_features, labels[train_indices])
        val_probabilities = model.predict_proba(val_features)[:, 1]
        threshold = select_youden_threshold(labels[val_indices], val_probabilities)
        test_probabilities = model.predict_proba(test_features)[:, 1]
        test_predictions = (test_probabilities >= threshold).astype(np.int64)
        metrics = classification_metrics(
            labels[test_indices],
            test_probabilities,
            threshold=threshold,
        )
        metrics.update({"fold": split["fold"], "threshold": threshold})
        fold_rows.append(metrics)
        oof_probabilities[test_indices] = test_probabilities
        oof_predictions[test_indices] = test_predictions

        fold_dir = model_dir / f"fold_{split['fold']:02d}"
        fold_dir.mkdir(exist_ok=True)
        with (fold_dir / "predictions.csv").open("w", newline="") as prediction_file:
            writer = csv.writer(prediction_file)
            writer.writerow(
                ["subject_id", "label", "probability", "threshold", "prediction"]
            )
            for index, probability, prediction in zip(
                test_indices,
                test_probabilities,
                test_predictions,
            ):
                writer.writerow(
                    [
                        subject_ids[index],
                        int(labels[index]),
                        float(probability),
                        threshold,
                        int(prediction),
                    ]
                )

    metric_names = ("ACC", "SEN", "SPE", "AUC")
    values = np.asarray([[row[name] for name in metric_names] for row in fold_rows])
    thresholds = np.asarray([row["threshold"] for row in fold_rows])
    with (model_dir / "fold_metrics.csv").open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["fold", *metric_names, "threshold"],
        )
        writer.writeheader()
        writer.writerows(fold_rows)
        writer.writerow(
            {
                "fold": "mean",
                **dict(zip(metric_names, values.mean(axis=0))),
                "threshold": thresholds.mean(),
            }
        )
        writer.writerow(
            {
                "fold": "std",
                **dict(zip(metric_names, values.std(axis=0))),
                "threshold": thresholds.std(),
            }
        )

    oof_metrics = classification_metrics_from_predictions(
        labels,
        oof_probabilities,
        oof_predictions,
    )
    with (model_dir / "oof_predictions.csv").open("w", newline="") as oof_file:
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
    with (model_dir / "oof_metrics.json").open("w") as metrics_file:
        json.dump(oof_metrics, metrics_file, indent=2)
    print(f"{model_name}: {oof_metrics}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bec_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--split_path",
        default="downstream_bec/splits/brainnetcnn_5fold_seed42.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.bec_path, allow_pickle=False)
    bec = data["bec"].astype(np.float64)
    labels = data["labels"].astype(np.int64)
    subject_ids = data["subject_ids"].astype(str)
    directed_mask = ~np.eye(bec.shape[-1], dtype=bool)
    features = bec[:, directed_mask]
    splits = load_split_indices(args.split_path, subject_ids)

    for model_name in ("logistic_regression", "random_forest"):
        run_model(
            model_name,
            features,
            labels,
            subject_ids,
            splits,
            output_dir,
            args.seed,
        )


if __name__ == "__main__":
    main()
