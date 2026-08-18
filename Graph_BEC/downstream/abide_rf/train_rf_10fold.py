import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


PRIMARY_METRICS = ("precision", "recall", "f1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="10-fold random forest classification using subject BEC matrices"
    )
    parser.add_argument(
        "--bec_path",
        default="/data/users/liulin/PythonCode/ST-MRI/FSTA/Graph_BEC/outputs/entropy/loss_alpha_0.01/seed_42/epochs_101/subject_bec.npz",
        help="FSTA_BEC.py 或 Phenotype_FSTA_BEC.py 生成的 subject_bec.npz",
    )
    parser.add_argument(
        "--output_dir",
        default="Graph_BEC/outputs/random_forest_10fold/entropy/loss_alpha_0.01/seed_42/epochs_101",
        help="分类指标输出目录",
    )
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def calculate_metrics(labels, predictions):
    return {
        "precision": precision_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def summarize(rows):
    values = np.asarray(
        [[row[name] for name in PRIMARY_METRICS] for row in rows],
        dtype=np.float64,
    )
    return values.mean(axis=0), values.std(axis=0, ddof=0)


def format_percent(value):
    return f"{value * 100:.2f}"


def format_mean_std(mean, std):
    return f"{mean * 100:.2f}±{std * 100:.2f}"


def load_bec_data(bec_path):
    data = np.load(bec_path, allow_pickle=False)
    missing_keys = {"bec", "labels"} - set(data.files)
    if missing_keys:
        raise ValueError(f"Missing arrays in {bec_path}: {sorted(missing_keys)}")

    bec = data["bec"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC shape [subjects, nodes, nodes], got {bec.shape}")
    if len(bec) != len(labels):
        raise ValueError("BEC and labels have different lengths")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, got {np.unique(labels)}")

    directed_mask = ~np.eye(bec.shape[-1], dtype=bool)
    features = bec[:, directed_mask]
    metadata = {
        "subjects": len(labels),
        "nodes": bec.shape[-1],
        "features": features.shape[1],
        "class_counts": {
            str(label): int(np.sum(labels == label))
            for label in np.unique(labels)
        },
    }
    if "phenotype_gate" in data.files:
        metadata["phenotype_gate"] = float(data["phenotype_gate"])
    if "phenotype_columns" in data.files:
        metadata["phenotype_columns"] = data["phenotype_columns"].tolist()
    return features, labels, metadata


def write_results(output_dir, fold_rows, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fold_metrics.csv").open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["fold", *PRIMARY_METRICS],
        )
        writer.writeheader()
        writer.writerows(
            {
                "fold": row["fold"],
                **{
                    name: format_percent(row[name])
                    for name in PRIMARY_METRICS
                },
            }
            for row in fold_rows
        )
        writer.writerow(
            {
                "fold": "mean±std",
                "precision": format_mean_std(
                    summary["precision_mean"],
                    summary["precision_std"],
                ),
                "recall": format_mean_std(
                    summary["recall_mean"],
                    summary["recall_std"],
                ),
                "f1": format_mean_std(
                    summary["f1_mean"],
                    summary["f1_std"],
                ),
            }
        )

    json_summary = {
        key: value
        for key, value in summary.items()
        if not key.endswith("_mean") and not key.endswith("_std")
    }
    for metric in PRIMARY_METRICS:
        json_summary[f"{metric}_mean_percent"] = format_percent(
            summary[f"{metric}_mean"]
        )
        json_summary[f"{metric}_std_percent"] = format_percent(
            summary[f"{metric}_std"]
        )
    with (output_dir / "summary.json").open("w") as summary_file:
        json.dump(json_summary, summary_file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    bec_path = Path(args.bec_path)
    output_dir = Path(args.output_dir)
    if not bec_path.is_file():
        raise FileNotFoundError(f"BEC file not found: {bec_path}")

    features, labels, bec_metadata = load_bec_data(bec_path)
    print(
        f"Loaded BEC: subjects={len(labels)}, nodes={bec_metadata['nodes']}, "
        f"directed_features={features.shape[1]}"
    )
    if "phenotype_gate" in bec_metadata:
        print(
            "Phenotype-conditioned BEC: "
            f"gate={bec_metadata['phenotype_gate']:.6f}, "
            f"columns={bec_metadata.get('phenotype_columns', [])}"
        )
    print("Metric average=weighted")

    splitter = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    fold_rows = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(features, labels),
        start=1,
    ):
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.seed + fold,
        )
        classifier.fit(features[train_indices], labels[train_indices])
        predictions = classifier.predict(features[test_indices])
        metrics = calculate_metrics(labels[test_indices], predictions)
        fold_rows.append({"fold": fold, **metrics})
        print(
            f"fold={fold:02d} "
            f"weighted_precision={metrics['precision'] * 100:.2f} "
            f"weighted_recall={metrics['recall'] * 100:.2f} "
            f"weighted_F1={metrics['f1'] * 100:.2f}"
        )

    means, stds = summarize(fold_rows)
    summary = {
        "metric_average": "weighted",
        "bec_path": str(bec_path),
        "n_splits": args.n_splits,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        **bec_metadata,
        **{
            f"{metric}_mean": float(mean)
            for metric, mean in zip(PRIMARY_METRICS, means)
        },
        **{
            f"{metric}_std": float(std)
            for metric, std in zip(PRIMARY_METRICS, stds)
        },
    }
    write_results(output_dir, fold_rows, summary)
    print(
        f"Precision={format_mean_std(means[0], stds[0])} "
        f"Recall={format_mean_std(means[1], stds[1])} "
        f"F1={format_mean_std(means[2], stds[2])}"
    )


if __name__ == "__main__":
    main()
