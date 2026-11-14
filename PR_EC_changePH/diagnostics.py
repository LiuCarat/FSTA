"""Post-retrieval, unweighted diagnosis diagnostics and auditable W6 outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from PR_EC_changePH.rebuttal import DATASET_DIRS

METRICS = {"ACC": "ACC", "AUC": "AUC", "SEN": "Recall", "PRE": "Precision", "F1": "F1"}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_partition(subject_ids, train_index, val_index, test_index):
    ids = np.asarray(subject_ids).astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Subject IDs must be unique")
    partitions = [set(np.asarray(index).tolist()) for index in (train_index, val_index, test_index)]
    if any(partitions[first] & partitions[second] for first, second in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Reference train, validation and test indices must be disjoint")
    if set.union(*partitions) != set(range(len(ids))):
        raise ValueError("Fold partition does not cover exactly the EC subjects")


def neighbor_consistency_rows(args, fold, subject_ids, labels, train_index, test_index, neighbor_indices):
    """Labels enter only here, after final PH+FC top-k indices are fixed."""
    ids = np.asarray(subject_ids).astype(str)
    labels = np.asarray(labels)
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    train_index, test_index = np.asarray(train_index), np.asarray(test_index)
    if set(train_index.tolist()) & set(test_index.tolist()):
        raise ValueError("Test subjects cannot be reference subjects")
    expected_k = min(args.reference_k, len(train_index))
    if neighbors.shape != (len(test_index), expected_k):
        raise ValueError("Unexpected final top-k shape")
    if np.any(neighbors < 0) or np.any(neighbors >= len(train_index)):
        raise ValueError("Neighbor index is outside the actual reference train set")
    if any(len(set(row)) != expected_k for row in neighbors.tolist()):
        raise ValueError("Duplicate neighbor indices")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Diagnosis diagnostics require the existing binary labels")
    reference_labels = labels[train_index]
    rows = []
    for subject_index, local_neighbors in zip(test_index, neighbors):
        selected = train_index[local_neighbors]
        label = int(labels[subject_index])
        same_count = int(np.sum(labels[selected] == label))
        base_count = int(np.sum(reference_labels == label))
        rows.append({
            "dataset": DATASET_DIRS[args.dataset],
            "phenotype_feature_set": args.phenotype_feature_set,
            "seed": args.seed, "fold": fold, "subject_id": ids[subject_index],
            "true_label": label,
            "diagnosis": ("ADHD" if args.dataset == "adhd200" else "ASD") if label else ("Control" if args.dataset == "adhd200" else "TC"),
            "same_diagnosis_count": same_count, "K": expected_k,
            "same_diagnosis_pct": 100.0 * same_count / expected_k,
            "reference_train_n": len(train_index), "reference_same_diagnosis_n": base_count,
            "base_rate_pct": 100.0 * base_count / len(train_index),
            "neighbor_subject_ids": json.dumps(ids[selected].tolist()),
        })
    return rows


def save_neighbor_diagnostics(args, rows):
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("OOF diagnosis rows must contain each test subject exactly once")
    write_csv(args.output_dir / "neighbor_consistency_subjects.csv", rows)
    summaries = []
    for fold in ["all", *sorted({row["fold"] for row in rows})]:
        members = rows if fold == "all" else [row for row in rows if row["fold"] == fold]
        summary = {
            "dataset": DATASET_DIRS[args.dataset],
            "phenotype_feature_set": args.phenotype_feature_set,
            "seed": args.seed, "fold": fold, "n_subjects": len(members),
            "same_diagnosis_pct": float(np.mean([row["same_diagnosis_pct"] for row in members])),
            "base_rate_pct": float(np.mean([row["base_rate_pct"] for row in members])),
        }
        summaries.append(summary)
    write_csv(args.output_dir / "neighbor_consistency_summary.csv", summaries[:1])
    write_csv(args.output_dir / "neighbor_consistency_folds.csv", summaries[1:])


def save_classification(args, fold_results):
    rows = []
    for fold, result in enumerate(fold_results, 1):
        if "qc_refined" not in result:
            continue
        rows.append({
            "dataset": DATASET_DIRS[args.dataset],
            "phenotype_feature_set": args.phenotype_feature_set,
            "seed": args.seed, "fold": fold, "representation": "qc_refined",
            **{f"{name}_pct": 100.0 * result["qc_refined"][source] for name, source in METRICS.items()},
        })
    if not rows:
        return
    summary = {key: rows[0][key] for key in ("dataset", "phenotype_feature_set", "seed", "representation")}
    summary.update(n_folds=len(rows), std_ddof=0)
    for metric in METRICS:
        values = [row[f"{metric}_pct"] for row in rows]
        mean, std = float(np.mean(values)), float(np.std(values, ddof=0))
        summary.update({f"{metric}_mean_pct": mean, f"{metric}_std_pct": std,
                        f"{metric}_display": f"{mean:.2f} ± {std:.2f}"})
    write_csv(args.output_dir / "classification_folds.csv", rows)
    write_csv(args.output_dir / "classification_summary.csv", [summary])


def save_run_manifest(args, data):
    """Prevent mixing subjects, settings or protocol in an existing result folder."""
    excluded = {"diagnostics_only", "representations", "gpu_id", "output_dir", "refined_ec_path", "qsr_refined_ec_path"}
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in excluded and key != "profile"}
    payload = json.dumps({
        "subject_ids": np.asarray(data["subject_ids"]).astype(str).tolist(),
        "labels": np.asarray(data["labels"]).astype(int).tolist(),
    }, sort_keys=True)
    manifest = {"config": config, "subjects_labels_sha256": hashlib.sha256(payload.encode()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "run_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output directory already contains a different W6 protocol/cohort; use a fresh directory")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def split_membership_rows(args, fold, data, train_index, val_index, test_index):
    validate_partition(data["subject_ids"], train_index, val_index, test_index)
    return [{
        "dataset": DATASET_DIRS[args.dataset], "seed": args.seed, "fold": fold,
        "archive_index": int(index), "subject_id": str(data["subject_ids"][index]), "role": role,
    } for role, indices in (("train", train_index), ("val", val_index), ("test", test_index)) for index in indices]
