"""Join completed W6 outputs; optionally reuse original seed-42 classification folds."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LOCAL_ROOT.parent
DATASETS = ("abide-i", "abide-ii", "adhd200")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def checked_single_row(path):
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one summary row: {path}")
    return rows[0]


def classification_display(directory, original_directory=None):
    summary_path = directory / "classification_summary.csv"
    if summary_path.exists():
        row = checked_single_row(summary_path)
        if int(row["n_folds"]) != 10 or int(row["seed"]) != 42 or row["representation"] != "qc_refined":
            raise ValueError(f"Expected 10-fold seed42 Full PR-EC: {summary_path}")
        return {metric: row[f"{metric}_display"] for metric in ("ACC", "AUC")}, str(summary_path)
    if original_directory is None:
        raise FileNotFoundError(summary_path)
    original_config = json.loads((original_directory / "summary.json").read_text())["config"]
    if int(original_config["seed"]) != 42 or int(original_config["n_splits"]) != 10:
        raise ValueError("Original classification results must use seed42 and ten folds")
    path = original_directory / "experiment_summary.csv"
    rows = read_rows(path)
    if len(rows) != 10 or {int(row["fold"]) for row in rows} != set(range(1, 11)):
        raise ValueError(f"Original results must contain ten unique folds: {path}")
    result = {}
    for metric in ("ACC", "AUC"):
        values = [100.0 * float(row[f"qc_refined_{metric}"]) for row in rows]
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
            raise ValueError(f"Invalid original {metric}: {path}")
        result[metric] = f"{statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}"
    return result, str(path)


def validate_oof_pair(original_dir, alternate_dir):
    manifests = [json.loads((directory / "run_manifest.json").read_text())
                 for directory in (original_dir, alternate_dir)]
    allowed = {"phenotype_feature_set", "continuous_weights"}
    configs = [{key: value for key, value in manifest["config"].items() if key not in allowed}
               for manifest in manifests]
    if configs[0] != configs[1] or manifests[0]["subjects_labels_sha256"] != manifests[1]["subjects_labels_sha256"]:
        raise ValueError("Non-PH settings or cohort differ between configurations")
    original_splits = read_rows(original_dir / "split_membership.csv")
    alternate_splits = read_rows(alternate_dir / "split_membership.csv")
    if original_splits != alternate_splits:
        raise ValueError("Original and Age/Sex/Site do not have identical subject order and train/val/test splits")
    subject_rows = []
    for directory in (original_dir, alternate_dir):
        rows = read_rows(directory / "neighbor_consistency_subjects.csv")
        if len({row["subject_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate OOF subjects: {directory}")
        subject_rows.append([(row["subject_id"], row["true_label"], row["fold"], row["seed"]) for row in rows])
    if subject_rows[0] != subject_rows[1]:
        raise ValueError("OOF subject IDs, labels, fold IDs or seeds differ across PH settings")
    membership = {}
    for row in original_splits:
        if int(row["seed"]) != 42:
            raise ValueError("Table R6 requires seed=42")
        membership.setdefault((row["fold"], row["role"]), set()).add(row["subject_id"])
    if {int(row["fold"]) for row in original_splits} != set(range(1, 11)):
        raise ValueError("Expected all ten folds")
    for fold in range(1, 11):
        train = membership[(str(fold), "train")]
        val = membership[(str(fold), "val")]
        test = membership[(str(fold), "test")]
        if train & val or train & test or val & test:
            raise ValueError("Split membership overlap")
    for directory in (original_dir, alternate_dir):
        test_ids = {(row["fold"], row["subject_id"]) for row in original_splits if row["role"] == "test"}
        rows = read_rows(directory / "neighbor_consistency_subjects.csv")
        observed = {(row["fold"], row["subject_id"]) for row in rows}
        if observed != test_ids:
            raise ValueError("Incomplete OOF diagnostics")
        labels = {row["subject_id"]: int(row["true_label"]) for row in rows}
        for row in rows:
            train = membership[(row["fold"], "train")]
            neighbors = json.loads(row["neighbor_subject_ids"])
            count = min(20, len(train))
            if len(neighbors) != count or len(set(neighbors)) != count or not set(neighbors).issubset(train):
                raise ValueError("Neighbors are not the expected unique train-only Top-K subjects")
            label = int(row["true_label"])
            same_count = sum(labels[subject] == label for subject in neighbors)
            base_count = sum(labels[subject] == label for subject in train)
            if (int(row["K"]) != count or int(row["same_diagnosis_count"]) != same_count
                    or int(row["reference_train_n"]) != len(train)
                    or int(row["reference_same_diagnosis_n"]) != base_count
                    or not math.isclose(float(row["same_diagnosis_pct"]), 100 * same_count / count, abs_tol=1e-8)
                    or not math.isclose(float(row["base_rate_pct"]), 100 * base_count / len(train), abs_tol=1e-8)):
                raise ValueError("Stored neighbor counts/proportions do not match the raw subject records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LOCAL_ROOT / "outputs/rebuttal_w6")
    parser.add_argument("--original-results-root", type=Path, default=REPO_ROOT / "PR_EC/outputs/seed42",
                        help="Original paper's seed42 result folders; read only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.root / "table_r6.csv").resolve()
    if not output.is_relative_to(LOCAL_ROOT):
        parser.error("Table output must stay within PR_EC_changePH")
    rows = []
    for dataset in DATASETS:
        validate_oof_pair(args.root / dataset / "original", args.root / dataset / "age_sex_site")
        for setting in ("original", "age_sex_site"):
            directory = args.root / dataset / setting
            neighbors = checked_single_row(directory / "neighbor_consistency_summary.csv")
            if neighbors["dataset"] != dataset or neighbors["phenotype_feature_set"] != setting or int(neighbors["seed"]) != 42:
                raise ValueError(f"Mismatched summary metadata: {directory}")
            subjects = read_rows(directory / "neighbor_consistency_subjects.csv")
            if int(neighbors["n_subjects"]) != len(subjects):
                raise ValueError("Neighbor summary subject count does not match OOF records")
            for metric in ("same_diagnosis_pct", "base_rate_pct"):
                if not math.isclose(float(neighbors[metric]), statistics.mean(float(row[metric]) for row in subjects), abs_tol=1e-8):
                    raise ValueError("Neighbor summary is not the subject-weighted OOF mean")
            classification, source = classification_display(
                directory, args.original_results_root / dataset if setting == "original" else None,
            )
            rows.append({
                "Dataset": dataset, "PH setting": "Original PH" if setting == "original" else "Age/Sex/Site PH",
                "Same-diagnosis neighbors (%)": f"{float(neighbors['same_diagnosis_pct']):.2f}",
                "Base rate (%)": f"{float(neighbors['base_rate_pct']):.2f}",
                "ACC (%)": classification["ACC"], "AUC (%)": classification["AUC"],
                "classification_source": source,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
