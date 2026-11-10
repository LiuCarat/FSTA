"""Build rebuttal Table R4 from fold-specific inductive OOF outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


DATASETS = {
    "ABIDE-I": "abide-i",
    "ABIDE-II": "abide-ii",
    "ADHD200": "adhd200",
}
METHOD_COLUMNS = {
    "initial": "original_AUC",
    "mpr": "refined_AUC",
    "full": "qc_refined_AUC",
}
COMPARISONS = {
    "mpr_minus_initial": ("mpr", "initial"),
    "full_minus_mpr": ("full", "mpr"),
    "full_minus_initial": ("full", "initial"),
}


def read_fold_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        raise ValueError(f"{path}: expected 10 folds, found {len(rows)}")
    folds = sorted(int(row["fold"]) for row in rows)
    if folds != list(range(1, 11)):
        raise ValueError(f"{path}: folds must be exactly 1..10, found {folds}")
    return sorted(rows, key=lambda row: int(row["fold"]))


def fold_interval(differences: np.ndarray) -> dict:
    differences = np.asarray(differences, dtype=np.float64)
    mean = float(differences.mean())
    standard_deviation = float(differences.std(ddof=1))
    critical = float(stats.t.ppf(0.975, df=len(differences) - 1))
    half_width = critical * standard_deviation / np.sqrt(len(differences))
    return {
        "mean_delta_auc": mean,
        "mean_delta_percentage_points": 100.0 * mean,
        "ci95_low_percentage_points": 100.0 * (mean - half_width),
        "ci95_high_percentage_points": 100.0 * (mean + half_width),
        "sample_std_percentage_points": 100.0 * standard_deviation,
        "positive_folds": int(np.count_nonzero(differences > 0)),
        "n_folds": int(len(differences)),
        "ci_method": "two-sided paired t interval over 10 same-fold AUC differences, df=9",
    }


def paired_subject_bootstrap(path: Path, samples: int, seed: int) -> dict:
    archive = np.load(path, allow_pickle=False)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    probabilities = {
        "initial": np.asarray(archive["initial_probability"], dtype=np.float64),
        "mpr": np.asarray(archive["mpr_probability"], dtype=np.float64),
        "full": np.asarray(archive["full_probability"], dtype=np.float64),
    }
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if len(class_indices) != 2:
        raise ValueError(f"{path}: bootstrap requires exactly two classes")
    rng = np.random.default_rng(seed)
    observed_aucs = {
        name: roc_auc_score(labels, values)
        for name, values in probabilities.items()
    }
    distributions = {name: np.empty(samples, dtype=np.float64) for name in COMPARISONS}
    for sample in range(samples):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        aucs = {
            name: roc_auc_score(labels[indices], values[indices])
            for name, values in probabilities.items()
        }
        for comparison, (left, right) in COMPARISONS.items():
            distributions[comparison][sample] = aucs[left] - aucs[right]
    return {
        name: {
            "mean_delta_percentage_points": 100.0 * (
                observed_aucs[COMPARISONS[name][0]]
                - observed_aucs[COMPARISONS[name][1]]
            ),
            "bootstrap_distribution_mean_percentage_points": 100.0 * float(values.mean()),
            "ci95_low_percentage_points": 100.0 * float(np.quantile(values, 0.025)),
            "ci95_high_percentage_points": 100.0 * float(np.quantile(values, 0.975)),
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "ci_method": "paired stratified subject bootstrap over the same OOF subjects",
        }
        for name, values in distributions.items()
    }


def format_interval(result: dict) -> str:
    return (
        f"{result['mean_delta_percentage_points']:+.2f} "
        f"[{result['ci95_low_percentage_points']:+.2f}, "
        f"{result['ci95_high_percentage_points']:+.2f}]"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    table_rows = []
    all_fold_rows = []
    fold_ci_rows = []
    bootstrap_rows = []
    for dataset, directory in DATASETS.items():
        dataset_dir = args.outputs_dir / directory
        if args.evaluation_run_name:
            dataset_dir = dataset_dir / "evaluations" / args.evaluation_run_name
        fold_path = dataset_dir / "experiment_summary.csv"
        prediction_path = dataset_dir / "oof_predictions.npz"
        if not fold_path.is_file() or not prediction_path.is_file():
            summaries[dataset] = {
                "status": "missing",
                "fold_path": str(fold_path),
                "prediction_path": str(prediction_path),
            }
            table_rows.append((dataset, None))
            continue
        rows = read_fold_rows(fold_path)
        aucs = {
            method: np.asarray([float(row[column]) for row in rows])
            for method, column in METHOD_COLUMNS.items()
        }
        fold_statistics = {
            name: fold_interval(aucs[left] - aucs[right])
            for name, (left, right) in COMPARISONS.items()
        }
        bootstrap = paired_subject_bootstrap(
            prediction_path, args.bootstrap_samples, args.bootstrap_seed
        )
        dataset_summary = {
            "status": "computed",
            "mean_auc": {
                name: float(values.mean()) for name, values in aucs.items()
            },
            "fold_level_paired_ci": fold_statistics,
            "oof_subject_paired_bootstrap_ci": bootstrap,
            "fold_source": str(fold_path),
            "oof_source": str(prediction_path),
        }
        summaries[dataset] = dataset_summary
        table_rows.append((dataset, dataset_summary))
        for index, row in enumerate(rows):
            all_fold_rows.append({
                "dataset": dataset,
                "fold": int(row["fold"]),
                "initial_auc": aucs["initial"][index],
                "mpr_auc": aucs["mpr"][index],
                "full_auc": aucs["full"][index],
                "mpr_minus_initial_percentage_points": 100.0 * (aucs["mpr"][index] - aucs["initial"][index]),
                "full_minus_mpr_percentage_points": 100.0 * (aucs["full"][index] - aucs["mpr"][index]),
                "full_minus_initial_percentage_points": 100.0 * (aucs["full"][index] - aucs["initial"][index]),
            })
        for comparison, result in fold_statistics.items():
            fold_ci_rows.append({"dataset": dataset, "comparison": comparison, **result})
        for comparison, result in bootstrap.items():
            bootstrap_rows.append({"dataset": dataset, "comparison": comparison, **result})

    lines = [
        "# Table R4. Classification performance and paired AUC differences under fold-specific inductive evaluation",
        "",
        "Main-table intervals are paired t intervals over the ten same-fold AUC differences, matching Table R3.",
        "All values are percentage points; absolute AUCs are ten-fold means.",
        "",
        "| Dataset | Initial → Full AUC | MPR − Initial ΔAUC | Full − MPR ΔAUC | Full − Initial ΔAUC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset, summary in table_rows:
        if summary is None:
            lines.append(f"| {dataset} | Unavailable | Unavailable | Unavailable | Unavailable |")
            continue
        means = summary["mean_auc"]
        intervals = summary["fold_level_paired_ci"]
        lines.append(
            f"| {dataset} | {100 * means['initial']:.2f} → {100 * means['full']:.2f} | "
            f"{format_interval(intervals['mpr_minus_initial'])} | "
            f"{format_interval(intervals['full_minus_mpr'])} | "
            f"{format_interval(intervals['full_minus_initial'])} |"
        )
    lines.extend([
        "",
        "## OOF subject-level paired bootstrap (secondary)",
        "",
        "These intervals resample the same held-out OOF subjects within class and are separate from the fold-level Table R4 intervals.",
        "",
        "| Dataset | MPR − Initial | Full − MPR | Full − Initial |",
        "| --- | --- | --- | --- |",
    ])
    for dataset, summary in table_rows:
        if summary is None:
            lines.append(f"| {dataset} | Unavailable | Unavailable | Unavailable |")
            continue
        bootstrap = summary["oof_subject_paired_bootstrap_ci"]
        lines.append(
            f"| {dataset} | {format_interval(bootstrap['mpr_minus_initial'])} | "
            f"{format_interval(bootstrap['full_minus_mpr'])} | "
            f"{format_interval(bootstrap['full_minus_initial'])} |"
        )
    (args.output_dir / "Table_R4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "Table_R4.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if all_fold_rows:
        write_csv(args.output_dir / "fold_auc_and_differences.csv", all_fold_rows)
    if fold_ci_rows:
        write_csv(args.output_dir / "fold_level_paired_ci.csv", fold_ci_rows)
    if bootstrap_rows:
        write_csv(args.output_dir / "oof_paired_bootstrap_ci.csv", bootstrap_rows)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--outputs-dir", type=Path, default=project_dir / "outputs")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "analysis/results")
    parser.add_argument(
        "--evaluation-run-name",
        help="Read each dataset from outputs/<dataset>/evaluations/RUN_NAME.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    run(args)
    print(f"Wrote {args.output_dir / 'Table_R4.md'}")


if __name__ == "__main__":
    main()
