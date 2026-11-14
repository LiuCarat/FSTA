"""Paired ten-fold AUC deltas and confidence intervals for rebuttal Table R3.

This script is read-only with respect to model outputs. It consumes the existing
experiment_summary.csv files and never retrains a model or changes a split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


DATASETS = {
    "ABIDE-I": "abide-i",
    "ABIDE-II": "abide-ii",
    "ADHD200": "adhd200",
}
COMPARISONS = {
    "mpr_minus_initial": ("refined_AUC", "original_AUC"),
    "qc_minus_mpr": ("qc_refined_AUC", "refined_AUC"),
    "qc_minus_initial": ("qc_refined_AUC", "original_AUC"),
}


def read_summary(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        raise ValueError(f"{path}: expected exactly 10 folds, found {len(rows)}")
    try:
        folds = [int(row["fold"]) for row in rows]
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid fold column") from error
    if sorted(folds) != list(range(1, 11)):
        raise ValueError(f"{path}: folds must be exactly 1..10, found {folds}")
    required = {column for pair in COMPARISONS.values() for column in pair}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing AUC columns {sorted(missing)}")
    return rows


def paired_statistics(rows: list[dict[str, float]], left: str, right: str) -> tuple[np.ndarray, dict]:
    differences = np.asarray(
        [float(row[left]) - float(row[right]) for row in rows], dtype=np.float64
    )
    if not np.isfinite(differences).all():
        raise ValueError(f"Non-finite AUC values in {left} - {right}")
    mean = float(differences.mean())
    std = float(differences.std(ddof=1))
    standard_error = std / np.sqrt(len(differences))
    critical = float(stats.t.ppf(0.975, df=len(differences) - 1))
    ci_low, ci_high = mean - critical * standard_error, mean + critical * standard_error
    return differences, {
        "mean_delta_auc": mean,
        "mean_delta_percentage_points": 100.0 * mean,
        "sample_std_auc": std,
        "sample_std_percentage_points": 100.0 * std,
        "ci95_low_percentage_points": 100.0 * ci_low,
        "ci95_high_percentage_points": 100.0 * ci_high,
        "positive_folds": int(np.count_nonzero(differences > 0)),
        "nonnegative_folds": int(np.count_nonzero(differences >= 0)),
        "n_folds": len(differences),
        "ci_method": "two-sided paired t interval over 10 fold-level differences, df=9",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(statistics: dict) -> str:
    if statistics is None:
        return "Unavailable"
    return (
        f"{statistics['mean_delta_percentage_points']:+.2f} "
        f"[{statistics['ci95_low_percentage_points']:+.2f}, "
        f"{statistics['ci95_high_percentage_points']:+.2f}], "
        f"{statistics['positive_folds']}/{statistics['n_folds']}"
    )


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    markdown_rows = []
    for display_name, directory_name in DATASETS.items():
        source = args.outputs_dir / directory_name / "experiment_summary.csv"
        if not source.is_file():
            all_summaries[display_name] = {"status": "missing", "path": str(source)}
            markdown_rows.append((display_name, None, None, None))
            continue
        rows = read_summary(source)
        dataset_summary = {"status": "computed", "source": str(source), "comparisons": {}}
        detail_rows = []
        for index, row in enumerate(rows):
            detail = {
                "fold": int(row["fold"]),
                "original_AUC": float(row["original_AUC"]),
                "refined_AUC": float(row["refined_AUC"]),
                "qc_refined_AUC": float(row["qc_refined_AUC"]),
            }
            for name, (left, right) in COMPARISONS.items():
                detail[f"{name}_auc"] = float(row[left]) - float(row[right])
                detail[f"{name}_percentage_points"] = 100.0 * detail[f"{name}_auc"]
            detail_rows.append(detail)
        for name, (left, right) in COMPARISONS.items():
            differences, statistics = paired_statistics(rows, left, right)
            dataset_summary["comparisons"][name] = statistics
        all_summaries[display_name] = dataset_summary
        write_csv(args.output_dir / f"{directory_name}_fold_differences.csv", detail_rows)
        markdown_rows.append(
            (
                display_name,
                dataset_summary["comparisons"]["mpr_minus_initial"],
                dataset_summary["comparisons"]["qc_minus_mpr"],
                dataset_summary["comparisons"]["qc_minus_initial"],
            )
        )

    lines = [
        "# Table R3. Paired AUC differences across ten folds",
        "",
        "Values are AUC percentage-point change, 10-fold mean delta [95% paired t CI], positive folds/10.",
        "The CI is computed from the ten same-fold paired differences, not from two independent method summaries.",
        "Fold training sets overlap; these fold-level CIs are stability summaries, not independent-sample significance tests.",
        "`QC-refined − MPR-only` compares two independently configured reference-based branches; it is not a strict isolated QSR causal contribution.",
        "",
        "| Dataset | MPR − Initial | QC-refined − MPR-only | QC-refined − Initial |",
        "| --- | --- | --- | --- |",
    ]
    for dataset, first, second, third in markdown_rows:
        lines.append(f"| {dataset} | {fmt(first)} | {fmt(second)} | {fmt(third)} |")
    lines.extend([
        "",
        "Format: `ΔAUC percentage points [95% CI], positive folds/10`.",
        "A positive fold means the left-hand representation has larger test AUC in that same fold.",
        "Statistics use sample SD (`ddof=1`) and a two-sided t critical value with df=9.",
        "",
        "## Missing outputs",
    ])
    for dataset, summary in all_summaries.items():
        if summary["status"] == "missing":
            lines.append(f"- {dataset}: missing `{summary['path']}`; no result was fabricated.")
    (args.output_dir / "Table_R3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "Table_R3.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return all_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--outputs-dir", type=Path, default=root / "outputs")
    parser.add_argument("--output-dir", type=Path, default=root / "analysis/rebuttal_stats/results")
    args = parser.parse_args()
    run(args)
    print(f"Wrote {args.output_dir / 'Table_R3.md'}")
    print(f"Wrote {args.output_dir / 'Table_R3.json'}")


if __name__ == "__main__":
    main()
