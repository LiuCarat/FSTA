"""Writers for QSR Refinement archives and experiment summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def save_pr_ec_archive(output_path, data, qc_refined_ec, fold_ids):
    """Save the held-out QSR Refinement results as a standalone archive."""
    qc_refined_ec = np.asarray(qc_refined_ec, dtype=np.float32)
    input_ec = np.asarray(data["ec"], dtype=np.float32)
    if qc_refined_ec.shape != input_ec.shape:
        raise ValueError(
            f"QSR Refinement shape {qc_refined_ec.shape} does not match "
            f"input shape {input_ec.shape}"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ec=qc_refined_ec,
        qc_refined_ec=qc_refined_ec,
        labels=np.asarray(data["labels"], dtype=np.int64),
        subject_ids=np.asarray(data["subject_ids"]).astype(str),
        site_ids=np.asarray(data["site_ids"]).astype(str),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        roi_names=np.asarray(
            data.get(
                "roi_names",
                [f"ROI_{index + 1:03d}" for index in range(qc_refined_ec.shape[1])],
            )
        ).astype(str),
        representation=np.asarray("PR-EC"),
    )
    return output_path


def save_results(args, fold_results, training_metrics):
    rows = []
    for fold, result in enumerate(fold_results, 1):
        row = {"fold": fold}
        names = [name for name, value in result.items() if isinstance(value, dict)]
        for name in names:
            row.update({f"{name}_{key}": value for key, value in result[name].items()})
        row.update({key: value for key, value in result.items() if key not in names})
        rows.append(row)

    summary = {
        "config": vars(args),
        "individual_ec_training": training_metrics,
        "folds": rows,
    }
    names = [
        name for name, value in fold_results[0].items() if isinstance(value, dict)
    ]
    for name in names:
        for metric in ("ACC", "Recall", "AUC", "Precision", "F1"):
            values = [row[f"{name}_{metric}"] for row in rows]
            mean = float(np.mean(values))
            std = float(np.std(values))
            summary[f"{name}_{metric}_mean"] = mean
            summary[f"{name}_{metric}_std"] = std
            summary[f"{name}_{metric}_display"] = f"{100 * mean:.2f}±{100 * std:.2f}"
    summary["representations"] = names

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.output_dir.iterdir():
        if (
            path.is_file()
            and path.suffix != ".npz"
            and path.name not in {"experiment_summary.csv", "summary.json"}
        ):
            path.unlink()
    with (args.output_dir / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


def print_summary_table(summary, title="mean±std (%)"):
    metrics = ("ACC", "Recall", "AUC", "Precision", "F1")
    print(f"\n{title}")
    display_names = {"Recall": "SEN", "Precision": "Precision"}
    print(
        "representation | "
        + " | ".join(display_names.get(name, name) for name in metrics)
    )
    for name in summary["representations"]:
        values = [summary[f"{name}_{metric}_display"] for metric in metrics]
        print(f"{name:20s} | " + " | ".join(values))
