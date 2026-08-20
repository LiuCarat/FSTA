"""Writers for refined BEC archives and experiment summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def save_refined_bec_archive(
    output_path, data, pgr_bec, qc_refined_bec, fold_ids, source_bec_path
):
    """Save aligned test-only OOF PGR and QC-refined BEC matrices."""
    pgr_bec = np.asarray(pgr_bec, dtype=np.float32)
    qc_refined_bec = np.asarray(qc_refined_bec, dtype=np.float32)
    if pgr_bec.shape != np.asarray(data["bec"]).shape:
        raise ValueError(
            f"OOF PGR-BEC shape {pgr_bec.shape} does not match input "
            f"shape {np.asarray(data['bec']).shape}"
        )
    if qc_refined_bec.shape != pgr_bec.shape:
        raise ValueError("PGR-BEC and QC-refined BEC shapes do not match")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        bec=pgr_bec,
        pgr_bec=pgr_bec,
        refined_bec=pgr_bec,
        qc_refined_bec=qc_refined_bec,
        original_bec=np.asarray(data["bec"], dtype=np.float32),
        labels=np.asarray(data["labels"], dtype=np.int64),
        subject_ids=np.asarray(data["subject_ids"]).astype(str),
        site_ids=np.asarray(data["site_ids"]).astype(str),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        roi_names=np.asarray(
            data.get(
                "roi_names",
                [f"ROI_{index + 1:03d}" for index in range(pgr_bec.shape[1])],
            )
        ).astype(str),
        representation=np.asarray("pgr_and_qc_refined"),
        source_bec_path=np.asarray(str(Path(source_bec_path).resolve())),
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
        "fsta_training": training_metrics,
        "folds": rows,
    }
    names = [
        name for name, value in fold_results[0].items() if isinstance(value, dict)
    ]
    for name in names:
        for metric in ("ACC", "SPE", "AUC", "Precision", "Recall", "F1"):
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
    metrics = ("ACC", "SPE", "AUC", "Precision", "Recall", "F1")
    print(f"\n{title}")
    print("representation | " + " | ".join(metrics))
    for name in summary["representations"]:
        values = [summary[f"{name}_{metric}_display"] for metric in metrics]
        print(f"{name:20s} | " + " | ".join(values))
