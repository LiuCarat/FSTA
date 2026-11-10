"""Writers for refined EC archives and experiment summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def save_refined_ec_archive(
    output_path, data, pgr_ec, qc_refined_ec, fold_ids, source_ec_path
):
    """Save aligned test-only OOF PGR and QC-refined EC matrices."""
    pgr_ec = np.asarray(pgr_ec, dtype=np.float32)
    qc_refined_ec = np.asarray(qc_refined_ec, dtype=np.float32)
    if pgr_ec.shape != np.asarray(data["ec"]).shape:
        raise ValueError(
            f"OOF PGR-EC shape {pgr_ec.shape} does not match input "
            f"shape {np.asarray(data['ec']).shape}"
        )
    if qc_refined_ec.shape != pgr_ec.shape:
        raise ValueError("PGR-EC and QC-refined EC shapes do not match")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ec=pgr_ec,
        pgr_ec=pgr_ec,
        refined_ec=pgr_ec,
        qc_refined_ec=qc_refined_ec,
        original_ec=np.asarray(data["ec"], dtype=np.float32),
        labels=np.asarray(data["labels"], dtype=np.int64),
        subject_ids=np.asarray(data["subject_ids"]).astype(str),
        site_ids=np.asarray(data["site_ids"]).astype(str),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        roi_names=np.asarray(
            data.get(
                "roi_names",
                [f"ROI_{index + 1:03d}" for index in range(pgr_ec.shape[1])],
            )
        ).astype(str),
        representation=np.asarray("pgr_and_qc_refined"),
        source_ec_path=np.asarray(str(Path(source_ec_path).resolve())),
    )
    return output_path


def save_qsr_ec_archive(
    output_path, data, qc_refined_ec, fold_ids, source_ec_path
):
    """Save the held-out QSR-refined ECs as a standalone archive."""
    qc_refined_ec = np.asarray(qc_refined_ec, dtype=np.float32)
    original_ec = np.asarray(data["ec"], dtype=np.float32)
    if qc_refined_ec.shape != original_ec.shape:
        raise ValueError(
            f"QSR-refined EC shape {qc_refined_ec.shape} does not match "
            f"input shape {original_ec.shape}"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ec=qc_refined_ec,
        refined_ec=qc_refined_ec,
        qc_refined_ec=qc_refined_ec,
        original_ec=original_ec,
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
        representation=np.asarray("qsr_refined"),
        source_ec_path=np.asarray(str(Path(source_ec_path).resolve())),
    )
    return output_path


def save_results(
    args,
    fold_results,
    training_metrics,
    csv_name="experiment_summary.csv",
    json_name="summary.json",
):
    rows = []
    for fallback_fold, result in enumerate(fold_results, 1):
        row = {"fold": int(result.get("fold_id", fallback_fold))}
        names = [name for name, value in result.items() if isinstance(value, dict)]
        for name in names:
            row.update({f"{name}_{key}": value for key, value in result[name].items()})
        row.update({
            key: value
            for key, value in result.items()
            if key not in names and key != "fold_id"
        })
        rows.append(row)

    summary = {
        "config": vars(args),
        "stf_ec_training": training_metrics,
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
    with (args.output_dir / csv_name).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / json_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


def save_inductive_ec_archive(output_path, data, ec, fold_ids, representation):
    ec = np.asarray(ec, dtype=np.float32)
    selected = np.asarray(fold_ids) > 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ec=ec[selected],
        labels=np.asarray(data["labels"], dtype=np.int64)[selected],
        subject_ids=np.asarray(data["subject_ids"]).astype(str)[selected],
        site_ids=np.asarray(data["site_ids"]).astype(str)[selected],
        fold_ids=np.asarray(fold_ids, dtype=np.int64)[selected],
        representation=np.asarray(representation),
        protocol=np.asarray("fold-specific-inductive"),
    )
    return output_path


def save_oof_predictions(output_dir, data, fold_ids, probabilities):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = np.asarray(fold_ids) > 0
    rows = []
    for index in np.flatnonzero(selected):
        rows.append({
            "subject_id": str(data["subject_ids"][index]),
            "fold": int(fold_ids[index]),
            "label": int(data["labels"][index]),
            "initial_probability": float(probabilities["original"][index]),
            "mpr_probability": float(probabilities["refined"][index]),
            "full_probability": float(probabilities["qc_refined"][index]),
        })
    write_path = output_dir / "oof_predictions.csv"
    with write_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        subject_ids=np.asarray(data["subject_ids"]).astype(str)[selected],
        fold_ids=np.asarray(fold_ids, dtype=np.int64)[selected],
        labels=np.asarray(data["labels"], dtype=np.int64)[selected],
        initial_probability=np.asarray(probabilities["original"])[selected],
        mpr_probability=np.asarray(probabilities["refined"])[selected],
        full_probability=np.asarray(probabilities["qc_refined"])[selected],
    )
    return write_path


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
