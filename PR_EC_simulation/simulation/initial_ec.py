"""Five-fold initial-encoder training and individual directed-graph scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

from ..model.individual_ec.training import train_stf_ec
from ..model.individual_ec.utils import extract_subject_ec
from ..utils.runtime import select_device


METRICS = ("auprc", "precision", "recall", "f1", "accuracy", "shd")
THRESHOLD_PROTOCOL = {
    "name": "FSTA-style relative threshold",
    "coefficient": 0.5,
    "formula": "min(off_diagonal_scores) + 0.5 * (max(off_diagonal_scores) - min(off_diagonal_scores))",
    "comparison": ">=",
    "scope": "per subject, off-diagonal directed edges only",
    "provenance": "Individual-level extension of FSTA's group-level rule; post-hoc correction of the original fixed threshold=0.1; no coefficient search or ground-truth tuning",
}
PAPER_ORIENTATION = {
    "prediction_orientation": "paper_initial_ec_attention_transpose",
    "edge_convention": "matrix[i,j] = source i -> target j; off-diagonal only",
    "orientation_conversion": "Use extract_subject_ec output (mean attention.T) directly, without an additional transpose",
    "direction_basis": "Paper initial-EC definition supplied by author; FSTA train_epoch returns adj_mean.T; attention query rows are targets and key columns are sources; processed adj[source,target]=1",
    "direction_caveat": "Decoder uses features @ attention, whose mixing direction is opposite to the attention query/key EC convention; decoder multiplication alone does not establish the paper's EC convention",
}


def load_network(path, window_length, n_splits=5):
    with np.load(path, allow_pickle=False) as archive:
        series = archive["bold"].astype(np.float32)
        graph = archive["adj"].copy()
    if series.ndim != 3 or series.shape[2] < 2:
        raise ValueError(f"{path}: expected bold shape (subjects, T, N), N >= 2")
    if not 2 <= n_splits <= series.shape[0]:
        raise ValueError(f"{path}: n_splits must be between 2 and the subject count ({series.shape[0]})")
    if series.shape[1] < window_length or graph.shape != (series.shape[2],) * 2:
        raise ValueError(f"{path}: incompatible window length or adj shape")
    if not np.isfinite(series).all() or not np.isin(graph, [0, 1]).all():
        raise ValueError(f"{path}: BOLD must be finite and adj must be binary")
    if np.any(np.diag(graph)):
        raise ValueError(f"{path}: adj must not contain self-edges")
    off_diagonal = graph[~np.eye(graph.shape[0], dtype=bool)]
    if np.unique(off_diagonal).size != 2:
        raise ValueError(f"{path}: directed AUPRC requires positive and negative edges")
    mean = series.mean(axis=1, keepdims=True)
    std = series.std(axis=1, keepdims=True)
    series = (series - mean) / np.maximum(std, 1e-6)
    return series, graph


def score_subjects(oof_ec, graph):
    """Score off-diagonal i -> j edges; SHD counts a reversal as two errors."""
    if oof_ec.ndim != 3 or oof_ec.shape[1:] != graph.shape:
        raise ValueError("OOF EC and ground truth shapes do not match")
    if not np.isfinite(oof_ec).all():
        raise ValueError("OOF EC contains non-finite values")
    mask = ~np.eye(graph.shape[0], dtype=bool)
    truth = graph[mask].astype(bool)
    rows = []
    for matrix in oof_ec:
        scores = matrix[mask]
        threshold = scores.min() + 0.5 * (scores.max() - scores.min())
        predicted = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        rows.append({
            "auprc": float(average_precision_score(truth, scores)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy_score(truth, predicted)),
            "shd": int(np.count_nonzero(truth != predicted)),
            "threshold": float(threshold),
        })
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(scores):
    return {
        metric: {
            "mean": float(np.mean([score[metric] for score in scores])),
            "std": float(np.std([score[metric] for score in scores], ddof=1)),
        }
        for metric in METRICS
    }


def write_experiment_tables(output_dir, summaries):
    rows = []
    for summary in summaries:
        row = {
            "network": summary["network"], "subjects": summary["subjects"], "nodes": summary["nodes"],
            "threshold_rule": THRESHOLD_PROTOCOL["name"],
            "prediction_orientation": summary["prediction_orientation"],
        }
        for metric, values in summary["metrics"].items():
            row[f"{metric}_mean"] = values["mean"]
            row[f"{metric}_std"] = values["std"]
        rows.append(row)
    write_csv(output_dir / "experiment_summary.csv", rows)
    lines = [
        "# Individual OOF EC results", "",
        "Primary: Directed AUPRC (average precision, threshold-independent).",
        "Auxiliary: F1 and SHD; Precision, Recall and Accuracy are also reported.",
        "FSTA-style relative threshold: per-subject off-diagonal min + 0.5 × (max − min), using >=.",
        "Individual-level extension of the group-level FSTA rule; post-hoc correction of fixed threshold=0.1, not a threshold search.",
        "Mean ± sample SD across subjects; SHD reversal costs 2.", "",
        "| Network | Subjects | Directed AUPRC | F1 | SHD | Precision | Recall | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        values = [f"{summary['metrics'][metric]['mean']:.6f} ± {summary['metrics'][metric]['std']:.6f}" for metric in ("auprc", "f1", "shd", "precision", "recall", "accuracy")]
        lines.append(f"| {summary['network']} | {summary['subjects']} | " + " | ".join(values) + " |")
    for summary in summaries:
        lines.extend(["", f"- {summary['network']}: {summary['prediction_orientation']}."])
        if "rescore_note" in summary:
            lines.append(f"  {summary['rescore_note']}")
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_network(args, network, device):
    name = f"Network{network}_amp"
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    series, graph = load_network(args.data_dir / f"{name}.npz", args.window_length, args.n_splits)
    fold_args = argparse.Namespace(**vars(args))
    fold_args.nodes_num = series.shape[2]
    oof_ec = np.full((len(series), *graph.shape), np.nan, dtype=np.float32)
    oof_mse = np.full(len(series), np.nan, dtype=np.float32)
    fold_ids = np.zeros(len(series), dtype=np.int64)
    folds = []
    splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(series), 1):
        print(f"\n{name} fold={fold}/{args.n_splits} train={len(train_indices)} test={len(test_indices)}")
        model, training_metrics = train_stf_ec(fold_args, series[train_indices], device)
        records = [SimpleNamespace(subject_id=int(index) + 1) for index in test_indices]
        extracted = extract_subject_ec(
            model, records, series[test_indices], args.window_length, args.stride, device
        )
        predicted = extracted["ec"]
        if predicted.shape != (len(test_indices), *graph.shape):
            raise RuntimeError("Unexpected held-out EC shape")
        if np.any(fold_ids[test_indices]):
            raise RuntimeError("A subject was predicted more than once")
        oof_ec[test_indices] = predicted
        oof_mse[test_indices] = extracted["reconstruction_mse"]
        fold_ids[test_indices] = fold
        folds.append({
            "fold": fold,
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_count": len(train_indices),
            "test_count": len(test_indices),
            "training_metrics": training_metrics,
        })
        del model
    if np.any(fold_ids == 0) or not np.isfinite(oof_mse).all():
        raise RuntimeError("OOF coverage is incomplete or reconstruction MSE is invalid")
    scores = score_subjects(oof_ec, graph)
    rows = [
        {"subject_index": index, "subject_id": index + 1, "fold": int(fold_ids[index]), **score}
        for index, score in enumerate(scores)
    ]
    summary = {
        "network": name,
        "subjects": len(series),
        "nodes": graph.shape[0],
        "seed": 42,
        "folds": args.n_splits,
        "nominal_train_fraction": (args.n_splits - 1) / args.n_splits,
        "nominal_test_fraction": 1 / args.n_splits,
        "threshold_protocol": dict(THRESHOLD_PROTOCOL),
        **PAPER_ORIENTATION,
        "auprc_definition": "sklearn average_precision_score (non-interpolated average precision)",
        "shd_definition": "off-diagonal directed adjacency Hamming distance; reversal costs 2",
        "std_ddof": 1,
        "normalization": "per-subject per-ROI temporal z-score, no between-subject fitted statistics",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(fold_args).items()},
        "metrics": metric_summary(scores),
    }
    np.savez_compressed(
        output_dir / "oof_ec.npz", oof_ec=oof_ec, ground_truth=graph,
        subject_indices=np.arange(len(series)), subject_ids=np.arange(1, len(series) + 1),
        fold_ids=fold_ids, reconstruction_mse=oof_mse,
        subject_thresholds=np.array([score["threshold"] for score in scores]),
        **{f"{metric}_scores": np.array([row[metric] for row in scores]) for metric in METRICS},
    )
    write_csv(output_dir / "subject_metrics.csv", rows)
    for filename, payload in (("summary.json", summary), ("folds.json", folds)):
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"\n{name}: individual OOF mean ± sample SD (n={len(series)})")
    for metric, values in summary["metrics"].items():
        print(f"  {metric.upper()}: {values['mean']:.6f} ± {values['std']:.6f}")
    return summary


def run_experiments(args):
    device = select_device(args.gpu_id)
    print(f"Initial EC only; device={device}; KFold={args.n_splits}; seed=42; FSTA-style relative threshold (coefficient=0.5)")
    summaries = [run_network(args, network, device) for network in dict.fromkeys(args.networks)]
    write_experiment_tables(args.output_dir, summaries)
