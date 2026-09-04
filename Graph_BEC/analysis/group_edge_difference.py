#!/usr/bin/env python3
"""Rank the strongest directed ASD-vs-TC BEC edges.

Canonical labels: 0 = TC, 1 = ASD.
Difference: ASD_mean - TC_mean.

The output contains the top ``K`` ASD-enhanced and top ``K`` TC-enhanced
directed edges in one CSV table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

TC_LABEL = 0
ASD_LABEL = 1
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEC = ROOT / "Graph_BEC/outputs/abide/abide_qsr_refined_subject_bec.npz"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/analysis/outputs/group_edge_difference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument(
        "--bec-key", choices=["pgr_bec", "qc_refined_bec", "bec"],
        default="bec",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-name", default="top_edges_abide.csv")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_archive(path: Path, bec_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"BEC archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {bec_key, "labels"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"BEC archive is missing: {sorted(missing)}")
        bec = np.asarray(archive[bec_key], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        roi_names = (
            np.asarray(archive["roi_names"]).astype(str).reshape(-1)
            if "roi_names" in archive.files
            else np.asarray([f"ROI {i + 1}" for i in range(bec.shape[1])])
        )
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC shape [subjects, nodes, nodes], got {bec.shape}")
    if len(labels) != len(bec) or len(roi_names) != bec.shape[1]:
        raise ValueError("BEC, labels, and roi_names have incompatible lengths")
    if not np.isfinite(bec).all():
        raise ValueError("BEC contains NaN or infinite values")
    return bec, labels, roi_names


def compute_difference(
    bec: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    asd_mask = labels == ASD_LABEL
    tc_mask = labels == TC_LABEL
    if not asd_mask.any() or not tc_mask.any():
        raise ValueError(
            f"Both groups are required; observed ASD n={asd_mask.sum()}, TC n={tc_mask.sum()}"
        )
    if np.any(~(asd_mask | tc_mask)):
        raise ValueError("Labels outside the canonical groups 0=TC and 1=ASD were found")
    asd_mean = bec[asd_mask].mean(axis=0, dtype=np.float64)
    tc_mean = bec[tc_mask].mean(axis=0, dtype=np.float64)
    return asd_mean, tc_mean, asd_mean - tc_mean


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.arange(1, len(sorted_values) + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(
        (sorted_values * len(sorted_values) / ranks)[::-1]
    )[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def compute_edge_statistics(
    bec: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch p-values and BH q-values for all directed edges."""
    asd_mask = labels == ASD_LABEL
    tc_mask = labels == TC_LABEL
    _, p_values = ttest_ind(
        bec[asd_mask], bec[tc_mask], axis=0, equal_var=False, nan_policy="raise"
    )
    off_diagonal = ~np.eye(p_values.shape[0], dtype=bool)
    q_values = np.full_like(p_values, np.nan, dtype=np.float64)
    q_values[off_diagonal] = benjamini_hochberg(p_values[off_diagonal])
    return p_values, q_values


def edge_rows(
    asd_mean: np.ndarray,
    tc_mean: np.ndarray,
    difference: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
    roi_names: np.ndarray,
    top_k: int,
) -> list[dict[str, object]]:
    candidates = []
    for source in range(difference.shape[0]):
        for target in range(difference.shape[1]):
            if source == target:
                continue
            value = float(difference[source, target])
            candidates.append({
                "source": str(roi_names[source]),
                "target": str(roi_names[target]),
                "asd_mean": float(asd_mean[source, target]),
                "tc_mean": float(tc_mean[source, target]),
                "difference_asd_minus_tc": value,
                "absolute_difference": abs(value),
                "p_value": float(p_values[source, target]),
                "fdr_q": float(q_values[source, target]),
                "source_index": source,
                "target_index": target,
            })
    positive = sorted(
        (row for row in candidates if row["difference_asd_minus_tc"] > 0),
        key=lambda row: row["difference_asd_minus_tc"], reverse=True,
    )[:top_k]
    negative = sorted(
        (row for row in candidates if row["difference_asd_minus_tc"] < 0),
        key=lambda row: row["difference_asd_minus_tc"],
    )[:top_k]
    rows = []
    for rank, row in enumerate(positive, 1):
        rows.append({"direction": "ASD_enhanced", "rank": rank, **row})
    for rank, row in enumerate(negative, 1):
        rows.append({"direction": "TC_enhanced", "rank": rank, **row})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "Direction", "Rank", "Source", "Target", "SourceIndex", "TargetIndex",
        "ASD_mean", "TC_mean", "Difference", "AbsoluteDifference",
        "PValue", "FDR_q",
    ]
    rows = [
        {
            "Direction": row["direction"],
            "Rank": row["rank"],
            "Source": row["source"],
            "Target": row["target"],
            "SourceIndex": row["source_index"],
            "TargetIndex": row["target_index"],
            "ASD_mean": row["asd_mean"],
            "TC_mean": row["tc_mean"],
            "Difference": row["difference_asd_minus_tc"],
            "AbsoluteDifference": row["absolute_difference"],
            "PValue": row["p_value"],
            "FDR_q": row["fdr_q"],
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    bec, labels, roi_names = load_archive(args.bec_path, args.bec_key)
    asd_mean, tc_mean, difference = compute_difference(bec, labels)
    p_values, q_values = compute_edge_statistics(bec, labels)
    rows = edge_rows(
        asd_mean, tc_mean, difference, p_values, q_values, roi_names, args.top_k
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.output_name
    write_csv(output, rows)
    print(f"ASD subjects: {int((labels == ASD_LABEL).sum())}")
    print(f"TC subjects: {int((labels == TC_LABEL).sum())}")
    print(f"Saved {len(rows)} rows ({args.top_k} per direction): {output.resolve()}")


if __name__ == "__main__":
    main()
