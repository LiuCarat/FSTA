#!/usr/bin/env python3
"""Rank ROIs with the largest ASD-vs-TC directed BEC differences.

Canonical labels: 0 = TC, 1 = ASD.
Difference: ASD_mean - TC_mean.

Each ROI is scored using all incident directed edges (incoming and outgoing).
The ranking uses the mean absolute incident difference, so it is not biased
by using a sum over a fixed number of connections.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

TC_LABEL = 0
ASD_LABEL = 1
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEC = ROOT / "Graph_BEC/outputs/refined_subject_bec.npz"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/analysis/outputs/group_roi_difference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument(
        "--bec-key", choices=["pgr_bec", "qc_refined_bec", "bec"],
        default="pgr_bec",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_archive(path: Path, bec_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"BEC archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if bec_key not in archive.files:
            raise ValueError(f"BEC archive is missing array: {bec_key}")
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


def compute_group_difference(
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


def roi_rows(
    difference: np.ndarray, roi_names: np.ndarray, top_k: int,
) -> list[dict[str, object]]:
    rows = []
    for node in range(difference.shape[0]):
        outgoing = np.delete(difference[node, :], node)
        incoming = np.delete(difference[:, node], node)
        incident = np.concatenate([outgoing, incoming])
        positive = np.maximum(incident, 0.0)
        negative = np.maximum(-incident, 0.0)
        total = float(np.mean(np.abs(incident)))
        asd_score = float(np.mean(positive))
        tc_score = float(np.mean(negative))
        net = float(np.mean(incident))

        best_position = int(np.argmax(np.abs(incident)))
        if best_position < len(outgoing):
            other = best_position if best_position < node else best_position + 1
            source, target = node, other
        else:
            position = best_position - len(outgoing)
            other = position if position < node else position + 1
            source, target = other, node
        best_difference = float(difference[source, target])
        if asd_score > tc_score:
            dominant = "ASD_enhanced"
        elif tc_score > asd_score:
            dominant = "TC_enhanced"
        else:
            dominant = "mixed"
        rows.append({
            "roi": str(roi_names[node]),
            "roi_index": node,
            "total_difference_score": total,
            "asd_enhanced_score": asd_score,
            "tc_enhanced_score": tc_score,
            "net_score_asd_minus_tc": net,
            "dominant_direction": dominant,
            "strongest_source": str(roi_names[source]),
            "strongest_target": str(roi_names[target]),
            "strongest_edge_difference": best_difference,
            "strongest_edge_direction": (
                "ASD_enhanced" if best_difference > 0 else "TC_enhanced"
            ),
        })
    rows.sort(key=lambda row: row["total_difference_score"], reverse=True)
    return rows[:top_k]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "Rank", "ROI", "ROIIndex", "TotalDifferenceScore",
        "ASDEnhancedScore", "TCEnhancedScore", "NetScoreASDMinusTC",
        "DominantDirection", "StrongestSource", "StrongestTarget",
        "StrongestEdgeDifference", "StrongestEdgeDirection",
    ]
    numbered = []
    for rank, row in enumerate(rows, 1):
        numbered.append({
            "Rank": rank,
            "ROI": row["roi"],
            "ROIIndex": row["roi_index"],
            "TotalDifferenceScore": row["total_difference_score"],
            "ASDEnhancedScore": row["asd_enhanced_score"],
            "TCEnhancedScore": row["tc_enhanced_score"],
            "NetScoreASDMinusTC": row["net_score_asd_minus_tc"],
            "DominantDirection": row["dominant_direction"],
            "StrongestSource": row["strongest_source"],
            "StrongestTarget": row["strongest_target"],
            "StrongestEdgeDifference": row["strongest_edge_difference"],
            "StrongestEdgeDirection": row["strongest_edge_direction"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(numbered)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    bec, labels, roi_names = load_archive(args.bec_path, args.bec_key)
    _, _, difference = compute_group_difference(bec, labels)
    rows = roi_rows(difference, roi_names, args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "top_rois_asd_vs_tc.csv"
    write_csv(output, rows)
    print(f"ASD subjects: {int((labels == ASD_LABEL).sum())}")
    print(f"TC subjects: {int((labels == TC_LABEL).sum())}")
    print(f"Saved {len(rows)} ROIs: {output.resolve()}")


if __name__ == "__main__":
    main()
