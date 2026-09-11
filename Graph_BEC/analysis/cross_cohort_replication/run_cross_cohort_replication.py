#!/usr/bin/env python3
"""Identify BEC edges replicated across ABIDE-I and ABIDE-II.

Replication is defined per identical directed edge (source -> target): both
cohorts must have BH-FDR q < alpha and the ASD-minus-HC effect must have the
same sign. Within-cohort rank is retained for description only and is never a
replication criterion.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

HC_LABEL = 0
ASD_LABEL = 1
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUTS = {
    "ABIDE-I": ROOT / "Graph_BEC/outputs/abide-i/abide_qsr_refined_subject_bec.npz",
    "ABIDE-II": ROOT / "Graph_BEC/outputs/abide-ii/abide_ii_qsr_refined_subject_bec.npz",
}
DEFAULT_OUTPUT_DIR = ROOT / "Graph_BEC/analysis/cross_cohort_replication/outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abide-i", type=Path, default=DEFAULT_INPUTS["ABIDE-I"])
    parser.add_argument("--abide-ii", type=Path, default=DEFAULT_INPUTS["ABIDE-II"])
    parser.add_argument(
        "--bec-key",
        choices=("bec", "refined_bec", "qc_refined_bec", "original_bec"),
        default="bec",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_archive(path: Path, bec_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"BEC archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = {bec_key, "labels"} - set(archive.files)
        if missing:
            raise ValueError(f"BEC archive is missing: {sorted(missing)}")
        bec = np.asarray(archive[bec_key], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        roi_names = (
            np.asarray(archive["roi_names"]).astype(str).reshape(-1)
            if "roi_names" in archive.files
            else np.asarray([f"ROI_{index + 1:03d}" for index in range(bec.shape[1])])
        )
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC shape [subjects, nodes, nodes], got {bec.shape}")
    if len(labels) != len(bec) or len(roi_names) != bec.shape[1]:
        raise ValueError("BEC, labels, and roi_names have incompatible lengths")
    if not np.isfinite(bec).all():
        raise ValueError("BEC contains NaN or infinite values")
    if np.any(~np.isin(labels, (HC_LABEL, ASD_LABEL))):
        raise ValueError("Labels outside 0=HC/TC and 1=ASD were found")
    if not (labels == HC_LABEL).any() or not (labels == ASD_LABEL).any():
        raise ValueError("Both HC/TC and ASD groups are required")
    return bec, labels, roi_names


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate((sorted_values * len(values) / ranks)[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def compute_effects_and_tests(
    bec: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    asd = bec[labels == ASD_LABEL]
    hc = bec[labels == HC_LABEL]
    n_asd, n_hc = len(asd), len(hc)
    degrees_of_freedom = n_asd + n_hc - 2
    asd_mean = asd.mean(axis=0)
    hc_mean = hc.mean(axis=0)
    mean_difference = asd_mean - hc_mean
    pooled_variance = (
        (n_asd - 1) * asd.var(axis=0, ddof=1)
        + (n_hc - 1) * hc.var(axis=0, ddof=1)
    ) / degrees_of_freedom
    pooled_sd = np.sqrt(pooled_variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        cohens_d = np.divide(
            mean_difference,
            pooled_sd,
            out=np.zeros_like(mean_difference),
            where=pooled_sd > 0,
        )
    correction = math.exp(
        math.lgamma(degrees_of_freedom / 2.0)
        - 0.5 * math.log(degrees_of_freedom / 2.0)
        - math.lgamma((degrees_of_freedom - 1) / 2.0)
    )
    hedges_g = cohens_d * correction

    _, p_values = ttest_ind(asd, hc, axis=0, equal_var=False, nan_policy="raise")
    off_diagonal = ~np.eye(p_values.shape[0], dtype=bool)
    q_values = np.full_like(p_values, np.nan, dtype=np.float64)
    q_values[off_diagonal] = benjamini_hochberg(p_values[off_diagonal])
    return asd_mean, hc_mean, hedges_g, q_values


def edge_table(
    dataset: str,
    asd_mean: np.ndarray,
    hc_mean: np.ndarray,
    hedges_g: np.ndarray,
    q_values: np.ndarray,
    roi_names: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for source in range(hedges_g.shape[0]):
        for target in range(hedges_g.shape[1]):
            if source == target:
                continue
            rows.append({
                "Dataset": dataset,
                "Source": str(roi_names[source]),
                "Target": str(roi_names[target]),
                "SourceIndex": source,
                "TargetIndex": target,
                "ASD_mean": float(asd_mean[source, target]),
                "HC_mean": float(hc_mean[source, target]),
                "Hedges_g": float(hedges_g[source, target]),
                "AbsHedges_g": abs(float(hedges_g[source, target])),
                "QValue": float(q_values[source, target]),
            })
    rows.sort(key=lambda row: (-row["AbsHedges_g"], row["SourceIndex"], row["TargetIndex"]))
    for rank, row in enumerate(rows, start=1):
        row["WithinCohortRank"] = rank
    return rows


def merge_tables(
    abide_i: list[dict[str, object]],
    abide_ii: list[dict[str, object]],
    alpha: float,
) -> list[dict[str, object]]:
    key = lambda row: (row["SourceIndex"], row["TargetIndex"])
    left = {key(row): row for row in abide_i}
    right = {key(row): row for row in abide_ii}
    if set(left) != set(right):
        raise ValueError("ABIDE-I and ABIDE-II do not contain the same directed edge set")

    merged = []
    for edge_key in sorted(left):
        row_i, row_ii = left[edge_key], right[edge_key]
        g_i, g_ii = float(row_i["Hedges_g"]), float(row_ii["Hedges_g"])
        q_i, q_ii = float(row_i["QValue"]), float(row_ii["QValue"])
        same_direction = np.sign(g_i) == np.sign(g_ii) and g_i != 0 and g_ii != 0
        replicated = q_i < alpha and q_ii < alpha and same_direction
        if replicated:
            direction = "replicated_ASD_enhanced" if g_i > 0 else "replicated_ASD_reduced"
        else:
            direction = "not_replicated"
        merged.append({
            "Source": row_i["Source"],
            "Target": row_i["Target"],
            "SourceIndex": edge_key[0],
            "TargetIndex": edge_key[1],
            "ABIDE_I_Hedges_g": g_i,
            "ABIDE_II_Hedges_g": g_ii,
            "ABIDE_I_AbsHedges_g": row_i["AbsHedges_g"],
            "ABIDE_II_AbsHedges_g": row_ii["AbsHedges_g"],
            "MeanAbsHedges_g": (float(row_i["AbsHedges_g"]) + float(row_ii["AbsHedges_g"])) / 2,
            "ABIDE_I_QValue": q_i,
            "ABIDE_II_QValue": q_ii,
            "ABIDE_I_Rank": row_i["WithinCohortRank"],
            "ABIDE_II_Rank": row_ii["WithinCohortRank"],
            "SameDirection": same_direction,
            "Replicated": replicated,
            "ReplicationClass": direction,
        })
    replicated_rows = [row for row in merged if row["Replicated"]]
    replicated_rows.sort(
        key=lambda row: (-row["MeanAbsHedges_g"], row["SourceIndex"], row["TargetIndex"])
    )
    for rank, row in enumerate(replicated_rows, start=1):
        row["ReplicatedRank"] = rank
    for row in merged:
        row.setdefault("ReplicatedRank", "")
    return merged


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")

    cohort_tables = {}
    for dataset, path in (("ABIDE-I", args.abide_i), ("ABIDE-II", args.abide_ii)):
        bec, labels, roi_names = load_archive(path, args.bec_key)
        asd_mean, hc_mean, hedges_g, q_values = compute_effects_and_tests(bec, labels)
        cohort_tables[dataset] = edge_table(
            dataset, asd_mean, hc_mean, hedges_g, q_values, roi_names
        )
        print(f"{dataset}: ASD n={(labels == ASD_LABEL).sum()}, HC/TC n={(labels == HC_LABEL).sum()}")

    merged = merge_tables(cohort_tables["ABIDE-I"], cohort_tables["ABIDE-II"], args.alpha)
    replicated = [row for row in merged if row["Replicated"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_fields = [
        "Source", "Target", "SourceIndex", "TargetIndex",
        "ABIDE_I_Hedges_g", "ABIDE_II_Hedges_g", "ABIDE_I_AbsHedges_g",
        "ABIDE_II_AbsHedges_g", "MeanAbsHedges_g", "ABIDE_I_QValue",
        "ABIDE_II_QValue", "ABIDE_I_Rank", "ABIDE_II_Rank", "SameDirection",
        "Replicated", "ReplicationClass", "ReplicatedRank",
    ]
    write_csv(args.output_dir / "all_edges_cross_cohort.csv", merged, merged_fields)
    write_csv(args.output_dir / "replicated_edges.csv", replicated, merged_fields)

    summary_rows = [
        {"Dataset": "ABIDE-I", "ASD_N": sum(row["Dataset"] == "ABIDE-I" for row in cohort_tables["ABIDE-I"]), "HC_N": "", "Significant_Q_lt_alpha": sum(row["QValue"] < args.alpha for row in cohort_tables["ABIDE-I"]), "Replicated": ""},
        {"Dataset": "ABIDE-II", "ASD_N": sum(row["Dataset"] == "ABIDE-II" for row in cohort_tables["ABIDE-II"]), "HC_N": "", "Significant_Q_lt_alpha": sum(row["QValue"] < args.alpha for row in cohort_tables["ABIDE-II"]), "Replicated": ""},
        {"Dataset": "Cross-cohort replicated", "ASD_N": "", "HC_N": "", "Significant_Q_lt_alpha": "", "Replicated": len(replicated)},
    ]
    # Replace the compact per-edge count placeholders with cohort sample sizes.
    for row, dataset, path in zip(summary_rows[:2], ("ABIDE-I", "ABIDE-II"), (args.abide_i, args.abide_ii)):
        with np.load(path, allow_pickle=False) as archive:
            labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        row["ASD_N"] = int((labels == ASD_LABEL).sum())
        row["HC_N"] = int((labels == HC_LABEL).sum())
    write_csv(
        args.output_dir / "replication_summary.csv",
        summary_rows,
        ["Dataset", "ASD_N", "HC_N", "Significant_Q_lt_alpha", "Replicated"],
    )
    enhanced = sum(row["ReplicationClass"] == "replicated_ASD_enhanced" for row in replicated)
    reduced = sum(row["ReplicationClass"] == "replicated_ASD_reduced" for row in replicated)
    print(f"Replicated edges: {len(replicated)} (ASD-enhanced={enhanced}, ASD-reduced={reduced})")
    print(f"Saved: {(args.output_dir / 'replicated_edges.csv').resolve()}")


if __name__ == "__main__":
    main()
