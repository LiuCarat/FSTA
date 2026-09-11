#!/usr/bin/env python3
"""Check whether common top-k edge cutoffs form natural effect-size gaps.

For each dataset, all off-diagonal directed BEC edges are compared between
ASD (label 1) and HC/TC (label 0). Edges are sorted by absolute Hedges' g,
and the values immediately around the requested ranks are reported.

Hedges' g uses the usual small-sample correction applied to Cohen's d with
the pooled within-group standard deviation. The sign is ASD minus HC/TC;
ranking is based on ``abs(g)``.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

HC_LABEL = 0
ASD_LABEL = 1
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS = {
    "ABIDE-I": ROOT / "Graph_BEC/outputs/abide-i/abide_qsr_refined_subject_bec.npz",
    "ABIDE-II": ROOT / "Graph_BEC/outputs/abide-ii/abide_ii_qsr_refined_subject_bec.npz",
}
DEFAULT_OUTPUT_DIR = ROOT / "Graph_BEC/analysis/topk_cutoff_check/outputs"
DEFAULT_RANKS = (10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--abide-i",
        type=Path,
        default=DEFAULT_DATASETS["ABIDE-I"],
        help="ABIDE-I BEC .npz archive",
    )
    parser.add_argument(
        "--abide-ii",
        type=Path,
        default=DEFAULT_DATASETS["ABIDE-II"],
        help="ABIDE-II BEC .npz archive",
    )
    parser.add_argument(
        "--bec-key",
        choices=("bec", "refined_bec", "qc_refined_bec", "original_bec"),
        default="bec",
        help="Array key used as the subject-level edge representation",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANKS),
        help="1-based ranks whose boundary gaps are checked",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
            else np.asarray([f"ROI {index + 1}" for index in range(bec.shape[1])])
        )
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC shape [subjects, nodes, nodes], got {bec.shape}")
    if len(labels) != len(bec) or len(roi_names) != bec.shape[1]:
        raise ValueError("BEC, labels, and roi_names have incompatible lengths")
    if not np.isfinite(bec).all():
        raise ValueError("BEC contains NaN or infinite values")
    if np.any(~np.isin(labels, (HC_LABEL, ASD_LABEL))):
        raise ValueError("Labels outside the canonical groups 0=HC/TC and 1=ASD were found")
    if not (labels == HC_LABEL).any() or not (labels == ASD_LABEL).any():
        raise ValueError("Both HC/TC and ASD groups are required")
    return bec, labels, roi_names


def hedges_g(bec: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return directed-edge Hedges' g for ASD minus HC/TC."""
    asd = bec[labels == ASD_LABEL]
    hc = bec[labels == HC_LABEL]
    n_asd, n_hc = len(asd), len(hc)
    degrees_of_freedom = n_asd + n_hc - 2
    if degrees_of_freedom <= 0:
        raise ValueError("At least two observations across the two groups are required")

    mean_difference = asd.mean(axis=0) - hc.mean(axis=0)
    asd_variance = asd.var(axis=0, ddof=1)
    hc_variance = hc.var(axis=0, ddof=1)
    pooled_variance = (
        (n_asd - 1) * asd_variance + (n_hc - 1) * hc_variance
    ) / degrees_of_freedom
    pooled_sd = np.sqrt(pooled_variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        cohens_d = np.divide(mean_difference, pooled_sd, out=np.zeros_like(mean_difference), where=pooled_sd > 0)

    # Exact gamma-ratio correction, J(df) = Gamma(df/2) / (sqrt(df/2) Gamma((df-1)/2)).
    # scipy is intentionally not required for this descriptive calculation.
    correction = float(np.exp(
        math.lgamma(degrees_of_freedom / 2.0)
        - 0.5 * np.log(degrees_of_freedom / 2.0)
        - math.lgamma((degrees_of_freedom - 1) / 2.0)
    ))
    return cohens_d * correction


def sorted_edge_rows(g_values: np.ndarray, roi_names: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in range(g_values.shape[0]):
        for target in range(g_values.shape[1]):
            if source == target:
                continue
            signed_g = float(g_values[source, target])
            rows.append({
                "rank": 0,
                "source": str(roi_names[source]),
                "target": str(roi_names[target]),
                "source_index": source,
                "target_index": target,
                "hedges_g": signed_g,
                "absolute_hedges_g": abs(signed_g),
            })
    rows.sort(key=lambda row: (-row["absolute_hedges_g"], row["source_index"], row["target_index"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def cutoff_rows(dataset: str, edge_rows: list[dict[str, object]], ranks: tuple[int, ...]) -> list[dict[str, object]]:
    by_rank = {int(row["rank"]): row for row in edge_rows}
    output = []
    for rank in ranks:
        current = by_rank[rank]
        next_edge = by_rank[rank + 1]
        output.append({
            "Dataset": dataset,
            "CutoffRank": rank,
            "Rank": rank,
            "NextRank": rank + 1,
            "AbsG": current["absolute_hedges_g"],
            "NextAbsG": next_edge["absolute_hedges_g"],
            "Delta": current["absolute_hedges_g"] - next_edge["absolute_hedges_g"],
            "G": current["hedges_g"],
            "NextG": next_edge["hedges_g"],
            "Source": current["source"],
            "Target": current["target"],
            "NextSource": next_edge["source"],
            "NextTarget": next_edge["target"],
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ranks = tuple(sorted(set(args.ranks)))
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("--ranks must contain positive 1-based ranks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_edges: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for dataset, path in (("ABIDE-I", args.abide_i), ("ABIDE-II", args.abide_ii)):
        bec, labels, roi_names = load_archive(path, args.bec_key)
        g_values = hedges_g(bec, labels)
        rows = sorted_edge_rows(g_values, roi_names)
        if any(rank + 1 > len(rows) for rank in ranks):
            raise ValueError(
                f"Requested rank exceeds available edges for {dataset}: {len(rows)}"
            )
        all_edges.extend({"Dataset": dataset, **row} for row in rows)
        summary.extend(cutoff_rows(dataset, rows, ranks))
        print(f"{dataset}: ASD n={(labels == ASD_LABEL).sum()}, HC/TC n={(labels == HC_LABEL).sum()}")

    write_csv(
        args.output_dir / "topk_cutoff_summary.csv",
        summary,
        ["Dataset", "CutoffRank", "Rank", "NextRank", "AbsG", "NextAbsG", "Delta", "G", "NextG", "Source", "Target", "NextSource", "NextTarget"],
    )
    write_csv(
        args.output_dir / "all_edges_ranked_by_absolute_hedges_g.csv",
        all_edges,
        ["Dataset", "rank", "source", "target", "source_index", "target_index", "hedges_g", "absolute_hedges_g"],
    )
    print(f"Saved summary: {(args.output_dir / 'topk_cutoff_summary.csv').resolve()}")
    print(f"Saved ranked edges: {(args.output_dir / 'all_edges_ranked_by_absolute_hedges_g.csv').resolve()}")


if __name__ == "__main__":
    main()
