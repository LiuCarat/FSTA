#!/usr/bin/env python3
"""Draw separate directed chord diagrams for top ASD-enhanced/diminished edges.

The analysis is descriptive: Difference = MeanBEC_ASD - MeanBEC_TC.
Positive edges are ASD-enhanced; negative edges are ASD-diminished
(TC-enhanced). Each diagram contains the selected directed Source -> Target
edges and all ROIs involved in those edges.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from group_edge_difference import (
    ASD_LABEL,
    TC_LABEL,
    compute_difference,
    load_archive,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEC = ROOT / "Graph_BEC/outputs/refined_subject_bec.npz"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/analysis/outputs/group_chord"

SURFACE = "#fbfbf8"
TEXT = "#30302d"
NODE = "#2f6fbd"       # blue ROI nodes
ENHANCED = "#d94b4b"    # ASD-enhanced
DIMINISHED = "#4c78a8"  # ASD-diminished / TC-enhanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument(
        "--bec-key", choices=["bec", "pgr_bec", "refined_bec", "qc_refined_bec", "original_bec"],
        default="pgr_bec",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def top_edges(asd_mean, tc_mean, difference, roi_names, top_k):
    rows = []
    for source in range(difference.shape[0]):
        for target in range(difference.shape[1]):
            if source == target:
                continue
            value = float(difference[source, target])
            if value == 0.0:
                continue
            rows.append({
                "Source": str(roi_names[source]),
                "Target": str(roi_names[target]),
                "SourceIndex": source,
                "TargetIndex": target,
                "ASD_mean": float(asd_mean[source, target]),
                "TC_mean": float(tc_mean[source, target]),
                "Difference": value,
                "AbsoluteDifference": abs(value),
            })
    positive = sorted(
        (row for row in rows if row["Difference"] > 0),
        key=lambda row: row["Difference"], reverse=True,
    )[:top_k]
    negative = sorted(
        (row for row in rows if row["Difference"] < 0),
        key=lambda row: row["Difference"],
    )[:top_k]
    for rank, row in enumerate(positive, 1):
        row.update(Direction="ASD_enhanced", Rank=rank)
    for rank, row in enumerate(negative, 1):
        row.update(Direction="ASD_diminished_TC_enhanced", Rank=rank)
    return positive, negative


def write_edge_csv(path: Path, rows):
    fields = [
        "Direction", "Rank", "Source", "Target", "SourceIndex", "TargetIndex",
        "ASD_mean", "TC_mean", "Difference", "AbsoluteDifference",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_roi_csv(path: Path, positive, negative):
    fields = ["Network", "ROI", "ROIIndex"]
    rows = []
    for network, edges in (("ASD_enhanced", positive), ("ASD_diminished_TC_enhanced", negative)):
        indices = {}
        for row in edges:
            indices[row["Source"]] = row["SourceIndex"]
            indices[row["Target"]] = row["TargetIndex"]
        rows.extend(
            {"Network": network, "ROI": roi, "ROIIndex": index}
            for roi, index in sorted(indices.items(), key=lambda item: item[1])
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _circle_positions(nodes):
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(nodes), endpoint=False)
    return {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(nodes, angles)}


def draw_network(ax, edges, title, color, group_label):
    nodes = sorted(
        {row["Source"] for row in edges} | {row["Target"] for row in edges},
        key=lambda name: next(
            row["SourceIndex"] if row["Source"] == name else row["TargetIndex"]
            for row in edges
            if row["Source"] == name or row["Target"] == name
        ),
    )
    positions = _circle_positions(nodes)
    max_delta = max((row["AbsoluteDifference"] for row in edges), default=1.0)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    # Draw weakest edges first so the strongest edge remains visible.
    for row in sorted(edges, key=lambda item: item["AbsoluteDifference"]):
        start, end = positions[row["Source"]], positions[row["Target"]]
        width = 1.5 + 5.0 * row["AbsoluteDifference"] / max_delta
        curvature = 0.18 if np.dot(start, end) >= 0 else 0.30
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            linewidth=width, color=color, alpha=0.78,
            connectionstyle=f"arc3,rad={curvature}", shrinkA=14, shrinkB=14,
        )
        ax.add_patch(arrow)

    for node, point in positions.items():
        ax.scatter(*point, s=260, c=NODE, edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(
            point[0] * 1.16, point[1] * 1.16, node,
            ha="center", va="center", fontsize=9, color=TEXT,
        )

    ax.set_title(title, fontsize=13, color=TEXT, pad=18)
    ax.text(
        0.5, -0.07,
        f"{group_label}; {len(edges)} directed edges; {len(nodes)} involved ROIs",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#6a6964",
    )
    return nodes


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    bec, labels, roi_names = load_archive(args.bec_path, args.bec_key)
    asd_mean, tc_mean, difference = compute_difference(bec, labels)
    positive, negative = top_edges(asd_mean, tc_mean, difference, roi_names, args.top_k)
    if not positive and not negative:
        raise ValueError("No non-zero directed group differences were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_edge_csv(args.output_dir / f"top{args.top_k}_edges_asd_enhanced.csv", positive)
    write_edge_csv(args.output_dir / f"top{args.top_k}_edges_asd_diminished.csv", negative)
    write_roi_csv(args.output_dir / f"top{args.top_k}_involved_rois.csv", positive, negative)

    if positive:
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=SURFACE)
        draw_network(ax, positive, f"Top {len(positive)} ASD-enhanced directed edges", ENHANCED, "ASD > TC")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"chord_top{len(positive)}_asd_enhanced.png", dpi=220, facecolor=SURFACE)
        plt.close(fig)
    if negative:
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=SURFACE)
        draw_network(ax, negative, f"Top {len(negative)} ASD-diminished directed edges", DIMINISHED, "ASD < TC / TC > ASD")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"chord_top{len(negative)}_asd_diminished.png", dpi=220, facecolor=SURFACE)
        plt.close(fig)

    print(f"ASD subjects: {int((labels == ASD_LABEL).sum())}")
    print(f"TC subjects: {int((labels == TC_LABEL).sum())}")
    print(f"Saved chord outputs to: {args.output_dir.resolve()}")
    print(f"ASD-enhanced edges: {len(positive)}; ASD-diminished edges: {len(negative)}")


if __name__ == "__main__":
    main()
