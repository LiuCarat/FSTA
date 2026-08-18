#!/usr/bin/env python3
"""Descriptive ASD/TC group comparison for individual Original BEC.

The only persistent outputs are:

* ``group_bec_table.csv`` with Source, Target, ASD_mean, TC_mean, Difference;
* ``group_bec_difference_heatmap.png`` showing MeanBEC_ASD - MeanBEC_TC.

No inferential/statistical significance test is performed.
"""
from __future__ import annotations

import argparse
import csv
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEC = (
    ROOT
    / "Graph_BEC/outputs/original/loss_alpha_0.8/seed_42/epochs_101/subject_bec.npz"
)
DEFAULT_OUTPUT = ROOT / "Graph_BEC/analysis/outputs/original_group_bec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--asd-label", type=int, choices=[0, 1], default=1,
        help="Label used for ASD; current ABIDE archive uses 1.",
    )
    parser.add_argument(
        "--tc-label", type=int, choices=[0, 1], default=0,
        help="Label used for TC; current ABIDE archive uses 0.",
    )
    parser.add_argument(
        "--cell-size", type=int, default=8,
        help="Pixel size of each heatmap cell (default: 8).",
    )
    return parser.parse_args()


def load_bec_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BEC archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"bec", "labels", "subject_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"BEC archive is missing arrays: {sorted(missing)}")
        bec = np.asarray(archive["bec"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        subject_ids = np.asarray(archive["subject_ids"]).astype(str).reshape(-1)
        roi_names = (
            np.asarray(archive["roi_names"]).astype(str).reshape(-1)
            if "roi_names" in archive.files
            else np.asarray([], dtype=str)
        )

    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC shape [subjects, nodes, nodes], got {bec.shape}")
    if len(labels) != len(bec) or len(subject_ids) != len(bec):
        raise ValueError("BEC, labels, and subject_ids must have equal length")
    if len(np.unique(subject_ids)) != len(subject_ids):
        raise ValueError("subject_ids are not unique")
    if not np.isfinite(bec).all():
        raise ValueError("BEC contains NaN or infinite values")
    if roi_names.size == 0:
        roi_names = np.asarray([f"ROI {i + 1}" for i in range(bec.shape[1])])
    if len(roi_names) != bec.shape[1]:
        raise ValueError(f"roi_names has length {len(roi_names)}, expected {bec.shape[1]}")
    return {"bec": bec, "labels": labels, "subject_ids": subject_ids, "roi_names": roi_names}


def compute_group_means(
    bec: np.ndarray, labels: np.ndarray, asd_label: int, tc_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if asd_label == tc_label:
        raise ValueError("--asd-label and --tc-label must differ")
    asd_mask = labels == asd_label
    tc_mask = labels == tc_label
    other_mask = ~(asd_mask | tc_mask)
    if other_mask.any():
        observed = sorted(int(value) for value in np.unique(labels[other_mask]))
        raise ValueError(f"Unexpected labels outside ASD/TC: {observed}")
    if not asd_mask.any() or not tc_mask.any():
        raise ValueError(
            f"Both groups must be present; ASD n={int(asd_mask.sum())}, "
            f"TC n={int(tc_mask.sum())}"
        )

    # One BEC per subject, equal subject weighting, no extra normalization.
    mean_asd = bec[asd_mask].mean(axis=0, dtype=np.float64)
    mean_tc = bec[tc_mask].mean(axis=0, dtype=np.float64)
    return mean_asd, mean_tc, mean_asd - mean_tc, asd_mask, tc_mask


def write_group_table(
    path: Path, mean_asd: np.ndarray, mean_tc: np.ndarray,
    difference: np.ndarray, roi_names: np.ndarray,
) -> None:
    rows = []
    for source in range(difference.shape[0]):
        for target in range(difference.shape[1]):
            if source == target:
                continue
            rows.append({
                "Source": str(roi_names[source]),
                "Target": str(roi_names[target]),
                "ASD_mean": float(mean_asd[source, target]),
                "TC_mean": float(mean_tc[source, target]),
                "Difference": float(difference[source, target]),
            })
    rows.sort(key=lambda row: abs(row["Difference"]), reverse=True)
    fields = ["Source", "Target", "ASD_mean", "TC_mean", "Difference"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_rgb_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw = b"".join(
        b"\x00" + b"".join(bytes(color) for color in row) for row in pixels
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _difference_color(value: float, maximum: float) -> tuple[int, int, int]:
    """Blue (negative) -> white (zero) -> red (positive)."""
    if maximum <= 0.0:
        return 255, 255, 255
    position = float(np.clip(0.5 + 0.5 * value / maximum, 0.0, 1.0))
    if position <= 0.5:
        intensity = position * 2.0
        return int(round(255 * intensity)), int(round(255 * intensity)), 255
    intensity = (position - 0.5) * 2.0
    return 255, int(round(255 * (1.0 - intensity))), int(round(255 * (1.0 - intensity)))


def write_difference_heatmap(path: Path, difference: np.ndarray, cell_size: int = 8) -> None:
    if cell_size < 1:
        raise ValueError("--cell-size must be at least 1")
    nodes = difference.shape[0]
    maximum = float(np.max(np.abs(difference)))
    margin = 20
    colorbar_width = 24
    gap = 16
    matrix_size = nodes * cell_size
    width = margin + matrix_size + gap + colorbar_width + margin
    height = margin + matrix_size + margin
    white = (255, 255, 255)
    pixels = [[white for _ in range(width)] for _ in range(height)]

    for source in range(nodes):
        for target in range(nodes):
            color = _difference_color(float(difference[source, target]), maximum)
            y0 = margin + source * cell_size
            x0 = margin + target * cell_size
            for y in range(y0, y0 + cell_size):
                pixels[y][x0:x0 + cell_size] = [color] * cell_size

    # Compact color bar: blue = TC > ASD, red = ASD > TC.
    bar_x = margin + matrix_size + gap
    for y in range(matrix_size):
        value = maximum * (1.0 - 2.0 * y / max(matrix_size - 1, 1))
        color = _difference_color(value, maximum)
        pixels[margin + y][bar_x:bar_x + colorbar_width] = [color] * colorbar_width

    # Thin black frame around the matrix.
    for x in range(margin, margin + matrix_size):
        pixels[margin - 1][x] = (0, 0, 0)
        pixels[margin + matrix_size][x] = (0, 0, 0)
    for y in range(margin, margin + matrix_size):
        pixels[y][margin - 1] = (0, 0, 0)
        pixels[y][margin + matrix_size] = (0, 0, 0)
    _write_rgb_png(path, pixels)


def main() -> None:
    args = parse_args()
    data = load_bec_archive(args.bec_path)
    mean_asd, mean_tc, difference, asd_mask, tc_mask = compute_group_means(
        data["bec"], data["labels"], args.asd_label, args.tc_label
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Remove only files produced by earlier versions of this same analysis.
    stale_names = {
        "summary.json", "group_bec_matrices.npz", "edgewise_group_difference.csv",
        "top_positive_edges.csv", "top_negative_edges.csv", "group_counts.csv",
        "group_mean_bec_heatmaps.png", "top_edge_differences.png",
    }
    for name in stale_names:
        (args.output_dir / name).unlink(missing_ok=True)

    table_path = args.output_dir / "group_bec_table.csv"
    heatmap_path = args.output_dir / "group_bec_difference_heatmap.png"
    write_group_table(table_path, mean_asd, mean_tc, difference, data["roi_names"])
    write_difference_heatmap(heatmap_path, difference, args.cell_size)

    print(f"ASD subjects: {int(asd_mask.sum())}")
    print(f"TC subjects: {int(tc_mask.sum())}")
    print(f"Difference range: {difference.min():.6g} to {difference.max():.6g}")
    print(f"Saved table: {table_path.resolve()}")
    print(f"Saved heatmap: {heatmap_path.resolve()}")


if __name__ == "__main__":
    main()
