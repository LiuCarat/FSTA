#!/usr/bin/env python3
"""Create publication-ready TIFF heatmaps for four representative subjects."""
from __future__ import annotations

import argparse
import csv
import re
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEC = ROOT / "Graph_BEC/outputs/abide-ii/abide_ii_qsr_refined_subject_bec.npz"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/analysis/outputs/bec_heatmaps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument("--bec-key", default="bec")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs=4, metavar="SUBJECT_ID")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def safe_filename(subject_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", subject_id)


def select_representative_subjects(
    bec: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray
) -> list[tuple[int, str, int]]:
    """Select two subjects per class closest to that class's mean connectivity."""
    off_diagonal = ~np.eye(bec.shape[1], dtype=bool)
    strengths = bec[:, off_diagonal].mean(axis=1)
    selected: list[tuple[int, str, int]] = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError(f"Need at least two subjects with label {label}")
        distances = np.abs(strengths[indices] - strengths[indices].mean())
        order = indices[np.argsort(distances, kind="stable")[:2]]
        selected.extend((int(index), str(subject_ids[index]), label) for index in order)
    return selected


def load_archive(path: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        for required in (key, "labels", "subject_ids"):
            if required not in archive.files:
                raise ValueError(f"BEC archive is missing array: {required}")
        bec = np.asarray(archive[key], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        subject_ids = np.asarray(archive["subject_ids"]).astype(str).reshape(-1)
    if bec.ndim != 3 or bec.shape[1] != bec.shape[2]:
        raise ValueError(f"Expected BEC shape (subjects, rois, rois), got {bec.shape}")
    if len(labels) != len(bec) or len(subject_ids) != len(bec):
        raise ValueError("BEC, labels, and subject_ids have incompatible lengths")
    return bec, labels, subject_ids


def resolve_subjects(
    bec: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    requested: list[str] | None,
) -> list[tuple[int, str, int]]:
    if requested is None:
        return select_representative_subjects(bec, labels, subject_ids)
    positions = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    missing = [subject_id for subject_id in requested if subject_id not in positions]
    if missing:
        raise ValueError(f"Subject ID(s) not found: {', '.join(missing)}")
    return [
        (positions[subject_id], subject_id, int(labels[positions[subject_id]]))
        for subject_id in requested
    ]


def turbo_rgb(values: np.ndarray) -> np.ndarray:
    """Approximate matplotlib's turbo map using evenly spaced anchor colors."""
    anchors = np.array(
        [[48, 18, 59], [35, 92, 185], [32, 166, 184], [139, 213, 72], [244, 227, 48], [122, 4, 3]],
        dtype=np.float64,
    )
    positions = np.linspace(0.0, 1.0, len(anchors))
    channels = [np.interp(values, positions, anchors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)


def write_rgb_tiff(image: np.ndarray, output: Path, dpi: int) -> None:
    """Write an uncompressed 8-bit RGB TIFF without requiring an image package."""
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("Expected an RGB image")
    pixel_bytes = image.tobytes(order="C")
    entries = 11
    ifd_offset = 8
    ifd_size = 2 + entries * 12 + 4
    bits_offset = ifd_offset + ifd_size
    xres_offset = bits_offset + 6
    yres_offset = xres_offset + 8
    pixel_offset = yres_offset + 8
    data = bytearray(b"II" + struct.pack("<H", 42) + struct.pack("<I", ifd_offset))
    data.extend(struct.pack("<H", entries))
    tags = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 3, bits_offset),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (273, 4, 1, pixel_offset),
        (277, 3, 1, 3),
        (278, 4, 1, height),
        (279, 4, 1, len(pixel_bytes)),
        (282, 5, 1, xres_offset),
        (283, 5, 1, yres_offset),
    ]
    for tag, type_code, count, value in tags:
        data.extend(struct.pack("<HHI", tag, type_code, count))
        if type_code == 3 and count == 1:
            data.extend(struct.pack("<H", value) + b"\x00\x00")
        else:
            data.extend(struct.pack("<I", value))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<HHH", 8, 8, 8))
    data.extend(struct.pack("<II", dpi, 1))
    data.extend(struct.pack("<II", dpi, 1))
    data.extend(pixel_bytes)
    output.write_bytes(data)


def plot_heatmap(matrix: np.ndarray, output: Path, vmin: float, vmax: float, dpi: int) -> None:
    normalized = np.nan_to_num((matrix - vmin) / (vmax - vmin), nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    image = turbo_rgb(normalized)
    scale = max(1, int(round(dpi / 100)))
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    write_rgb_tiff(image, output, dpi)


def main() -> None:
    args = parse_args()
    bec, labels, subject_ids = load_archive(args.bec_path, args.bec_key)
    selected = resolve_subjects(bec, labels, subject_ids, args.subjects)
    finite_values = bec[np.isfinite(bec)]
    vmin = float(np.percentile(finite_values, 1))
    vmax = float(np.percentile(finite_values, 99))
    if vmax <= vmin:
        vmax = vmin + np.finfo(float).eps

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "label", "group", "file"])
        writer.writeheader()
        for index, subject_id, label in selected:
            filename = f"{safe_filename(subject_id)}_bec.tif"
            output = args.output_dir / filename
            plot_heatmap(bec[index], output, vmin, vmax, args.dpi)
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "group": "TC" if label == 0 else "ASD",
                    "file": filename,
                }
            )
            print(f"{subject_id}\t{'TC' if label == 0 else 'ASD'}\t{output}")
    print(f"Shared color range: [{vmin:.6g}, {vmax:.6g}]")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
