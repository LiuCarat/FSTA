"""Preprocess the MTLnet real resting-state fMRI dataset for MemoryEC."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "dataset" / "MTLnet" / "functional"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "real" / "MTLnet" / "processed"
SIDES = {"left": "L", "right": "R"}
ROI_NAMES = ("CA1", "CA2", "DG", "CA3", "SUB", "ERC", "PHC")
GROUND_TRUTH = np.array(
    [
        [0, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def subject_number(path: Path) -> int:
    match = re.fullmatch(r"S(\d+)", path.name)
    if match is None:
        raise ValueError(f"Unexpected subject directory: {path}")
    return int(match.group(1))


def roi_name(raw_name: str) -> str:
    return raw_name.rsplit("_", maxsplit=1)[-1]


def load_subject(path: Path) -> tuple[np.ndarray, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty MTLnet file: {path}")
    declared_rows, declared_timepoints = map(int, lines[0].split())
    if len(lines) != declared_rows + 1:
        raise ValueError(
            f"{path}: declared {declared_rows} rows, found {len(lines) - 1}"
        )

    values = []
    names = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != declared_timepoints + 1:
            raise ValueError(
                f"{path}: expected {declared_timepoints} values, found {len(fields) - 1}"
            )
        names.append(roi_name(fields[0]))
        values.append([float(value) for value in fields[1:]])

    array = np.asarray(values, dtype=np.float32)
    missing = [name for name in ROI_NAMES if name not in names]
    if missing:
        raise ValueError(f"{path}: missing reduced ROIs {missing}")
    indices = [names.index(name) for name in ROI_NAMES]
    reduced = array[indices]
    if not np.isfinite(reduced).all():
        raise ValueError(f"{path}: selected reduced ROIs contain non-finite values")
    return reduced.T, [names[index] for index in indices]


def preprocess_side(source_dir: Path, output_dir: Path, side: str) -> dict:
    suffix = SIDES[side]
    subject_dirs = sorted(
        (path for path in source_dir.glob("S*") if path.is_dir()),
        key=subject_number,
    )
    subject_ids = [subject_number(path) for path in subject_dirs]
    if subject_ids != list(range(1, len(subject_dirs) + 1)):
        raise ValueError(f"MTLnet subject IDs are not contiguous: {subject_ids}")

    subjects = []
    expected_rois = None
    for subject_dir in subject_dirs:
        bold, names = load_subject(subject_dir / f"res_tcourse_{suffix}.txt")
        if expected_rois is None:
            expected_rois = names
        elif names != expected_rois:
            raise ValueError(f"{subject_dir}: inconsistent ROI order")
        subjects.append(bold)

    if not subjects:
        raise ValueError(f"No MTLnet subjects found in {source_dir}")
    lengths = np.asarray([subject.shape[0] for subject in subjects], dtype=np.int32)
    padded = np.zeros((len(subjects), int(lengths.max()), len(ROI_NAMES)), dtype=np.float32)
    for index, subject in enumerate(subjects):
        padded[index, : subject.shape[0]] = subject

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{side}.npz"
    temporary_path = output_path.with_suffix(".npz.part")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            bold=padded,
            lengths=lengths,
            subject_ids=np.asarray(subject_ids, dtype=np.int32),
            roi_names=np.asarray(ROI_NAMES),
            adj=GROUND_TRUTH,
        )
    os.replace(temporary_path, output_path)
    return {
        "side": side,
        "subjects": len(subjects),
        "timepoint_lengths": sorted(set(lengths.tolist())),
        "roi_names": list(ROI_NAMES),
        "ground_truth_edges": int(GROUND_TRUTH.sum()),
        "output": str(output_path),
    }


def preprocess(source_dir: Path, output_dir: Path) -> list[dict]:
    summaries = [preprocess_side(source_dir, output_dir, side) for side in SIDES]
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"dataset": "MTLnet", "hemispheres": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for summary in preprocess(arguments.source, arguments.output):
        print(
            f"{summary['side']}: subjects={summary['subjects']}, "
            f"lengths={summary['timepoint_lengths']}, ROIs={summary['roi_names']}, "
            f"edges={summary['ground_truth_edges']}"
        )
