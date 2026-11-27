"""Preprocess the MTLnet real resting-state fMRI dataset for MemoryEC."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "dataset" / "MTLnet" / "functional"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "MTLnet" / "processed"
SIDES = {"left": "L", "right": "R"}
ROI_SOURCES = {
    "CA1": ("CA1",),
    "CA23DG": ("CA2", "CA3", "DG"),
    "SUB": ("SUB",),
    "ERC": ("ERC",),
    "BA35": ("BA35",),
    "BA36": ("BA36",),
    "PHC": ("PHC",),
}
ROI_NAMES = tuple(ROI_SOURCES)
PAPER_REPORTED_TIMEPOINTS = 421
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
    return int(match.group(1))


def roi_name(raw_name: str) -> str:
    return raw_name.rsplit("_", maxsplit=1)[-1]


def load_subject(path: Path) -> tuple[np.ndarray, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values = []
    names = []
    for line in lines[1:]:
        fields = line.split()
        names.append(roi_name(fields[0]))
        values.append([float(value) for value in fields[1:]])

    array = np.asarray(values, dtype=np.float32)
    reduced = np.stack([
        array[[names.index(name) for name in source_names]].mean(axis=0)
        for source_names in ROI_SOURCES.values()
    ])
    return reduced.T, list(ROI_NAMES)


def complete_subject_dirs(source_dir: Path) -> tuple[list[Path], list[int], int]:
    subject_dirs = sorted(
        (path for path in source_dir.glob("S*") if path.is_dir()),
        key=subject_number,
    )
    lengths = {}
    for subject_dir in subject_dirs:
        with (subject_dir / "res_tcourse_L.txt").open(encoding="utf-8") as file:
            lengths[subject_dir] = int(file.readline().split()[1])
    complete_length = max(lengths.values())
    selected = [path for path in subject_dirs if lengths[path] == complete_length]
    excluded = [subject_number(path) for path in subject_dirs if lengths[path] != complete_length]
    return selected, excluded, complete_length


def preprocess_side(
    output_dir: Path,
    side: str,
    subject_dirs: list[Path],
    excluded_subject_ids: list[int],
    numeric_timepoints: int,
) -> dict:
    suffix = SIDES[side]
    subject_ids = [subject_number(path) for path in subject_dirs]

    subjects = [
        load_subject(subject_dir / f"res_tcourse_{suffix}.txt")[0]
        for subject_dir in subject_dirs
    ]
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
        "subject_ids": subject_ids,
        "excluded_incomplete_subject_ids": excluded_subject_ids,
        "paper_reported_timepoints": PAPER_REPORTED_TIMEPOINTS,
        "numeric_timepoints_loaded": numeric_timepoints,
        "timepoint_lengths": sorted(set(lengths.tolist())),
        "roi_names": list(ROI_NAMES),
        "roi_construction": {name: list(sources) for name, sources in ROI_SOURCES.items()},
        "ground_truth_edges": int(GROUND_TRUTH.sum()),
        "output": str(output_path),
    }


def preprocess(source_dir: Path, output_dir: Path) -> list[dict]:
    subject_dirs, excluded_subject_ids, numeric_timepoints = complete_subject_dirs(source_dir)
    summaries = [
        preprocess_side(
            output_dir,
            side,
            subject_dirs,
            excluded_subject_ids,
            numeric_timepoints,
        )
        for side in SIDES
    ]
    metadata_path = Path(__file__).resolve().parent / "metadata.json"
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
