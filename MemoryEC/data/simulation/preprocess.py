"""Preprocess Sanchez-Romero 2019 simple-network simulations."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "dataset" / "SanchezRomero2019_SimpleNetworks"
SIMULATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SIMULATION_DIR / "processed"
DEFAULT_SPLITS = SIMULATION_DIR / "splits.json"
SCENE_PATTERN = re.compile(r"Network(\d+)(?:_|$)")
EDGE_PATTERN = re.compile(r"X(\d+)\s*-->\s*X(\d+)")

BASE_SPLITS = {
    "train": [1, 2, 3, 4, 5, 6],
    "val": [7],
    "test": [8, 9],
}


def base_network(scene_name: str) -> int:
    match = SCENE_PATTERN.match(scene_name)
    return int(match.group(1))


def subject_number(path: Path) -> int:
    match = re.fullmatch(r"BOLDfslfilter_(\d+)\.txt", path.name)
    return int(match.group(1))


def load_bold(scene_dir: Path) -> tuple[np.ndarray, list[str]]:
    data_dir = scene_dir / "data_fslfilter"
    files = sorted(data_dir.glob("BOLDfslfilter_*.txt"), key=subject_number)
    with files[0].open(encoding="utf-8") as file:
        roi_names = file.readline().strip().split()
    subjects = [np.loadtxt(path, skiprows=1, dtype=np.float32) for path in files]
    return np.stack(subjects), roi_names


def load_graph(scene_dir: Path, roi_names: list[str]) -> np.ndarray:
    graph_files = sorted((scene_dir / "graph").glob("*.txt"))
    text = graph_files[0].read_text(encoding="utf-8")
    adj = np.zeros((len(roi_names), len(roi_names)), dtype=np.uint8)
    for source, target in EDGE_PATTERN.findall(text):
        source_index = int(source) - 1
        target_index = int(target) - 1
        adj[source_index, target_index] = 1
    return adj


def write_scene(output_path: Path, bold: np.ndarray, adj: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".npz.part")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, bold=bold, adj=adj)
    os.replace(temporary_path, output_path)


def make_splits(scene_names: list[str]) -> dict[str, dict[str, list]]:
    return {
        split: {
            "base_networks": networks,
            "scenes": [
                scene for scene in scene_names if base_network(scene) in networks
            ],
        }
        for split, networks in BASE_SPLITS.items()
    }


def preprocess(source_dir: Path, output_dir: Path, splits_path: Path) -> None:
    scene_dirs = sorted(
        (path for path in source_dir.glob("Network*") if path.is_dir()),
        key=lambda path: (base_network(path.name), path.name),
    )
    summaries = []
    for scene_dir in scene_dirs:
        bold, roi_names = load_bold(scene_dir)
        adj = load_graph(scene_dir, roi_names)
        write_scene(output_dir / f"{scene_dir.name}.npz", bold, adj)
        summaries.append(
            f"{scene_dir.name}: bold={tuple(bold.shape)}, adj={tuple(adj.shape)}, "
            f"edges={int(adj.sum())}"
        )

    splits = make_splits([path.name for path in scene_dirs])
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Wrote scenes:")
    for summary in summaries:
        print(f"  {summary}")
    print(f"Wrote splits: {splits_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    preprocess(arguments.source, arguments.output, arguments.splits)
