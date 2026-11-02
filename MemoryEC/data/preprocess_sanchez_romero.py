"""Preprocess Sanchez-Romero 2019 simple-network simulations."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "dataset" / "SanchezRomero2019_SimpleNetworks"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "processed"
DEFAULT_SPLITS = Path(__file__).resolve().parent / "splits.json"
SCENE_PATTERN = re.compile(r"Network(\d+)(?:_|$)")
EDGE_PATTERN = re.compile(r"X(\d+)\s*-->\s*X(\d+)")
NODE_PATTERN = re.compile(r"X(\d+)")

BASE_SPLITS = {
    "train": [1, 2, 3, 4, 5, 6],
    "val": [7],
    "test": [8, 9],
}


def base_network(scene_name: str) -> int:
    match = SCENE_PATTERN.match(scene_name)
    if match is None:
        raise ValueError(f"Invalid scene directory name: {scene_name}")
    return int(match.group(1))


def subject_number(path: Path) -> int:
    match = re.fullmatch(r"BOLDfslfilter_(\d+)\.txt", path.name)
    if match is None:
        raise ValueError(f"Unexpected BOLD filename: {path}")
    return int(match.group(1))


def load_bold(scene_dir: Path) -> tuple[np.ndarray, list[str]]:
    data_dir = scene_dir / "data_fslfilter"
    files = sorted(data_dir.glob("BOLDfslfilter_*.txt"), key=subject_number)
    if not files:
        raise ValueError(f"No single-subject BOLD files found in {data_dir}")

    subject_ids = [subject_number(path) for path in files]
    expected_ids = list(range(1, len(files) + 1))
    if subject_ids != expected_ids:
        raise ValueError(
            f"{scene_dir.name}: subject IDs are not contiguous: {subject_ids}"
        )

    subjects: list[np.ndarray] = []
    expected_header: list[str] | None = None
    expected_shape: tuple[int, int] | None = None
    for path in files:
        with path.open(encoding="utf-8") as file:
            header = file.readline().strip().split()
        values = np.loadtxt(path, skiprows=1, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{path}: expected a 2D [T,N] table, got {values.shape}")
        if len(header) != values.shape[1]:
            raise ValueError(f"{path}: header and ROI count do not match")
        if expected_header is None:
            expected_header = header
            expected_shape = values.shape
        if header != expected_header or values.shape != expected_shape:
            raise ValueError(f"{path}: inconsistent header or [T,N] shape")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: BOLD data contain non-finite values")
        subjects.append(values)

    assert expected_header is not None
    expected_rois = [f"X{index}" for index in range(1, len(expected_header) + 1)]
    if expected_header != expected_rois:
        raise ValueError(
            f"{scene_dir.name}: expected ROI order {expected_rois}, got {expected_header}"
        )
    return np.stack(subjects), expected_header


def load_graph(scene_dir: Path, roi_names: list[str]) -> np.ndarray:
    graph_files = sorted((scene_dir / "graph").glob("*.txt"))
    if len(graph_files) != 1:
        raise ValueError(f"{scene_dir.name}: expected exactly one graph text file")

    text = graph_files[0].read_text(encoding="utf-8")
    node_section = text.split("Graph Edges:", maxsplit=1)[0]
    graph_nodes = [f"X{node}" for node in NODE_PATTERN.findall(node_section)]
    if graph_nodes != roi_names:
        raise ValueError(
            f"{scene_dir.name}: graph nodes {graph_nodes} do not match {roi_names}"
        )

    adj = np.zeros((len(roi_names), len(roi_names)), dtype=np.uint8)
    for source, target in EDGE_PATTERN.findall(text):
        source_index = int(source) - 1
        target_index = int(target) - 1
        if not 0 <= source_index < len(roi_names):
            raise ValueError(f"{scene_dir.name}: invalid source node X{source}")
        if not 0 <= target_index < len(roi_names):
            raise ValueError(f"{scene_dir.name}: invalid target node X{target}")
        if source_index == target_index:
            raise ValueError(f"{scene_dir.name}: graph contains a self-edge")
        adj[source_index, target_index] = 1
    return adj


def verify_amatrices(scene_dir: Path, adj: np.ndarray, subjects: int) -> None:
    contents = loadmat(scene_dir / "Amatrices.mat")
    variables = {key: value for key, value in contents.items() if not key.startswith("__")}
    if set(variables) != {"aa"}:
        raise ValueError(f"{scene_dir.name}: unexpected MAT variables {sorted(variables)}")

    matrices = np.asarray(variables["aa"])
    nodes = adj.shape[0]
    if matrices.shape != (subjects, nodes, nodes):
        raise ValueError(
            f"{scene_dir.name}: aa shape {matrices.shape}, expected "
            f"{(subjects, nodes, nodes)}"
        )
    if not np.isfinite(matrices).all():
        raise ValueError(f"{scene_dir.name}: aa contains non-finite values")
    if not np.all(np.diagonal(matrices, axis1=1, axis2=2) == -1):
        raise ValueError(f"{scene_dir.name}: aa diagonal is not uniformly -1")

    matrix_support = matrices != 0
    diagonal = np.eye(nodes, dtype=bool)
    expected_support = np.logical_or(adj.astype(bool), diagonal)
    if not np.all(matrix_support == expected_support[None, :, :]):
        raise ValueError(
            f"{scene_dir.name}: graph direction disagrees with aa support; "
            "expected aa[:, i, j] != 0 to mean Xi -> Xj"
        )


def write_scene(output_path: Path, bold: np.ndarray, adj: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".npz.part")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, bold=bold, adj=adj)
    os.replace(temporary_path, output_path)


def make_splits(scene_names: list[str]) -> dict[str, dict[str, list]]:
    assigned = set().union(*map(set, BASE_SPLITS.values()))
    available = {base_network(scene) for scene in scene_names}
    if assigned != available:
        raise ValueError(
            f"Split base networks {sorted(assigned)} do not match available "
            f"networks {sorted(available)}"
        )

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
    if not scene_dirs:
        raise ValueError(f"No scene directories found in {source_dir}")

    summaries = []
    for scene_dir in scene_dirs:
        bold, roi_names = load_bold(scene_dir)
        adj = load_graph(scene_dir, roi_names)
        verify_amatrices(scene_dir, adj, bold.shape[0])
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

    print("Validated and wrote scenes:")
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
