"""Utilities for loading preprocessed Sanchez-Romero scenes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_PATH = DATA_DIR / "splits.json"


def load_scene(
    scene: str,
    processed_dir: str | Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one scene as ``bold [S,T,N]`` and ``adj [N,N]``."""
    path = Path(processed_dir) / f"{scene}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Processed scene not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"bold", "adj"}:
            raise ValueError(f"{path} must contain exactly 'bold' and 'adj'")
        bold = data["bold"]
        adj = data["adj"]

    if bold.ndim != 3:
        raise ValueError(f"{path}: bold must have shape [S,T,N], got {bold.shape}")
    if adj.shape != (bold.shape[-1], bold.shape[-1]):
        raise ValueError(
            f"{path}: adj shape {adj.shape} does not match N={bold.shape[-1]}"
        )
    if not np.isfinite(bold).all() or not np.isfinite(adj).all():
        raise ValueError(f"{path}: arrays contain non-finite values")
    if np.any(np.diag(adj) != 0):
        raise ValueError(f"{path}: adj diagonal must be zero")

    return bold, adj


def load_splits(path: str | Path = SPLITS_PATH) -> dict[str, dict[str, list]]:
    """Load scene splits and verify that base networks do not overlap."""
    split_path = Path(path)
    with split_path.open(encoding="utf-8") as file:
        splits = json.load(file)

    expected = {"train", "val", "test"}
    if set(splits) != expected:
        raise ValueError(f"{split_path}: expected split keys {sorted(expected)}")

    seen_networks: set[int] = set()
    seen_scenes: set[str] = set()
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        if set(split) != {"base_networks", "scenes"}:
            raise ValueError(
                f"{split_path}: {split_name} must contain base_networks and scenes"
            )
        networks = set(split["base_networks"])
        scenes = set(split["scenes"])
        if networks & seen_networks:
            raise ValueError(f"{split_path}: base networks overlap across splits")
        if scenes & seen_scenes:
            raise ValueError(f"{split_path}: scenes overlap across splits")
        seen_networks.update(networks)
        seen_scenes.update(scenes)

    return splits
