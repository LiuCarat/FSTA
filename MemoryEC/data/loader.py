"""Utilities for loading preprocessed Sanchez-Romero scenes."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

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


def base_network(scene: str) -> int:
    """Return the integer family identifier from a scene name."""
    match = re.match(r"Network(\d+)(?:_|$)", scene)
    if match is None:
        raise ValueError(f"Invalid scene name: {scene}")
    return int(match.group(1))


class SanchezRomeroDataset:
    """Subject-level samples backed by the preprocessed scene arrays."""

    def __init__(
        self,
        scenes: list[str],
        processed_dir: str | Path = PROCESSED_DIR,
    ) -> None:
        self.samples: list[dict] = []
        self.indices_by_network: dict[int, list[int]] = defaultdict(list)
        self.indices_by_nodes: dict[int, list[int]] = defaultdict(list)

        for scene in scenes:
            bold, adj = load_scene(scene, processed_dir)
            network = base_network(scene)
            for subject in range(bold.shape[0]):
                index = len(self.samples)
                self.samples.append(
                    {
                        "bold": bold[subject],
                        "adj": adj,
                        "scene": scene,
                        "subject": subject,
                        "base_network": network,
                        "nodes": bold.shape[-1],
                    }
                )
                self.indices_by_network[network].append(index)
                self.indices_by_nodes[bold.shape[-1]].append(index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class BalancedNetworkBatchSampler:
    """Sample base networks uniformly while keeping one ROI size per batch."""

    def __init__(
        self,
        dataset: SanchezRomeroDataset,
        batch_size: int,
        seed: int,
        samples_per_epoch: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.networks = sorted(dataset.indices_by_network)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        buffers: dict[int, list[int]] = defaultdict(list)
        yielded = 0
        for _ in range(self.samples_per_epoch):
            network = int(rng.choice(self.networks))
            candidates = self.dataset.indices_by_network[network]
            index = int(rng.choice(candidates))
            nodes = self.dataset.samples[index]["nodes"]
            buffers[nodes].append(index)
            if len(buffers[nodes]) == self.batch_size:
                yield buffers.pop(nodes)
                yielded += self.batch_size

        for nodes in sorted(buffers):
            if buffers[nodes]:
                yield buffers[nodes]
                yielded += len(buffers[nodes])
        if yielded != self.samples_per_epoch:
            raise RuntimeError("Balanced sampler yielded an unexpected sample count")


class GroupedBatchSampler:
    """Deterministic evaluation batches grouped by ROI count."""

    def __init__(self, dataset: SanchezRomeroDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.dataset.indices_by_nodes.values()
        )

    def __iter__(self) -> Iterator[list[int]]:
        for nodes in sorted(self.dataset.indices_by_nodes):
            indices = self.dataset.indices_by_nodes[nodes]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]


def collate_samples(samples: list[dict], permute_rois: bool = False) -> dict:
    """Stack equal-size samples and optionally apply synchronized ROI permutations."""
    import torch

    node_counts = {sample["nodes"] for sample in samples}
    if len(node_counts) != 1:
        raise ValueError(f"A batch must have one ROI count, got {sorted(node_counts)}")

    bold = torch.from_numpy(np.stack([sample["bold"] for sample in samples])).float()
    adj = torch.from_numpy(np.stack([sample["adj"] for sample in samples])).float()
    if permute_rois:
        for batch_index in range(len(samples)):
            permutation = torch.randperm(bold.shape[-1])
            bold[batch_index] = bold[batch_index, :, permutation]
            adj[batch_index] = adj[batch_index][permutation][:, permutation]

    return {
        "bold": bold,
        "adj": adj,
        "scene": [sample["scene"] for sample in samples],
        "subject": torch.tensor([sample["subject"] for sample in samples]),
        "base_network": torch.tensor(
            [sample["base_network"] for sample in samples]
        ),
    }


def make_dataloader(
    split: str,
    batch_size: int = 8,
    seed: int = 42,
    permute_rois: bool = False,
    balanced: bool | None = None,
    processed_dir: str | Path = PROCESSED_DIR,
    splits_path: str | Path = SPLITS_PATH,
):
    """Build a DataLoader without padding different ROI counts."""
    from functools import partial

    from torch.utils.data import DataLoader

    splits = load_splits(splits_path)
    if split not in splits:
        raise ValueError(f"Unknown split: {split}")
    dataset = SanchezRomeroDataset(splits[split]["scenes"], processed_dir)
    use_balanced = split == "train" if balanced is None else balanced
    if use_balanced:
        batch_sampler = BalancedNetworkBatchSampler(dataset, batch_size, seed)
    else:
        batch_sampler = GroupedBatchSampler(dataset, batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate_samples, permute_rois=permute_rois),
        num_workers=0,
    )
    return loader
