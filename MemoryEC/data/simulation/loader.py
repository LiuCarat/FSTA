"""Utilities for loading preprocessed Sanchez-Romero simulation scenes."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent
SIMULATION_DIR = DATA_DIR / "simulation"
PROCESSED_DIR = SIMULATION_DIR / "processed"
SPLITS_PATH = SIMULATION_DIR / "splits.json"


def load_scene(
    scene: str,
    processed_dir: str | Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one scene as ``bold [S,T,N]`` and ``adj [N,N]``."""
    path = Path(processed_dir) / f"{scene}.npz"
    with np.load(path, allow_pickle=False) as data:
        bold = data["bold"]
        adj = data["adj"]
    return bold, adj


def load_splits(path: str | Path = SPLITS_PATH) -> dict[str, dict[str, list]]:
    """Load scene splits."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def base_network(scene: str) -> int:
    """Return the integer family identifier from a scene name."""
    match = re.match(r"Network(\d+)(?:_|$)", scene)
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
        for _ in range(self.samples_per_epoch):
            network = int(rng.choice(self.networks))
            candidates = self.dataset.indices_by_network[network]
            index = int(rng.choice(candidates))
            nodes = self.dataset.samples[index]["nodes"]
            buffers[nodes].append(index)
            if len(buffers[nodes]) == self.batch_size:
                yield buffers.pop(nodes)

        for nodes in sorted(buffers):
            if buffers[nodes]:
                yield buffers[nodes]


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
    scenes: list[str] | None = None,
):
    """Build a DataLoader without padding different ROI counts."""
    from functools import partial

    from torch.utils.data import DataLoader

    if scenes is None:
        splits = load_splits(splits_path)
        scenes = splits[split]["scenes"]
    dataset = SanchezRomeroDataset(
        scenes, processed_dir
    )
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
