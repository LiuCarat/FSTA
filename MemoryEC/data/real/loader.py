"""Loader for MemoryEC real-data evaluation datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent
MTLNET_PROCESSED_DIR = DATA_DIR / "real" / "MTLnet" / "processed"


def load_mtlnet(
    side: str,
    processed_dir: str | Path = MTLNET_PROCESSED_DIR,
) -> dict[str, np.ndarray]:
    path = Path(processed_dir) / f"{side}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {
            name: data[name]
            for name in ("bold", "lengths", "subject_ids", "roi_names", "adj")
        }
