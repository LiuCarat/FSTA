"""Loader for MemoryEC real-data evaluation datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent
MTLNET_PROCESSED_DIR = DATA_DIR / "real" / "MTLnet" / "processed"
MTLNET_ROI_NAMES = ("CA1", "CA23DG", "SUB", "ERC", "BA35", "BA36", "PHC")


def load_mtlnet(
    side: str,
    processed_dir: str | Path = MTLNET_PROCESSED_DIR,
) -> dict[str, np.ndarray]:
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown MTLnet hemisphere: {side}")
    path = Path(processed_dir) / f"{side}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Preprocessed MTLnet data not found: {path}. "
            "Run MemoryEC/data/preprocess_mtlnet.py first."
        )
    with np.load(path, allow_pickle=False) as data:
        expected = {"bold", "lengths", "subject_ids", "roi_names", "adj"}
        if set(data.files) != expected:
            raise ValueError(f"{path}: expected arrays {sorted(expected)}")
        result = {name: data[name] for name in expected}

    bold = result["bold"]
    lengths = result["lengths"]
    adj = result["adj"]
    if bold.ndim != 3 or lengths.shape != (bold.shape[0],):
        raise ValueError(f"{path}: invalid bold/length shapes")
    if adj.shape != (bold.shape[-1], bold.shape[-1]):
        raise ValueError(f"{path}: ground truth does not match ROI count")
    if np.any(lengths < 4) or np.any(lengths > bold.shape[1]):
        raise ValueError(f"{path}: invalid subject timepoint lengths")
    if bold.shape[0] != 23 or not np.all(lengths == 420):
        raise ValueError(f"{path}: expected 23 complete subjects with 420 numeric timepoints")
    if tuple(result["roi_names"].tolist()) != MTLNET_ROI_NAMES:
        raise ValueError(f"{path}: unexpected ROI order {result['roi_names'].tolist()}")
    if not np.isfinite(bold).all() or not np.isfinite(adj).all():
        raise ValueError(f"{path}: contains non-finite values")
    return result
