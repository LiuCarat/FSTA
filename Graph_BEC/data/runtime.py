"""Runtime utilities shared by training and evaluation stages."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Reset every RNG used by one independently comparable stage."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def select_device(gpu_id):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == "auto":
        free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
        gpu_id = int(np.argmax(free))
    return torch.device(f"cuda:{int(gpu_id)}")

