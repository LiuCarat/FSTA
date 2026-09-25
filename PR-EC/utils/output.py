from __future__ import annotations

from pathlib import Path

import numpy as np


def save_pr_ec_archive(output_path, data, pr_ec, fold_ids):
    pr_ec = np.asarray(pr_ec, dtype=np.float32)
    input_ec = np.asarray(data["individual_ec"], dtype=np.float32)
    if pr_ec.shape != input_ec.shape:
        raise ValueError(
            f"PR-EC shape {pr_ec.shape} does not match "
            f"input shape {input_ec.shape}"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pr_ec=pr_ec,
        labels=np.asarray(data["labels"], dtype=np.int64),
        subject_ids=np.asarray(data["subject_ids"]).astype(str),
        site_ids=np.asarray(data["site_ids"]).astype(str),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        roi_names=np.asarray(
            data.get(
                "roi_names",
                [f"ROI_{index + 1:03d}" for index in range(pr_ec.shape[1])],
            )
        ).astype(str),
        representation=np.asarray("PR-EC"),
    )
    return output_path
