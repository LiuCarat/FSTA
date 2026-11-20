
from __future__ import annotations

import numpy as np

from pr_ec.data import load_pipeline_data
from pr_ec.utils import select_device, set_seed
from pr_ec.utils.output import (
    save_pr_ec_archive,
)
from pr_ec.workflow import run_cross_validation


def run(args):
    device = select_device(args.gpu_id)
    set_seed(args.seed)
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Device: {device}")
    print("[INFO] Loading raw fMRI time series...")
    data, _ = load_pipeline_data(args, device)
    if np.unique(data["labels"]).size != 2:
        raise ValueError("The dataset must contain exactly two patient/control labels")
    print(f"[INFO] Individual-EC shape: {list(data['ec'].shape)}")
    print(f"[INFO] Labels: {np.bincount(data['labels']).tolist()}")

    print("[STAGE 3/4] Fold-local QSR refinement")
    experiment = run_cross_validation(args, data, device)
    print("[STAGE 3/4] All folds completed")
    print(f"[INFO] PR-EC shape: {list(experiment['oof_qc'].shape)}")
    pr_ec_path = save_pr_ec_archive(
        args.pr_ec_path, data, experiment["oof_qc"],
        experiment["fold_ids"],
    )
    print(f"[INFO] Saving QSR Refinement")
    print(f"[INFO] pr_ec_path={pr_ec_path.resolve()}")
    metric_names = ("ACC", "Recall", "AUC", "Precision", "F1")
    print("[STAGE 4/4] Downstream classification completed")
    print("[INFO] QSR Refinement classification metrics (mean ± std)")
    for name in metric_names:
        values = [result["pr_ec"][name] for result in experiment["fold_results"]]
        print(f"  {name:<10}: {100 * np.mean(values):.2f}% ± {100 * np.std(values):.2f}%")
    print("[INFO] Pipeline completed successfully")
