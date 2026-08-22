#!/usr/bin/env python3
"""Leakage-safe fold-wise FSTA-Graph-BEC and BrainNetCNN runner.

Each fold fits PGR/QC on training subjects only, immediately classifies the
fold-local representations, and contributes only held-out BECs to the OOF NPZ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_BEC.config import parse_args
from Graph_BEC.data import load_pipeline_data
from Graph_BEC.utils import select_device, set_seed
from Graph_BEC.utils.output import (
    print_summary_table,
    save_refined_bec_archive,
    save_qsr_bec_archive,
    save_results,
)
from Graph_BEC.workflow import run_cross_validation


def main():
    args = parse_args(__doc__)
    device = select_device(args.gpu_id)
    set_seed(args.seed)
    print(
        f"Dataset={args.dataset}; "
        f"loading data from {args.data_root} "
        f"with phenotype {args.phenotype_csv}..."
    )
    data, fsta_metrics = load_pipeline_data(args, device)
    if np.unique(data["labels"]).size != 2:
        raise ValueError("The dataset must contain exactly two patient/control labels")
    print(
        f"Input={args.input_mode}; subjects={len(data['bec'])}; "
        f"BEC={data['bec'].shape}; labels={np.bincount(data['labels'])}; device={device}"
    )

    print("\n===== fold refinement + BrainNetCNN =====")
    experiment = run_cross_validation(args, data, device)
    refined_path = qsr_refined_path = None
    if "refined" in args.representations and "qc_refined" in args.representations:
        refined_path = save_refined_bec_archive(
            args.refined_bec_path, data, experiment["oof_pgr"],
            experiment["oof_qc"], experiment["fold_ids"], args.bec_path,
        )
        print(f"Saved refined NPZ: {refined_path.resolve()}")
    if "qc_refined" in args.representations:
        qsr_refined_path = save_qsr_bec_archive(
            args.qsr_refined_bec_path, data, experiment["oof_qc"],
            experiment["fold_ids"], args.bec_path,
        )
        print(f"Saved QSR-refined NPZ: {qsr_refined_path.resolve()}")
    training_summary = {
        "fsta": fsta_metrics,
        "refinement_folds": experiment["refinement_metrics"],
        "representations": args.representations,
        "refined_bec_path": str(refined_path.resolve()) if refined_path else None,
        "qsr_refined_bec_path": str(qsr_refined_path.resolve()) if qsr_refined_path else None,
        "classification_protocol": (
            "fold-local train/val/test representations; test-only OOF NPZ"
        ),
    }
    summary = save_results(args, experiment["fold_results"], training_summary)
    print_summary_table(summary, title="fold-local mean±std (%)")

    print(
        "\nQSR-BEC configuration:\n"
        f"  QC columns: {', '.join(args.qsr_qc_columns)}\n"
        f"  Training: epochs={args.qsr_epochs}; lr={args.qsr_lr:g}; "
        f"hidden_channels={args.qsr_hidden_channels}\n"
        f"  Pseudo-target: eta={args.qsr_eta:g}; r_max={args.qsr_r_max:g}\n"
        f"  Synthetic corruption: scale={args.qsr_corruption_scale:g}"
    )


if __name__ == "__main__":
    main()
