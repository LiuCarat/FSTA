"""Shared execution flow for the dataset-specific Graph-EC entry points."""
from __future__ import annotations

import numpy as np

from PR_EC.data import load_pipeline_data
from PR_EC.utils import select_device, set_seed
from PR_EC.utils.output import (
    print_summary_table,
    save_qsr_ec_archive,
    save_refined_ec_archive,
    save_results,
)
from PR_EC.workflow import run_cross_validation


def run(args):
    device = select_device(args.gpu_id)
    set_seed(args.seed)
    print(
        f"Dataset={args.dataset}; loading data from {args.data_root} "
        f"with phenotype {args.phenotype_csv}..."
    )
    data, stf_metrics = load_pipeline_data(args, device)
    if np.unique(data["labels"]).size != 2:
        raise ValueError("The dataset must contain exactly two patient/control labels")
    print(
        f"Input={args.input_mode}; subjects={len(data['ec'])}; "
        f"EC={data['ec'].shape}; labels={np.bincount(data['labels'])}; device={device}"
    )

    print("\n===== fold refinement + BrainNetCNN =====")
    experiment = run_cross_validation(args, data, device)
    refined_path = qsr_refined_path = None
    if "refined" in args.representations and "qc_refined" in args.representations:
        refined_path = save_refined_ec_archive(
            args.refined_ec_path, data, experiment["oof_pgr"],
            experiment["oof_qc"], experiment["fold_ids"], args.ec_path,
        )
        print(f"Saved refined NPZ: {refined_path.resolve()}")
    if "qc_refined" in args.representations:
        qsr_refined_path = save_qsr_ec_archive(
            args.qsr_refined_ec_path, data, experiment["oof_qc"],
            experiment["fold_ids"], args.ec_path,
        )
        print(f"Saved QSR-refined NPZ: {qsr_refined_path.resolve()}")
    training_summary = {
        "stf_ec": stf_metrics,
        "refinement_folds": experiment["refinement_metrics"],
        "representations": args.representations,
        "refined_ec_path": str(refined_path.resolve()) if refined_path else None,
        "qsr_refined_ec_path": str(qsr_refined_path.resolve()) if qsr_refined_path else None,
        "classification_protocol": "fold-local train/val/test representations; test-only OOF NPZ",
    }
    summary = save_results(args, experiment["fold_results"], training_summary)
    print_summary_table(summary, title="fold-local mean±std (%)")
    print(
        "\nQSR-EC configuration:\n"
        f"  QC columns: {', '.join(args.qsr_qc_columns)}\n"
        f"  Training: epochs={args.qsr_epochs}; lr={args.qsr_lr:g}; "
        f"hidden_channels={args.qsr_hidden_channels}\n"
        f"  Pseudo-target: eta={args.qsr_eta:g}; r_max={args.qsr_r_max:g}\n"
        f"  Synthetic perturbation: scale={args.qsr_perturbation_scale:g}"
    )
