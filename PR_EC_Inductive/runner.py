"""Shared execution flow for the dataset-specific Graph-EC entry points."""
from __future__ import annotations

import numpy as np

from PR_EC_Inductive.data import load_inductive_data
from PR_EC_Inductive.utils import select_device, set_seed
from PR_EC_Inductive.utils.output import (
    print_summary_table,
    save_inductive_ec_archive,
    save_oof_predictions,
    save_results,
)
from PR_EC_Inductive.workflow import run_cross_validation


def run(args):
    device = select_device(args.gpu_id)
    set_seed(args.seed)
    if args.encoder_cache_dir is None:
        args.encoder_cache_dir = args.output_dir / "fold_encoders"
    print(
        f"Dataset={args.dataset}; loading data from {args.data_root} "
        f"with phenotype {args.phenotype_csv}..."
    )
    if args.input_mode != "raw":
        raise ValueError("Inductive evaluation requires raw time series")
    args.representations = list(dict.fromkeys(args.representations))
    data = load_inductive_data(args)
    if data["subjects"] is None:
        raise RuntimeError("Raw subject time series were not attached")
    if np.unique(data["labels"]).size != 2:
        raise ValueError("The dataset must contain exactly two patient/control labels")
    print(
        f"Protocol=fold-specific-inductive; "
        f"subjects={len(data['labels'])}; "
        f"labels={np.bincount(data['labels'])}; device={device}"
    )

    print("\n===== fold refinement + BrainNetCNN =====")
    experiment = run_cross_validation(args, data, device)
    initial_path = save_inductive_ec_archive(
        args.output_dir / "inductive_oof_initial_ec.npz",
        data, experiment["oof_original"], experiment["fold_ids"],
        "initial_ec",
    )
    refined_path = None
    if "refined" in args.representations:
        refined_path = save_inductive_ec_archive(
            args.output_dir / "inductive_oof_mpr_ec.npz",
            data, experiment["oof_pgr"], experiment["fold_ids"],
            "mpr_only",
        )
    qsr_refined_path = None
    if "qc_refined" in args.representations:
        qsr_refined_path = save_inductive_ec_archive(
            args.output_dir / "inductive_oof_full_ec.npz",
            data, experiment["oof_qc"], experiment["fold_ids"],
            "full_pr_ec",
        )
    prediction_path = save_oof_predictions(
        args.output_dir, data, experiment["fold_ids"], experiment["oof_probabilities"]
    )
    print(f"Saved OOF predictions: {prediction_path.resolve()}")
    training_summary = {
        "fold_specific_stf_ec": experiment["encoder_metrics"],
        "refinement_folds": experiment["refinement_metrics"],
        "representations": args.representations,
        "initial_ec_path": str(initial_path.resolve()),
        "refined_ec_path": str(refined_path.resolve()) if refined_path else None,
        "qsr_refined_ec_path": str(qsr_refined_path.resolve()) if qsr_refined_path else None,
        "classification_protocol": "fold-specific inductive STF-EC; fold-local MPR/QSR/classifier; test-only OOF outputs",
        "completed_all_folds": experiment["completed"],
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
