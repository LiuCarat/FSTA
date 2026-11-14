"""Transductive STF fusion ablation with the unchanged original PR-EC downstream."""
from __future__ import annotations

import numpy as np

from PR_EC.utils import select_device, set_seed
from PR_EC.workflow import run_cross_validation
from PR_EC_E2E.data import load_pipeline_data
from PR_EC_E2E.utils.output import (
    print_summary_table, save_qsr_ec_archive, save_refined_ec_archive, save_results,
)
from PR_EC_E2E.w7 import (
    initialize_w7_run, save_split_membership, save_w7_classification,
    validate_reference_subjects, write_json,
)


def run_w7(args):
    initialize_w7_run(args)
    device = select_device(args.gpu_id)
    set_seed(args.seed)
    print(f"W7 {args.dataset}: fusion={args.stf_fusion_mode}; stage={args.w7_stage}; output={args.output_dir}")
    data, stf_metrics = load_pipeline_data(args, device)
    reference = validate_reference_subjects(args, data)
    expected_folds = save_split_membership(args, data, reference)
    if args.input_mode == "raw":
        write_json(args.output_dir / "pretraining_metrics.json", stf_metrics)
    if args.w7_stage == "pretrain":
        write_json(args.output_dir / "pretraining_complete.json", {
            "stf_fusion_mode": args.stf_fusion_mode, "initial_ec_path": args.ec_path,
            "seed": args.seed, "epochs": args.epochs, "stf_checkpoint": args.stf_checkpoint,
            "subjects": len(data["subject_ids"]),
        })
        print(f"Pretraining/extraction complete; no downstream fitting. Initial EC: {args.ec_path}")
        return
    if np.unique(data["labels"]).size != 2:
        raise ValueError("W7 requires the original binary diagnostic labels")
    experiment = run_cross_validation(args, data, device)
    if not np.array_equal(experiment["fold_ids"], expected_folds):
        raise ValueError("Downstream OOF fold IDs differ from the original protocol")
    save_refined_ec_archive(
        args.refined_ec_path, data, experiment["oof_pgr"], experiment["oof_qc"],
        experiment["fold_ids"], args.ec_path,
    )
    save_qsr_ec_archive(
        args.qsr_refined_ec_path, data, experiment["oof_qc"], experiment["fold_ids"], args.ec_path,
    )
    summary = save_results(args, experiment["fold_results"], {
        "stf_ec": stf_metrics, "refinement_folds": experiment["refinement_metrics"],
        "pretraining_protocol": "transductive_full_cohort_unlabeled",
        "entropy_attention": "batch_mean", "stf_fusion_mode": args.stf_fusion_mode,
        "classification_protocol": "original seed42 fold-local train/val/test; test-only OOF EC",
    })
    save_w7_classification(args, experiment["fold_results"])
    write_json(args.output_dir / "downstream_complete.json", {
        "stf_fusion_mode": args.stf_fusion_mode, "seed": args.seed, "n_splits": args.n_splits,
        "initial_ec_path": args.ec_path, "refined_ec_path": args.refined_ec_path,
        "qsr_refined_ec_path": args.qsr_refined_ec_path,
    })
    print_summary_table(summary, title="W7 Initial / MPR / Full PR-EC mean±std (%)")
