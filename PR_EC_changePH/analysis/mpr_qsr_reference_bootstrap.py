#!/usr/bin/env python3
"""Assess MPR-reference/QSR training-pool sensitivity without refitting initial EC."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PR_EC.data import load_pipeline_data
from PR_EC.main_abide_i import parse_args as parse_abide_i_args
from PR_EC.main_abide_ii import parse_args as parse_abide_ii_args
from PR_EC.utils import select_device, set_seed
from PR_EC.utils.output import save_qsr_ec_archive
from PR_EC.workflow import build_fold_reference
from PR_EC.utils import make_stratified_splits, prepare_fold_arrays
from PR_EC.model.qsr.qsr_refiner import apply_qsr_refiner, train_qsr_refiner

sys.path.insert(0, str(Path(__file__).resolve().parent / "cross_cohort_replication"))
from run_covariate_adjusted_replication import (
    FORMULA, DEFAULT_PHENOTYPES, align_covariates, fit_edges, file_hash,
)
from run_cross_cohort_replication import (
    DEFAULT_INPUTS, load_archive, write_csv, compute_effects_and_tests, edge_table, merge_tables,
)


def run_bootstrap_qsr_cross_validation(args, data, device, bootstrap_seed):
    """Run fold-fixed, training-pool subject bootstrap QSR OOF inference.

    The original ten-fold train/validation/test partitions are regenerated from
    ``args.seed`` for every call.  Within each fold only the training pool is
    sampled with replacement; validation and test subjects remain unchanged.
    Consequently no test subject can enter the corresponding fold's bootstrap
    training set, while each bootstrap replicate gets a new MPR reference and
    QSR fit.
    """
    if "qc_refined" not in tuple(args.representations):
        raise ValueError("Bootstrap QSR requires the qc_refined representation")

    oof_qc = np.full_like(data["ec"], np.nan, dtype=np.float32)
    fold_ids = np.full(len(data["ec"]), -1, dtype=np.int64)
    sampled_indices = {}
    refinement_metrics = []
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    prepare_arrays = prepare_fold_arrays

    for fold, train_index, val_index, test_index in make_stratified_splits(
        data["labels"], args.n_splits, args.seed, args.validation_size
    ):
        fold_seed = args.seed + fold * 1000
        set_seed(fold_seed)
        sampled_train_index = bootstrap_rng.choice(
            train_index, size=len(train_index), replace=True
        )
        if np.intersect1d(sampled_train_index, np.concatenate((val_index, test_index))).size:
            raise RuntimeError("Bootstrap sample overlaps the held-out test fold")
        sampled_indices[str(fold)] = sampled_train_index.astype(np.int64)

        arrays = prepare_arrays(
            data["ec"][sampled_train_index], data["ec"][val_index], data["ec"][test_index],
            data["continuous"][sampled_train_index], data["continuous"][val_index], data["continuous"][test_index],
            data["categorical_raw"][sampled_train_index], data["categorical_raw"][val_index], data["categorical_raw"][test_index],
        )
        fmri_arrays = None
        if args.graph_mode == "fusion":
            fmri_arrays = tuple(
                data["fmri_features"][index]
                for index in (sampled_train_index, val_index, test_index)
            )
        reference = build_fold_reference(args, arrays, fmri_arrays)

        qsr_train_confound = data["qsr_confound_values"][sampled_train_index]

        set_seed(fold_seed + 1)
        qsr_refiner, _, sensitive_map, qsr_metrics = train_qsr_refiner(
            args, arrays["train_ec"], reference["train_neighbor"],
            data["qsr_qc"][sampled_train_index], qsr_train_confound,
            data["site_ids"][sampled_train_index], device, fold_seed + 1,
        )
        test_qsr = apply_qsr_refiner(
            qsr_refiner, arrays["test_ec"], reference["test_neighbor"],
            sensitive_map, device,
        )
        restored_qc = (
            test_qsr * arrays["ec_std"] + arrays["ec_mean"]
        ).astype(np.float32)
        diagonal = np.arange(restored_qc.shape[-1])
        restored_qc[:, diagonal, diagonal] = 0.0
        oof_qc[test_index] = restored_qc
        fold_ids[test_index] = fold
        refinement_metrics.append({"fold": fold, **qsr_metrics})
        print(f"{args.dataset} fold {fold}: n={len(sampled_train_index)}, unique={len(np.unique(sampled_train_index))}, loss={qsr_metrics['qsr_loss']:.6g}", flush=True)

    if not np.isfinite(oof_qc).all() or np.any(fold_ids < 0):
        raise RuntimeError("Bootstrap QSR OOF array is incomplete")
    return {
        "oof_qc": oof_qc,
        "fold_ids": fold_ids,
        "sampled_indices": sampled_indices,
        "refinement_metrics": refinement_metrics,
    }

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mpr_qsr_reference_bootstrap"
ORIGINAL_EDGES = Path(__file__).resolve().parent / "cross_cohort_replication/outputs/replicated_edges.csv"

MTG_THA_EDGES = {(source, target) for source in (84, 85) for target in (76, 77)}
MTG_THA_EDGES |= {(source, target) for source in (76, 77) for target in (84, 85)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "pilot_5_20261111")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def build_profile_args(parse_function, gpu_id):
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "--input-mode", "ec", "--gpu-id", gpu_id,
                    "--representations", "qc_refined"]
        args = parse_function()
    finally:
        sys.argv = original_argv
    args.gpu_id = gpu_id
    args.representations = ["qc_refined"]
    args.n_splits = 10
    saved_config = json.loads((args.profile.output_dir / "summary.json").read_text())["config"]
    for name in ("seed", "n_splits", "validation_size", "reference_k", "graph_mode",
                 "fusion_beta", "reference_bandwidth", "categorical_penalty",
                 "continuous_weights", "permute_phenotype"):
        setattr(args, name, saved_config[name])
    for name, value in saved_config.items():
        if name.startswith("qsr_"):
            setattr(args, name, value)
    if args.n_splits != 10:
        raise ValueError("Expected the saved ten-fold protocol")
    return args


def load_cohort(parse_function, gpu_id, device):
    args = build_profile_args(parse_function, gpu_id)
    set_seed(args.seed)
    data, _ = load_pipeline_data(args, device)
    if np.unique(data["labels"]).size != 2:
        raise ValueError(f"{args.profile.name} does not contain both diagnostic groups")
    with np.load(args.profile.qsr_refined_ec_path, allow_pickle=False) as baseline:
        for name in ("subject_ids", "site_ids", "labels"):
            np.testing.assert_array_equal(data[name], baseline[name])
        np.testing.assert_array_equal(data["ec"], baseline["original_ec"])
        fold_ids = np.zeros(len(data["ec"]), dtype=np.int64)
        for fold, _, _, test_index in make_stratified_splits(data["labels"], args.n_splits, args.seed, args.validation_size):
            fold_ids[test_index] = fold
        np.testing.assert_array_equal(fold_ids, baseline["fold_ids"])
    return args, data


def jsonable_sampled_indices(sampled_indices, subject_ids):
    return {
        fold: [str(subject_ids[index]) for index in indices]
        for fold, indices in sampled_indices.items()
    }


def run_replication(archive_i, archive_ii, output_dir, alpha):
    output_dir.mkdir(parents=True, exist_ok=False)
    original = {
        (int(row["SourceIndex"]), int(row["TargetIndex"])): row
        for row in read_edges(ORIGINAL_EDGES)
    }
    if len(original) != 213 or not MTG_THA_EDGES.issubset(original):
        raise ValueError("Expected 213 original connections including all eight MTG–thalamus directions")
    tables, summaries = {}, {}
    reference_rois = None
    for cohort, archive_path in (("ABIDE-I", archive_i), ("ABIDE-II", archive_ii)):
        ec, labels, roi_names = load_archive(archive_path, "ec")
        if reference_rois is not None:
            np.testing.assert_array_equal(roi_names, reference_rois)
        reference_rois = roi_names
        audit, keep = align_covariates(archive_path, DEFAULT_PHENOTYPES[cohort], labels)
        beta, standard_error, p_values, q_values, diagnostics = fit_edges(ec[keep], labels[keep], audit)
        source, target = np.where(~np.eye(ec.shape[1], dtype=bool))
        rows = []
        for edge_index, (source_index, target_index) in enumerate(zip(source, target)):
            rows.append({
                "Source": str(roi_names[source_index]), "Target": str(roi_names[target_index]),
                "SourceIndex": int(source_index), "TargetIndex": int(target_index),
                "ASD_Beta": float(beta[edge_index]), "ASD_SE": float(standard_error[edge_index]),
                "ASD_T": float(beta[edge_index] / standard_error[edge_index]),
                "PValue": float(p_values[edge_index]), "QValue": float(q_values[edge_index]),
            })
        tables[cohort] = rows
        write_csv(output_dir / f"{cohort.lower()}_edge_statistics.csv", rows, list(rows[0]))
        write_csv(output_dir / f"{cohort.lower()}_subject_audit.csv", audit, list(audit[0]))
        summaries[cohort] = {
            **diagnostics, "archive_sha256": file_hash(archive_path),
            "phenotype_sha256": file_hash(DEFAULT_PHENOTYPES[cohort]),
            "included_n": int(keep.sum()), "excluded_n": int((~keep).sum()),
            "significant_edges": int((q_values < alpha).sum()),
        }
    merged = []
    for row_i, row_ii in zip(tables["ABIDE-I"], tables["ABIDE-II"]):
        key = row_i["SourceIndex"], row_i["TargetIndex"]
        if key != (row_ii["SourceIndex"], row_ii["TargetIndex"]):
            raise ValueError("Directed edge order mismatch")
        beta_i, beta_ii = row_i["ASD_Beta"], row_ii["ASD_Beta"]
        same = bool(beta_i != 0 and beta_ii != 0 and np.sign(beta_i) == np.sign(beta_ii))
        baseline = original.get(key)
        preserves = bool(baseline is not None and same and
                         np.sign(beta_i) == np.sign(float(baseline["ABIDE_I_Hedges_g"])) and
                         np.sign(beta_ii) == np.sign(float(baseline["ABIDE_II_Hedges_g"])))
        merged.append({
            **{name: row_i[name] for name in ("Source", "Target", "SourceIndex", "TargetIndex")},
            "ABIDE_I_ASD_Beta": beta_i, "ABIDE_II_ASD_Beta": beta_ii,
            "ABIDE_I_PValue": row_i["PValue"], "ABIDE_II_PValue": row_ii["PValue"],
            "ABIDE_I_QValue": row_i["QValue"], "ABIDE_II_QValue": row_ii["QValue"],
            "SameDirection": same,
            "Replicated": bool(same and row_i["QValue"] < alpha and row_ii["QValue"] < alpha),
            "OriginallyReplicated": baseline is not None,
            "PreservesOriginalDirectionBothCohorts": preserves,
        })
    comparison = [row for row in merged if row["OriginallyReplicated"]]
    replicated = [row for row in merged if row["Replicated"]]
    for filename, rows in (("all_edges_cross_cohort.csv", merged), ("replicated_edges.csv", replicated),
                           ("original_213_comparison.csv", comparison)):
        write_csv(output_dir / filename, rows, list(merged[0]))
    summary = {
        "formula": FORMULA, "alpha": alpha,
        "adjusted_replicated_n": len(replicated), "cohorts": summaries,
        "original_edges_sha256": file_hash(ORIGINAL_EDGES),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_edges(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_replication(summary, comparison_path, alpha):
    comparison = read_edges(comparison_path)
    original_rows = [row for row in comparison if row["OriginallyReplicated"] == "True"]
    retained = [row for row in original_rows if row["Replicated"] == "True"]
    original_direction_ok = [
        row for row in original_rows
        if row["PreservesOriginalDirectionBothCohorts"] == "True"
    ]
    mtg_rows = [
        row for row in original_rows
        if (int(row["SourceIndex"]), int(row["TargetIndex"])) in MTG_THA_EDGES
    ]
    mtg_direction_ok = [row for row in mtg_rows if row["PreservesOriginalDirectionBothCohorts"] == "True"]
    mtg_significant = [row for row in mtg_rows if row["Replicated"] == "True"]
    return {
        "adjusted_replicated_n": int(summary["adjusted_replicated_n"]),
        "original_213_retained_n": len(retained),
        "original_213_direction_consistent_n": len(original_direction_ok),
        "original_213_direction_consistency_rate": len(original_direction_ok) / 213.0,
        "original_213_between_cohort_direction_rate": sum(row["SameDirection"] == "True" for row in original_rows) / 213.0,
        "mtg_thalamus_edge_n": len(mtg_rows),
        "mtg_thalamus_direction_consistent_n": len(mtg_direction_ok),
        "mtg_thalamus_direction_consistency_rate": len(mtg_direction_ok) / max(len(mtg_rows), 1),
        "mtg_thalamus_significant_n": len(mtg_significant),
        "mtg_thalamus_significance_pass_rate": len(mtg_significant) / max(len(mtg_rows), 1),
        "alpha": alpha,
    }


def main():
    cli = parse_args()
    if cli.repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0 < cli.alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if cli.output_dir.exists() and any(cli.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {cli.output_dir}")
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(cli.gpu_id)
    args_i, data_i = load_cohort(parse_abide_i_args, cli.gpu_id, device)
    args_ii, data_ii = load_cohort(parse_abide_ii_args, cli.gpu_id, device)
    baseline_tables = {}
    for cohort, path in DEFAULT_INPUTS.items():
        ec, labels, roi_names = load_archive(path, "ec")
        baseline_tables[cohort] = edge_table(cohort, *compute_effects_and_tests(ec, labels), roi_names)
    recomputed = merge_tables(baseline_tables["ABIDE-I"], baseline_tables["ABIDE-II"], cli.alpha)
    expected_keys = {(int(row["SourceIndex"]), int(row["TargetIndex"])) for row in read_edges(ORIGINAL_EDGES)}
    if { (row["SourceIndex"], row["TargetIndex"]) for row in recomputed if row["Replicated"] } != expected_keys:
        raise ValueError("Original 213 connections could not be reproduced from original archives")
    baseline_summary = run_replication(DEFAULT_INPUTS["ABIDE-I"], DEFAULT_INPUTS["ABIDE-II"], cli.output_dir / "baseline_adjusted", cli.alpha)
    provenance = {
        "device": str(device), "python": sys.executable, "numpy": np.__version__,
        "seed": cli.seed, "original_edges_sha256": file_hash(ORIGINAL_EDGES),
        "original_replication_reproduced": True,
        "baseline_adjusted": baseline_summary,
        "protocol": "Fixed saved test and validation folds; iid subject bootstrap of the original post-validation training subset, same size with replacement; fixed original fold QSR RNG seeds; fresh scalers, MPR reference, QC patterns and QSR in every fold.",
        "cohorts": {},
    }
    for name, args, data in (("abide_i", args_i, data_i), ("abide_ii", args_ii, data_ii)):
        partitions = {}
        for fold, train_index, val_index, test_index in make_stratified_splits(data["labels"], args.n_splits, args.seed, args.validation_size):
            partitions[f"fold_{fold}_train_pool"] = train_index
            partitions[f"fold_{fold}_val"] = val_index
            partitions[f"fold_{fold}_test"] = test_index
        np.savez_compressed(cli.output_dir / f"{name}_fixed_partitions.npz", subject_ids=data["subject_ids"], **partitions)
        provenance["cohorts"][name] = {
            "config": vars(args), "initial_ec_sha256": file_hash(args.ec_path),
            "original_qsr_ec_sha256": file_hash(args.profile.qsr_refined_ec_path),
        }
    (cli.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    records = []
    mtg_observations = []
    for repeat in range(1, cli.repeats + 1):
        repeat_dir = cli.output_dir / f"repeat_{repeat:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        print(f"===== MPR/QSR reference resampling repeat {repeat}/{cli.repeats} =====")
        result_i = run_bootstrap_qsr_cross_validation(args_i, data_i, device, cli.seed + repeat * 100000 + 1)
        result_ii = run_bootstrap_qsr_cross_validation(args_ii, data_ii, device, cli.seed + repeat * 100000 + 2)

        archive_i = save_qsr_ec_archive(
            repeat_dir / "abide-i_qsr_bootstrap_oof.npz", data_i, result_i["oof_qc"],
            result_i["fold_ids"], args_i.ec_path,
        )
        archive_ii = save_qsr_ec_archive(
            repeat_dir / "abide-ii_qsr_bootstrap_oof.npz", data_ii, result_ii["oof_qc"],
            result_ii["fold_ids"], args_ii.ec_path,
        )
        (repeat_dir / "bootstrap_training_subjects.json").write_text(
            json.dumps({
                "repeat": repeat,
                "seed_abide_i": cli.seed + repeat * 100000 + 1,
                "seed_abide_ii": cli.seed + repeat * 100000 + 2,
                "abide_i": jsonable_sampled_indices(result_i["sampled_indices"], data_i["subject_ids"]),
                "abide_ii": jsonable_sampled_indices(result_ii["sampled_indices"], data_ii["subject_ids"]),
            }, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (repeat_dir / "qsr_training_metrics.json").write_text(
            json.dumps({"abide_i": result_i["refinement_metrics"], "abide_ii": result_ii["refinement_metrics"]},
                       indent=2, ensure_ascii=False), encoding="utf-8",
        )
        replication_dir = repeat_dir / "covariate_adjusted_replication"
        summary = run_replication(archive_i, archive_ii, replication_dir, cli.alpha)
        metrics = summarize_replication(summary, replication_dir / "original_213_comparison.csv", cli.alpha)
        metrics["repeat"] = repeat
        (repeat_dir / "repeat_summary.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        records.append(metrics)
        for row in read_edges(replication_dir / "original_213_comparison.csv"):
            if (int(row["SourceIndex"]), int(row["TargetIndex"])) in MTG_THA_EDGES:
                mtg_observations.append({"repeat": repeat, **row})
        write_csv(cli.output_dir / "mtg_thalamus_per_repeat.csv", mtg_observations, list(mtg_observations[0]))
        write_csv(cli.output_dir / "replicate_summary.csv", records, list(records[0]))
        print(json.dumps(metrics, ensure_ascii=False))

    fields = [
        "repeat", "adjusted_replicated_n", "original_213_retained_n",
        "original_213_direction_consistency_rate", "mtg_thalamus_direction_consistency_rate",
        "mtg_thalamus_significance_pass_rate",
    ]
    with (cli.output_dir / "replicate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)
    summary = {
        "purpose": "Sensitivity of discovered connections to MPR training-reference subject selection and QSR retraining",
        "interpretation_warning": "These five checks are a training-reference selection sensitivity analysis, not formal bootstrap confidence intervals accounting for all subject-sampling dependence.",
        "repeats": cli.repeats,
        "fixed_initial_ec": True,
        "fixed_test_folds": True,
        "n_splits": 10,
        "original_edges_n": 213,
        "mtg_thalamus_edges": sorted([list(edge) for edge in MTG_THA_EDGES]),
        "replicates": records,
        "mean": {field: float(np.mean([row[field] for row in records])) for field in fields if field != "repeat"},
        "sd": {field: float(np.std([row[field] for row in records], ddof=1)) for field in fields if field != "repeat"},
    }
    (cli.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    frequencies = []
    for source, target in sorted(MTG_THA_EDGES):
        rows = [row for row in mtg_observations if (int(row["SourceIndex"]), int(row["TargetIndex"])) == (source, target)]
        frequencies.append({
            "SourceIndex": source, "TargetIndex": target,
            "Source": rows[0]["Source"], "Target": rows[0]["Target"], "repeats": len(rows),
            "OriginalDirectionBothCohortsRate": sum(row["PreservesOriginalDirectionBothCohorts"] == "True" for row in rows) / len(rows),
            "BetweenCohortDirectionRate": sum(row["SameDirection"] == "True" for row in rows) / len(rows),
            "ABIDE_I_FDRPassFrequency": sum(float(row["ABIDE_I_QValue"]) < cli.alpha for row in rows) / len(rows),
            "ABIDE_II_FDRPassFrequency": sum(float(row["ABIDE_II_QValue"]) < cli.alpha for row in rows) / len(rows),
            "BothCohortsFDRPassFrequency": sum(float(row["ABIDE_I_QValue"]) < cli.alpha and float(row["ABIDE_II_QValue"]) < cli.alpha for row in rows) / len(rows),
            "ReplicationPassFrequency": sum(row["Replicated"] == "True" for row in rows) / len(rows),
        })
    write_csv(cli.output_dir / "mtg_thalamus_edge_frequencies.csv", frequencies, list(frequencies[0]))
    print(f"Saved bootstrap sensitivity results to {cli.output_dir}")


if __name__ == "__main__":
    main()
