from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import random
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
)
from torch.utils.data import DataLoader
from src.data import (
    ABIDEWindowDataset,
    NumericPreprocessor,
    build_manifest,
    parse_bool_int,
    parse_columns,
)
from src.train import extract_bec, train_model


MODES = {
    "1": "fmri_only",
    "fmri_only": "fmri_only",
    "2": "fmri_pheno",
    "fmri_pheno": "fmri_pheno",
    "3": "fmri_pheno_qcadv",
    "fmri_pheno_qcadv": "fmri_pheno_qcadv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_mode",
        required=True,
        choices=sorted(MODES),
    )
    parser.add_argument(
        "--roi_root",
        default=str(REPO_ROOT / "dataset/ABIDE-I/cpac/filt_noglobal"),
    )
    parser.add_argument(
        "--phenotypic_csv",
        default=str(
            REPO_ROOT / "dataset/ABIDE-I/Phenotypic_V1_0b_preprocessed1.csv"),
    )
    parser.add_argument("--result_dir")
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument(
        "--save_individual_files",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--atlas", default="AAL")
    parser.add_argument("--num_rois", type=int, default=90)
    parser.add_argument(
        "--allow_aal116_to_aal90",
        type=parse_bool_int,
        default=True,
    )
    parser.add_argument(
        "--phenotype_columns",
        default="AGE_AT_SCAN,SEX,FIQ",
    )
    parser.add_argument(
        "--qc_columns",
        default=(
            "func_mean_fd,func_dvars,"
            "func_outlier,func_perc_fd"
        ),
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument(
        "--stratify_by_dx",
        type=parse_bool_int,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window_length", type=int, default=78)
    parser.add_argument("--context_ratio", type=float, default=0.60)
    parser.add_argument("--eval_windows", type=int, default=5)
    parser.add_argument("--phenotype_hidden", type=int, default=64)
    parser.add_argument("--phenotype_dim", type=int, default=32)
    parser.add_argument("--temporal_hidden", type=int, default=32)
    parser.add_argument("--roi_dim", type=int, default=64)
    parser.add_argument("--common_dim", type=int, default=32)
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--max_incoming_edges", type=int, default=20)
    parser.add_argument("--adversary_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--lambda_sparse", type=float, default=0.05)
    parser.add_argument("--lambda_directional", type=float, default=0.1)
    parser.add_argument("--lambda_consistency", type=float, default=0.05)
    parser.add_argument("--lambda_modal", type=float, default=0.02)
    parser.add_argument("--lambda_qc", type=float, default=0.01)
    parser.add_argument("--modality_grl_strength", type=float, default=1.0)
    parser.add_argument("--qc_grl_strength", type=float, default=1.0)
    parser.add_argument("--adversarial_warmup_epochs", type=int, default=20)

    parser.add_argument("--edge_presence_threshold", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def configure_mode(args):
    mode = MODES[args.experiment_mode]
    args.experiment_mode_name = mode
    args.use_phenotype = mode != "fmri_only"
    args.use_qc_adversary = mode == "fmri_pheno_qcadv"
    args.effective_lambda_modal = (
        args.lambda_modal if args.use_phenotype else 0.0
    )
    args.effective_lambda_qc = (
        args.lambda_qc if args.use_qc_adversary else 0.0
    )
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_npz(path, result):
    np.savez_compressed(path, **result)


def main():
    args = configure_mode(parse_args())
    if args.num_rois != 90:
        raise ValueError("This version is fixed to AAL90.")

    set_seed(args.seed)
    args.device = torch.device(args.device)

    result_dir = (
        Path(args.result_dir)
        if args.result_dir
        else PROJECT_ROOT / "results" / args.experiment_mode_name
    )
    args.result_dir = str(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest, report = build_manifest(
        args.roi_root,
        args.phenotypic_csv,
        args.atlas,
    )
    if args.max_subjects is not None and args.max_subjects < len(manifest):
        selected, _ = train_test_split(
            np.arange(len(manifest)),
            train_size=args.max_subjects,
            random_state=args.seed,
            stratify=manifest["label"],
        )
        manifest = manifest.iloc[np.sort(selected)].reset_index(drop=True)
        report["selected_subjects"] = len(manifest)
    manifest.to_csv(result_dir / "matched_manifest.csv", index=False)
    (result_dir / "matching_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    phenotype_columns = (
        parse_columns(args.phenotype_columns)
        if args.use_phenotype
        else []
    )
    qc_columns = (
        parse_columns(args.qc_columns)
        if args.use_qc_adversary
        else []
    )

    config = vars(args).copy()
    config["device"] = str(args.device)
    config["phenotype_columns_used"] = phenotype_columns
    config["qc_columns_used"] = qc_columns
    (result_dir / "run_configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    labels = manifest["label"].to_numpy(dtype=np.int64)
    if args.stratify_by_dx:
        splitter = StratifiedKFold(
            args.folds,
            shuffle=True,
            random_state=args.seed,
        )
        splits = splitter.split(np.zeros(len(labels)), labels)
    else:
        splitter = KFold(
            args.folds,
            shuffle=True,
            random_state=args.seed,
        )
        splits = splitter.split(np.zeros(len(labels)))

    all_bec = np.zeros(
        (len(manifest), args.num_rois, args.num_rois),
        dtype=np.float32,
    )
    all_self = np.zeros(
        (len(manifest), args.num_rois),
        dtype=np.float32,
    )
    all_mae = np.zeros(len(manifest), dtype=np.float32)
    all_rmse = np.zeros(len(manifest), dtype=np.float32)
    all_density = np.zeros(len(manifest), dtype=np.float32)
    all_asymmetry = np.zeros(len(manifest), dtype=np.float32)
    all_directionality = np.zeros(len(manifest), dtype=np.float32)
    all_reciprocity = np.zeros(len(manifest), dtype=np.float32)
    all_qc_true = np.empty(
        (len(manifest), len(qc_columns)),
        dtype=np.float32,
    )
    all_qc_pred = np.empty(
        (len(manifest), len(qc_columns)),
        dtype=np.float32,
    )

    fold_rows = []

    for fold, (train_valid, test_indices) in enumerate(splits, start=1):
        fold_dir = result_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        stratify = labels[train_valid] if args.stratify_by_dx else None
        train_indices, valid_indices = train_test_split(
            train_valid,
            test_size=args.validation_fraction,
            random_state=args.seed + fold,
            stratify=stratify,
        )

        if args.use_phenotype:
            pheno_pre = NumericPreprocessor(
                phenotype_columns,
                "phenotype",
            ).fit(manifest.iloc[train_indices])
            phenotype_array = pheno_pre.transform(manifest)
            pheno_pre.save(fold_dir / "phenotype_preprocessor.json")
        else:
            phenotype_array = np.zeros((len(manifest), 1), dtype=np.float32)

        if args.use_qc_adversary:
            qc_pre = NumericPreprocessor(
                qc_columns,
                "qc",
            ).fit(manifest.iloc[train_indices])
            qc_array = qc_pre.transform(manifest)
            qc_pre.save(fold_dir / "qc_preprocessor.json")
        else:
            qc_array = np.zeros((len(manifest), 1), dtype=np.float32)

        train_set = ABIDEWindowDataset(
            manifest,
            train_indices,
            phenotype_array,
            qc_array,
            args.num_rois,
            args.window_length,
            True,
            args.allow_aal116_to_aal90,
        )
        valid_set = ABIDEWindowDataset(
            manifest,
            valid_indices,
            phenotype_array,
            qc_array,
            args.num_rois,
            args.window_length,
            False,
            args.allow_aal116_to_aal90,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model, best_valid = train_model(
            train_loader,
            valid_loader,
            len(phenotype_columns),
            len(qc_columns),
            args,
            fold_dir,
        )

        train_result = extract_bec(
            model,
            manifest,
            train_valid,
            phenotype_array,
            qc_array,
            args,
        )
        test_result = extract_bec(
            model,
            manifest,
            test_indices,
            phenotype_array,
            qc_array,
            args,
        )

        save_npz(fold_dir / "train_individual_bec.npz", train_result)
        save_npz(fold_dir / "test_individual_bec.npz", test_result)

        all_bec[test_indices] = test_result["bec"]
        all_self[test_indices] = test_result["self_coeff"]
        all_mae[test_indices] = test_result["reconstruction_mae"]
        all_rmse[test_indices] = test_result["reconstruction_rmse"]
        all_density[test_indices] = test_result["edge_density"]
        all_asymmetry[test_indices] = test_result["asymmetry"]
        all_directionality[test_indices] = test_result[
            "directionality_index"
        ]
        all_reciprocity[test_indices] = test_result["reciprocity"]

        if args.use_qc_adversary:
            all_qc_true[test_indices] = test_result[
                "qc_true_standardized"
            ]
            all_qc_pred[test_indices] = test_result[
                "qc_adversary_prediction"
            ]

        fold_rows.append(
            {
                "mode": args.experiment_mode_name,
                "fold": fold,
                "best_valid_loss": best_valid["loss"],
                "best_valid_reconstruction": best_valid[
                    "reconstruction"
                ],
                "test_mae": float(
                    test_result["reconstruction_mae"].mean()
                ),
                "test_rmse": float(
                    test_result["reconstruction_rmse"].mean()
                ),
                "edge_density": float(
                    test_result["edge_density"].mean()
                ),
                "asymmetry": float(
                    test_result["asymmetry"].mean()
                ),
                "directionality_index": float(
                    test_result["directionality_index"].mean()
                ),
                "reciprocity": float(
                    test_result["reciprocity"].mean()
                ),
            }
        )

    np.savez_compressed(
        result_dir / "oof_individual_bec_AAL90.npz",
        experiment_mode=args.experiment_mode_name,
        bec=all_bec,
        self_coeff=all_self,
        reconstruction_mae=all_mae,
        reconstruction_rmse=all_rmse,
        edge_density=all_density,
        asymmetry=all_asymmetry,
        directionality_index=all_directionality,
        reciprocity=all_reciprocity,
        phenotype_columns=np.asarray(phenotype_columns),
        qc_columns=np.asarray(qc_columns),
        qc_true_standardized=all_qc_true,
        qc_adversary_prediction=all_qc_pred,
        label_metadata=labels,
        subject_id=manifest["subject_id"].astype(str).to_numpy(),
        site_id=manifest["site_id"].astype(str).to_numpy(),
        matrix_convention=(
            "bec[source, target] = source ROI -> target ROI"
        ),
    )

    if args.save_individual_files:
        individual_dir = result_dir / "individual_bec"
        individual_dir.mkdir(exist_ok=True)
        for index, row in manifest.iterrows():
            np.savez_compressed(
                individual_dir
                / f"{row['subject_id']}_AAL90_individual_BEC.npz",
                bec=all_bec[index],
                self_coeff=all_self[index],
                subject_id=str(row["subject_id"]),
                site_id=str(row.get("site_id", "")),
                label_metadata=int(row["label"]),
                reconstruction_mae=all_mae[index],
                reconstruction_rmse=all_rmse[index],
                edge_density=all_density[index],
                directionality_index=all_directionality[index],
                reciprocity=all_reciprocity[index],
                matrix_convention=(
                    "bec[source, target] = source ROI -> target ROI"
                ),
            )

    pd.DataFrame(fold_rows).to_csv(
        result_dir / "fold_metrics.csv",
        index=False,
    )
    print(f"Saved to: {result_dir.resolve()}")


if __name__ == "__main__":
    main()
