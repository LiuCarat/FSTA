from __future__ import annotations
from src.data import (
    NumericPreprocessor,
    build_manifest,
    parse_bool_int,
    parse_columns,
    read_roi_1d,
)

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roi_root",
        default=REPO_ROOT / "dataset/ABIDE-I/cpac/filt_noglobal",
        type=Path,
    )
    parser.add_argument(
        "--phenotypic_csv",
        default=REPO_ROOT / "dataset/ABIDE-I/Phenotypic_V1_0b_preprocessed1.csv",
        type=Path,
    )
    parser.add_argument("--num_rois", type=int, default=90)
    parser.add_argument(
        "--allow_aal116_to_aal90",
        type=parse_bool_int,
        default=True,
    )
    parser.add_argument(
        "--phenotype_columns",
        default="AGE_AT_SCAN,SEX,FIQ", help="表型列名列表"
    )
    parser.add_argument(
        "--qc_columns",
        default=(
            "func_mean_fd,func_dvars,"
            "func_outlier,func_perc_fd"
        ),
        help="QC列名列表"
    )
    args = parser.parse_args()

    manifest, report = build_manifest(
        args.roi_root,
        args.phenotypic_csv,
    )
    print(report)
    pheno = NumericPreprocessor(
        parse_columns(args.phenotype_columns),
        "phenotype",
    ).fit(manifest)
    qc = NumericPreprocessor(
        parse_columns(args.qc_columns),
        "qc",
    ).fit(manifest)

    print("\nPhenotype missingness:")
    for key, value in pheno.stats.items():
        print(f"{key}: {value.missing_fraction:.3f}")

    print("\nQC missingness:")
    for key, value in qc.stats.items():
        print(f"{key}: {value.missing_fraction:.3f}")

    print("\nSample shapes:")
    for _, row in manifest.head(10).iterrows():
        data = read_roi_1d(
            row["roi_path"],
            args.num_rois,
            args.allow_aal116_to_aal90,
        )
        print(row["subject_id"], data.shape)


if __name__ == "__main__":
    main()
