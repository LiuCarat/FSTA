"""CLI safety and phenotype-only configuration for the W6/Q1 ablation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

LOCAL_ROOT = Path(__file__).resolve().parent
DATASET_DIRS = {"abide": "abide-i", "abide_ii": "abide-ii", "adhd200": "adhd200"}
DEFAULT_OUTPUT_ROOT = LOCAL_ROOT / "outputs" / "rebuttal_w6"


def add_rebuttal_arguments(parser):
    parser.add_argument(
        "--phenotype-feature-set", choices=("original", "age_sex_site"),
        default="original",
    )
    parser.add_argument(
        "--diagnostics-only", action="store_true",
        help="Build the identical fold-local PH+FC references without training any model.",
    )


def finalize_rebuttal_arguments(parser, args):
    count = 1 if args.phenotype_feature_set == "age_sex_site" else len(args.profile.continuous_columns)
    defaults = [1.0] if count == 1 else [1.0] + [0.3] * (count - 1)
    if args.continuous_weights is None:
        args.continuous_weights = defaults
    if len(args.continuous_weights) != count:
        parser.error(f"{args.phenotype_feature_set} requires {count} continuous weight(s)")
    if not np.isfinite(args.continuous_weights).all() or min(args.continuous_weights) < 0:
        parser.error("--continuous-weights must be finite and nonnegative")
    if args.diagnostics_only or args.phenotype_feature_set == "age_sex_site":
        if args.input_mode != "ec" or args.graph_mode != "fusion" or args.permute_phenotype:
            parser.error("W6 diagnostics/age_sex_site require --input-mode ec, fusion, and unpermuted PH")
        if args.seed != 42 or args.n_splits != 10 or args.reference_k != 20:
            parser.error("W6 requires seed=42, n-splits=10, reference-k=20")
        if not args.diagnostics_only and "qc_refined" not in args.representations:
            parser.error("W6 classification requires the qc_refined representation")
    args.output_dir = (args.output_dir or (
        DEFAULT_OUTPUT_ROOT / DATASET_DIRS[args.dataset] / args.phenotype_feature_set
    )).resolve()
    if not args.output_dir.is_relative_to(LOCAL_ROOT):
        parser.error("Outputs must stay inside PR_EC_changePH (never the original PR_EC outputs)")
    args.refined_ec_path = (args.refined_ec_path or (
        args.output_dir / f"{args.dataset}_refined_subject_ec.npz"
    )).resolve()
    args.qsr_refined_ec_path = (args.qsr_refined_ec_path or (
        args.output_dir / f"{args.dataset}_qsr_refined_subject_ec.npz"
    )).resolve()
    for path in (args.refined_ec_path, args.qsr_refined_ec_path):
        if not path.is_relative_to(args.output_dir):
            parser.error("Refined EC paths must be inside the isolated --output-dir")
    if args.input_mode == "raw" and args.ec_path == args.profile.ec_path:
        args.ec_path = args.output_dir / f"{args.dataset}_subject_ec.npz"
    if args.input_mode == "raw" and not args.ec_path.resolve().is_relative_to(args.output_dir):
        parser.error("Raw-mode EC writes must be inside the isolated --output-dir")
    if args.ec_path.resolve() in (args.refined_ec_path, args.qsr_refined_ec_path):
        parser.error("Input EC and refined EC output paths must differ")
    if args.refined_ec_path == args.qsr_refined_ec_path:
        parser.error("PGR and QSR output paths must differ")
    return args
