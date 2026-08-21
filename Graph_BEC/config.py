"""Command-line configuration for dataset-specific Graph-BEC experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from Graph_BEC.baseline.FSTA_EC import add_fsta_arguments
from Graph_BEC.experiments.configuration import PROFILES, get_profile


def parse_args(description=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=sorted(PROFILES), default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=sorted(PROFILES), default=profile.name)
    parser.add_argument("--input-mode", choices=["bec", "raw"])
    parser.add_argument("--bec-path", type=Path)
    parser.add_argument("--refined-bec-path", type=Path)
    parser.add_argument("--qsr-refined-bec-path", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--phenotype-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--validation-size", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--gpu-id")
    add_fsta_arguments(parser)

    parser.add_argument("--reference-k", type=int)
    parser.add_argument("--graph-mode", choices=["phenotype", "fusion"])
    parser.add_argument("--fusion-beta", type=float)
    parser.add_argument("--reference-bandwidth", type=float)
    parser.add_argument("--categorical-penalty", type=float)
    parser.add_argument("--continuous-weights", type=float, nargs=2)
    parser.add_argument("--permute-phenotype", action="store_true")
    parser.add_argument("--refiner-epochs", type=int)
    parser.add_argument("--refiner-lr", type=float)
    parser.add_argument("--gate-max", type=float)
    parser.add_argument("--gate-l1-weight", type=float)
    parser.add_argument("--anchor-weight", type=float)
    parser.add_argument("--variance-weight", type=float)
    parser.add_argument("--variance-retention", type=float)
    parser.add_argument("--qsr-qc-columns", nargs="+")
    parser.add_argument("--qsr-epochs", type=int)
    parser.add_argument("--qsr-lr", type=float)
    parser.add_argument("--qsr-hidden-channels", type=int)
    parser.add_argument("--qsr-eta", type=float)
    parser.add_argument("--qsr-r-max", type=float)
    parser.add_argument("--qsr-corruption-scale", type=float)
    parser.add_argument("--qsr-gate-max", type=float)
    parser.add_argument("--qsr-gate-weight", type=float)
    parser.add_argument("--qsr-variance-weight", type=float)
    parser.add_argument("--qsr-variance-retention", type=float)
    parser.add_argument("--qsr-basis-ridge", type=float)
    parser.add_argument("--classifier-epochs", type=int)
    parser.add_argument("--classifier-patience", type=int)
    parser.add_argument("--classifier-lr", type=float)
    parser.add_argument("--classifier-repeats", type=int)
    parser.add_argument("--patient-label", type=int, choices=[0, 1])
    parser.add_argument("--control-label", type=int, choices=[0, 1])

    defaults = dict(profile.defaults)
    defaults.update(
        {
            "bec_path": profile.bec_path,
            "refined_bec_path": profile.refined_bec_path,
            "qsr_refined_bec_path": profile.qsr_refined_bec_path,
            "data_root": profile.data_root,
            "phenotype_csv": profile.phenotype_path,
            "output_dir": profile.output_dir,
        }
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    # Keep the selected profile attached to the run and support the old internal names.
    args.profile = get_profile(args.dataset)
    if args.patient_label == args.control_label:
        parser.error("--patient-label and --control-label must be different")
    args.asd_label = args.patient_label
    args.tc_label = args.control_label
    if args.seeds is not None and len(args.seeds) > 1:
        parser.error("Strict fold-local refinement supports one seed per run")
    if args.seeds:
        args.seed = args.seeds[0]
    return args
