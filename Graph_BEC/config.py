"""Command-line configuration for dataset-specific Graph-BEC experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from Graph_BEC.profiles.configuration import PROFILES, get_profile


def add_fsta_arguments(parser):
    """Register the shared FSTA-EC command-line parameters."""
    group = parser.add_argument_group("FSTA-EC model")
    group.add_argument("--window-length", type=int, default=78)
    group.add_argument("--stride", type=int, default=39)
    group.add_argument("--epochs", type=int, default=101)
    group.add_argument("--fsta-checkpoint", choices=["final", "best"], default="final")
    group.add_argument("--loss-mode", choices=["original", "entropy"], default="entropy")
    group.add_argument("--loss-alpha", type=float, default=0.01)
    group.add_argument("--batch-size", type=int, default=32)
    group.add_argument("--log-every", type=int, default=10)
    group.add_argument("--d-model", type=int, default=16)
    group.add_argument("--d-inner-hid", type=int, default=64)
    group.add_argument("--d-k", type=int, default=8)
    group.add_argument("--d-v", type=int, default=8)
    group.add_argument("--n-head", type=int, default=2)
    group.add_argument("--dropout", type=float, default=0.2)
    group.add_argument("--n-warmup-steps", type=int, default=4000)
    group.add_argument("--lr-mul", type=float, default=1.2)
    group.add_argument("--weight-decay", type=float, default=0.0)
    group.add_argument("--adam-beta1", type=float, default=0.9)
    group.add_argument("--adam-beta2", type=float, default=0.98)
    group.add_argument("--num-hidden-layers", type=int, default=1)
    group.add_argument("--num-attention-heads", type=int, default=2)
    group.add_argument("--hidden-act", default="gelu")
    group.add_argument("--attention-probs-dropout-prob", type=float, default=0.5)
    group.add_argument("--hidden-dropout-prob", type=float, default=0.5)
    group.add_argument("--initializer-range", type=float, default=0.02)
    group.add_argument("--no-filters", action="store_true")


def parse_args(description=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--dataset", choices=sorted(PROFILES), default="abide")
    selected, _ = selector.parse_known_args()
    profile = get_profile(selected.dataset)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=sorted(PROFILES), default=profile.name)
    parser.add_argument("--input-mode", choices=["bec", "raw"])
    parser.add_argument("--representations", choices=["original", "refined", "qc_refined"], nargs="+")
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
    parser.add_argument("--continuous-weights", type=float, nargs=3 if selected.dataset == "adhd200" else 2)
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
