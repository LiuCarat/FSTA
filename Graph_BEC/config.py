"""Command-line configuration for the Graph-BEC experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from Graph_BEC.baseline.FSTA_EC import add_fsta_arguments
from Graph_BEC.data import ASD_LABEL, TC_LABEL
from Graph_BEC.model.qc import DEFAULT_QC_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEC = ROOT / "Graph_BEC/outputs/seed_42/subject_bec.npz"
DEFAULT_REFINED_BEC = ROOT / "Graph_BEC/outputs/refined_subject_bec.npz"
DEFAULT_DATA_ROOT = ROOT / "dataset/ABIDE-I"
DEFAULT_PHENOTYPE = ROOT / "dataset/ABIDE-I/Phenotypic_Processing_filled.csv"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/outputs"


def parse_args(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-mode", choices=["bec", "raw"], default="bec")
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument("--refined-bec-path", type=Path, default=DEFAULT_REFINED_BEC)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--phenotype-csv", type=Path, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--gpu-id", default="auto")
    add_fsta_arguments(parser)

    parser.add_argument("--reference-k", type=int, default=20)
    parser.add_argument("--graph-mode", choices=["phenotype", "fusion"], default="fusion")
    parser.add_argument("--fusion-beta", type=float, default=0.6)
    parser.add_argument("--reference-bandwidth", type=float, default=2.0)
    parser.add_argument("--categorical-penalty", type=float, default=4.0)
    parser.add_argument("--continuous-weights", type=float, nargs=2, default=[1.0, 0.3])
    parser.add_argument("--permute-phenotype", action="store_true")

    parser.add_argument("--refiner-epochs", type=int, default=80)
    parser.add_argument("--refiner-lr", type=float, default=1e-2)
    parser.add_argument("--gate-max", type=float, default=0.5)
    parser.add_argument("--gate-l1-weight", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--variance-retention", type=float, default=0.85)

    parser.add_argument("--qsr-qc-columns", nargs="+", default=list(DEFAULT_QC_COLUMNS))
    parser.add_argument("--qsr-epochs", type=int, default=80)
    parser.add_argument("--qsr-lr", type=float, default=3e-3)
    parser.add_argument("--qsr-hidden-channels", type=int, default=8)
    parser.add_argument("--qsr-eta", type=float, default=0.15)
    parser.add_argument("--qsr-r-max", type=float, default=0.03)
    parser.add_argument("--qsr-corruption-scale", type=float, default=0.5)
    parser.add_argument("--qsr-gate-max", type=float, default=0.5)
    parser.add_argument("--qsr-gate-weight", type=float, default=1e-3)
    parser.add_argument("--qsr-variance-weight", type=float, default=0.1)
    parser.add_argument("--qsr-variance-retention", type=float, default=0.85)
    parser.add_argument("--qsr-basis-ridge", type=float, default=1e-3)

    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-repeats", type=int, default=1)
    parser.add_argument("--asd-label", type=int, default=ASD_LABEL, choices=[0, 1])
    parser.add_argument("--tc-label", type=int, default=TC_LABEL, choices=[0, 1])
    args = parser.parse_args()
    if args.asd_label != ASD_LABEL or args.tc_label != TC_LABEL:
        parser.error("Graph_BEC uses canonical labels 0=TC and 1=ASD.")
    if args.seeds is not None and len(args.seeds) > 1:
        parser.error(
            "Strict fold-local refinement supports one seed per run; "
            "run main.py separately for each seed."
        )
    if args.seeds:
        args.seed = args.seeds[0]
    return args
