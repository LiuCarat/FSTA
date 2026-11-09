"""Strict subject-level OOF evaluation of the initial EC estimator only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(ROOT.parent))
    __package__ = ROOT.name

from .simulation.initial_ec import run_experiments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "MemoryEC/data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/initial_ec_oof_paper_relative")
    parser.add_argument("--network", "--networks", dest="networks", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1], help="Network ID(s) to run; default: 1 only")
    parser.add_argument("--n-splits", type=int, default=5, help="OOF folds; training fraction is approximately (K-1)/K")
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--window-length", type=int, default=78)
    parser.add_argument("--stride", type=int, default=39)
    parser.add_argument("--epochs", type=int, default=301)
    parser.add_argument("--stf-checkpoint", choices=["final", "best"], default="final")
    parser.add_argument("--loss-alpha", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--d-inner-hid", type=int, default=64)
    parser.add_argument("--d-k", type=int, default=8)
    parser.add_argument("--d-v", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-warmup-steps", type=int, default=4000)
    parser.add_argument("--lr-mul", type=float, default=1.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int, default=2)
    parser.add_argument("--hidden-act", default="gelu")
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0.5)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.5)
    parser.add_argument("--initializer-range", type=float, default=0.02)
    parser.add_argument("--no-filters", action="store_true")
    args = parser.parse_args()
    args.seed = 42
    if args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    for name in ("window_length", "stride", "epochs", "batch_size", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    run_experiments(parse_args())
