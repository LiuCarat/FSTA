"""Produce a threshold-only legacy audit and a separate paper-direction result table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(ROOT.parent))
    __package__ = ROOT.name

from .simulation.rescore import run_rescore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/initial_ec_oof")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/initial_ec_oof_relative")
    parser.add_argument("--paper-output-dir", type=Path, default=ROOT / "outputs/initial_ec_oof_paper_relative")
    parser.add_argument("--network", "--networks", dest="networks", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1])
    return parser.parse_args()


if __name__ == "__main__":
    run_rescore(parse_args())
