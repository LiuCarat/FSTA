"""NetSim Sim21: pair-held-out evaluation of individual difference preservation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(ROOT.parent))
    __package__ = ROOT.name

from .main_simulation import build_parser, parse_args
from .simulation.sim21 import run_experiment


def sim21_args():
    parser = build_parser(include_network_options=False)
    parser.description = __doc__
    parser.set_defaults(output_dir=ROOT / "outputs/sim21")
    parser.add_argument("--mat-path", type=Path, default=ROOT.parent / "dataset/NetSim/sim21.mat")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--reuse-cache", action="store_true", help="Require matching fold EC caches; never train the encoder")
    parser.add_argument("--reference-k", type=int, default=15, help="FC top-k neighbors (1-40); default: 5")
    args = parse_args(parser)
    if args.n_splits != 5:
        parser.error("Sim21 protocol fixes five matched-pair folds")
    if not 1 <= args.reference_k <= 40:
        parser.error("--reference-k must be between 1 and 40 (training subjects per fold)")
    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "encoder_cache"
    return args


if __name__ == "__main__":
    run_experiment(sim21_args())
