
import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE = types.ModuleType("PR_EC")
PACKAGE.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules["PR_EC"] = PACKAGE

from PR_EC.dataset_configs import ExperimentProfile
from PR_EC.downstream import add_classifier_arguments

DATASET_CONFIG = ExperimentProfile(
    name="abide_i",
    data_root=ROOT / "dataset/ABIDE-I",
    phenotype_path=ROOT / "dataset/ABIDE-I/Phenotypic_Processing.csv",
    output_dir=ROOT / "PR-EC/outputs/abide-i",
    PR_EC_PATH=ROOT / "PR-EC/outputs/abide-i/abide_pr_ec.npz",
    phenotype_format="csv",
    phenotype_id_column="FILE_ID",
    patient_column="DX_GROUP",
    control_column="DX_GROUP",
    patient_values=("2",),
    control_values=("1",),
    site_column="SITE_ID",
    sex_column="SEX",
    continuous_columns=("FIQ", "PIQ"),
    confound_columns=("AGE_AT_SCAN", "SEX", "FIQ", "PIQ"),
    qc_columns=("func_mean_fd", "func_dvars", "func_quality"),
    source_roi_count=116,
    roi_count=90,
    exclude_subjects=(),
)



def add_individual_ec_arguments(parser):
    group = parser.add_argument_group("Individual-EC encoder")
    group.add_argument("--window-length", type=int, default=78)
    group.add_argument("--stride", type=int, default=39)
    group.add_argument("--epochs", type=int, default=81)
    group.add_argument("--loss-alpha", type=float, default=0.01)
    group.add_argument("--batch-size", type=int, default=32)
    group.add_argument("--log-every", type=int, default=20)
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-ec-path", type=Path, dest="PR_EC_PATH", default=DATASET_CONFIG.PR_EC_PATH)
    parser.add_argument("--data-root", type=Path, default=DATASET_CONFIG.data_root)
    parser.add_argument("--phenotype-csv", type=Path, default=DATASET_CONFIG.phenotype_path)
    parser.add_argument("--output-dir", type=Path, default=DATASET_CONFIG.output_dir)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--gpu-id", default='auto')

    parser.add_argument("--reference-k", type=int, default=20)
    parser.add_argument("--fusion-beta", type=float, default=0.6)
    parser.add_argument("--reference-bandwidth", type=float, default=2.0)
    parser.add_argument("--categorical-penalty", type=float, default=4.0)
    parser.add_argument("--continuous-weights", type=float, nargs=len(DATASET_CONFIG.continuous_columns), default=[1.0, 0.3])

    parser.add_argument("--qsr-qc-columns", nargs="+", default=list(['func_mean_fd', 'func_dvars', 'func_quality']))
    parser.add_argument("--qsr-epochs", type=int, default=80)
    parser.add_argument("--qsr-lr", type=float, default=0.003)
    parser.add_argument("--qsr-hidden-channels", type=int, default=8)
    parser.add_argument("--qsr-eta", type=float, default=0.15)
    parser.add_argument("--qsr-r-max", type=float, default=0.03)
    parser.add_argument("--qsr-perturbation-scale", type=float, default=0.5)
    parser.add_argument("--qsr-gate-max", type=float, default=0.5)
    parser.add_argument("--qsr-gate-weight", type=float, default=0.001)
    parser.add_argument("--qsr-variance-weight", type=float, default=0.1)
    parser.add_argument("--qsr-variance-retention", type=float, default=0.85)
    parser.add_argument("--qsr-basis-ridge", type=float, default=0.001)

    add_classifier_arguments(parser)
    add_individual_ec_arguments(parser)
    args = parser.parse_args()
    args.individual_ec_checkpoint = "final"
    args.graph_mode = "fusion"
    args.permute_phenotype = False
    args.no_filters = False
    args.dataset, args.profile = DATASET_CONFIG.name, DATASET_CONFIG
    if args.patient_label == args.control_label:
        parser.error("--patient-label and --control-label must be different")
    args.asd_label, args.tc_label = args.patient_label, args.control_label
    if args.seeds is not None and len(args.seeds) > 1:
        parser.error("Strict fold-local refinement supports one seed per run")
    if args.seeds:
        args.seed = args.seeds[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    from PR_EC.runner import run
    run(args)
