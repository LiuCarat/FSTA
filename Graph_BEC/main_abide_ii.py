"""Run the ABIDE-II Graph-BEC experiment."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_BEC.dataset_configs import ExperimentProfile

DATASET_CONFIG = ExperimentProfile(
    name="abide_ii",
    data_root=ROOT / "dataset/ABIDE-II",
    phenotype_path=ROOT / "dataset/ABIDE-II/Phenotypic_Processing.csv",
    output_dir=ROOT / "Graph_BEC/outputs/abide-ii",
    bec_path=ROOT / "Graph_BEC/outputs/abide-ii/abide_ii_subject_bec.npz",
    refined_bec_path=ROOT / "Graph_BEC/outputs/abide-ii/abide_ii_refined_subject_bec.npz",
    qsr_refined_bec_path=ROOT / "Graph_BEC/outputs/abide-ii/abide_ii_qsr_refined_subject_bec.npz",
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

def add_stf_arguments(parser):
    group = parser.add_argument_group("STF-BEC encoder")
    group.add_argument("--window-length", type=int, default=80) #80
    group.add_argument("--stride", type=int, default=40) #40
    group.add_argument("--epochs", type=int, default=111)
    group.add_argument("--stf-checkpoint", choices=["final", "best"], default='final')
    group.add_argument("--loss-mode", choices=["original", "entropy"], default='entropy')
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=["bec", "raw"], default='bec')
    parser.add_argument("--representations", choices=["original", "refined", "qc_refined"], nargs="+", default=['original', 'refined', "qc_refined"])
    parser.add_argument("--bec-path", type=Path, default=DATASET_CONFIG.bec_path)
    parser.add_argument("--refined-bec-path", type=Path, default=DATASET_CONFIG.refined_bec_path)
    parser.add_argument("--qsr-refined-bec-path", type=Path, default=DATASET_CONFIG.qsr_refined_bec_path)
    parser.add_argument("--data-root", type=Path, default=DATASET_CONFIG.data_root)
    parser.add_argument("--phenotype-csv", type=Path, default=DATASET_CONFIG.phenotype_path)
    parser.add_argument("--output-dir", type=Path, default=DATASET_CONFIG.output_dir)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--gpu-id", default='auto')

    parser.add_argument("--reference-k", type=int, default=20)
    parser.add_argument("--graph-mode", choices=["phenotype", "fusion"], default='fusion')
    parser.add_argument("--fusion-beta", type=float, default=0.6)
    parser.add_argument("--reference-bandwidth", type=float, default=2.0)
    parser.add_argument("--categorical-penalty", type=float, default=4.0)
    parser.add_argument("--continuous-weights", type=float, nargs=len(DATASET_CONFIG.continuous_columns), default=[1.0, 0.3])
    parser.add_argument("--permute-phenotype", action="store_true", default=False)

    parser.add_argument("--refiner-epochs", type=int, default=80)
    parser.add_argument("--refiner-lr", type=float, default=0.01)
    parser.add_argument("--gate-max", type=float, default=0.5)
    parser.add_argument("--gate-l1-weight", type=float, default=0.001)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--variance-retention", type=float, default=0.85)

    parser.add_argument("--qsr-qc-columns", nargs="+", default=list(['func_mean_fd', 'func_dvars', 'func_quality']))
    parser.add_argument("--qsr-epochs", type=int, default=80)
    parser.add_argument("--qsr-lr", type=float, default=0.003)
    parser.add_argument("--qsr-hidden-channels", type=int, default=8)
    parser.add_argument("--qsr-eta", type=float, default=0.3)
    parser.add_argument("--qsr-r-max", type=float, default=0.04)
    parser.add_argument("--qsr-corruption-scale", type=float, default=0.5)
    parser.add_argument("--qsr-gate-max", type=float, default=0.4)
    parser.add_argument("--qsr-gate-weight", type=float, default=0.001)
    parser.add_argument("--qsr-variance-weight", type=float, default=0.25)
    parser.add_argument("--qsr-variance-retention", type=float, default=0.85)
    parser.add_argument("--qsr-basis-ridge", type=float, default=0.001)

    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=0.001)
    parser.add_argument("--classifier-repeats", type=int, default=1)
    parser.add_argument("--patient-label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--control-label", type=int, choices=[0, 1], default=0)
    add_stf_arguments(parser)
    args = parser.parse_args()
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
    from Graph_BEC.runner import run
    run(args)
