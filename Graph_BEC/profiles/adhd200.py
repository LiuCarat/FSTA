"""Complete ADHD200 experiment configuration."""
from pathlib import Path

from Graph_BEC.profiles.configuration import ExperimentProfile

# adhd200.py -> profiles -> Graph_BEC -> repository root
ROOT = Path(__file__).resolve().parents[2]
QC_COLUMNS = ("QC_Athena", "QC_NIAK")

PROFILE = ExperimentProfile(
    name="adhd200",
    data_root=ROOT / "dataset/ADHD200",
    phenotype_path=ROOT / "dataset/ADHD200/adhd200_preprocessed_phenotypics.tsv",
    output_dir=ROOT / "Graph_BEC/outputs/adhd200",
    bec_path=ROOT / "Graph_BEC/outputs/adhd200/adhd200_subject_bec.npz",
    refined_bec_path=ROOT / "Graph_BEC/outputs/adhd200/adhd200_refined_subject_bec.npz",
    qsr_refined_bec_path=ROOT / "Graph_BEC/outputs/adhd200/adhd200_qsr_refined_subject_bec.npz",
    phenotype_format="tsv",
    phenotype_id_column="ScanDir ID",
    patient_column="DX",
    control_column="DX",
    patient_values=("1", "2", "3"),
    control_values=("0",),
    site_column="Site",
    sex_column="Gender",
    continuous_columns=("Age", "Full4 IQ", "Handedness"),
    confound_columns=("Age", "Gender", "Full4 IQ", "Handedness"),
    qc_columns=QC_COLUMNS,
    source_roi_count=116,
    roi_count=90,
    exclude_subjects=(),
    defaults={
        "representations": ["original", "refined", "qc_refined"],
        "input_mode": "bec", "n_splits": 10, "validation_size": 0.2,
        "seed": 42, "gpu_id": "auto", "reference_k": 20,
        "graph_mode": "fusion", "fusion_beta": 0.6,
        "reference_bandwidth": 2.0, "categorical_penalty": 4.0,
        "continuous_weights": [1.0, 0.3, 0.3], "permute_phenotype": False,
        "refiner_epochs": 80, "refiner_lr": 1e-2, "gate_max": 0.5,
        "gate_l1_weight": 1e-3, "anchor_weight": 1.0,
        "variance_weight": 1.0, "variance_retention": 0.85,
        "qsr_epochs": 80, "qsr_lr": 3e-3, "qsr_hidden_channels": 8,
        "qsr_eta": 0.15, "qsr_r_max": 0.03, "qsr_corruption_scale": 0.5,
        "qsr_gate_max": 0.5, "qsr_gate_weight": 1e-3,
        "qsr_variance_weight": 0.1, "qsr_variance_retention": 0.85,
        "qsr_basis_ridge": 1e-3, "qsr_qc_columns": list(QC_COLUMNS),
        "classifier_epochs": 60, "classifier_patience": 12,
        "classifier_lr": 1e-3, "classifier_repeats": 1,
        "patient_label": 1, "control_label": 0,
        "window_length": 50, "stride": 25, "epochs": 91, # 91
        "fsta_checkpoint": "final", "loss_mode": "entropy", "loss_alpha": 0.03, # 0.03
        "batch_size": 32, "log_every": 20, "d_model": 16,
        "d_inner_hid": 64, "d_k": 8, "d_v": 8, "n_head": 2,
        "dropout": 0.2, "n_warmup_steps": 4000, "lr_mul": 1.2,
        "weight_decay": 0.0, "adam_beta1": 0.9, "adam_beta2": 0.98,
    },
)
