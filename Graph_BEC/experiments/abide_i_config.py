"""Complete ABIDE-I experiment configuration."""
from pathlib import Path

from Graph_BEC.experiments.configuration import ExperimentProfile

# abide_i_config.py -> experiments -> Graph_BEC -> repository root
ROOT = Path(__file__).resolve().parents[2]

PROFILE = ExperimentProfile(
    name="abide",
    data_root=ROOT / "dataset/ABIDE-I",
    phenotype_path=ROOT / "dataset/ABIDE-I/Phenotypic_Processing_filled.csv",
    output_dir=ROOT / "Graph_BEC/outputs/abide-i",
    bec_path=ROOT / "Graph_BEC/outputs/abide-i/abide_subject_bec.npz",
    refined_bec_path=ROOT / "Graph_BEC/outputs/abide-i/abide_refined_subject_bec.npz",
    qsr_refined_bec_path=ROOT / "Graph_BEC/outputs/abide-i/abide_qsr_refined_subject_bec.npz",
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
    defaults={
        "representations": ["original", "refined", "qc_refined"],
        "input_mode": "bec", "n_splits": 10, "validation_size": 0.2,
        "seed": 42, "gpu_id": "auto", "reference_k": 20,
        "graph_mode": "fusion", "fusion_beta": 0.6,
        "reference_bandwidth": 2.0, "categorical_penalty": 4.0,
        "continuous_weights": [1.0, 0.3], "permute_phenotype": False,
        "refiner_epochs": 80, "refiner_lr": 1e-2, "gate_max": 0.5,
        "gate_l1_weight": 1e-3, "anchor_weight": 1.0,
        "variance_weight": 1.0, "variance_retention": 0.85,
        "qsr_epochs": 80, "qsr_lr": 3e-3, "qsr_hidden_channels": 8,
        "qsr_eta": 0.15, "qsr_r_max": 0.03, "qsr_corruption_scale": 0.5,
        "qsr_gate_max": 0.5, "qsr_gate_weight": 1e-3,
        "qsr_variance_weight": 0.1, "qsr_variance_retention": 0.85,
        "qsr_basis_ridge": 1e-3,
        "qsr_qc_columns": ["func_mean_fd", "func_dvars", "func_quality"],
        "classifier_epochs": 100, "classifier_patience": 20,
        "classifier_lr": 1e-3, "classifier_repeats": 1,
        "patient_label": 1, "control_label": 0,
        "window_length": 78, "stride": 39, "epochs": 101,
        "fsta_checkpoint": "final", "loss_mode": "entropy", "loss_alpha": 0.01,
        "batch_size": 32, "log_every": 10, "d_model": 16,
        "d_inner_hid": 64, "d_k": 8, "d_v": 8, "n_head": 2,
        "dropout": 0.2, "n_warmup_steps": 4000, "lr_mul": 1.2,
        "weight_decay": 0.0, "adam_beta1": 0.9, "adam_beta2": 0.98,
    },
)
