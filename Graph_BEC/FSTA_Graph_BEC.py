#!/usr/bin/env python3
"""End-to-end FSTA-Graph-BEC experiment runner.

Raw mode: ROI time series -> FSTA -> BEC -> label-free graph refinement.
BEC mode: existing subject_bec.npz -> label-free graph refinement.
Diagnosis labels are used only in final statistics and downstream classification.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from Graph_BEC.normative_bec import (
    bec_separability, edge_effect_sizes, normative_reference,
)
from Graph_BEC.data import (
    load_subject_dataset, load_bec_archive, make_stratified_splits,
    prepare_fold_arrays, select_device, set_seed,
)
from Graph_BEC.downstream import train_classifier
from Graph_BEC.model import (
    PGRBECStatic, QSRBECRefiner,
    train_fsta, extract_subject_bec,
)
from Graph_BEC.model.pgr_bec_static import static_refinement_loss
from Graph_BEC.model.qsr_bec import qsr_refinement_loss
from Graph_BEC.qc import (
    DEFAULT_QC_COLUMNS,
    build_confound_design,
    build_pseudo_target,
    build_qc_sensitive_map,
    fit_qc_artifact_basis,
    fit_qc_scaler,
    load_aligned_qc,
    qc_corrupt,
    relative_change,
    sample_joint_qc_delta,
    transform_qc_badness,
)
from Graph_BEC.phenotype import (
    build_reference_graph,
    fused_graph, load_aligned_phenotypes, load_phenotypes,
    subject_fc_features, topk_graph,
)

DEFAULT_BEC = ROOT / "downstream_abide_i/outputs/entropy/loss_alpha_0.01/seed_42/epochs_101/subject_bec.npz"
# downstream_abide_i/outputs/original/loss_alpha_0.8/seed_42/epochs_31/subject_bec.npz
DEFAULT_DATA_ROOT = ROOT / "dataset/ABIDE-I"
DEFAULT_PHENOTYPE = ROOT / "dataset/ABIDE-I/Phenotypic_Processing_filled.csv"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/outputs"


def parse_args():
    # =========================== 数据 & 实验配置 ===========================
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=["bec", "raw"], default="bec")
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--phenotype-csv", type=Path, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--pipeline", default="cpac") # 预处理流水线名称
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True) # 是否标准化
    parser.add_argument("--max-subjects", type=int, default=None) # 限制加载的受试者数量上限（None=全部） 
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-size", type=float, default=0.2) # 验证集比例
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="auto")

    # =========================== FSTA 训练参数 ===========================
    parser.add_argument("--window-length", type=int, default=78)
    parser.add_argument("--stride", type=int, default=39)
    parser.add_argument("--epochs", type=int, default=101)
    parser.add_argument("--fsta-checkpoint", choices=["final", "best"], default="final")
    parser.add_argument("--loss-mode", choices=["original", "entropy"], default="entropy") # 损失函数类型
    parser.add_argument("--loss-alpha", type=float, default=0.01) # 损失函数权重
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=16) # FSTA Transformer 隐藏层维度
    parser.add_argument("--d-inner-hid", type=int, default=64) # FFN 内部隐藏层维度，通常为 d_model 的 4 倍
    parser.add_argument("--d-k", type=int, default=8) # FSTA Transformer key 的维度
    parser.add_argument("--d-v", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-warmup-steps", type=int, default=4000) # AdamW 的预热步数
    parser.add_argument("--lr-mul", type=float, default=1.2) # 学习率倍数
    parser.add_argument("--weight-decay", type=float, default=0.0) # 权重衰减，0 表示不用
    parser.add_argument("--adam-beta1", type=float, default=0.9) # AdamW 的 beta1 参数（一阶动量衰减率）
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--num-hidden-layers", type=int, default=1) # FourierAtt 编码层数量
    parser.add_argument("--num-attention-heads", type=int, default=2)
    parser.add_argument("--hidden-act", default="gelu") # 前馈层激活函数
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0.5)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.5)
    parser.add_argument("--initializer-range", type=float, default=0.02) # 初始化范围
    parser.add_argument("--no-filters", action="store_true") # 不使用傅里叶注意力

    # =========================== PGR-BEC 修正模块 ===========================
    parser.add_argument("--refiner-epochs", type=int, default=80) # 修正模块训练轮数
    parser.add_argument("--refiner-lr", type=float, default=1e-2)
    parser.add_argument("--gate-max", type=float, default=0.5) # 门控输出的最大值，0.5
    parser.add_argument("--gate-l1-weight", type=float, default=1e-3) # 门控损失权重，增大稀疏，减小密集
    parser.add_argument("--anchor-weight", type=float, default=1.0) # 锚定损失权重，增大保守，减小激进
    parser.add_argument("--variance-weight", type=float, default=1.0) # 方差损失权重，增大保守，减小激进
    parser.add_argument("--variance-retention", type=float, default=0.85)

    # =========================== 表型邻域参考图 ===========================
    parser.add_argument("--reference-k", type=int, default=20)
    parser.add_argument("--graph-mode", choices=["phenotype", "fusion"], default="fusion")
    parser.add_argument("--fusion-beta", type=float, default=0.6)
    parser.add_argument("--reference-bandwidth", type=float, default=2.0) # 2 减小: 只有最近邻获得显著权重
    # 与 --reference-k 配合：大 k+小 σ=稀疏大邻域，小 k+大 σ=均匀小邻域
    parser.add_argument("--categorical-penalty", type=float, default=4.0) # 4 增大: 不同性别的受试者更难成为邻居
    parser.add_argument("--continuous-weights", type=float, nargs=2, default=[1.0, 0.3]) # [1.0, 0.3] 表示 FIQ 主导、PIQ 辅助构建邻域
    parser.add_argument("--permute-phenotype", action="store_true") # 随机打乱训练集的表型-受试者对应关系，以增加鲁棒性

    # =========================== QSR-BEC QC 弱监督 ===========================
    parser.add_argument("--qsr-qc-columns", nargs="+", default=list(DEFAULT_QC_COLUMNS))
    parser.add_argument("--qsr-epochs", type=int, default=80)
    parser.add_argument("--qsr-lr", type=float, default=3e-3) # 3e-3
    parser.add_argument("--qsr-hidden-channels", type=int, default=8) # 8
    parser.add_argument("--qsr-eta", type=float, default=0.15)
    parser.add_argument("--qsr-r-max", type=float, default=0.03)
    parser.add_argument("--qsr-corruption-scale", type=float, default=0.5)
    parser.add_argument("--qsr-gate-max", type=float, default=0.5) # 0.5
    parser.add_argument("--qsr-gate-weight", type=float, default=1e-3) # 1e-3
    parser.add_argument("--qsr-variance-weight", type=float, default=0.1)
    parser.add_argument("--qsr-variance-retention", type=float, default=0.85)
    parser.add_argument("--qsr-basis-ridge", type=float, default=1e-3)

    # =========================== BrainNetCNN 分类器 ===========================
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-patience", type=int, default=20)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-repeats", type=int, default=1) # 重复训练次数
    parser.add_argument("--tc-label", type=int, default=1, choices=[0, 1]) # ASD=1, TC=0
    return parser.parse_args()


def load_pipeline_data(args, device):
    fsta_metrics = None
    if args.input_mode == "raw":
        subjects = load_subject_dataset(args.data_root, args.pipeline, args.strategy, args.derivative, args.standardize, args.max_subjects)
        print(f"Training FSTA from {len(subjects['records'])} subject time series...")
        fsta_model, fsta_metrics = train_fsta(args, subjects["time_series"], device)
        extracted = extract_subject_bec(fsta_model, subjects["records"], subjects["time_series"], args.window_length, args.stride, device)
        data = {
            "bec": extracted["bec"], "labels": subjects["labels"],
            "subject_ids": subjects["subject_ids"], "site_ids": subjects["site_ids"],
            "reconstruction_mse": extracted["reconstruction_mse"],
        }
        if args.bec_path.is_file():
            archived = load_bec_archive(args.bec_path)
            if np.array_equal(data["subject_ids"].astype(str), archived["subject_ids"].astype(str)):
                difference = np.abs(data["bec"] - archived["bec"])
                print(
                    "raw-vs-archive BEC: "
                    f"max_abs={difference.max():.3e}, mean_abs={difference.mean():.3e}"
                )
    else:
        data = load_bec_archive(args.bec_path)
        if args.max_subjects is not None:
            data = {key: value[:args.max_subjects] if hasattr(value, "__len__") and len(value) == len(data["bec"]) else value for key, value in data.items()}
    if args.graph_mode == "fusion":
        if args.input_mode == "raw":
            graph_time_series = subjects["time_series"]
        else:
            graph_subjects = load_subject_dataset(
                args.data_root, args.pipeline, args.strategy, args.derivative,
                args.standardize, args.max_subjects,
            )
            by_subject = {
                str(subject_id): series
                for subject_id, series in zip(
                    graph_subjects["subject_ids"], graph_subjects["time_series"]
                )
            }
            try:
                graph_time_series = [by_subject[str(subject_id)] for subject_id in data["subject_ids"]]
            except KeyError as error:
                raise ValueError(
                    "Fusion mode requires raw ROI time series for every BEC subject"
                ) from error
        data["fmri_features"] = subject_fc_features(graph_time_series)
    phenotype = load_phenotypes(args.phenotype_csv, data["subject_ids"], data["site_ids"])
    data.update(phenotype)
    data["qsr_qc"] = load_aligned_qc(
        args.phenotype_csv, data["subject_ids"], args.qsr_qc_columns
    )
    data["qsr_confound_values"] = load_aligned_phenotypes(
        args.phenotype_csv,
        data["subject_ids"],
        ["AGE_AT_SCAN", "SEX", "FIQ", "PIQ"],
    ).astype(np.float32)
    data["bec"] = np.asarray(data["bec"], dtype=np.float32)
    data["labels"] = np.asarray(data["labels"], dtype=np.int64)
    return data, fsta_metrics


def build_fold_reference(args, arrays, fmri_arrays=None):
    common = dict(
        k=args.reference_k, bandwidth=args.reference_bandwidth,
        categorical_penalty=args.categorical_penalty,
        continuous_weights=args.continuous_weights,
        permute=args.permute_phenotype, seed=args.seed,
    )
    if args.graph_mode == "phenotype":
        train_weights, _ = build_reference_graph(
            arrays["train_cont"], arrays["train_cat"],
            arrays["train_cont"], arrays["train_cat"], **common
        )
        _, val_weights = build_reference_graph(
            arrays["train_cont"], arrays["train_cat"],
            arrays["val_cont"], arrays["val_cat"], **common
        )
        _, test_weights = build_reference_graph(
            arrays["train_cont"], arrays["train_cat"],
            arrays["test_cont"], arrays["test_cat"], **common
        )
        train_neighbor, _ = normative_reference(arrays["train_bec"], train_weights)
        val_neighbor, _ = normative_reference(arrays["train_bec"], val_weights)
        test_neighbor, _ = normative_reference(arrays["train_bec"], test_weights)
        return {
            "train_neighbor": train_neighbor,
            "val_neighbor": val_neighbor,
            "test_neighbor": test_neighbor,
            "train_weights": train_weights,
            "val_weights": val_weights,
            "test_weights": test_weights,
        }
    if fmri_arrays is None:
        raise ValueError("Fusion graph mode requires fold fMRI features")
    train_fmri, val_fmri, test_fmri = fmri_arrays
    fmri_mean = train_fmri.mean(axis=0)
    fmri_std = train_fmri.std(axis=0)
    fmri_std[~np.isfinite(fmri_std) | (fmri_std < 1e-6)] = 1.0
    train_fmri = ((train_fmri - fmri_mean) / fmri_std).astype(np.float32)
    val_fmri = ((val_fmri - fmri_mean) / fmri_std).astype(np.float32)
    test_fmri = ((test_fmri - fmri_mean) / fmri_std).astype(np.float32)
    phenotype_train_weights, _ = build_reference_graph(
        arrays["train_cont"], arrays["train_cat"],
        arrays["train_cont"], arrays["train_cat"], **common
    )
    _, phenotype_val_weights = build_reference_graph(
        arrays["train_cont"], arrays["train_cat"],
        arrays["val_cont"], arrays["val_cat"], **common
    )
    _, phenotype_test_weights = build_reference_graph(
        arrays["train_cont"], arrays["train_cat"],
        arrays["test_cont"], arrays["test_cat"], **common
    )
    fmri_train_weights = topk_graph(
        train_fmri, train_fmri, args.reference_k, exclude_self=True
    )
    fmri_val_weights = topk_graph(train_fmri, val_fmri, args.reference_k)
    fmri_test_weights = topk_graph(train_fmri, test_fmri, args.reference_k)
    fused_train_weights = fused_graph(
        fmri_train_weights, phenotype_train_weights, args.fusion_beta,
        args.reference_k,
    )
    fused_val_weights = fused_graph(
        fmri_val_weights, phenotype_val_weights, args.fusion_beta,
        args.reference_k,
    )
    fused_test_weights = fused_graph(
        fmri_test_weights, phenotype_test_weights, args.fusion_beta,
        args.reference_k,
    )
    train_neighbor, _ = normative_reference(arrays["train_bec"], fused_train_weights)
    val_neighbor, _ = normative_reference(arrays["train_bec"], fused_val_weights)
    test_neighbor, _ = normative_reference(arrays["train_bec"], fused_test_weights)
    return {
        "train_neighbor": train_neighbor,
        "val_neighbor": val_neighbor,
        "test_neighbor": test_neighbor,
        "train_weights": fused_train_weights,
        "val_weights": fused_val_weights,
        "test_weights": fused_test_weights,
    }


def train_refiner(args, bec, neighbor, device):
    """Train the label-free Static phenotype gate."""
    model = PGRBECStatic(
        bec.shape[-1], hidden_channels=16, gate_max=args.gate_max
    ).to(device)
    original = torch.from_numpy(bec).float().to(device)
    neighbor_tensor = torch.from_numpy(neighbor).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.refiner_lr)
    best_state, best_loss, metrics = None, float("inf"), {}
    for _ in range(args.refiner_epochs):
        optimizer.zero_grad()
        refined, gate, _ = model(original, neighbor_tensor, return_parts=True)
        total, parts = static_refinement_loss(
            refined, original, gate, args.variance_retention,
            args.anchor_weight, args.gate_l1_weight, args.variance_weight
        )
        total.backward()
        optimizer.step()
        metrics = {
            "refiner_loss": float(total.item()),
            **{key: float(value.item()) for key, value in parts.items()},
        }
        if metrics["refiner_loss"] < best_loss:
            best_loss = metrics["refiner_loss"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        refined, gate, _ = model(original, neighbor_tensor, return_parts=True)
    metrics.update({
        "gate_mean": float(gate.mean()),
        "gate_max": float(gate.max()),
        "gate_fraction_above_0p01": float((gate > 0.01).float().mean()),
    })
    return model, refined.cpu().numpy(), metrics


def train_qsr_refiner(args, bec, neighbor, train_qc, confound_values, site_ids, device, seed):
    """Train QSR using QC only within this training fold."""
    qc_scaler = fit_qc_scaler(train_qc)
    qc_badness = transform_qc_badness(train_qc, qc_scaler)
    confounds = build_confound_design(site_ids, confound_values)
    qc_basis = fit_qc_artifact_basis(
        bec, qc_badness, confounds, ridge=args.qsr_basis_ridge
    )
    qc_sensitive_map = build_qc_sensitive_map(qc_basis)
    pseudo_target = build_pseudo_target(
        bec, qc_badness, qc_basis, args.qsr_eta, args.qsr_r_max
    )

    model = QSRBECRefiner(
        bec.shape[-1], args.qsr_hidden_channels, args.qsr_gate_max
    ).to(device)
    original = torch.from_numpy(bec).float().to(device)
    neighbor_tensor = torch.from_numpy(neighbor).float().to(device)
    pseudo_tensor = torch.from_numpy(pseudo_target).float().to(device)
    sensitive_tensor = torch.from_numpy(qc_sensitive_map).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.qsr_lr)
    rng = np.random.default_rng(seed)
    best_state, best_loss, metrics = None, float("inf"), {}
    for _ in range(args.qsr_epochs):
        qc_delta = sample_joint_qc_delta(qc_badness, rng)
        corrupted = qc_corrupt(
            pseudo_target, qc_basis, qc_delta, args.qsr_corruption_scale,
            maximum_ratio=max(2.0 * args.qsr_r_max, args.qsr_r_max),
        )
        corrupted_tensor = torch.from_numpy(corrupted).float().to(device)
        optimizer.zero_grad()
        original_refined, original_gate, _, _ = model(
            original, neighbor_tensor, sensitive_tensor, return_parts=True
        )
        corrupted_refined, corrupted_gate, _, _ = model(
            corrupted_tensor, neighbor_tensor, sensitive_tensor, return_parts=True
        )
        total, parts = qsr_refinement_loss(
            original_refined, corrupted_refined, pseudo_tensor, original,
            original_gate, corrupted_gate,
            args.qsr_variance_retention, args.qsr_gate_weight,
            args.qsr_variance_weight,
        )
        total.backward()
        optimizer.step()
        metrics = {
            "qsr_loss": float(total.item()),
            **{key: float(value.item()) for key, value in parts.items()},
        }
        if metrics["qsr_loss"] < best_loss:
            best_loss = metrics["qsr_loss"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        refined, gate, direction, _ = model(
            original, neighbor_tensor, sensitive_tensor, return_parts=True
        )
    effective_coefficient = gate * direction
    refined_array = refined.cpu().numpy()
    metrics.update({
        "qsr_gate_mean": float(gate.mean().item()),
        "qsr_gate_max": float(gate.max().item()),
        "qsr_direction_abs_mean": float(direction.abs().mean().item()),
        "qsr_effective_coefficient_mean": float(effective_coefficient.mean().item()),
        "qsr_effective_coefficient_abs_mean": float(
            effective_coefficient.abs().mean().item()
        ),
        "qsr_pseudo_relative_change": relative_change(bec, pseudo_target),
        "qsr_refined_relative_change": relative_change(bec, refined_array),
        "qsr_sensitive_map_mean": float(qc_sensitive_map.mean()),
        "qsr_basis_abs_mean": float(np.abs(qc_basis).mean()),
    })
    return model, refined_array, qc_sensitive_map, metrics


def apply_qsr_refiner(model, bec, neighbor, qc_sensitive_map, device):
    """Apply a trained QSR model without reading validation/test QC."""
    with torch.no_grad():
        return model(
            torch.from_numpy(bec).float().to(device),
            torch.from_numpy(neighbor).float().to(device),
            torch.from_numpy(qc_sensitive_map).float().to(device),
        ).cpu().numpy()


def run_fold(args, fold, data, train_index, val_index, test_index, device):
    fold_seed = args.seed + fold * 1000
    set_seed(fold_seed)
    arrays = prepare_fold_arrays(
        data["bec"][train_index], data["bec"][val_index], data["bec"][test_index],
        data["continuous"][train_index], data["continuous"][val_index], data["continuous"][test_index],
        data["categorical_raw"][train_index], data["categorical_raw"][val_index], data["categorical_raw"][test_index],
    )
    fmri_arrays = None
    if args.graph_mode == "fusion":
        fmri_arrays = (
            data["fmri_features"][train_index],
            data["fmri_features"][val_index],
            data["fmri_features"][test_index],
        )
    # Fusion/phenotype graph is fixed independently of QC for all three outputs.
    base_reference = build_fold_reference(args, arrays, fmri_arrays)
    set_seed(fold_seed)
    base_refiner, train_base_refined, base_refinement_metrics = train_refiner(
        args, arrays["train_bec"], base_reference["train_neighbor"], device,
    )
    with torch.no_grad():
        val_base_refined = base_refiner(
            torch.from_numpy(arrays["val_bec"]).float().to(device),
            torch.from_numpy(base_reference["val_neighbor"]).float().to(device),
        ).cpu().numpy()
        test_base_refined = base_refiner(
            torch.from_numpy(arrays["test_bec"]).float().to(device),
            torch.from_numpy(base_reference["test_neighbor"]).float().to(device),
        ).cpu().numpy()

    representations = {
        "original": {
            "train": arrays["train_bec"],
            "val": arrays["val_bec"],
            "test": arrays["test_bec"],
            "reference": base_reference,
            "refinement": {},
        },
        "refined": {
            "train": train_base_refined,
            "val": val_base_refined,
            "test": test_base_refined,
            "reference": base_reference,
            "refinement": base_refinement_metrics,
        },
    }
    set_seed(fold_seed + 1)
    qsr_refiner, train_qsr_refined, qc_sensitive_map, qsr_metrics = train_qsr_refiner(
        args,
        arrays["train_bec"],
        base_reference["train_neighbor"],
        data["qsr_qc"][train_index],
        data["qsr_confound_values"][train_index],
        data["site_ids"][train_index],
        device,
        fold_seed + 1,
    )
    print(
        f"fold {fold}: QSR loss={qsr_metrics['qsr_loss']:.6f}, "
        f"pseudo={qsr_metrics['pseudo_loss']:.6f}, "
        f"restore={qsr_metrics['restore_loss']:.6f}, "
        f"gate_mean={qsr_metrics['qsr_gate_mean']:.6f}, "
        f"direction_abs_mean={qsr_metrics['qsr_direction_abs_mean']:.6f}, "
        f"effective_abs_mean={qsr_metrics['qsr_effective_coefficient_abs_mean']:.6f}, "
        f"refined_change={100.0 * qsr_metrics['qsr_refined_relative_change']:.2f}%, "
        f"pseudo_change={100.0 * qsr_metrics['qsr_pseudo_relative_change']:.2f}%"
    )
    val_qsr_refined = apply_qsr_refiner(
        qsr_refiner, arrays["val_bec"], base_reference["val_neighbor"],
        qc_sensitive_map, device,
    )
    test_qsr_refined = apply_qsr_refiner(
        qsr_refiner, arrays["test_bec"], base_reference["test_neighbor"],
        qc_sensitive_map, device,
    )
    representations["qc_refined"] = {
        "train": train_qsr_refined,
        "val": val_qsr_refined,
        "test": test_qsr_refined,
        "reference": base_reference,
        "refinement": qsr_metrics,
    }
    train_labels, test_labels = data["labels"][train_index], data["labels"][test_index]
    classifier_args = dict(
        device=device, max_epochs=args.classifier_epochs,
        patience=args.classifier_patience, batch_size=args.batch_size,
        learning_rate=args.classifier_lr,
    )
    metric_runs = {name: [] for name in representations}
    for repeat in range(args.classifier_repeats):
        classifier_seed = fold_seed + repeat + 1
        for name, representation in representations.items():
            metrics, _ = train_classifier(
                representation["train"], train_labels,
                representation["val"], data["labels"][val_index],
                representation["test"], test_labels,
                seed=classifier_seed, **classifier_args
            )
            metric_runs[name].append(metrics)
    metrics_by_name = {}
    for name, runs in metric_runs.items():
        metric_names = runs[0].keys()
        metrics_by_name[name] = {
            key: float(np.mean([run[key] for run in runs]))
            for key in metric_names
        }

    result = {name: metrics for name, metrics in metrics_by_name.items()}
    original_test = representations["original"]["test"]
    original_edge, original_effect = edge_effect_sizes(
        original_test, test_labels, args.tc_label
    )
    for name, representation in representations.items():
        group = bec_separability(representation["test"], test_labels, args.tc_label)
        edge, effect = edge_effect_sizes(
            representation["test"], test_labels, args.tc_label
        )
        result.update({f"{name}_{key}": value for key, value in group.items()})
        result.update({f"{name}_{key}": value for key, value in edge.items()})
        result[f"{name}_variance_retention"] = float(
            np.var(representation["test"])
            / max(np.var(original_test), 1e-12)
        )
        result[f"{name}_edge_abs_d_change"] = float(
            np.mean(np.abs(effect)) - np.mean(np.abs(original_effect))
        )
        result[f"{name}_paired_auc_delta_mean"] = float(
            metrics_by_name[name]["AUC"] - metrics_by_name["original"]["AUC"]
        )
        result.update({
            f"{name}_{key}": value
            for key, value in representation["refinement"].items()
        })
    return result


def save_results(args, fold_results, fsta_metrics):
    rows = []
    for fold, result in enumerate(fold_results, 1):
        row = {"fold": fold}
        representation_names = [
            name for name, value in result.items() if isinstance(value, dict)
        ]
        for name in representation_names:
            row.update({f"{name}_{key}": value for key, value in result[name].items()})
        row.update({
            key: value for key, value in result.items()
            if key not in representation_names
        })
        rows.append(row)
    summary = {"config": vars(args), "fsta_training": fsta_metrics, "folds": rows}
    representation_names = [
        name for name, value in fold_results[0].items() if isinstance(value, dict)
    ]
    for name in representation_names:
        for metric in ("ACC", "SPE", "AUC", "Precision", "Recall", "F1"):
            values = [row[f"{name}_{metric}"] for row in rows]
            summary[f"{name}_{metric}_mean"] = float(np.mean(values)); summary[f"{name}_{metric}_std"] = float(np.std(values))
            summary[f"{name}_{metric}_display"] = f"{100 * summary[f'{name}_{metric}_mean']:.2f}±{100 * summary[f'{name}_{metric}_std']:.2f}"
    summary["representations"] = representation_names
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.output_dir.iterdir():
        if path.is_file() and path.name not in {"experiment_summary.csv", "summary.json"}: path.unlink()
    with (args.output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main():
    args = parse_args(); set_seed(args.seed); device = select_device(args.gpu_id)
    print(f"Loading data from {args.data_root} with phenotype {args.phenotype_csv}...")
    data, fsta_metrics = load_pipeline_data(args, device)
    if np.unique(data["labels"]).size != 2:
        raise ValueError(
            "The selected subjects contain only one diagnosis class. "
            "Use the full dataset or choose a max-subjects subset containing both labels."
        )
    print(
        f"Input={args.input_mode}; graph={args.graph_mode}; subjects={len(data['bec'])}; "
        f"BEC={data['bec'].shape}; labels={np.bincount(data['labels'])}; device={device}"
    )
    results = []
    for fold, train_index, val_index, test_index in make_stratified_splits(data["labels"], args.n_splits, args.seed, args.validation_size):
        result = run_fold(args, fold, data, train_index, val_index, test_index, device); results.append(result)
        fold_report = " | ".join(
            f"{name} AUC={result[name]['AUC']:.4f}"
            for name, value in result.items() if isinstance(value, dict)
        )
        print(f"fold {fold}: {fold_report}")
    summary = save_results(args, results, fsta_metrics)
    report_metrics = ("ACC", "SPE", "AUC", "Precision", "Recall", "F1")
    print(
        "\nQSR-BEC QC weak-supervision configuration:\n"
        f"  QC columns: {', '.join(args.qsr_qc_columns)}\n"
        f"  Training: epochs={args.qsr_epochs}; lr={args.qsr_lr:g}; "
        f"hidden_channels={args.qsr_hidden_channels}\n"
        f"  Pseudo-target: eta={args.qsr_eta:g}; r_max={args.qsr_r_max:g}\n"
        f"  Synthetic corruption: scale={args.qsr_corruption_scale:g}\n"
        f"  Refiner: gate_max={args.qsr_gate_max:g}; "
        f"gate_weight={args.qsr_gate_weight:g}\n"
        f"  Variance: weight={args.qsr_variance_weight:g}; "
        f"retention={args.qsr_variance_retention:g}\n"
        f"  QC basis: ridge={args.qsr_basis_ridge:g}"
    )
    print("\nmean±std (%)")
    print("representation | " + " | ".join(report_metrics))
    for name in summary["representations"]:
        values = [summary[f"{name}_{metric}_display"] for metric in report_metrics]
        print(f"{name:20s} | " + " | ".join(values))


if __name__ == "__main__": main()
