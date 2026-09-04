"""Training and inference helpers for PGR-BEC and QC/QSR refinement."""
from __future__ import annotations

import copy

import numpy as np
import torch

from Graph_BEC.model.pgr_bec_ablation import PGRBECStatic, static_refinement_loss
from Graph_BEC.model.qsr_bec import QSRBECRefiner, qsr_refinement_loss
from Graph_BEC.model.qsr_bec.qc_prior import (
    build_confound_design,
    build_pseudo_target,
    build_qc_sensitive_map,
    fit_qc_artifact_basis,
    fit_qc_scaler,
    qc_corrupt,
    relative_change,
    sample_joint_qc_delta,
    transform_qc_badness,
)


def train_pgr_refiner(args, bec, neighbor, device):
    """Train the label-free static phenotype-guided refiner."""
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
            args.anchor_weight, args.gate_l1_weight, args.variance_weight,
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


def apply_pgr_refiner(model, bec, neighbor, device):
    with torch.no_grad():
        return model(
            torch.from_numpy(bec).float().to(device),
            torch.from_numpy(neighbor).float().to(device),
        ).cpu().numpy()


def train_qsr_refiner(
    args, bec, neighbor, train_qc, confound_values, site_ids, device, seed
):
    """Train QSR using QC information from the training fold only."""
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
            original_gate, corrupted_gate, args.qsr_variance_retention,
            args.qsr_gate_weight, args.qsr_variance_weight,
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
    refined_array = refined.cpu().numpy()
    effective_coefficient = gate * direction
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
