"""QC-guided self-supervised weak refinement for directed BEC matrices."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Graph_BEC.downstream.brainnetcnn import DirectedE2E
from .qc_prior import (
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


class QSRBECRefiner(nn.Module):
    """Refine BEC with fold-level QC sensitivity and neighbor context."""

    def __init__(self, nodes_num=90, hidden_channels=8, gate_max=0.5):
        super().__init__()
        self.nodes_num = int(nodes_num)
        self.gate_max = float(gate_max)
        self.input_projection = nn.Conv2d(4, hidden_channels, kernel_size=1)
        self.context_first = DirectedE2E(hidden_channels, hidden_channels, self.nodes_num)
        self.context_second = DirectedE2E(hidden_channels, hidden_channels, self.nodes_num)
        self.gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.direction_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, current_bec, neighbor_bec, qc_sensitive_map, return_parts=False):
        if current_bec.shape != neighbor_bec.shape:
            raise ValueError("current_bec and neighbor_bec must have matching shapes")
        if current_bec.ndim != 3 or current_bec.shape[-1] != self.nodes_num:
            raise ValueError("BEC inputs must have shape [subjects, nodes, nodes]")
        qc_map = torch.as_tensor(
            qc_sensitive_map, device=current_bec.device, dtype=current_bec.dtype
        )
        if qc_map.ndim == 2:
            qc_map = qc_map.unsqueeze(0).expand(current_bec.shape[0], -1, -1)
        if qc_map.shape != current_bec.shape:
            raise ValueError("qc_sensitive_map must be [nodes, nodes] or match BEC inputs")
        difference = neighbor_bec - current_bec
        inputs = torch.stack((current_bec, neighbor_bec, difference.abs(), qc_map), dim=1)
        features = F.leaky_relu(self.input_projection(inputs), negative_slope=0.1)
        features = F.leaky_relu(self.context_first(features), negative_slope=0.1)
        features = F.leaky_relu(self.context_second(features), negative_slope=0.1)
        gate = self.gate_max * torch.sigmoid(self.gate_head(features)).squeeze(1)
        direction = torch.tanh(self.direction_head(features)).squeeze(1)
        diagonal = torch.eye(self.nodes_num, device=current_bec.device, dtype=torch.bool)[None]
        gate = gate.masked_fill(diagonal, 0.0)
        direction = direction.masked_fill(diagonal, 0.0)
        refined = (current_bec + gate * direction * difference).masked_fill(diagonal, 0.0)
        if return_parts:
            return refined, gate, direction, difference
        return refined


def qsr_refinement_loss(
    original_refined,
    corrupted_refined,
    pseudo_target,
    original_bec,
    original_gate,
    corrupted_gate,
    variance_retention=0.85,
    gate_weight=0.1,
    variance_weight=0.1,
):
    """Combine pseudo-target restoration, sparse gating, and variance retention."""
    pseudo = F.smooth_l1_loss(original_refined, pseudo_target)
    restore = F.smooth_l1_loss(corrupted_refined, pseudo_target)
    gate = 0.5 * (original_gate.abs().mean() + corrupted_gate.abs().mean())
    original_variance = original_bec.flatten(1).var(dim=0, unbiased=False).mean()
    refined_variance = original_refined.flatten(1).var(dim=0, unbiased=False).mean()
    variance = F.relu(variance_retention * original_variance - refined_variance)
    total = pseudo + restore + gate_weight * gate + variance_weight * variance
    return total, {
        "pseudo_loss": pseudo,
        "restore_loss": restore,
        "gate_loss": gate,
        "variance_loss": variance,
    }


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
