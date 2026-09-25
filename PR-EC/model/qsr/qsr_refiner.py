from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PR_EC.downstream.brainnetcnn import DirectedE2E
from .qc_target import (
    build_confound_design,
    build_pseudo_target,
    build_qc_sensitive_map,
    fit_qc_sensitivity_basis,
    fit_qc_scaler,
    apply_qc_perturbation,
    relative_change,
    sample_joint_qc_delta,
    transform_qc_sensitivity,
)


class QSRECRefiner(nn.Module):
    def __init__(self, nodes_num=90, hidden_channels=8, gate_max=0.5):
        super().__init__()
        self.nodes_num = int(nodes_num)
        self.gate_max = float(gate_max)
        self.input_projection = nn.Conv2d(4, hidden_channels, kernel_size=1)
        self.context_first = DirectedE2E(hidden_channels, hidden_channels, self.nodes_num)
        self.context_second = DirectedE2E(hidden_channels, hidden_channels, self.nodes_num)
        self.gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.direction_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, current_ec, neighbor_ec, qc_sensitive_map, return_parts=False):
        if current_ec.shape != neighbor_ec.shape:
            raise ValueError("current_ec and neighbor_ec must have matching shapes")
        if current_ec.ndim != 3 or current_ec.shape[-1] != self.nodes_num:
            raise ValueError("EC inputs must have shape [subjects, nodes, nodes]")
        qc_map = torch.as_tensor(
            qc_sensitive_map, device=current_ec.device, dtype=current_ec.dtype
        )
        if qc_map.ndim == 2:
            qc_map = qc_map.unsqueeze(0).expand(current_ec.shape[0], -1, -1)
        if qc_map.shape != current_ec.shape:
            raise ValueError("qc_sensitive_map must be [nodes, nodes] or match EC inputs")
        difference = neighbor_ec - current_ec
        inputs = torch.stack((current_ec, neighbor_ec, difference.abs(), qc_map), dim=1)
        features = F.leaky_relu(self.input_projection(inputs), negative_slope=0.1)
        features = F.leaky_relu(self.context_first(features), negative_slope=0.1)
        features = F.leaky_relu(self.context_second(features), negative_slope=0.1)
        gate = self.gate_max * torch.sigmoid(self.gate_head(features)).squeeze(1)
        direction = torch.tanh(self.direction_head(features)).squeeze(1)
        diagonal = torch.eye(self.nodes_num, device=current_ec.device, dtype=torch.bool)[None]
        gate = gate.masked_fill(diagonal, 0.0)
        direction = direction.masked_fill(diagonal, 0.0)
        refined = (current_ec + gate * direction * difference).masked_fill(diagonal, 0.0)
        if return_parts:
            return refined, gate, direction, difference
        return refined


def qsr_refinement_loss(
    original_refined,
    perturbed_refined,
    pseudo_target,
    original_ec,
    original_gate,
    perturbed_gate,
    variance_retention=0.85,
    gate_weight=0.1,
    variance_weight=0.1,
):
    pseudo = F.smooth_l1_loss(original_refined, pseudo_target)
    restore = F.smooth_l1_loss(perturbed_refined, pseudo_target)
    gate = 0.5 * (original_gate.abs().mean() + perturbed_gate.abs().mean())
    original_variance = original_ec.flatten(1).var(dim=0, unbiased=False).mean()
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
    args, ec, neighbor, train_qc, confound_values, site_ids, device, seed,
    fold=1, total_folds=1,
):
    qc_scaler = fit_qc_scaler(train_qc)
    qc_sensitivity = transform_qc_sensitivity(train_qc, qc_scaler)
    confounds = build_confound_design(site_ids, confound_values)
    qc_basis = fit_qc_sensitivity_basis(
        ec, qc_sensitivity, confounds, ridge=args.qsr_basis_ridge
    )
    qc_sensitive_map = build_qc_sensitive_map(qc_basis)
    pseudo_target = build_pseudo_target(
        ec, qc_sensitivity, qc_basis, args.qsr_eta, args.qsr_r_max
    )

    model = QSRECRefiner(
        ec.shape[-1], args.qsr_hidden_channels, args.qsr_gate_max
    ).to(device)
    original = torch.from_numpy(ec).float().to(device)
    neighbor_tensor = torch.from_numpy(neighbor).float().to(device)
    pseudo_tensor = torch.from_numpy(pseudo_target).float().to(device)
    sensitive_tensor = torch.from_numpy(qc_sensitive_map).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.qsr_lr)
    rng = np.random.default_rng(seed)
    best_state, best_loss, metrics = None, float("inf"), {}
    log_interval = 20
    for epoch in range(1, args.qsr_epochs + 1):
        qc_delta = sample_joint_qc_delta(qc_sensitivity, rng)
        perturbed = apply_qc_perturbation(
            pseudo_target, qc_basis, qc_delta, args.qsr_perturbation_scale,
            maximum_ratio=max(2.0 * args.qsr_r_max, args.qsr_r_max),
        )
        perturbed_tensor = torch.from_numpy(perturbed).float().to(device)
        optimizer.zero_grad()
        original_refined, original_gate, _, _ = model(
            original, neighbor_tensor, sensitive_tensor, return_parts=True
        )
        perturbed_refined, perturbed_gate, _, _ = model(
            perturbed_tensor, neighbor_tensor, sensitive_tensor, return_parts=True
        )
        total, parts = qsr_refinement_loss(
            original_refined, perturbed_refined, pseudo_tensor, original,
            original_gate, perturbed_gate, args.qsr_variance_retention,
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
        if epoch == 1 or epoch % log_interval == 0 or epoch == args.qsr_epochs:
            print(
                f"[Fold {fold}/{total_folds}][QSR] Epoch {epoch}/{args.qsr_epochs} | "
                f"loss={metrics['qsr_loss']:.6f} | "
                f"pseudo={metrics['pseudo_loss']:.6f} | "
                f"restore={metrics['restore_loss']:.6f} | "
                f"gate={metrics['gate_loss']:.6f}"
            )
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
        "qsr_pseudo_relative_change": relative_change(ec, pseudo_target),
        "pr_ec_relative_change": relative_change(ec, refined_array),
        "qsr_sensitive_map_mean": float(qc_sensitive_map.mean()),
        "qsr_basis_abs_mean": float(np.abs(qc_basis).mean()),
    })
    return model, refined_array, qc_sensitive_map, metrics


def apply_qsr_refiner(model, ec, neighbor, qc_sensitive_map, device):
    with torch.no_grad():
        return model(
            torch.from_numpy(ec).float().to(device),
            torch.from_numpy(neighbor).float().to(device),
            torch.from_numpy(qc_sensitive_map).float().to(device),
        ).cpu().numpy()
