from __future__ import annotations
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGRBECStatic(nn.Module):
    """Predict a subject-specific edge gate from [A, N, |N-A|]."""
    def __init__(self, nodes_num=90, hidden_channels=16, gate_max=0.2):
        super().__init__()
        self.nodes_num = nodes_num
        self.gate_max = float(gate_max)
        self.gate_network = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, initial_bec, neighbor_bec, return_parts=False):
        difference = neighbor_bec - initial_bec
        gate_input = torch.stack(
            (initial_bec, neighbor_bec, difference.abs()), dim=1
        )
        gate = self.gate_max * torch.sigmoid(self.gate_network(gate_input)).squeeze(1)
        diagonal = torch.eye(self.nodes_num, device=initial_bec.device, dtype=torch.bool)[None]
        gate = gate.masked_fill(diagonal, 0.0)
        refined = (initial_bec + gate * difference).masked_fill(diagonal, 0.0)
        if return_parts:
            return refined, gate, difference
        return refined


def static_refinement_loss(refined, initial, gate, variance_retention=0.85,
                           anchor_weight=1.0, gate_weight=1e-3,
                           variance_weight=1.0):
    """Unsupervised loss; no ASD/TC term is present."""
    anchor = F.smooth_l1_loss(refined, initial.detach())
    sparse = gate.abs().mean()
    original_variance = initial.flatten(1).var(dim=0, unbiased=False).mean()
    refined_variance = refined.flatten(1).var(dim=0, unbiased=False).mean()
    variance = F.relu(variance_retention * original_variance - refined_variance)
    total = anchor_weight * anchor + gate_weight * sparse + variance_weight * variance
    return total, {"anchor_loss": anchor, "gate_loss": sparse, "variance_loss": variance}


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
    """Apply a trained PGR refiner to BEC and neighbor reference arrays."""
    with torch.no_grad():
        return model(
            torch.from_numpy(bec).float().to(device),
            torch.from_numpy(neighbor).float().to(device),
        ).cpu().numpy()
