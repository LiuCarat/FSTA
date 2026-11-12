"""Supervised neighbor aggregation of frozen, fold-standardized Initial EC."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .brainnetcnn import DirectedBrainNetCNN
from PR_EC.model.mpr.population_reference import to_directed_channels
from PR_EC.utils.runtime import set_seed


class NeighborAttention(nn.Module):
    def __init__(self, nodes_num, hidden_dim=32):
        super().__init__()
        self.query = nn.Linear(4 * nodes_num, hidden_dim, bias=False)
        self.key = nn.Linear(4 * nodes_num, hidden_dim, bias=False)
        self.residual_logit = nn.Parameter(torch.zeros(()))

    @staticmethod
    def descriptors(ec):
        return torch.cat((
            ec.mean(dim=-1), ec.mean(dim=-2),
            ec.square().mean(dim=-1).sqrt(),
            ec.square().mean(dim=-2).sqrt(),
        ), dim=-1)

    def forward(self, own_ec, neighbor_ec, mask):
        query = torch.tanh(self.query(self.descriptors(own_ec)))
        keys = torch.tanh(self.key(self.descriptors(neighbor_ec)))
        scores = (query[:, None, :] * keys).sum(-1) / math.sqrt(query.shape[-1])
        weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        aggregate = torch.einsum("bk,bkij->bij", weights, neighbor_ec)
        fused = own_ec + self.residual_logit.sigmoid() * (aggregate - own_ec)
        return fused, weights


class NeighborClassifier(nn.Module):
    def __init__(self, nodes_num, hidden_dim=32):
        super().__init__()
        self.classifier = DirectedBrainNetCNN(nodes_num=nodes_num, dropout=0.3)
        self.aggregator = NeighborAttention(nodes_num, hidden_dim)

    def forward(self, own_ec, neighbor_ec, mask):
        fused, _ = self.aggregator(own_ec, neighbor_ec, mask)
        return self.classifier(to_directed_channels(fused))


def predict(model, ec, indices, mask, bank, device, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(ec), batch_size):
            stop = start + batch_size
            logits = model(
                torch.as_tensor(ec[start:stop], device=device),
                bank[torch.as_tensor(indices[start:stop], device=device)],
                torch.as_tensor(mask[start:stop], device=device),
            )
            outputs.append(logits.cpu())
    return torch.cat(outputs)


def fit_neighbor_classifier(train_ec, train_labels, val_ec, val_labels,
                            train_indices, train_mask, val_indices, val_mask,
                            device, seed, max_epochs, patience, batch_size,
                            learning_rate, hidden_dim=32):
    """No test inputs or labels are accepted by the fitting interface."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("Epochs and patience must be positive")
    set_seed(seed)
    model = NeighborClassifier(train_ec.shape[-1], hidden_dim).to(device)
    bank = torch.as_tensor(train_ec, device=device)
    neighbor_index = torch.as_tensor(train_indices, device=device)
    neighbor_mask = torch.as_tensor(train_mask, device=device)
    labels = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_ec))), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    best_state, best_loss, waiting, best_epoch = None, float("inf"), 0, 0
    history, audit = [], {}
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for (batch_indices,) in loader:
            batch_indices = batch_indices.to(device)
            inputs = bank[batch_indices]
            neighbors = bank[neighbor_index[batch_indices]]
            mask = neighbor_mask[batch_indices]
            optimizer.zero_grad()
            loss = criterion(model(inputs, neighbors, mask), labels[batch_indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training BCE")
            loss.backward()
            first_step = not audit
            if first_step:
                before = {name: parameter.detach().clone()
                          for name, parameter in model.named_parameters()}
                with torch.no_grad():
                    _, attention_before = model.aggregator(inputs, neighbors, mask)
                gradients = {}
                for prefix in ("aggregator.query", "aggregator.key",
                               "aggregator.residual_logit", "classifier"):
                    gradients[prefix] = sum(
                        float(parameter.grad.detach().square().sum())
                        for name, parameter in model.named_parameters()
                        if name.startswith(prefix) and parameter.grad is not None
                    ) ** 0.5
                if not all(np.isfinite(value) and value > 0 for value in gradients.values()):
                    raise RuntimeError(f"Joint backpropagation failed: {gradients}")
            optimizer.step()
            if first_step:
                updates = {
                    name: float((parameter.detach() - before[name]).abs().max())
                    for name, parameter in model.named_parameters()
                }
                with torch.no_grad():
                    _, attention_after = model.aggregator(inputs, neighbors, mask)
                change = float((attention_after - attention_before).abs().max())
                if change <= 0 or not all(updates[name] > 0 for name in (
                    "aggregator.query.weight", "aggregator.key.weight",
                    "aggregator.residual_logit", "classifier.fc2.weight",
                )):
                    raise RuntimeError("Optimizer did not update attention and classifier")
                audit = {"gradient_norms": gradients, "first_step_parameter_updates": updates,
                         "attention_weight_max_change": change}
            loss_sum += loss.item() * len(batch_indices)
        validation_logits = predict(
            model, val_ec, val_indices, val_mask, bank, device, batch_size,
        )
        validation_loss = criterion(
            validation_logits, torch.as_tensor(val_labels, dtype=torch.float32),
        ).item()
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite validation BCE")
        history.append({"epoch": epoch, "train_bce": loss_sum / len(train_ec),
                        "validation_bce": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, waiting = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waiting += 1
        if waiting >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    audit.update({"best_epoch": best_epoch, "epochs_run": epoch,
                  "best_validation_bce": best_loss, "history": history,
                  "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "attention_parameter_count": sum(
                      parameter.numel() for parameter in model.aggregator.parameters())})
    return model, bank, audit
