from __future__ import annotations
import numpy as np
import torch

def _as_2d(values, name):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{name} must be [N, features], got {values.shape}")
    return values

def fit_continuous_scaler(values):
    values = _as_2d(values, "continuous values").astype(np.float64)
    median = np.nanmedian(values, axis=0)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(np.isfinite(values), values, median)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return {"median": median.astype(np.float32), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}

def apply_continuous_scaler(values, scaler):
    values = _as_2d(values, "continuous values").astype(np.float32)
    filled = np.where(np.isfinite(values), values, scaler["median"])
    return ((filled - scaler["mean"]) / scaler["std"]).astype(np.float32)

def _pairwise_distance(query_cont, ref_cont, query_cat, ref_cat, weights, categorical_penalty):
    distance = np.zeros((len(query_cont), len(ref_cont)), dtype=np.float64)
    if query_cont.shape[1]:
        difference = query_cont[:, None, :] - ref_cont[None, :, :]
        distance += ((difference ** 2) * np.asarray(weights)[None, None, :]).sum(axis=-1)
    if query_cat.shape[1]:
        distance += categorical_penalty * (query_cat[:, None, :] != ref_cat[None, :, :]).sum(axis=-1)
    return distance

def reference_weights(query_cont, query_cat, reference_continuous, reference_categorical, k=20, bandwidth=1.0, categorical_penalty=4.0, continuous_weights=None, self_indices=None):
    query_cont = _as_2d(query_cont, "query_cont").astype(np.float64)
    ref_cont = _as_2d(reference_continuous, "reference_continuous").astype(np.float64)
    query_cat = _as_2d(query_cat, "query_cat").astype(np.int64)
    ref_cat = _as_2d(reference_categorical, "reference_categorical").astype(np.int64)
    if len(ref_cont) == 0:
        raise ValueError("The phenotype reference set is empty")
    weights = np.ones(ref_cont.shape[1]) if continuous_weights is None else np.asarray(continuous_weights)
    distance = _pairwise_distance(query_cont, ref_cont, query_cat, ref_cat, weights, categorical_penalty)
    if self_indices is not None:
        for row, ref_index in enumerate(np.asarray(self_indices).tolist()):
            if 0 <= ref_index < distance.shape[1]:
                distance[row, ref_index] = np.inf
    count = min(max(int(k), 1), distance.shape[1])
    nearest = np.argpartition(distance, count - 1, axis=1)[:, :count]
    nearest_distance = np.take_along_axis(distance, nearest, axis=1)
    affinity = np.exp(-nearest_distance / max(float(bandwidth), 1e-6))
    affinity[~np.isfinite(affinity)] = 0.0
    affinity /= np.maximum(affinity.sum(axis=1, keepdims=True), 1e-12)
    output = np.zeros_like(distance)
    np.put_along_axis(output, nearest, affinity, axis=1)
    return output.astype(np.float32)

def normative_reference(population_reference, weights):
    population_reference = np.asarray(population_reference, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if population_reference.ndim != 3 or weights.ndim != 2 or weights.shape[1] != len(population_reference):
        raise ValueError("population_reference and weights have incompatible shapes")
    flat = weights @ population_reference.reshape(len(population_reference), -1)
    subject_reference = flat.reshape(len(weights), *population_reference.shape[1:])
    global_mean = np.average(population_reference, axis=0, weights=np.maximum(weights.sum(axis=0), 1e-8))
    return subject_reference.astype(np.float32), global_mean.astype(np.float32)

def to_directed_channels(ec):
    ec = torch.as_tensor(ec) if not torch.is_tensor(ec) else ec
    if ec.ndim != 3:
        raise ValueError(f"Expected EC [N, nodes, nodes], got {tuple(ec.shape)}")
    return torch.stack((ec, ec.transpose(-1, -2)), dim=1)
