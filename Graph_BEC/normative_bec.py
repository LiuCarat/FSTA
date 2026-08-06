"""Unsupervised matrix-level normative refinement for FSTA BEC."""
from __future__ import annotations
import csv
import numpy as np
import torch


# 将输入的数据转换为 NumPy 的二维数组格式。如果输入的数据已经是二维数组，则直接返回；如果输入的数据是一维数组，则在第二个维度上添加一个维度；如果输入的数据不是二维数组，则抛出一个 ValueError 异常。
def _as_2d(values, name):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{name} must be [N, features], got {values.shape}")
    return values


# 用于对连续值数据进行标准化处理，通过计算每个特征的中位数、均值和标准差来生成缩放参数
def fit_continuous_scaler(values):
    values = _as_2d(values, "continuous values").astype(np.float64) # 确保输入数据是二维数组格式，并将数据类型转换为 float64
    median = np.nanmedian(values, axis=0) # 计算每个特征的中位数
    median[~np.isfinite(median)] = 0.0 # 将非有限值的中位数设置为 0
    filled = np.where(np.isfinite(values), values, median) # 将非有限值替换为对应特征的中位数
    mean, std = filled.mean(axis=0), filled.std(axis=0) # 计算每个特征的均值和标准差
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0 # 将非有限值或小于 1e-6 的标准差设置为 1
    return {"median": median.astype(np.float32), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


# 用于对连续值数据进行标准化处理，将非有限值替换为中位数，并将数据转换为均值为0、标准差为1的标准化形式
def apply_continuous_scaler(values, scaler):
    values = _as_2d(values, "continuous values").astype(np.float32)
    filled = np.where(np.isfinite(values), values, scaler["median"]) # 将非有限值替换为对应特征的中位数
    return ((filled - scaler["mean"]) / scaler["std"]).astype(np.float32) # 将数据标准化为均值为 0，标准差为 1


# 从CSV文件中加载指定的分类变量表型数据，并将其进行整数编码
def load_categorical_phenotypes(phenotype_csv, subject_ids, columns=("SEX", "SITE_ID")):
    subject_ids = np.asarray(subject_ids).astype(str)
    with open(phenotype_csv, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        for key in ("SUB_ID", "subject_id", "SUBJECT_ID", "subject"):
            if row.get(key):
                by_id[str(row[key]).strip()] = row
                break
    encoded, codebooks = [], {}
    for column in columns:
        raw = [str(by_id.get(subject, {}).get(column, "__MISSING__")).strip() for subject in subject_ids]
        categories = sorted(set(raw))
        codebooks[column] = {value: index for index, value in enumerate(categories)}
        encoded.append([codebooks[column][value] for value in raw])
    return np.asarray(encoded, dtype=np.int64).T, codebooks


def _pairwise_distance(query_cont, ref_cont, query_cat, ref_cat, weights, categorical_penalty):
    distance = np.zeros((len(query_cont), len(ref_cont)), dtype=np.float64)
    if query_cont.shape[1]:
        difference = query_cont[:, None, :] - ref_cont[None, :, :]
        distance += ((difference ** 2) * np.asarray(weights)[None, None, :]).sum(axis=-1)
    if query_cat.shape[1]:
        distance += categorical_penalty * (query_cat[:, None, :] != ref_cat[None, :, :]).sum(axis=-1)
    return distance


def reference_weights(query_cont, query_cat, reference_continuous, reference_categorical, k=20,
                      bandwidth=1.0, categorical_penalty=4.0, continuous_weights=None,
                      self_indices=None):
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


def normative_reference(reference_bec, weights):
    reference_bec = np.asarray(reference_bec, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if reference_bec.ndim != 3 or weights.ndim != 2 or weights.shape[1] != len(reference_bec):
        raise ValueError("reference_bec and weights have incompatible shapes")
    flat = weights @ reference_bec.reshape(len(reference_bec), -1)
    subject_reference = flat.reshape(len(weights), *reference_bec.shape[1:])
    global_mean = np.average(reference_bec, axis=0, weights=np.maximum(weights.sum(axis=0), 1e-8))
    return subject_reference.astype(np.float32), global_mean.astype(np.float32)


def reference_diagnostics(weights, reference_labels=None):
    weights = np.asarray(weights, dtype=np.float32)
    return {"reference_mean_neighbors": float((weights > 0).sum(axis=1).mean()), "reference_effective_sample_size": float((1.0 / np.maximum((weights ** 2).sum(axis=1), 1e-8)).mean())}


def to_directed_channels(bec):
    bec = torch.as_tensor(bec) if not torch.is_tensor(bec) else bec
    if bec.ndim != 3:
        raise ValueError(f"Expected BEC [N, nodes, nodes], got {tuple(bec.shape)}")
    return torch.stack((bec, bec.transpose(-1, -2)), dim=1)


def bec_separability(bec, labels, tc_label=1):
    bec = np.asarray(bec, dtype=np.float64).reshape(len(bec), -1)
    labels = np.asarray(labels)
    first, second = bec[labels != tc_label], bec[labels == tc_label]
    if len(first) == 0 or len(second) == 0:
        return {"bec_centroid_distance": np.nan, "bec_within_dispersion": np.nan, "bec_fisher_ratio": np.nan}
    first_mean, second_mean = first.mean(0), second.mean(0)
    distance = np.linalg.norm(first_mean - second_mean)
    within = 0.5 * (np.linalg.norm(first - first_mean, axis=1).mean() + np.linalg.norm(second - second_mean, axis=1).mean())
    return {"bec_centroid_distance": float(distance), "bec_within_dispersion": float(within), "bec_fisher_ratio": float(distance ** 2 / max(within ** 2, 1e-12))}


def edge_effect_sizes(bec, labels, tc_label=1):
    bec = np.asarray(bec, dtype=np.float64)
    labels = np.asarray(labels)
    first, second = bec[labels != tc_label], bec[labels == tc_label]
    mean_difference = first.mean(0) - second.mean(0)
    pooled = np.sqrt((first.var(0) + second.var(0)) / 2.0)
    effect = mean_difference / np.maximum(pooled, 1e-8)
    return {"mean_abs_d": float(np.mean(np.abs(effect))), "max_abs_d": float(np.max(np.abs(effect))), "edges_abs_d_gt_0p5": int(np.sum(np.abs(effect) > 0.5))}, effect.astype(np.float32)
