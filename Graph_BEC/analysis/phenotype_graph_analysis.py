#!/usr/bin/env python3
"""Visualize and evaluate the diagnosis-free SEX/FIQ/PIQ patient graph.

The graph is built without DX_GROUP. Labels are used only after graph
construction to color the plots and calculate evaluation metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_BEC.phenotype import load_phenotypes
from Graph_BEC.normative_bec import reference_weights

DEFAULT_BEC = ROOT / "downstream_abide_i/outputs/entropy/loss_alpha_0.01/seed_42/epochs_101/subject_bec.npz"
DEFAULT_PHENOTYPE = ROOT / "dataset/ABIDE-I/Phenotypic_Processing_filled.csv"
DEFAULT_OUTPUT = ROOT / "Graph_BEC/analysis/outputs"

ASD_COLOR = "#ef5b55"
TC_COLOR = "#58bf88"
EDGE_COLOR = "#9b8acb"
FRAME_COLOR = "#6f89c5"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bec-path", type=Path, default=DEFAULT_BEC)
    parser.add_argument("--phenotype-csv", type=Path, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--bandwidth", type=float, default=2.0)
    parser.add_argument("--categorical-penalty", type=float, default=4.0)
    parser.add_argument("--continuous-weights", type=float, nargs=2, default=[1.0, 0.3])
    parser.add_argument("--tc-label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--network-nodes", type=int, default=300)
    parser.add_argument("--network-edges", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_inputs(args):
    archive = np.load(args.bec_path, allow_pickle=False)
    required = {"subject_ids", "labels", "site_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"BEC archive is missing: {sorted(missing)}")
    subject_ids = archive["subject_ids"].astype(str)
    labels = archive["labels"].astype(int)
    site_ids = archive["site_ids"].astype(str)
    phenotype = load_phenotypes(args.phenotype_csv, subject_ids, site_ids)
    continuous = np.asarray(phenotype["continuous"], dtype=np.float64)
    continuous = np.where(np.isfinite(continuous), continuous, np.nan)
    medians = np.nanmedian(continuous, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    continuous = np.where(np.isfinite(continuous), continuous, medians)
    mean = continuous.mean(axis=0)
    std = continuous.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    continuous = (continuous - mean) / std
    sex = np.asarray(phenotype["categorical_raw"]).reshape(-1)
    sex = np.asarray([float(value) if str(value) != "nan" else np.nan for value in sex])
    if np.isnan(sex).any():
        sex[np.isnan(sex)] = 0.0
    return subject_ids, labels, continuous.astype(np.float32), sex.astype(np.float32)


def build_current_graph(continuous, sex, args):
    """Reproduce the current train-only phenotype distance as a full graph."""
    categorical = np.rint(sex).astype(np.int64)[:, None]
    indices = np.arange(len(continuous), dtype=np.int64)
    adjacency = reference_weights(
        continuous,
        categorical,
        reference_continuous=continuous,
        reference_categorical=categorical,
        self_indices=indices,
        k=args.k,
        bandwidth=args.bandwidth,
        categorical_penalty=args.categorical_penalty,
        continuous_weights=np.asarray(args.continuous_weights),
    )
    adjacency = np.asarray(adjacency, dtype=np.float64)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def build_rbf_knn_graph(values, k, bandwidth):
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    squared_distance = ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=-1)
    similarity = np.exp(-squared_distance / (2.0 * max(float(bandwidth), 1e-8) ** 2))
    np.fill_diagonal(similarity, 0.0)
    k = min(max(1, int(k)), len(values) - 1)
    adjacency = np.zeros_like(similarity)
    for row in range(len(values)):
        neighbors = np.argpartition(similarity[row], -k)[-k:]
        adjacency[row, neighbors] = similarity[row, neighbors]
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def build_multiview_graph(continuous, sex, args):
    """Unsupervised DeepASD-like multi-view fusion.

    DeepASD learns a projection and a weight for each modality. Here we use
    fixed diagnosis-free views and equal fusion weights, so this graph is an
    honest unsupervised diagnostic baseline rather than a supervised replica.
    """
    fiq_graph = build_rbf_knn_graph(continuous[:, 0], args.k, args.bandwidth)
    piq_graph = build_rbf_knn_graph(continuous[:, 1], args.k, args.bandwidth)
    sex_graph = build_rbf_knn_graph(sex, args.k, 1.0)
    adjacency = (fiq_graph + piq_graph + sex_graph) / 3.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def graph_embedding(adjacency, seed):
    embedding = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        random_state=seed,
    ).fit_transform(adjacency + np.eye(len(adjacency), dtype=np.float32))
    return embedding.astype(np.float32)


def cluster_graph(adjacency, seed):
    cluster = SpectralClustering(
        n_clusters=2,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=20,
    ).fit_predict(adjacency + np.eye(len(adjacency), dtype=np.float32))
    return cluster.astype(int)


def purity_score(labels, clusters):
    labels = np.asarray(labels)
    clusters = np.asarray(clusters)
    total = 0
    for cluster in np.unique(clusters):
        values = labels[clusters == cluster]
        if len(values):
            _, counts = np.unique(values, return_counts=True)
            total += counts.max()
    return float(total / len(labels))


def graph_metrics(adjacency, embedding, clusters, labels, tc_label):
    diagnosis = (labels == tc_label).astype(int)
    same = diagnosis[:, None] == diagnosis[None, :]
    upper = np.triu(np.ones_like(adjacency, dtype=bool), k=1)
    weights = adjacency[upper]
    same_weights = adjacency[upper & same]
    different_weights = adjacency[upper & ~same]
    total_weight = float(weights.sum())
    metrics = {
        "nodes": int(len(labels)),
        "edges_nonzero": int(np.count_nonzero(np.triu(adjacency, 1))),
        "mean_degree": float((adjacency > 0).sum(axis=1).mean()),
        "weighted_same_label_fraction": float(same_weights.sum() / max(total_weight, 1e-12)),
        "same_to_different_edge_weight_ratio": float(
            same_weights.sum() / max(different_weights.sum(), 1e-12)
        ),
        "neighbor_label_purity": float(
            (adjacency * same).sum() / max(adjacency.sum(), 1e-12)
        ),
        "ARI": float(adjusted_rand_score(diagnosis, clusters)),
        "NMI": float(normalized_mutual_info_score(diagnosis, clusters)),
        "homogeneity": float(homogeneity_score(diagnosis, clusters)),
        "completeness": float(completeness_score(diagnosis, clusters)),
        "purity": purity_score(diagnosis, clusters),
        "silhouette_embedding": float(silhouette_score(embedding, clusters)),
    }
    return metrics


def plot_embedding(path, embeddings, labels, tc_label, seed):
    diagnosis = labels == tc_label
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220)
    for axis, (title, embedding) in zip(axes, embeddings.items()):
        axis.scatter(embedding[~diagnosis, 0], embedding[~diagnosis, 1], s=11,
                     c=ASD_COLOR, alpha=0.72, linewidths=0, label="ASD")
        axis.scatter(embedding[diagnosis, 0], embedding[diagnosis, 1], s=11,
                     c=TC_COLOR, alpha=0.72, linewidths=0, label="TC")
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(FRAME_COLOR)
            spine.set_linewidth(1.5)
        axis.legend(frameon=False, loc="best", markerscale=1.3)
    fig.suptitle("ASD/TC distribution in diagnosis-free phenotype graphs", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def select_network_nodes(labels, max_nodes, seed):
    if len(labels) <= max_nodes:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        count = max(1, int(round(max_nodes * len(candidates) / len(labels))))
        selected.extend(rng.choice(candidates, size=min(count, len(candidates)), replace=False))
    selected = np.asarray(selected, dtype=int)
    if len(selected) > max_nodes:
        selected = rng.choice(selected, size=max_nodes, replace=False)
    return np.sort(selected)


def plot_network(path, adjacency, embedding, labels, tc_label, args):
    selected = select_network_nodes(labels, args.network_nodes, args.seed)
    local = adjacency[np.ix_(selected, selected)]
    upper = np.triu_indices(len(selected), k=1)
    edge_order = np.argsort(local[upper])[::-1]
    edge_order = edge_order[local[upper][edge_order] > 0]
    edge_order = edge_order[:args.network_edges]
    fig, axis = plt.subplots(figsize=(8, 7), dpi=220)
    for edge in edge_order:
        i, j = upper[0][edge], upper[1][edge]
        weight = local[i, j]
        alpha = 0.04 + 0.20 * weight / max(float(local.max()), 1e-8)
        axis.plot(
            [embedding[selected[i], 0], embedding[selected[j], 0]],
            [embedding[selected[i], 1], embedding[selected[j], 1]],
            color=EDGE_COLOR, linewidth=0.35, alpha=alpha, zorder=1,
        )
    diagnosis = labels[selected] == tc_label
    axis.scatter(embedding[selected[~diagnosis], 0], embedding[selected[~diagnosis], 1],
                 s=18, c=ASD_COLOR, alpha=0.82, linewidths=0, label="ASD", zorder=2)
    axis.scatter(embedding[selected[diagnosis], 0], embedding[selected[diagnosis], 1],
                 s=18, c=TC_COLOR, alpha=0.82, linewidths=0, label="TC", zorder=2)
    axis.set_title(
        f"Phenotype patient graph ({len(selected)} nodes, {len(edge_order)} strongest edges)",
        fontsize=13, fontweight="bold",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(frameon=False, loc="best", markerscale=1.3)
    for spine in axis.spines.values():
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metrics(path, metrics):
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    fields = sorted({key for item in metrics.values() for key in item})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["graph"] + fields)
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"graph": name, **values})


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, labels, continuous, sex = load_inputs(args)
    current_graph = build_current_graph(continuous, sex, args)
    multiview_graph = build_multiview_graph(continuous, sex, args)
    current_embedding = graph_embedding(current_graph, args.seed)
    multiview_embedding = graph_embedding(multiview_graph, args.seed)
    current_cluster = cluster_graph(current_graph, args.seed)
    multiview_cluster = cluster_graph(multiview_graph, args.seed)
    metrics = {
        "current_sex_fiq_piq": graph_metrics(
            current_graph, current_embedding, current_cluster, labels, args.tc_label
        ),
        "deepasd_like_equal_multiview": graph_metrics(
            multiview_graph, multiview_embedding, multiview_cluster, labels, args.tc_label
        ),
    }
    plot_embedding(
        args.output_dir / "phenotype_embedding.png",
        {"Current SEX + FIQ + PIQ": current_embedding,
         "DeepASD-like multi-view": multiview_embedding},
        labels, args.tc_label, args.seed,
    )
    plot_network(
        args.output_dir / "phenotype_network.png", multiview_graph,
        multiview_embedding, labels, args.tc_label, args,
    )
    save_metrics(args.output_dir / "clustering_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
