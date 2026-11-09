"""Preservation of known Sim21 group effects and within-group heterogeneity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat
from scipy.special import gammaln
from sklearn.model_selection import KFold
import torch

from ..model.individual_ec.training import train_stf_ec
from ..model.individual_ec.utils import extract_subject_ec
from ..model.mpr.population_similarity import subject_fc_features, topk_graph
from ..utils.runtime import select_device
from .initial_ec import PAPER_ORIENTATION, write_csv


METHODS = ("Initial EC", "FC-neighbor Reference", "Random-neighbor Reference", "Global Reference")
METRICS = ("hedges_g", "paired_sign_accuracy", "within_group_r", "spurious_group_difference")
ENCODER_FIELDS = (
    "seed", "window_length", "stride", "epochs", "stf_checkpoint", "loss_alpha", "batch_size",
    "d_model", "d_inner_hid", "d_k", "d_v", "n_head", "dropout", "n_warmup_steps", "lr_mul",
    "weight_decay", "adam_beta1", "adam_beta2", "num_hidden_layers", "num_attention_heads",
    "hidden_act", "attention_probs_dropout_prob", "hidden_dropout_prob", "initializer_range", "no_filters",
)


def dump_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def load_sim21(path, window_length):
    data = loadmat(path)
    dimensions = tuple(int(np.asarray(data[key]).item()) for key in ("Nsubjects", "Ntimepoints", "Nnodes"))
    if dimensions != (50, 200, 5) or data["ts"].shape != (10000, 5) or data["net"].shape != (50, 5, 5):
        raise ValueError("Expected Sim21 with 50 subjects, 200 timepoints and 5 nodes")
    series = data["ts"].reshape(dimensions).astype(np.float32)
    truth = data["net"].astype(np.float64)
    if not np.isfinite(series).all() or not np.isfinite(truth).all() or not 1 <= window_length <= 200:
        raise ValueError("Invalid data or window length")
    unchanged = np.ones((5, 5), dtype=bool)
    unchanged[1, 2] = False
    if not np.allclose(truth[:25, unchanged], truth[25:, unchanged], atol=1e-10, rtol=0):
        raise ValueError("Matched pairs differ outside the prespecified target edge")
    if not np.all(truth[:25, 1, 2] > truth[25:, 1, 2]):
        raise ValueError("Sim21 does not have the expected paired effect on net[:,1,2]")
    series = (series - series.mean(axis=1, keepdims=True)) / np.maximum(series.std(axis=1, keepdims=True), 1e-6)
    return series, truth


def paired_splits():
    for train_pairs, test_pairs in KFold(5, shuffle=True, random_state=42).split(np.arange(25)):
        yield np.r_[train_pairs, train_pairs + 25], np.r_[test_pairs, test_pairs + 25]


def correlation(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    first, second = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= np.finfo(float).eps:
        return None
    return float(np.clip(first @ second / denominator, -1, 1))


def effect_metrics(predictions, truth):
    if predictions.shape != (50, 5, 5) or not np.isfinite(predictions).all():
        raise ValueError("Expected finite (50,5,5) OOF predictions")
    edge = predictions[:, 1, 2].astype(np.float64)
    first, second = edge[:25], edge[25:]
    difference = float(first.mean() - second.mean())
    pooled_std = np.sqrt((first.var(ddof=1) + second.var(ddof=1)) / 2)
    correction = np.exp(gammaln(24) - .5 * np.log(24) - gammaln(23.5))
    hedges = float(correction * difference / pooled_std) if pooled_std > np.finfo(float).eps else None
    first_r = correlation(first, truth[:25, 1, 2])
    second_r = correlation(second, truth[25:, 1, 2])
    combined_r = None
    if first_r is not None and second_r is not None:
        combined_r = float(np.tanh(np.mean(np.arctanh(np.clip([first_r, second_r], -1 + 1e-12, 1 - 1e-12)))))
    mask = ~np.eye(5, dtype=bool)
    mask[1, 2] = False
    spurious = predictions[:25].astype(np.float64).mean(axis=0) - predictions[25:].astype(np.float64).mean(axis=0)
    return {
        "hedges_g": hedges, "paired_sign_accuracy": float(np.mean(first > second)),
        "within_group_r": combined_r, "within_group_r_a": first_r, "within_group_r_b": second_r,
        "spurious_group_difference": float(np.mean(np.abs(spurious[mask]))),
        "spurious_group_difference_sum": float(np.sum(np.abs(spurious[mask]))),
        "target_group_difference": difference,
    }


def cached_ec(args, series, train, test, fold, device):
    folder = args.cache_dir / f"fold_{fold:02d}"
    identity = {
        "version": 1, "normalized_bold_sha256": hashlib.sha256(np.ascontiguousarray(series).tobytes()).hexdigest(),
        "encoder": {key: getattr(args, key) for key in ENCODER_FIELDS},
        "train_indices": train.tolist(), "test_indices": test.tolist(),
        "orientation": PAPER_ORIENTATION["prediction_orientation"], "nodes_num": 5,
    }
    if args.reuse_cache:
        if not (folder / "ec.npz").exists() or not (folder / "metadata.json").exists():
            raise ValueError(f"Missing fold cache {folder}; will not retrain with --reuse-cache")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata["identity"] != identity:
            raise ValueError(f"Fold cache configuration/data/split/orientation mismatch: {folder}")
        with np.load(folder / "ec.npz", allow_pickle=False) as archive:
            train_ec, test_ec = archive["train_ec"].copy(), archive["test_ec"].copy()
        metrics = metadata["training_metrics"]
    else:
        if folder.exists():
            raise ValueError(f"Cache exists: {folder}; use --reuse-cache or a new cache directory")
        options = argparse.Namespace(**vars(args))
        options.nodes_num = 5
        model, metrics = train_stf_ec(options, series[train], device)
        extracted = []
        for indices in (train, test):
            records = [SimpleNamespace(subject_id=int(index) + 1) for index in indices]
            extracted.append(extract_subject_ec(model, records, series[indices], args.window_length, args.stride, device)["ec"])
        train_ec, test_ec = extracted
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), folder / "encoder.pt")
        np.savez_compressed(folder / "ec.npz", train_ec=train_ec, test_ec=test_ec)
        dump_json(folder / "metadata.json", {"identity": identity, "training_metrics": metrics})
    for matrices, count in ((train_ec, 40), (test_ec, 10)):
        if matrices.shape != (count, 5, 5) or not np.isfinite(matrices).all():
            raise ValueError("Invalid cached EC matrices")
    return train_ec, test_ec, metrics


def run_experiment(args):
    series, truth = load_sim21(args.mat_path, args.window_length)
    device = select_device(args.gpu_id)
    fc = subject_fc_features(series)
    predictions = {method: np.full((50, 5, 5), np.nan, dtype=np.float32) for method in METHODS if method != "Random-neighbor Reference"}
    random_predictions = np.full((20, 50, 5, 5), np.nan, dtype=np.float32)
    fold_ids = np.zeros(50, dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for fold, (train, test) in enumerate(paired_splits(), 1):
        print(f"Sim21 fold={fold}/5: 40 train / 10 test; matched pairs kept together", flush=True)
        train_ec, test_ec, metrics = cached_ec(args, series, train, test, fold, device)
        mean, std = fc[train].mean(axis=0), fc[train].std(axis=0)
        std[std < 1e-6] = 1
        weights = topk_graph((fc[train] - mean) / std, (fc[test] - mean) / std, args.reference_k)
        reference = (weights @ train_ec.reshape(40, -1)).reshape(10, 5, 5)
        predictions["Initial EC"][test] = test_ec
        predictions["FC-neighbor Reference"][test] = reference
        predictions["Global Reference"][test] = train_ec.mean(axis=0)
        random_weights = np.zeros((20, 10, 40), dtype=np.float32)
        for seed in range(20):
            generator = np.random.default_rng(np.random.SeedSequence([seed, fold]))
            for row, scores in enumerate(weights):
                values = scores[scores > 0]
                indices = generator.choice(40, size=len(values), replace=False)
                random_weights[seed, row, indices] = generator.permutation(values)
            random_predictions[seed, test] = (random_weights[seed] @ train_ec.reshape(40, -1)).reshape(10, 5, 5)
        same_group = (test[:, None] < 25) == (train[None, :] < 25)
        selected_fraction = float(np.mean(np.sum((weights > 0) & same_group, axis=1) / np.sum(weights > 0, axis=1)))
        diagnostics.append({"fold": fold, "train_indices": train.tolist(), "test_indices": test.tolist(),
                            "encoder_metrics": metrics, "fc_same_group_neighbor_fraction": selected_fraction})
        np.savez_compressed(args.output_dir / f"fold_{fold:02d}_references.npz", train_indices=train, test_indices=test,
                            fc_weights=weights, random_weights=random_weights, fc_mean=mean, fc_std=std)
        fold_ids[test] = fold
    if np.any(fold_ids == 0) or not np.isfinite(random_predictions).all():
        raise RuntimeError("Incomplete OOF coverage")
    random_metrics = [effect_metrics(matrices, truth) for matrices in random_predictions]
    method_metrics = {method: effect_metrics(matrices, truth) for method, matrices in predictions.items()}
    aggregate_random = {}
    for metric in random_metrics[0]:
        values = [row[metric] for row in random_metrics if row[metric] is not None]
        aggregate_random[metric] = {"mean": float(np.mean(values)) if values else None,
                                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "valid_seeds": len(values)}
    rows, table = [], []
    for method in METHODS:
        if method == "Random-neighbor Reference":
            result = {key: value["mean"] for key, value in aggregate_random.items()}
        else:
            result = method_metrics[method]
        rows.append({"method": method, **result, **{
            f"{metric}_seed_std": aggregate_random[metric]["std"] if method == "Random-neighbor Reference" else None
            for metric in METRICS
        }})
        cells = []
        for metric in METRICS:
            value = result[metric]
            cell = "N/A" if value is None else f"{value:.2f}"
            if method == "Random-neighbor Reference" and value is not None and aggregate_random[metric]["std"] is not None:
                cell += f" ± {aggregate_random[metric]['std']:.2f}"
            cells.append(cell)
        table.append([method, *cells])
    headers = ["Method", "Group Hedges' g ↑", "Paired Sign Acc. ↑", "Within-group r ↑", "Spurious Group Difference ↓"]
    widths = [max(len(row[index]) for row in [headers, *table]) for index in range(len(headers))]
    print(f"\nTable R2. Preservation of known individual/group differences on NetSim Sim21 (Top-{args.reference_k})", flush=True)
    for row in [headers, *table]:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)), flush=True)
    print("Random: mean ± sample SD across 20 fixed seeds; other methods: one 50-subject OOF estimate.", flush=True)
    write_csv(args.output_dir / "table_R2.csv", rows)
    write_csv(args.output_dir / "random_seed_metrics.csv", [{"seed": seed, **metrics} for seed, metrics in enumerate(random_metrics)])
    subject_rows = []
    all_runs = [(method, None, matrices) for method, matrices in predictions.items()]
    all_runs.extend(("Random-neighbor Reference", seed, matrices) for seed, matrices in enumerate(random_predictions))
    for method, seed, matrices in all_runs:
        for index in range(50):
            subject_rows.append({"method": method, "random_seed": seed, "subject_index": index, "subject_id": index + 1,
                                 "pair_id": index % 25, "group": "A" if index < 25 else "B", "fold": int(fold_ids[index]),
                                 "predicted_target_weight": float(matrices[index, 1, 2]), "true_target_weight": float(truth[index, 1, 2])})
    write_csv(args.output_dir / "subject_target_weights.csv", subject_rows)
    spurious_rows = []
    for method, seed, matrices in all_runs:
        difference = matrices[:25].astype(np.float64).mean(0) - matrices[25:].astype(np.float64).mean(0)
        for source in range(5):
            for target in range(5):
                if source != target and (source, target) != (1, 2):
                    spurious_rows.append({"method": method, "random_seed": seed, "source_index": source, "target_index": target,
                                          "group_difference": float(difference[source, target]), "absolute_difference": float(abs(difference[source, target]))})
    write_csv(args.output_dir / "spurious_edge_effects.csv", spurious_rows)
    np.savez_compressed(args.output_dir / "oof_predictions.npz", initial=predictions["Initial EC"],
                        fc_reference=predictions["FC-neighbor Reference"], global_reference=predictions["Global Reference"],
                        random_reference=random_predictions, ground_truth=truth, fold_ids=fold_ids,
                        subject_indices=np.arange(50), pair_ids=np.arange(50) % 25, random_seeds=np.arange(20))
    summary = {
        "dataset": "NetSim Sim21", "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "methods": method_metrics, "random_metrics_across_seeds": aggregate_random, "ground_truth_metrics": effect_metrics(truth, truth),
        "true_target_means": [float(truth[:25, 1, 2].mean()), float(truth[25:, 1, 2].mean())],
        "target_edge": [1, 2], "folds": diagnostics, **PAPER_ORIENTATION,
        "protocol": {
            "scope": "Group references only, not trained MPR/QSR refinement. No GT, group labels or pair labels in encoder loss/FC neighbor selection.",
            "split": "5-fold KFold of 25 matched pairs, seed 42; each pair kept together; 40 train / 10 test",
            "fc": f"Pearson FC upper triangle; train-fold z-score; clipped cosine top-{args.reference_k}, row normalized; nonpositive similarities receive zero weight; all-zero rows use uniform top-{args.reference_k} fallback",
            "random": "Seeds 0..19, SeedSequence([seed, fold]); training identities sampled without replacement; same nonzero neighbor count and weight multiset as FC; no seed selection; metrics computed per seed BEFORE aggregation",
            "hedges_g": "Signed group A-B pooled-SD standardized difference, exact gamma correction J(df=48); undefined zero variance -> null; not a paired standardized effect",
            "paired_sign_accuracy": "Fraction of 25 pairs with predicted A edge > predicted B edge; ties incorrect; effect sign, NOT neural direction accuracy",
            "within_group_r": "Pearson r separately in each group, then equal-weight Fisher-z average; constant predictions -> null, not zero",
            "spurious_group_difference": "Mean absolute A-B group mean difference over the OTHER 19 off-diagonal directed edges; sum also exported; raw prediction scale, not causal parameter error",
            "caveats": "OOF scores come from five separately fitted encoders; scales may differ across folds. Pair differences are within-fold. Global reference is identical within each test fold, hence paired sign accuracy is 0 with strict >. Nonzero pooled within-group r for Global may reflect fold composition, not personalization. No GT-based transpose or hyperparameter selection.",
        },
    }
    dump_json(args.output_dir / "summary.json", summary)
    lines = ["# Table R2. Preservation of known subject- and group-level connectivity differences on NetSim Sim21", "",
             "Random reference: mean ± sample SD across 20 fixed seeds (not across subjects). Other rows: one OOF estimate.",
             f"K={args.reference_k}; matched-pair five-fold OOF; no QSR or MPR gate training. Target edge net[:,1,2].", "",
             "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in table)
    lines.extend(["", "Within-group r is the Fisher-z average of group A and B correlations; separate r values are in table_R2.csv.",
                  "Spurious difference is the mean absolute group effect on the other 19 off-diagonal edges (prediction scale).",
                  "Strict paired > treats ties as incorrect; Global is constant within each test fold, so its paired sign accuracy is 0.",
                  "Five encoder scales and fold composition can affect pooled correlations. No significance claims or hyperparameter search."])
    (args.output_dir / "table_R2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.output_dir / 'table_R2.md'}", flush=True)
    return summary
