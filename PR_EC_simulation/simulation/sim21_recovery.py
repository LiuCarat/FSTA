"""Evaluate actual Initial/reference/refined EC against each Sim21 subject's GT."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch

from ..model.ablation.mpr_refinement import PGRECStatic, train_pgr_refiner
from ..model.mpr.population_similarity import subject_fc_features, topk_graph
from ..utils.runtime import set_seed
from .initial_ec import PAPER_ORIENTATION, THRESHOLD_PROTOCOL, score_subjects, write_csv
from .sim21 import dump_json, effect_metrics, load_sim21, paired_splits


TOPOLOGY_METRICS = ("auprc", "f1", "shd", "precision", "recall", "accuracy")
MPR_SETTINGS = {
    "refiner_epochs": 80, "refiner_lr": 0.03, "gate_max": 0.4,
    "gate_l1_weight": 0.38, "anchor_weight": 0.8,
    "variance_weight": 1.0, "variance_retention": 0.85,
}


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def evaluate_method(matrices, truth):
    rows = []
    for matrix, graph in zip(matrices, truth):
        binary = (graph != 0).astype(np.uint8)
        np.fill_diagonal(binary, 0)
        rows.append(score_subjects(matrix[None], binary)[0])
    effects = effect_metrics(matrices, truth)
    return rows, {
        metric: {"mean": float(np.mean([row[metric] for row in rows])),
                 "std": float(np.std([row[metric] for row in rows], ddof=1))}
        for metric in TOPOLOGY_METRICS
    }, effects


def load_inputs(args):
    summary_path = args.source_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Missing saved Sim21 experiment: {summary_path}. No training will be started.")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("dataset") != "NetSim Sim21" or source_summary.get("prediction_orientation") != PAPER_ORIENTATION["prediction_orientation"]:
        raise ValueError("Expected Sim21 data with the paper-definition EC orientation; refusing to guess a transpose")
    series, truth = load_sim21(args.mat_path, source_summary["configuration"]["window_length"])
    fingerprint = array_digest(series)
    with np.load(args.source_dir / "oof_predictions.npz", allow_pickle=False) as archive:
        initial, reference = archive["initial"].copy(), archive["fc_reference"].copy()
        fold_ids = archive["fold_ids"].copy()
        np.testing.assert_array_equal(truth, archive["ground_truth"])
        np.testing.assert_array_equal(np.arange(50), archive["subject_indices"])
    for matrices in (initial, reference):
        if matrices.shape != (50, 5, 5) or not np.isfinite(matrices).all():
            raise ValueError("Invalid saved OOF matrix shape or values")
    folds = []
    expected_ids = np.zeros(50, dtype=np.int64)
    for number, (train, test) in enumerate(paired_splits(), 1):
        cached = args.cache_dir / f"fold_{number:02d}"
        identity = json.loads((cached / "metadata.json").read_text(encoding="utf-8"))["identity"]
        if identity["orientation"] != source_summary["prediction_orientation"] or identity["normalized_bold_sha256"] != fingerprint:
            raise ValueError("Encoder cache data or orientation mismatch")
        np.testing.assert_array_equal(train, identity["train_indices"])
        np.testing.assert_array_equal(test, identity["test_indices"])
        with np.load(cached / "ec.npz", allow_pickle=False) as archive:
            train_ec = archive["train_ec"].copy()
            np.testing.assert_array_equal(initial[test], archive["test_ec"])
        if train_ec.shape != (40, 5, 5) or not np.isfinite(train_ec).all():
            raise ValueError("Invalid training EC cache")
        with np.load(args.source_dir / f"fold_{number:02d}_references.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(train, archive["train_indices"])
            np.testing.assert_array_equal(test, archive["test_indices"])
            weights = archive["fc_weights"].copy()
            fc_mean, fc_std = archive["fc_mean"].copy(), archive["fc_std"].copy()
        if weights.shape != (10, 40) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Invalid fold-local FC reference weights")
        np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)
        reconstructed = (weights @ train_ec.reshape(40, -1)).reshape(10, 5, 5)
        np.testing.assert_allclose(reference[test], reconstructed, atol=1e-7, rtol=1e-6)
        expected_ids[test] = number
        folds.append({"fold": number, "train": train, "test": test, "encoder_identity": identity, "weights": weights,
                      "train_ec": train_ec, "train_series": series[train], "fc_mean": fc_mean, "fc_std": fc_std})
    np.testing.assert_array_equal(fold_ids, expected_ids)
    return initial, reference, truth, fold_ids, folds, source_summary


def train_fold_mpr(args, folds, reference_k):
    if args.mpr_dir is None:
        raise ValueError("An MPR checkpoint directory is required for training")
    for fold in folds:
        folder = args.mpr_dir / f"fold_{fold['fold']:02d}"
        if folder.exists():
            raise ValueError(f"MPR cache already exists: {folder}. Omit --train-mpr to evaluate it, or choose a new --mpr-dir; will not overwrite.")
    options = SimpleNamespace(**MPR_SETTINGS)
    for fold in folds:
        train_ec = fold["train_ec"]
        train_fc = subject_fc_features(fold["train_series"])
        mean, std = train_fc.mean(0), train_fc.std(0)
        std[std < 1e-6] = 1
        np.testing.assert_allclose(mean, fold["fc_mean"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(std, fold["fc_std"], rtol=1e-6, atol=1e-7)
        normalized = (train_fc - mean) / std
        weights = topk_graph(normalized, normalized, reference_k, exclude_self=True)
        if np.any(np.diag(weights) != 0):
            raise ValueError("Training reference includes self-neighbors")
        neighbor = (weights @ train_ec.reshape(len(train_ec), -1)).reshape(train_ec.shape)
        seed = 42 + fold["fold"] * 1000
        set_seed(seed)
        print(f"Training MPR only fold={fold['fold']}/5; 40 train EC; self excluded; Encoder is not loaded or trained", flush=True)
        model, _, metrics = train_pgr_refiner(options, train_ec, neighbor, torch.device("cpu"))
        folder = args.mpr_dir / f"fold_{fold['fold']:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        torch.save(model.state_dict(), folder / "mpr.pt")
        np.savez_compressed(folder / "training_reference.npz", train_indices=fold["train"],
                            train_weights=weights, train_neighbor=neighbor)
        dump_json(folder / "metadata.json", {
            "dataset": "NetSim Sim21", "prediction_orientation": PAPER_ORIENTATION["prediction_orientation"],
            "train_indices": fold["train"].tolist(), "test_indices": fold["test"].tolist(),
            "encoder_identity": fold["encoder_identity"], "fc_weights_sha256": array_digest(fold["weights"]),
            "gate_max": options.gate_max, "hidden_channels": 16, "reference_k": reference_k,
            "training_settings": MPR_SETTINGS, "training_seed": seed, "training_metrics": metrics,
            "training_protocol": "Train on 40 cached Initial EC and leave-self-out FC references only. No GT, test EC, test reference or group labels in loss/checkpoint selection. Encoder not loaded, updated or trained.",
        })


def apply_saved_mpr(args, initial, reference, folds):
    if args.mpr_dir is None:
        return None, [], "Pending: no Sim21 MPR checkpoint directory supplied; reference R is NOT a final refined EC."
    for fold in folds:
        folder = args.mpr_dir / f"fold_{fold['fold']:02d}"
        if not (folder / "mpr.pt").exists() or not (folder / "metadata.json").exists():
            raise ValueError(f"Missing trained MPR checkpoint/provenance at {folder}; no training or partial OOF substitution allowed")
    output = np.full_like(initial, np.nan)
    diagnostics = []
    for fold in folds:
        folder = args.mpr_dir / f"fold_{fold['fold']:02d}"
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset") != "NetSim Sim21" or metadata.get("prediction_orientation") != PAPER_ORIENTATION["prediction_orientation"]:
            raise ValueError("MPR checkpoint dataset/orientation mismatch; Sanchez checkpoints cannot be substituted")
        np.testing.assert_array_equal(metadata["train_indices"], fold["train"])
        np.testing.assert_array_equal(metadata["test_indices"], fold["test"])
        if metadata["encoder_identity"] != fold["encoder_identity"] or metadata["fc_weights_sha256"] != array_digest(fold["weights"]):
            raise ValueError("MPR checkpoint encoder/reference provenance mismatch")
        gate_max = float(metadata["gate_max"])
        hidden_channels = int(metadata["hidden_channels"])
        if not np.isfinite(gate_max) or not 0 <= gate_max <= 1 or hidden_channels < 1:
            raise ValueError("Invalid MPR architecture metadata")
        model = PGRECStatic(5, hidden_channels=hidden_channels, gate_max=gate_max)
        model.load_state_dict(torch.load(folder / "mpr.pt", map_location="cpu", weights_only=True))
        model.eval()
        test = fold["test"]
        with torch.no_grad():
            refined, gate, _ = model(torch.from_numpy(initial[test]).float(), torch.from_numpy(reference[test]).float(), return_parts=True)
        output[test] = refined.numpy()
        difference = np.abs(output[test] - initial[test])
        diagnostic = {"fold": fold["fold"], "mean_absolute_change": float(difference.mean()),
                      "max_absolute_change": float(difference.max()), "gate_mean": float(gate.mean()),
                      "gate_max": float(gate.max()), "exact_identity": bool(np.array_equal(output[test], initial[test]))}
        diagnostics.append(diagnostic)
        print(f"MPR fold={fold['fold']}: mean change={difference.mean():.6e}; max change={difference.max():.6e}; gate mean={gate.mean():.6e}", flush=True)
    if not np.isfinite(output).all():
        raise ValueError("Invalid or incomplete MPR predictions")
    return output, diagnostics, "Evaluated saved fold-local PGRECStatic checkpoints; inference only, not the full QSR model."


def run_evaluation(args):
    initial, reference, truth, fold_ids, folds, source_summary = load_inputs(args)
    initial_digest = array_digest(initial)
    if getattr(args, "train_mpr", False):
        train_fold_mpr(args, folds, source_summary["configuration"]["reference_k"])
    refined, diagnostics, status = apply_saved_mpr(args, initial, reference, folds)
    if array_digest(initial) != initial_digest:
        raise RuntimeError("Initial EC baseline was modified during MPR processing")
    predictions = {"Initial EC": initial, "MPR Group Reference": reference}
    if refined is not None:
        predictions["Initial EC + MPR"] = refined
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results, csv_rows, display, all_subject_rows, deltas = {}, [], [], [], []
    baseline_rows = evaluate_method(initial, truth)[0]
    headers = ["Method", "Directed AUPRC ↑", "F1 ↑", "SHD ↓", "True strength r ↑", "Paired Sign Acc. ↑", "Mean abs ΔEC"]
    for method, matrices in predictions.items():
        rows, metrics, effects = evaluate_method(matrices, truth)
        difference = np.abs(matrices - initial)
        result = {"topology": metrics, "strength": effects,
                  "mean_absolute_change_from_initial": float(difference.mean()),
                  "max_absolute_change_from_initial": float(difference.max()),
                  "exact_identity_with_initial": bool(np.array_equal(matrices, initial))}
        results[method] = result
        csv_rows.append({"method": method, "status": "evaluated", **{
            f"{metric}_{statistic}": values[statistic] for metric, values in metrics.items() for statistic in ("mean", "std")
        }, **effects, "mean_absolute_change_from_initial": result["mean_absolute_change_from_initial"]})
        cells = [f"{metrics[metric]['mean']:.2f} ± {metrics[metric]['std']:.2f}" for metric in ("auprc", "f1", "shd")]
        cells.extend(["N/A" if effects["within_group_r"] is None else f"{effects['within_group_r']:.2f}",
                      f"{effects['paired_sign_accuracy']:.0%}", f"{difference.mean():.6e}"])
        display.append([method, *cells])
        for index, row in enumerate(rows):
            all_subject_rows.append({"method": method, "subject_index": index, "subject_id": index + 1, "fold": int(fold_ids[index]),
                                     **row, "true_target_weight": float(truth[index, 1, 2]),
                                     "predicted_target_weight": float(matrices[index, 1, 2]),
                                     "mean_absolute_change_from_initial": float(difference[index].mean())})
        if method != "Initial EC":
            for index, (first, second) in enumerate(zip(baseline_rows, rows)):
                deltas.append({"method": method, "subject_index": index, "fold": int(fold_ids[index]),
                               **{f"delta_{metric}": second[metric] - first[metric] for metric in TOPOLOGY_METRICS}})
    if refined is None:
        display.append(["Initial EC + MPR", *(["Pending"] * 6)])
        csv_rows.append({key: ("Initial EC + MPR" if key == "method" else "pending_no_sim21_mpr_checkpoint" if key == "status" else None) for key in csv_rows[0]})
    elif results["Initial EC + MPR"]["exact_identity_with_initial"]:
        status += " WARNING: all final EC matrices exactly equal Initial EC; no actual refinement improvement."
    widths = [max(len(row[index]) for row in [headers, *display]) for index in range(len(headers))]
    mode = "Encoder unchanged; MPR trained on cached training EC only" if getattr(args, "train_mpr", False) else "NO TRAINING"
    print(f"\nTable R2. Effect of MPR on individual EC recovery (Sim21; {mode})", flush=True)
    for row in [headers, *display]:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)), flush=True)
    print(status, flush=True)
    write_csv(output / "table_R2_recovery.csv", csv_rows)
    write_csv(output / "subject_metrics.csv", all_subject_rows)
    write_csv(output / "paired_topology_differences.csv", deltas)
    np.savez_compressed(output / "oof_comparison.npz", initial=initial, fc_reference=reference,
                        **({"mpr_refined": refined} if refined is not None else {}),
                        ground_truth=truth, fold_ids=fold_ids, subject_indices=np.arange(50))
    summary = {"methods": results, "mpr_status": status, "mpr_fold_diagnostics": diagnostics,
               "initial_uses_neighbors": False, "initial_sha256": initial_digest,
               "mpr_trained_this_run": bool(getattr(args, "train_mpr", False)),
               "mpr_training_settings": MPR_SETTINGS if getattr(args, "train_mpr", False) else None,
               "source_directory": str(args.source_dir.resolve()), "source_configuration": source_summary["configuration"],
               **PAPER_ORIENTATION, "threshold_protocol": THRESHOLD_PROTOCOL,
               "protocol": "Initial EC is unmodified Encoder-only output and never uses neighbors. Encoder never retrained here. Optional --train-mpr trains only fold-local PGRECStatic with existing fixed settings; no GT tuning. Each subject evaluated against its own weighted GT support (net != 0), excluding diagonal. Same fixed FSTA 0.5 relative threshold as R1. Strength r retains the previous within-group Fisher-z average, not pooled all-edge correlation. Paired effect sign is not neural direction accuracy. Topology mean/sample SD across 50 subjects; strength r is one OOF statistic. No claimed improvement without actual refined outputs."}
    dump_json(output / "summary.json", summary)
    lines = ["# Table R2. Effect of MPR on individual EC recovery", "",
             mode + ". Initial is Encoder-only and never uses neighbors. Topology: 50-subject mean ± sample SD; strength r: within-group Fisher-z average on net[:,1,2].",
             "F1/SHD use the same fixed FSTA-style 0.5 relative threshold as R1. Off-diagonal support: net != 0.",
             "Reference R is not final refined EC. Missing MPR predictions are Pending, never filled with R or A.", "",
             "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in display)
    lines.extend(["", status, "", "Changes use scientific notation so a small nonzero change is not hidden by two-decimal display."])
    (output / "table_R2_recovery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
