"""Single entry point for Pair Memory, ablations and sparsity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from data.loader import PROCESSED_DIR, base_network, load_scene, make_dataloader
from model.memory_ec import MemoryEC
from train import resolve_device, train
from utils import MODEL_TYPES, adaptive_threshold, configure_determinism, directed_metrics, save_json, zero_diagonal


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
LAMBDAS = (0.0, 0.005, 0.02)
METHODS = ("validation_selected", "fixed_0.5", "fsta_adaptive")
METRICS = (
    "precision", "recall", "directed_f1", "accuracy", "shd", "auprc", "auroc",
    "fp", "fn", "predicted_edges", "true_edges",
)
CONSOLE_METRICS = {
    "precision": "Precision (higher is better)",
    "recall": "Recall (higher is better)",
    "directed_f1": "F1 (higher is better)",
    "accuracy": "Accuracy (higher is better)",
    "shd": "SHD (lower is better)",
}
TRAIN_KEYS = (
    "hidden_dim", "memory_slots", "attention_heads", "dropout", "learning_rate",
    "weight_decay", "batch_size", "epochs", "patience", "min_delta",
)
LABELS = {
    "st_no_memory": "No Memory", "st_global_memory": "Global Memory",
    "st_pair_memory": "Pair Memory",
}


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def family_scenes() -> dict[int, list[str]]:
    families = {family: [] for family in range(1, 10)}
    for path in sorted(PROCESSED_DIR.glob("Network*.npz")):
        family = base_network(path.stem)
        if family not in families:
            raise ValueError(f"Unexpected family: {path}")
        families[family].append(path.stem)
    if any(not scenes for scenes in families.values()):
        raise ValueError("Processed data must contain every family Network1..9")
    return families


def fold_split(fold: int, families: dict[int, list[str]]) -> dict[str, list[str]]:
    validation = {fold % 9 + 1, (fold + 1) % 9 + 1}
    train_families = set(families) - validation - {fold}
    return {
        "train": [scene for family in sorted(train_families) for scene in families[family]],
        "val": [scene for family in sorted(validation) for scene in families[family]],
        "test": families[fold],
    }


def aggregate_predictions(model: MemoryEC, scenes: list[str], args: argparse.Namespace) -> dict:
    loader = make_dataloader("test", batch_size=args.batch_size, seed=args.seed,
                             balanced=False, scenes=scenes)
    probabilities = {}
    targets = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            scores = model(batch["bold"].to(args.device)).sigmoid().cpu().numpy()
            for index, scene in enumerate(batch["scene"]):
                probabilities.setdefault(scene, []).append(scores[index])
                targets[scene] = batch["adj"][index].numpy()
    return {
        scene: (zero_diagonal(np.mean(probabilities[scene], axis=0)), targets[scene])
        for scene in scenes
    }


def threshold_from_validation(data: dict) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    families = sorted({base_network(scene) for scene in data})
    scores = []
    for threshold in grid:
        family_f1 = [np.mean([
            directed_metrics(target, probability, float(threshold))["directed_f1"]
            for scene, (probability, target) in data.items()
            if base_network(scene) == family
        ]) for family in families]
        scores.append(float(np.mean(family_f1)))
    best = int(np.argmax(scores))
    return float(grid[best]), scores[best]


def evaluate_method(data: dict, method: str, selected: float,
                    fsta_ratio: float = 0.5) -> list[dict]:
    rows = []
    for scene, (probability, target) in sorted(data.items()):
        threshold = {
            "validation_selected": selected, "fixed_0.5": 0.5,
            "fsta_adaptive": adaptive_threshold(probability, fsta_ratio),
        }[method]
        mask = ~np.eye(target.shape[0], dtype=bool)
        labels = target[mask].astype(bool)
        prediction = probability[mask] >= threshold
        rows.append({
            "scene": scene, "network": base_network(scene), "method": method,
            "threshold": threshold, **directed_metrics(target, probability, threshold),
            "tp": int((prediction & labels).sum()),
            "fp": int((prediction & ~labels).sum()),
            "fn": int((~prediction & labels).sum()),
            "predicted_edges": int(prediction.sum()), "true_edges": int(labels.sum()),
        })
    return rows


def load_model(path: Path, device: str) -> tuple[MemoryEC, dict]:
    configure_determinism()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MemoryEC(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def candidate_validation(args: argparse.Namespace, split: dict, directory: Path,
                         sparse_lambda: float) -> dict:
    options = vars(args).copy()
    options.update(output_dir=directory, split_scenes=split, sparse_lambda=sparse_lambda)
    train(argparse.Namespace(**options))
    checkpoint_path = directory / f"{args.model_type}_best.pt"
    model, checkpoint = load_model(checkpoint_path, args.device)
    validation = aggregate_predictions(model, split["val"], args)
    threshold, score = threshold_from_validation(validation)
    return {
        "sparse_lambda": sparse_lambda, "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_selection": "val_loss",
        "threshold": threshold, "validation_family_macro_f1": score,
        "validation_metrics": evaluate_method(validation, "validation_selected", threshold),
    }


def run_model(args: argparse.Namespace, split: dict, directory: Path,
              sparse_search: bool) -> list[dict]:
    result_path = directory / "results.json"
    candidates = []
    for coefficient in LAMBDAS if sparse_search else (args.sparse_lambda,):
        candidate_dir = directory / f"lambda_{coefficient:g}" if sparse_search else directory
        candidates.append(candidate_validation(args, split, candidate_dir, coefficient))
        candidate = candidates[-1]
        print(f"  lambda={coefficient:g}, best_epoch={candidate['checkpoint_epoch']}, "
              f"validation_threshold={candidate['threshold']:.2f}, "
              f"validation_family_F1={candidate['validation_family_macro_f1']:.4f}", flush=True)
    winner = max(candidates, key=lambda item: (
        item["validation_family_macro_f1"], -item["sparse_lambda"]
    ))
    save_json({
        "candidates": candidates, "selected_lambda": winner["sparse_lambda"],
        "selection_criterion": "validation family-macro F1; ties choose smaller lambda",
        "threshold_tie_rule": "lowest threshold in 0.05..0.95 step 0.01",
        "test_used_for_selection": False,
    }, directory / "validation_selection.json")
    if sparse_search:
        print(f"  Validation-locked lambda={winner['sparse_lambda']:g}; test not used for selection.", flush=True)
    rows = []
    for candidate in candidates:
        model, _ = load_model(Path(candidate["checkpoint"]), args.device)
        test_data = aggregate_predictions(model, split["test"], args)
        variant = f"lambda={candidate['sparse_lambda']:g}" if sparse_search else args.model_type
        for method in METHODS:
            for metrics in evaluate_method(test_data, method, candidate["threshold"], args.fsta_ratio):
                row = {
                    "repeat": args.repeat, "seed": args.seed, "fold": args.fold,
                    "model": args.model_type, "variant": variant,
                    "sparse_lambda": candidate["sparse_lambda"],
                    "fsta_ratio": args.fsta_ratio,
                    "checkpoint_epoch": candidate["checkpoint_epoch"],
                    "checkpoint_selection": candidate["checkpoint_selection"],
                    "validation_threshold": candidate["threshold"],
                    "validation_family_macro_f1": candidate["validation_family_macro_f1"],
                    **metrics,
                }
                rows.append(row)
                if sparse_search and candidate is winner:
                    rows.append({**row, "variant": "validation_selected_lambda"})
    save_json(rows, result_path)
    return rows


def protocol(args: argparse.Namespace, families: dict) -> dict:
    return {
        "protocol": "unified_cv_v1", "mode": args.mode,
        "runtime": "strict_deterministic_math_sdp_v1",
        "models": list(MODEL_TYPES) if args.mode == "ablation" else [args.model_type],
        "repeats": args.repeats, "seeds": [args.seed + repeat for repeat in range(args.repeats)],
        "training": {key: getattr(args, key) for key in TRAIN_KEYS},
        "sparse_lambdas": list(LAMBDAS) if args.mode == "sparse" else [args.sparse_lambda],
        "splits": {str(fold): fold_split(fold, families) for fold in range(1, 10)},
        "checkpoint_selection": "minimum unpenalized validation BCE; early stopping on the same metric",
        "threshold_selection": "validation family-macro F1; grid 0.05..0.95 step 0.01; lowest threshold wins ties",
        "primary_method": "validation_selected", "fsta_ratio": args.fsta_ratio,
    }


def print_summary(family_rows: list[dict], fsta_ratio: float = 0.5) -> None:
    variants = list(dict.fromkeys(row["variant"] for row in family_rows))
    networks = sorted({row["network"] for row in family_rows})
    repeats = sorted({row["repeat"] for row in family_rows})
    width = max(22, *(len(LABELS.get(variant, variant)) + 2 for variant in variants))
    print("\n=== Cross-network results (two decimals) ===")
    print(f"Completed test families: {networks}; repetitions: {len(repeats)}")
    print("Network rows: scene-variant means, then repeat mean +/- repeat SD (ddof=0).")
    print("Console: Precision, Recall, F1, Accuracy, SHD only; full metrics remain in CSV/JSON.")
    print("Scores remain on a 0..1 scale; SHD is the scene-mean edge error count.")
    for method in METHODS:
        titles = {
            "validation_selected": "Validation-selected threshold [PRIMARY]",
            "fixed_0.5": "Fixed threshold=0.50 [comparison]",
            "fsta_adaptive": f"FSTA per-scene threshold, ratio={fsta_ratio:.2f} [comparison]",
        }
        print(f"\n--- {titles[method]} ---")
        for metric, title in CONSOLE_METRICS.items():
            print(f"\n{title}")
            print(f"{'Network':<16}" + "".join(f"{LABELS.get(variant, variant):>{width}}" for variant in variants))
            stats = {}
            for variant in variants:
                group = [row for row in family_rows if row["variant"] == variant and row["method"] == method]
                per_family = [[row[metric] for row in group if row["network"] == network] for network in networks]
                stats[variant] = per_family
            for index, network in enumerate(networks):
                cells = []
                for variant in variants:
                    values = stats[variant][index]
                    cell = f"{np.mean(values):.2f}"
                    if len(repeats) > 1:
                        cell += f" +/- {np.std(values):.2f}"
                    cells.append(cell)
                print(f"{'Network' + str(network):<16}" + "".join(f"{cell:>{width}}" for cell in cells))

def summarize(directory: Path, config: dict) -> None:
    rows = []
    expected = {
        (repeat + 1, fold, model)
        for repeat in range(config["repeats"]) for fold in range(1, 10)
        for model in config["models"]
    }
    present = {}
    for repeat, fold, model in sorted(expected):
        path = directory / f"repeat_{repeat:02d}" / f"fold_{fold:02d}" / model / "results.json"
        if path.is_file():
            present[(repeat, fold, model)] = read_json(path)
    complete_folds = [fold for fold in range(1, 10) if all(
        (repeat + 1, fold, model) in present
        for repeat in range(config["repeats"]) for model in config["models"]
    )]
    for key, records in sorted(present.items()):
        if key[1] in complete_folds:
            rows.extend(records)
    if not rows:
        print("No fully completed fold (all selected models and repeats) to summarize.")
        return
    keys = list(dict.fromkeys((row["repeat"], row["network"], row["variant"], row["method"]) for row in rows))
    family_rows = []
    for repeat, network, variant, method in keys:
        group = [row for row in rows if (row["repeat"], row["network"], row["variant"], row["method"]) == (repeat, network, variant, method)]
        family_rows.append({
            "repeat": repeat, "seed": group[0]["seed"], "network": network,
            "variant": variant, "method": method,
            "sparse_lambda": group[0]["sparse_lambda"],
            **{metric: float(np.mean([row[metric] for row in group])) for metric in METRICS},
        })
    write_csv(rows, directory / "scene_results.csv")
    write_csv(family_rows, directory / "family_results.csv")
    variants = list(dict.fromkeys(row["variant"] for row in family_rows))
    network_rows = []
    for variant in variants:
        for method in METHODS:
            group = [row for row in family_rows if row["variant"] == variant and row["method"] == method]
            for network in complete_folds:
                entries = [row for row in group if row["network"] == network]
                network_rows.append({
                    "network": network, "variant": variant, "method": method,
                    **{f"{metric}_mean": float(np.mean([row[metric] for row in entries])) for metric in METRICS},
                    **{f"{metric}_repeat_std": float(np.std([row[metric] for row in entries])) for metric in METRICS},
                })
    write_csv(network_rows, directory / "network_results.csv")
    wide_rows = []
    for network in complete_folds:
        wide = {"Network": f"Network{network}"}
        for row in network_rows:
            if row["network"] != network:
                continue
            for metric in METRICS:
                for statistic in ("mean", "repeat_std"):
                    wide[f"{row['variant']}_{row['method']}_{metric}_{statistic}"] = row[f"{metric}_{statistic}"]
        wide_rows.append(wide)
    write_csv(wide_rows, directory / "network_table.csv")
    save_json({"complete_test_families": complete_folds, "complete_nine_fold": len(complete_folds) == 9,
               "repeats": config["repeats"],
               "primary": "validation_selected; in sparse mode use validation_selected_lambda",
               "warning": "Per-lambda test comparisons are exploratory, never use test metrics to choose lambda."},
              directory / "summary.json")
    print_summary(family_rows, config.get("fsta_ratio", 0.5))
    print(f"\nSaved summaries: {directory}")


def diagnose_existing(source: Path) -> None:
    rows = []
    for fold in range(1, 10):
        for record in read_json(source / f"fold_{fold:02d}" / "results.json"):
            _, target = load_scene(record["scene"])
            count = int(target[~np.eye(target.shape[0], dtype=bool)].sum())
            tp = int(round(record["recall"] * count))
            fn = count - tp
            fp = int(round(record["shd"])) - fn
            prediction = tp + fp
            precision = tp / prediction if prediction else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
            accuracy = 1 - (fp + fn) / (target.size - target.shape[0])
            if fp < 0 or not np.allclose([precision, f1, accuracy],
                    [record["precision"], record["directed_f1"], record["accuracy"]], atol=1e-8):
                raise ValueError(f"Inconsistent saved metrics for {record['scene']}")
            rows.append({**record, "tp": tp, "fp": fp, "fn": fn,
                         "predicted_edges": prediction, "true_edges": count})
    write_csv(rows, source / "edge_diagnostics_scenes.csv")
    print(f"BCE pos_weight=None (effective positive-class weight=1). Saved: {source / 'edge_diagnostics_scenes.csv'}")


def prepare_training_output(directory: Path, config: dict, folds: list[int]) -> None:
    config_path = directory / "config.json"
    if directory.exists() and (not config_path.is_file() or read_json(config_path) != config):
        shutil.rmtree(directory)
    for repeat in range(1, config["repeats"] + 1):
        for fold in folds:
            fold_dir = directory / f"repeat_{repeat:02d}" / f"fold_{fold:02d}"
            if fold_dir.exists():
                shutil.rmtree(fold_dir)
    for filename in ("scene_results.csv", "family_results.csv", "network_results.csv",
                     "network_table.csv", "summary.json"):
        (directory / filename).unlink(missing_ok=True)
    save_json(config, config_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "ablation", "sparse", "evaluate", "diagnose"), default="train")
    parser.add_argument("--model-type", choices=("st_no_memory", "st_global_memory", "st_pair_memory"), default="st_pair_memory")
    parser.add_argument("--fold", type=int, choices=range(1, 10), help="Test family; omit for all nine folds")
    parser.add_argument("--repeats", type=int, choices=(1, 5, 20), default=5)

    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-3) #优化器权重衰减
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5) #早停判断中，验证损失需要降低的最小幅度
    parser.add_argument("--sparse-lambda", type=float, default=0.0)
    parser.add_argument("--fsta-ratio", "--fsta_ratio", type=float, default=0.4)
    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for key in ("hidden_dim", "memory_slots", "attention_heads", "batch_size", "epochs", "patience"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.hidden_dim % args.attention_heads:
        parser.error("hidden-dim must be divisible by attention-heads")
    if not math.isfinite(args.fsta_ratio) or not 0 <= args.fsta_ratio <= 1:
        parser.error("fsta-ratio must be finite and in [0,1]")
    for key in ("sparse_lambda", "learning_rate", "weight_decay", "min_delta", "dropout"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and non-negative")
    if args.dropout >= 1 or args.seed < 0 or args.seed + args.repeats - 1 >= 2**32:
        parser.error("dropout must be < 1; seeds must fit numpy's unsigned 32-bit range")
    if args.mode == "sparse" and (args.model_type != "st_pair_memory" or args.sparse_lambda != 0):
        parser.error("sparse mode uses Pair Memory and the fixed lambda grid (0, 0.005, 0.02)")
    return args


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    args.device = "cuda"
    if args.mode == "diagnose":
        diagnose_existing(OUTPUTS / "cross_network_cv")
        return
    name = f"cv_{args.mode}_{args.model_type}_r{args.repeats}_lambda{args.sparse_lambda:g}"
    if args.fsta_ratio != 0.5:
        name += f"_fsta{args.fsta_ratio!r}"
    if Path(name).name != name or name in (".", "..", "cross_network_cv"):
        raise ValueError("Invalid automatically generated output directory")
    directory = OUTPUTS / name
    config_path = directory / "config.json"
    if args.mode == "evaluate":
        summarize(directory, read_json(config_path))
        return
    families = family_scenes()
    config = protocol(args, families)
    folds = [args.fold] if args.fold else list(range(1, 10))
    print(f"Mode={args.mode}, models={config['models']}, repeats={args.repeats}, seeds={config['seeds']}")
    print(f"Sparse lambdas={config['sparse_lambdas']}; output={directory}")
    print(f"FSTA ratio={args.fsta_ratio:.2f} [comparison]", flush=True)
    print("Checkpoint: minimum validation BCE; primary threshold: validation family-macro F1.")
    for fold in folds:
        print(f"Fold {fold}: " + "; ".join(
            f"{part}={sorted({base_network(scene) for scene in scenes})}"
            for part, scenes in config["splits"][str(fold)].items()
        ))
    configure_determinism()
    device_object = resolve_device("cuda")
    device = str(device_object)
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    gpu_properties = torch.cuda.get_device_properties(gpu_index)
    print(
        f"GPU: cuda:{gpu_index} | {gpu_name} | "
        f"VRAM={gpu_properties.total_memory / (1024 ** 3):.2f} GB",
        flush=True,
    )
    print("Training from scratch; overwriting selected folds (no backup).", flush=True)
    prepare_training_output(directory, config, folds)
    for repeat, seed in enumerate(config["seeds"], 1):
        for fold in folds:
            split = config["splits"][str(fold)]
            fold_dir = directory / f"repeat_{repeat:02d}" / f"fold_{fold:02d}"
            save_json({"repeat": repeat, "seed": seed, "scenes": split,
                       "families": {part: sorted({base_network(scene) for scene in scenes})
                                    for part, scenes in split.items()}}, fold_dir / "split.json")
            for model_type in config["models"]:
                options = vars(args).copy()
                options.update(repeat=repeat, seed=seed, fold=fold, model_type=model_type, device=device)
                print(f"\nRepeat {repeat}/{args.repeats}, fold={fold}, model={model_type}, seed={seed}", flush=True)
                run_model(argparse.Namespace(**options), split, fold_dir / model_type, args.mode == "sparse")
    summarize(directory, config)


if __name__ == "__main__":
    main()
