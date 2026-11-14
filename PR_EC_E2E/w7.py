"""W7 protocol locks, configuration provenance and isolated artifact paths."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

LOCAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_ROOT.parent
OUTPUT_ROOT = LOCAL_ROOT / "outputs/rebuttal_w7"
DATASETS = {"abide": "abide-i", "abide_ii": "abide-ii", "adhd200": "adhd200"}
PROTOCOL = "transductive_full_cohort_unlabeled"
STF_KEYS = (
    "dataset", "seed", "epochs", "stf_checkpoint", "window_length", "stride",
    "loss_alpha", "batch_size", "d_model", "d_inner_hid", "d_k", "d_v", "n_head",
    "dropout", "n_warmup_steps", "lr_mul", "weight_decay", "adam_beta1", "adam_beta2",
    "num_hidden_layers", "num_attention_heads", "hidden_act", "attention_probs_dropout_prob",
    "hidden_dropout_prob", "initializer_range", "no_filters", "stf_fusion_mode",
)
METRICS = {"ACC": "ACC", "AUC": "AUC", "SEN": "Recall", "PRE": "Precision", "F1": "F1"}


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_w7_arguments(parser):
    parser.add_argument("--stf-fusion-mode", choices=("batch_mean", "subject_specific"), default="batch_mean")
    parser.add_argument("--w7-stage", choices=("pretrain", "downstream", "all"),
                        help="W7 only: pretrain/extract, cached-EC classification, or both")
    parser.add_argument("--reference-ec-path", type=Path,
                        help="Original seed42 EC; read-only reference for subject identity and ROI order")
    parser.add_argument("--reference-oof-path", type=Path,
                        help="Original seed42 QSR OOF; read-only check of labels and test fold IDs")


def finalize_w7_arguments(parser, args):
    args.w7_enabled = args.stf_fusion_mode == "subject_specific" or args.w7_stage is not None
    if not args.w7_enabled:
        return args
    if args.w7_stage is None:
        args.w7_stage = "all" if args.input_mode == "raw" else "downstream"
    expected_input = "ec" if args.w7_stage == "downstream" else "raw"
    if args.input_mode != expected_input:
        parser.error(f"W7 {args.w7_stage} requires --input-mode {expected_input}")
    if (args.seed != 42 or args.n_splits != 10 or args.validation_size != 0.2
            or args.stf_checkpoint != "final" or args.batch_size != 32
            or args.epochs != (141 if args.dataset == "adhd200" else 81)):
        parser.error("W7 fixes seed42, ten folds, validation-size=.2, batch-size32, final checkpoint and 81/81/141 epochs")
    if args.qsr_disable_perturbation_restoration or args.qsr_eta == 0:
        parser.error("W7 uses Full PR-EC, not QSR restoration/target-correction ablations")
    if args.graph_mode != "fusion" or args.reference_k != 20 or args.permute_phenotype:
        parser.error("W7 retains the unpermuted original PH+FC fusion graph and K=20")
    args.representations = ["original", "refined", "qc_refined"]
    dataset = DATASETS[args.dataset]
    if args.output_dir == args.profile.output_dir:
        args.output_dir = OUTPUT_ROOT / dataset / args.stf_fusion_mode
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error("All W7 writes must stay under PR_EC_E2E/outputs/rebuttal_w7")
    args.reference_ec_path = (args.reference_ec_path or (
        REPO_ROOT / "PR_EC/outputs/seed42" / dataset / f"{args.dataset}_subject_ec.npz"
    )).resolve()
    args.reference_oof_path = (args.reference_oof_path or (
        args.reference_ec_path.parent / f"{args.dataset}_qsr_refined_subject_ec.npz"
    )).resolve()
    if args.ec_path == args.profile.ec_path:
        args.ec_path = (args.reference_ec_path if args.stf_fusion_mode == "batch_mean" and expected_input == "ec"
                        else args.output_dir / f"{args.dataset}_{args.stf_fusion_mode}_subject_ec.npz")
    args.ec_path = args.ec_path.resolve()
    if args.refined_ec_path == args.profile.refined_ec_path:
        args.refined_ec_path = args.output_dir / f"{args.dataset}_refined_subject_ec.npz"
    if args.qsr_refined_ec_path == args.profile.qsr_refined_ec_path:
        args.qsr_refined_ec_path = args.output_dir / f"{args.dataset}_qsr_refined_subject_ec.npz"
    args.refined_ec_path, args.qsr_refined_ec_path = args.refined_ec_path.resolve(), args.qsr_refined_ec_path.resolve()
    writes = [args.refined_ec_path, args.qsr_refined_ec_path]
    if expected_input == "raw" or args.stf_fusion_mode == "subject_specific":
        writes.append(args.ec_path)
    if any(not path.is_relative_to(args.output_dir) for path in writes):
        parser.error("New Initial/refined EC paths must stay inside the W7 output directory")
    if len(set([args.ec_path, args.refined_ec_path, args.qsr_refined_ec_path])) != 3:
        parser.error("Initial, PGR and QSR EC paths must be distinct")
    if any(path in (args.reference_ec_path, args.reference_oof_path) for path in writes):
        parser.error("Never overwrite original reference EC archives")
    return args


def stf_configuration(args):
    return {**{key: getattr(args, key, "batch_mean" if key == "stf_fusion_mode" else None) for key in STF_KEYS},
            "pretraining_protocol": PROTOCOL, "entropy_attention": "batch_mean"}


def initial_ec_metadata(args, training_metrics):
    return {
        "stf_fusion_mode": np.asarray(getattr(args, "stf_fusion_mode", "batch_mean")),
        "stf_config_json": np.asarray(json.dumps(stf_configuration(args), sort_keys=True)),
        "stf_training_metrics_json": np.asarray(json.dumps(training_metrics, sort_keys=True)),
        "pretraining_protocol": np.asarray(PROTOCOL),
    }


def validate_initial_ec_mode(args, data):
    requested_mode = getattr(args, "stf_fusion_mode", "batch_mean")
    mode = str(np.asarray(data.get("stf_fusion_mode", "batch_mean")).item())
    if mode != requested_mode:
        raise ValueError(f"EC fusion mode {mode!r} does not match requested {requested_mode!r}")
    if requested_mode == "subject_specific" or "stf_config_json" in data:
        if "stf_config_json" not in data:
            raise ValueError("Subject-specific EC requires recorded pretraining provenance; old EC cannot be relabeled")
        config = json.loads(str(np.asarray(data["stf_config_json"]).item()))
        if config != stf_configuration(args):
            raise ValueError("Cached Initial EC has a different pretraining configuration")


def audit_original_configuration(args):
    source = args.reference_ec_path.parent / "summary.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    config = original["config"]
    if config.get("input_mode") != "raw" or config.get("seed") != 42 or config.get("stf_checkpoint") != "final":
        raise ValueError("Original source must document transductive raw pretraining, seed42 and final checkpoint")
    stf_metrics = original.get("stf_ec_training", {}).get("stf_ec") or {}
    if stf_metrics.get("epoch") != args.epochs:
        raise ValueError("Original source does not confirm the requested completed pretraining epoch")
    excluded = {"profile", "input_mode", "ec_path", "refined_ec_path", "qsr_refined_ec_path", "output_dir", "gpu_id", "log_every", "seeds"}
    current = jsonable(vars(args))
    differences = {}
    for key, expected in config.items():
        if key in excluded:
            continue
        actual = current.get(key)
        if key in ("data_root", "phenotype_csv"):
            expected, actual = str(Path(expected).resolve()), str(Path(actual).resolve())
        if actual != expected:
            differences[key] = {"original": expected, "requested": actual}
    if differences:
        raise ValueError(f"Non-fusion settings differ from original seed42: {differences}")
    return {"reference_summary": str(source), "original_config": config,
            "confirmed_pretraining_epoch": args.epochs, "non_fusion_differences": differences}


def read_reference_metadata(args):
    with np.load(args.reference_ec_path, allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in ("subject_ids", "labels", "site_ids")}
        if "roi_names" in archive:
            metadata["roi_names"] = archive["roi_names"]
    with np.load(args.reference_oof_path, allow_pickle=False) as archive:
        for key in ("subject_ids", "labels", "site_ids"):
            if not np.array_equal(metadata[key], archive[key]):
                raise ValueError(f"Original Initial and QSR archive {key} differ; resolve reference provenance first")
        metadata["fold_ids"] = archive["fold_ids"]
        if "roi_names" in archive:
            roi_names = archive["roi_names"]
            if "roi_names" in metadata and not np.array_equal(metadata["roi_names"], roi_names):
                raise ValueError("Original Initial and QSR ROI names/order differ")
            metadata["roi_names"] = roi_names
    return metadata


def validate_reference_subjects(args, data):
    reference = read_reference_metadata(args)
    for key in ("subject_ids", "labels", "site_ids"):
        if not np.array_equal(data[key], reference[key]):
            raise ValueError(f"W7 {key} or ordering differs from the original seed42 cohort")
    if "roi_names" in reference:
        if "roi_names" in data and not np.array_equal(data["roi_names"], reference["roi_names"]):
            raise ValueError("W7 ROI names/order differs from the original seed42 archive")
        data["roi_names"] = reference["roi_names"]
    from PR_EC.utils import make_stratified_splits

    fold_ids = np.full(len(data["subject_ids"]), -1, dtype=np.int64)
    for fold, _, _, test in make_stratified_splits(data["labels"], args.n_splits, args.seed, args.validation_size):
        fold_ids[test] = fold
    if not np.array_equal(fold_ids, reference["fold_ids"]):
        raise ValueError("W7 split generation differs from the original saved QSR OOF folds")
    return reference


def initialize_w7_run(args):
    from PR_EC_E2E.data import FIXED_DATA_CONFIG

    audit = audit_original_configuration(args)
    config = {key: value for key, value in jsonable(vars(args)).items()
              if key not in {"input_mode", "w7_stage", "gpu_id", "log_every"}}
    fixed_config = {"fixed_data_config": FIXED_DATA_CONFIG,
                    "pretraining_protocol": PROTOCOL, "entropy_attention": "batch_mean",
                    "downstream_implementation": "PR_EC.workflow.run_cross_validation"}
    config.update(fixed_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "protocol_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("W7 output directory contains a different protocol; use a fresh output directory")
    if args.input_mode == "raw" and args.ec_path.exists():
        raise FileExistsError("W7 Initial EC already exists; use --w7-stage downstream --input-mode ec to reuse it")
    write_json(path, config)
    stage_config = "pretraining_config.json" if args.input_mode == "raw" else "downstream_config.json"
    write_json(args.output_dir / stage_config, {**vars(args), **fixed_config})
    write_json(args.output_dir / "original_config_audit.json", audit)


def save_split_membership(args, data, reference):
    from PR_EC.utils import make_stratified_splits

    ids = np.asarray(data["subject_ids"]).astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("W7 subject IDs must be unique")
    fold_ids = np.full(len(ids), -1, dtype=np.int64)
    rows = []
    for fold, train, val, test in make_stratified_splits(data["labels"], args.n_splits, args.seed, args.validation_size):
        groups = [set(indices.tolist()) for indices in (train, val, test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Train/validation/test overlap")
        fold_ids[test] = fold
        for role, indices in (("train", train), ("val", val), ("test", test)):
            rows.extend({"fold": fold, "subject_id": ids[index], "archive_index": int(index),
                         "label": int(data["labels"][index]), "role": role} for index in indices)
    if not np.array_equal(fold_ids, reference["fold_ids"]):
        raise ValueError("W7 test folds differ from the original saved QSR OOF folds")
    write_csv(args.output_dir / "split_membership.csv", rows)
    return fold_ids


def save_w7_classification(args, fold_results):
    rows = []
    for fold, result in enumerate(fold_results, 1):
        for representation in args.representations:
            rows.append({"dataset": DATASETS[args.dataset], "training_fusion": args.stf_fusion_mode,
                         "seed": args.seed, "fold": fold, "representation": representation,
                         **{f"{name}_pct": 100 * result[representation][source] for name, source in METRICS.items()}})
    summaries = []
    for representation in args.representations:
        selected = [row for row in rows if row["representation"] == representation]
        summary = {key: selected[0][key] for key in ("dataset", "training_fusion", "seed", "representation")}
        summary.update(n_folds=len(selected), std_ddof=0)
        for metric in METRICS:
            values = [row[f"{metric}_pct"] for row in selected]
            mean, std = float(np.mean(values)), float(np.std(values, ddof=0))
            summary.update({f"{metric}_mean_pct": mean, f"{metric}_std_pct": std,
                            f"{metric}_display": f"{mean:.2f} ± {std:.2f}"})
        summaries.append(summary)
    write_csv(args.output_dir / "classification_folds.csv", rows)
    write_csv(args.output_dir / "classification_summary.csv", summaries)
