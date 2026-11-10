"""Raw time-series to Original-EC generation for the STF-EC encoder."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .training import build_stf_encoder, train_stf_ec
from .utils import extract_subject_ec


def generate_subject_ec(args, subjects, device):
    """Train STF-EC and extract one directed Original EC per subject."""
    model, training_metrics = train_stf_ec(
        args,
        subjects["time_series"],
        device,
        subjects.get("window_ranges"),
    )
    extracted = extract_subject_ec(
        model,
        subjects["records"],
        subjects["time_series"],
        args.window_length,
        args.stride,
        device,
        subjects.get("window_ranges"),
    )
    return {
        "ec": extracted["ec"],
        "labels": subjects["labels"],
        "subject_ids": subjects["subject_ids"],
        "site_ids": subjects["site_ids"],
        "reconstruction_mse": extracted["reconstruction_mse"],
    }, training_metrics


def save_subject_ec(path, data):
    """Persist a reusable EC archive for Graph-EC's ``ec`` mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ec=data["ec"],
        labels=data["labels"],
        subject_ids=data["subject_ids"],
        site_ids=data["site_ids"],
        reconstruction_mse=data["reconstruction_mse"],
    )


def _subset_subjects(subjects, indices):
    indices = np.asarray(indices, dtype=np.int64)
    subset = {
        "records": [subjects["records"][index] for index in indices],
        "time_series": [subjects["time_series"][index] for index in indices],
        "labels": subjects["labels"][indices],
        "subject_ids": subjects["subject_ids"][indices],
        "site_ids": subjects["site_ids"][indices],
    }
    ranges = subjects.get("window_ranges")
    subset["window_ranges"] = (
        None if ranges is None else [ranges[index] for index in indices]
    )
    return subset


def _fold_signature(args, fold, fold_seed, subjects, split_indices):
    encoder_fields = (
        "window_length", "stride", "epochs", "stf_checkpoint", "loss_alpha",
        "batch_size", "d_model", "d_inner_hid", "d_k", "d_v", "n_head",
        "dropout", "n_warmup_steps", "lr_mul", "weight_decay", "adam_beta1",
        "adam_beta2", "num_hidden_layers", "num_attention_heads", "hidden_act",
        "attention_probs_dropout_prob", "hidden_dropout_prob", "initializer_range",
        "no_filters",
    )
    payload = {
        "protocol": "fold-specific-inductive-v1",
        "dataset": args.dataset,
        "fold": int(fold),
        "fold_seed": int(fold_seed),
        "encoder": {field: getattr(args, field) for field in encoder_fields},
        "splits": {
            name: subjects["subject_ids"][indices].astype(str).tolist()
            for name, indices in split_indices.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def generate_fold_subject_ec(
    args,
    fold,
    fold_seed,
    subjects,
    train_index,
    val_index,
    test_index,
    device,
    include_test=True,
):
    """Fit on train subjects and extract only the splits needed by this stage."""
    split_indices = {
        "train": np.asarray(train_index, dtype=np.int64),
        "val": np.asarray(val_index, dtype=np.int64),
        "test": np.asarray(test_index, dtype=np.int64),
    }
    split_sets = {name: set(values.tolist()) for name, values in split_indices.items()}
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"]:
        raise ValueError("Encoder train subjects overlap validation or test subjects")
    if split_sets["val"] & split_sets["test"]:
        raise ValueError("Validation and test subjects overlap")

    fold_cache_dir = Path(args.encoder_cache_dir) / f"fold_{fold:02d}"
    legacy_manifest = fold_cache_dir / "manifest.json"
    legacy_cache = fold_cache_dir / "fold_ec.npz"
    manifest, signature = _fold_signature(
        args, fold, fold_seed, subjects, split_indices
    )
    cache_dir = fold_cache_dir / signature[:16]
    if legacy_manifest.is_file() and legacy_cache.is_file():
        legacy = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        if legacy.get("signature") == signature:
            cache_dir = fold_cache_dir
    cache_path = cache_dir / "fold_ec.npz"
    manifest_path = cache_dir / "manifest.json"
    checkpoint_path = cache_dir / "stf_ec.pt"
    required_splits = ("train", "val", "test") if include_test else ("train", "val")
    fold_args = type(args)(**vars(args))
    fold_args.seed = fold_seed
    if args.resume_fold_encoders and cache_path.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            with np.load(cache_path, allow_pickle=False) as archive:
                cached = {key: np.asarray(archive[key]) for key in archive.files}
            missing_splits = [
                name for name in required_splits if f"{name}_ec" not in cached
            ]
            if missing_splits and checkpoint_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model = build_stf_encoder(fold_args, device)
                model.load_state_dict(checkpoint["state_dict"])
                for name in missing_splits:
                    split_subjects = _subset_subjects(subjects, split_indices[name])
                    result = extract_subject_ec(
                        model,
                        split_subjects["records"],
                        split_subjects["time_series"],
                        args.window_length,
                        args.stride,
                        device,
                        split_subjects.get("window_ranges"),
                    )
                    cached[f"{name}_ec"] = result["ec"]
                    cached[f"{name}_reconstruction_mse"] = result[
                        "reconstruction_mse"
                    ]
                np.savez_compressed(cache_path, **cached)
                existing["extracted_splits"] = sorted(
                    name for name in split_indices if f"{name}_ec" in cached
                )
                manifest_path.write_text(
                    json.dumps(existing, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    f"fold {fold}: extended matching STF-EC cache with "
                    f"{', '.join(missing_splits)}"
                )
            if all(f"{name}_ec" in cached for name in required_splits):
                print(f"fold {fold}: loaded matching fold-specific STF-EC cache")
                return {
                    f"{name}_ec": np.asarray(cached[f"{name}_ec"], dtype=np.float32)
                    for name in required_splits
                } | {
                    "training_metrics": existing["training_metrics"],
                    "manifest": existing,
                    "cache_path": cache_path,
                }

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_subjects = _subset_subjects(subjects, split_indices["train"])
    model, training_metrics = train_stf_ec(
        fold_args,
        train_subjects["time_series"],
        device,
        train_subjects.get("window_ranges"),
    )
    extracted = {}
    reconstruction = {}
    for name in required_splits:
        indices = split_indices[name]
        split_subjects = _subset_subjects(subjects, indices)
        result = extract_subject_ec(
            model,
            split_subjects["records"],
            split_subjects["time_series"],
            args.window_length,
            args.stride,
            device,
            split_subjects.get("window_ranges"),
        )
        extracted[f"{name}_ec"] = result["ec"]
        reconstruction[f"{name}_reconstruction_mse"] = result["reconstruction_mse"]

    torch.save(
        {
            "state_dict": model.state_dict(),
            "fold": fold,
            "fold_seed": fold_seed,
            "signature": signature,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        cache_path,
        **extracted,
        **reconstruction,
        train_subject_ids=subjects["subject_ids"][split_indices["train"]].astype(str),
        val_subject_ids=subjects["subject_ids"][split_indices["val"]].astype(str),
        test_subject_ids=subjects["subject_ids"][split_indices["test"]].astype(str),
    )
    complete_manifest = {
        **manifest,
        "signature": signature,
        "training_metrics": training_metrics,
        "cache_path": str(cache_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "extracted_splits": list(required_splits),
        "leakage_audit": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "encoder_fit_subject_count": len(split_indices["train"]),
        },
    }
    manifest_path.write_text(
        json.dumps(complete_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        **extracted,
        "training_metrics": training_metrics,
        "manifest": complete_manifest,
        "cache_path": cache_path,
    }
