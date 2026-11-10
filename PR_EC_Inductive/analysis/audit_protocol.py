"""Audit fold caches and OOF outputs for the inductive protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


DATASETS = ("abide-i", "abide-ii", "adhd200")


def find_fold_manifests(dataset_dir: Path) -> list[tuple[Path, dict]]:
    summary_path = dataset_dir / "summary.json"
    recorded_signatures = {}
    if summary_path.is_file():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        encoder_runs = run_summary.get("stf_ec_training", {}).get(
            "fold_specific_stf_ec", []
        )
        for encoder_run in encoder_runs:
            fold = int(encoder_run["fold"])
            if fold in recorded_signatures:
                raise ValueError(f"{summary_path}: duplicate encoder record for fold {fold}")
            recorded_signatures[fold] = encoder_run["manifest"]["signature"]
        if recorded_signatures and set(recorded_signatures) != set(range(1, 11)):
            raise ValueError(f"{summary_path}: encoder records must contain folds 1..10")

    cache_root = dataset_dir / "fold_encoders"
    manifest_paths = sorted(
        set(cache_root.glob("fold_*/manifest.json"))
        | set(cache_root.glob("fold_*/*/manifest.json"))
    )
    selected = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fold = int(manifest["fold"])
        if fold not in range(1, 11):
            raise ValueError(f"{manifest_path}: fold must be in 1..10")
        if recorded_signatures and manifest.get("signature") != recorded_signatures[fold]:
            continue
        relative_parts = manifest_path.relative_to(cache_root).parts
        if relative_parts[0] != f"fold_{fold:02d}":
            raise ValueError(f"{manifest_path}: directory does not match fold {fold}")
        if len(relative_parts) == 3 and relative_parts[1] != manifest["signature"][:16]:
            raise ValueError(f"{manifest_path}: directory does not match encoder signature")
        for filename in ("stf_ec.pt", "fold_ec.npz"):
            if not (manifest_path.parent / filename).is_file():
                raise ValueError(f"{manifest_path}: missing cache artifact {filename}")
        if fold in selected:
            raise ValueError(
                f"{dataset_dir}: multiple matching caches for fold {fold}; "
                "retain one cache or provide the matching run summary.json"
            )
        selected[fold] = (manifest_path, manifest)
    if set(selected) != set(range(1, 11)):
        raise ValueError(
            f"{dataset_dir}: expected exactly 10 valid folds (1..10), "
            f"found {sorted(selected)}"
        )
    return [selected[fold] for fold in range(1, 11)]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {}
    for dataset in DATASETS:
        dataset_dir = project_dir / "outputs" / dataset
        with np.load(dataset_dir / "oof_predictions.npz", allow_pickle=False) as archive:
            subject_ids = archive["subject_ids"].astype(str)
            labels = np.asarray(archive["labels"], dtype=np.int64)
            raw_fold_ids = np.asarray(archive["fold_ids"])
            fold_ids = raw_fold_ids.astype(np.int64)
        if (
            subject_ids.ndim != 1
            or labels.shape != subject_ids.shape
            or fold_ids.shape != subject_ids.shape
        ):
            raise ValueError(f"{dataset_dir}: OOF IDs, labels and fold IDs must be aligned vectors")
        if len(set(subject_ids.tolist())) != len(subject_ids):
            raise ValueError(f"{dataset_dir}: duplicate OOF subject IDs")
        if not np.array_equal(raw_fold_ids, fold_ids) or set(fold_ids.tolist()) != set(range(1, 11)):
            raise ValueError(f"{dataset_dir}: OOF fold IDs must be exactly 1..10")
        all_subjects = set(subject_ids.tolist())
        expected = {}
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_pool, test_index) in enumerate(
            splitter.split(labels, labels), 1
        ):
            train_index, val_index = train_test_split(
                train_pool,
                test_size=0.2,
                stratify=labels[train_pool],
                random_state=42 + fold,
            )
            expected[fold] = {
                "train": set(subject_ids[train_index].tolist()),
                "val": set(subject_ids[val_index].tolist()),
                "test": set(subject_ids[test_index].tolist()),
            }
        dataset_rows = []
        for manifest_path, manifest in find_fold_manifests(dataset_dir):
            splits = {}
            for name in ("train", "val", "test"):
                values = manifest["splits"][name]
                if not isinstance(values, list) or not values:
                    raise ValueError(f"{manifest_path}: {name} must be a nonempty subject list")
                splits[name] = {str(subject_id) for subject_id in values}
                if len(splits[name]) != len(values):
                    raise ValueError(f"{manifest_path}: duplicate {name} subject IDs")
            fold = int(manifest["fold"])
            oof_subjects = set(subject_ids[fold_ids == fold].tolist())
            row = {
                "dataset": dataset,
                "fold": fold,
                "fold_seed": int(manifest["fold_seed"]),
                "encoder_fit_subjects": len(splits["train"]),
                "validation_subjects": len(splits["val"]),
                "test_subjects": len(splits["test"]),
                "train_val_overlap": len(splits["train"] & splits["val"]),
                "train_test_overlap": len(splits["train"] & splits["test"]),
                "val_test_overlap": len(splits["val"] & splits["test"]),
                "oof_matches_manifest_test": oof_subjects == splits["test"],
                "matches_seed42_train_val_test_split": splits == expected[fold],
                "signature": manifest["signature"],
                "manifest_path": str(manifest_path.resolve()),
            }
            for overlap in ("train_val_overlap", "train_test_overlap", "val_test_overlap"):
                if row[overlap]:
                    raise ValueError(f"{manifest_path}: {overlap}={row[overlap]}")
            if not row["oof_matches_manifest_test"]:
                raise ValueError(f"{manifest_path}: test subjects do not match OOF fold {fold}")
            if set.union(*splits.values()) != all_subjects:
                raise ValueError(f"{manifest_path}: train/val/test do not partition the OOF cohort")
            rows.append(row)
            dataset_rows.append(row)
        summary[dataset] = {
            "folds": len(dataset_rows),
            "oof_subjects": len(subject_ids),
            "unique_oof_subjects": len(set(subject_ids.tolist())),
            "fold_ids": sorted(np.unique(fold_ids).tolist()),
            "all_train_test_overlaps_zero": all(
                row["train_test_overlap"] == 0 for row in dataset_rows
            ),
            "all_train_val_overlaps_zero": all(
                row["train_val_overlap"] == 0 for row in dataset_rows
            ),
            "all_val_test_overlaps_zero": all(
                row["val_test_overlap"] == 0 for row in dataset_rows
            ),
            "all_oof_subjects_match_manifest_tests": all(
                row["oof_matches_manifest_test"] for row in dataset_rows
            ),
            "all_folds_match_seed42_original_splitter": all(
                row["matches_seed42_train_val_test_split"] for row in dataset_rows
            ),
            "all_signatures_unique": len({row["signature"] for row in dataset_rows})
            == len(dataset_rows),
        }
    with (output_dir / "protocol_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "protocol_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
