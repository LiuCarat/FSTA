#!/usr/bin/env python3
"""Validate and normalize the local ADHD200 AAL time-series dataset.

The script never deletes or edits the raw dataset. It creates a cleaned view
under ``cleaned/`` using symlinks, normalizes subject IDs, validates phenotype
and ROI matches, and writes manifests for data that is still unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

VALID_DX = {"0", "1", "2", "3"}
EXCLUDED_SUBJECTS = set()


def normalize_id(value: str) -> str:
    value = str(value).strip()
    if value.isdigit():
        return str(int(value))
    return value


def normalize_subject_key(value: str) -> str:
    site, separator, subject = str(value).strip().partition("/")
    if not separator:
        return normalize_id(value)
    return f"{site}/{normalize_id(subject)}"


NORMALIZED_EXCLUDED_SUBJECTS = {
    normalize_subject_key(subject_id) for subject_id in EXCLUDED_SUBJECTS
}


def read_phenotype(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = {"ScanDir ID", "Site", "DX"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Missing phenotype columns: {sorted(missing)}")
    by_id = {}
    duplicates = []
    for row in rows:
        subject_id = normalize_id(row["ScanDir ID"])
        if subject_id in by_id:
            duplicates.append(subject_id)
        by_id[subject_id] = row
        row["ScanDir ID"] = subject_id
    if duplicates:
        raise ValueError(f"Duplicate normalized phenotype IDs: {sorted(set(duplicates))}")
    return fieldnames, rows, by_id


def choose_roi_file(subject_dir: Path):
    candidates = sorted(
        path for path in subject_dir.glob("*_aal_TCs.1D")
        if "*" not in path.name and path.stat().st_size > 0
    )
    if not candidates:
        return None
    preferred = [path for path in candidates if path.name.startswith("sfnwmrda")]
    return (preferred or candidates)[0]


def validate_roi(path: Path, source_roi_count: int = 116):
    try:
        values = np.loadtxt(
            path,
            dtype=np.float32,
            skiprows=1,
            usecols=np.arange(2, 2 + source_roi_count),
        )
    except Exception as error:
        return False, None, f"read_error: {error}"
    if values.ndim != 2 or values.shape[1] != source_roi_count:
        return False, tuple(values.shape), "wrong_roi_shape"
    if not np.isfinite(values).all():
        return False, tuple(values.shape), "non_finite"
    return True, tuple(values.shape), "ok"


def write_tsv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, target: Path, copy_files: bool):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if copy_files:
        shutil.copytree(source, target)
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def clean_dataset(input_root: Path, output_root: Path, copy_files: bool = False):
    phenotype_path = input_root / "adhd200_preprocessed_phenotypics.tsv"
    raw_root = input_root / "AAL_TCs_filtfix"
    phenotype_fields, phenotype_rows, phenotype_by_id = read_phenotype(phenotype_path)

    output_root.mkdir(parents=True, exist_ok=True)
    clean_aal_root = output_root / "AAL_TCs_filtfix"
    clean_aal_root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    matched_ids = set()
    selected_rows = []
    status_counts = Counter()

    for site_dir in sorted(path for path in raw_root.iterdir() if path.is_dir() and path.name != "templates"):
        clean_site_dir = clean_aal_root / site_dir.name
        clean_site_dir.mkdir(parents=True, exist_ok=True)
        for subject_dir in sorted(path for path in site_dir.iterdir() if path.is_dir()):
            normalized_id = normalize_id(subject_dir.name)
            subject_key = f"{site_dir.name}/{normalized_id}"
            row = phenotype_by_id.get(normalized_id)
            status = "ok"
            roi_path = choose_roi_file(subject_dir)
            shape = None
            reason = ""
            if subject_key in NORMALIZED_EXCLUDED_SUBJECTS:
                status, reason = "excluded", "known_empty_subject"
            elif row is None:
                status, reason = "unmatched", "no_phenotype_or_invalid_id"
            elif str(row["DX"]).strip() not in VALID_DX:
                status, reason = "invalid_dx", f"DX={row['DX']}"
            elif roi_path is None:
                status, reason = "missing_roi", "no_aal_TCs_file"
            else:
                valid, shape, reason = validate_roi(roi_path)
                if not valid:
                    status = "invalid_roi"
                else:
                    matched_ids.add(normalized_id)
                    selected_row = dict(row)
                    selected_row["Site"] = str(row["Site"]).strip()
                    selected_rows.append(selected_row)
                    link_or_copy(subject_dir, clean_site_dir / normalized_id, copy_files)
            status_counts[status] += 1
            report_rows.append({
                "site": site_dir.name,
                "subject_id": normalized_id,
                "subject_key": subject_key,
                "status": status,
                "reason": reason,
                "roi_file": str(roi_path) if roi_path else "",
                "time_points": shape[0] if shape else "",
                "roi_columns": shape[1] if shape else "",
            })

    missing_rows = []
    for row in phenotype_rows:
        subject_id = normalize_id(row["ScanDir ID"])
        dx = str(row["DX"]).strip()
        if dx in VALID_DX and subject_id not in matched_ids:
            missing_rows.append({
                "subject_id": subject_id,
                "site_code": row["Site"],
                "dx": dx,
                "reason": "no_valid_local_aal_time_series",
            })

    download_manifest = output_root / "download_manifest.tsv"
    with download_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["site", "subject_id", "dx", "url", "filename", "sha256"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in missing_rows:
            writer.writerow({
                "site": row["site_code"],
                "subject_id": row["subject_id"],
                "dx": row["dx"],
                "url": "",
                "filename": "",
                "sha256": "",
            })

    # Keep phenotype rows aligned to the cleaned subject set and preserve order.
    selected_by_id = {row["ScanDir ID"]: row for row in selected_rows}
    selected_rows = [selected_by_id[key] for key in sorted(selected_by_id)]
    write_tsv(output_root / "adhd200_phenotypics_clean.tsv", phenotype_fields, selected_rows)

    with (output_root / "subject_manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(report_rows)
    with (output_root / "missing_subjects.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "site_code", "dx", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(missing_rows)

    valid_lengths = [
        int(row["time_points"])
        for row in report_rows
        if row["status"] == "ok" and row["time_points"]
    ]
    summary = {
        "phenotype_rows": len(phenotype_rows),
        "valid_dx_rows": sum(str(row["DX"]).strip() in VALID_DX for row in phenotype_rows),
        "clean_subjects": len(selected_rows),
        "missing_valid_subjects": len(missing_rows),
        "min_time_points": min(valid_lengths) if valid_lengths else None,
        "max_time_points": max(valid_lengths) if valid_lengths else None,
        "status_counts": dict(status_counts),
        "copy_files": copy_files,
    }
    (output_root / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--copy-files", action="store_true", help="Copy files instead of creating symlinks")
    args = parser.parse_args()
    output_root = args.output_root or args.input_root / "cleaned"
    clean_dataset(args.input_root, output_root, args.copy_files)


if __name__ == "__main__":
    main()
