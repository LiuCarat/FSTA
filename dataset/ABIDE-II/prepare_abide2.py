"""Prepare one ABIDE-II BIDS view and Graph-BEC phenotype table."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


ALIASES = {
    "site": ("site_id", "SITE_ID", "site"),
    "subject": ("participant_id", "SUB_ID", "subject_id", "subject"),
    "dx": ("dx_group", "DX_GROUP", "diagnosis", "DX"),
    "age": ("age_at_scan", "AGE_AT_SCAN", "age_at_scan "),
    "sex": ("sex", "SEX", "gender", "Gender"),
    "fiq": ("fiq", "FIQ", "Full4 IQ"),
    "viq": ("viq", "VIQ", "Verbal IQ"),
    "piq": ("piq", "PIQ", "Performance IQ"),
}
OUTPUT_COLUMNS = (
    "FILE_ID", "SUB_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX",
    "FIQ", "VIQ", "PIQ", "func_mean_fd", "func_fd_gt_0_2",
    "func_fd_gt_0_5", "func_mean_dvars", "func_quality",
)


def find_column(fieldnames, aliases, required=True):
    normalized = {str(name).strip().lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.strip().lower() in normalized:
            return normalized[alias.strip().lower()]
    if required:
        raise ValueError(f"Missing field; tried {aliases}")
    return None


def normalize_subject(value):
    subject = str(value).strip()
    if subject.endswith(".0"):
        subject = subject[:-2]
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def numeric(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(result) else f"{result:g}"


def build_phenotype(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    if not source:
        raise ValueError(f"No phenotype rows found in {input_path}")
    columns = {
        name: find_column(source[0], aliases, required=name in {"site", "subject", "dx", "age", "sex"})
        for name, aliases in ALIASES.items()
    }
    rows = {}
    for row in source:
        subject = normalize_subject(row[columns["subject"]])
        raw_dx = numeric(row[columns["dx"]])
        if raw_dx not in {"1", "2"}:
            continue
        site = str(row[columns["site"]]).strip().removeprefix("ABIDEII-")
        output = {
            "FILE_ID": subject,
            "SUB_ID": subject,
            "SITE_ID": site,
            "DX_GROUP": "2" if raw_dx == "1" else "1",
            "AGE_AT_SCAN": numeric(row[columns["age"]]),
            "SEX": numeric(row[columns["sex"]]),
            "FIQ": numeric(row[columns["fiq"]]) if columns["fiq"] else "",
            "VIQ": numeric(row[columns["viq"]]) if columns["viq"] else "",
            "PIQ": numeric(row[columns["piq"]]) if columns["piq"] else "",
            "func_mean_fd": "",
            "func_fd_gt_0_2": "",
            "func_fd_gt_0_5": "",
            "func_mean_dvars": "",
            "func_quality": "",
        }
        rows[subject] = output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())
    print(f"Phenotype rows: {len(rows)}")
    print(f"Saved: {output_path.resolve()}")


def prepare_bids(raw_root: Path, bids_root: Path, copy_files: bool) -> None:
    subjects = []
    bids_root.mkdir(parents=True, exist_ok=True)
    for site_root in sorted(raw_root.glob("ABIDEII-*")):
        if not site_root.is_dir():
            continue
        site = site_root.name.removeprefix("ABIDEII-")
        for subject_root in sorted(site_root.glob("sub-*")):
            if not subject_root.is_dir():
                continue
            subject = subject_root.name
            destination = bids_root / subject
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != subject_root.resolve():
                    raise FileExistsError(f"Subject collision: {subject}")
            elif copy_files:
                shutil.copytree(subject_root, destination)
            else:
                destination.symlink_to(subject_root.resolve(), target_is_directory=True)
            subjects.append((site, subject))
    (bids_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "ABIDE-II",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
                "GeneratedBy": [{"Name": "ABIDE-II preparation"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = bids_root / "abideii_bids_manifest.tsv"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("site\tsubject\n")
        handle.writelines(f"{site}\t{subject}\n" for site, subject in subjects)
    print(f"BIDS subjects: {len(subjects)}")
    print(f"BIDS root: {bids_root.resolve()}")
    print(f"Manifest: {manifest.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("ABIDEII/raw"))
    parser.add_argument("--bids-root", type=Path, default=Path("ABIDEII/bids"))
    parser.add_argument("--phenotype-input", type=Path, default=Path("ABIDEII/phenotype/ABIDEII_phenotype_merged.csv"))
    parser.add_argument("--phenotype-output", type=Path, default=Path("dataset/ABIDE-II/ABIDEII_phenotype_graphbec.csv"))
    parser.add_argument("--copy", action="store_true", help="copy subjects instead of creating symlinks")
    args = parser.parse_args()
    prepare_bids(args.raw_root, args.bids_root, args.copy)
    build_phenotype(args.phenotype_input, args.phenotype_output)


if __name__ == "__main__":
    main()
