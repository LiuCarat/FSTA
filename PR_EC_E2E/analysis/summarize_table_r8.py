"""Table R8: supplied paper baseline plus completed, measured W7 classification."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PR_EC_E2E.w7 import (
    OUTPUT_ROOT, read_reference_metadata, validate_initial_ec_mode, write_csv, write_json,
)

PAPER_BASELINE = {
    "abide-i": ("66.20 ± 4.32", "60.39 ± 5.12", "66.85 ± 4.32"),
    "abide-ii": ("66.53 ± 3.68", "60.62 ± 5.14", "66.83 ± 3.92"),
    "adhd200": ("60.68 ± 6.19", "58.62 ± 6.36", "61.21 ± 5.85"),
}
DISPLAY_NAMES = {"abide-i": "ABIDE-I", "abide-ii": "ABIDE-II", "adhd200": "ADHD200"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def display(values):
    if len(values) != 10 or not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
        raise ValueError("Expected ten finite fold percentages in [0,100]")
    return f"{float(np.mean(values)):.2f} ± {float(np.std(values, ddof=0)):.2f}"


def measured_result(directory, dataset):
    complete = json.loads((directory / "downstream_complete.json").read_text())
    config = json.loads((directory / "protocol_config.json").read_text())
    if (complete["stf_fusion_mode"] != "subject_specific" or complete["seed"] != 42
            or complete["n_splits"] != 10 or config["stf_fusion_mode"] != "subject_specific"
            or config["stf_checkpoint"] != "final"):
        raise ValueError(f"Not the requested completed W7 protocol: {directory}")
    config["reference_ec_path"] = Path(config["reference_ec_path"])
    config["reference_oof_path"] = Path(config["reference_oof_path"])
    args = SimpleNamespace(**config)
    with np.load(config["ec_path"], allow_pickle=False) as archive:
        initial = {key: archive[key] for key in ("stf_fusion_mode", "stf_config_json", "subject_ids", "labels", "site_ids")}
        if "roi_names" in archive:
            initial["roi_names"] = archive["roi_names"]
    validate_initial_ec_mode(args, initial)
    reference = read_reference_metadata(args)
    with np.load(config["qsr_refined_ec_path"], allow_pickle=False) as archive:
        for key in ("subject_ids", "labels", "site_ids"):
            if not np.array_equal(initial[key], reference[key]) or not np.array_equal(initial[key], archive[key]):
                raise ValueError(f"Cohort identity differs: {key}")
        if not np.array_equal(archive["fold_ids"], reference["fold_ids"]):
            raise ValueError("Subject-specific and original OOF test folds differ")
        if "roi_names" in reference and not np.array_equal(reference["roi_names"], archive["roi_names"]):
            raise ValueError("ROI order differs from the original")
    rows = read_csv(directory / "classification_folds.csv")
    if len(rows) != 30:
        raise ValueError("Expected ten folds for each of original/refined/qc_refined")
    for row in rows:
        if row["dataset"] != dataset or row["training_fusion"] != "subject_specific" or int(row["seed"]) != 42:
            raise ValueError("Classification fold metadata does not match W7")
    values = {}
    for representation in ("original", "refined", "qc_refined"):
        selected = [row for row in rows if row["representation"] == representation]
        if len(selected) != 10 or {int(row["fold"]) for row in selected} != set(range(1, 11)):
            raise ValueError("Missing or duplicate classification folds")
        for metric in ("ACC", "AUC", "SEN", "PRE", "F1"):
            values[(representation, metric)] = display([float(row[f"{metric}_pct"]) for row in selected])
    summaries = read_csv(directory / "classification_summary.csv")
    if len(summaries) != 3:
        raise ValueError("Missing classification representation summaries")
    for row in summaries:
        if int(row["n_folds"]) != 10 or int(row["std_ddof"]) != 0:
            raise ValueError("Classification summary uses a different aggregation rule")
        for metric in ("ACC", "AUC", "SEN", "PRE", "F1"):
            if row[f"{metric}_display"] != values[(row["representation"], metric)]:
                raise ValueError("Classification summaries do not match the actual fold metrics")
    return (values[("original", "AUC")], values[("qc_refined", "ACC")], values[("qc_refined", "AUC")])


def original_result(directory):
    config = json.loads((directory / "summary.json").read_text())["config"]
    if config["seed"] != 42 or config["n_splits"] != 10:
        raise ValueError("Original saved classification must be seed42 / ten folds")
    rows = read_csv(directory / "experiment_summary.csv")
    if len(rows) != 10 or {int(row["fold"]) for row in rows} != set(range(1, 11)):
        raise ValueError("Original results do not contain ten unique folds")
    return tuple(display([100 * float(row[key]) for row in rows])
                 for key in ("original_AUC", "qc_refined_ACC", "qc_refined_AUC"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--original-source", choices=("paper", "saved"), default="paper",
                        help="paper uses the baseline numbers explicitly supplied in the W7 request")
    parser.add_argument("--original-results-root", type=Path, default=REPO_ROOT / "PR_EC/outputs/seed42")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.root / "table_r8.csv").resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error("All Table R8 writes must stay under PR_EC_E2E/outputs/rebuttal_w7")
    rows, sources = [], {}
    for dataset in PAPER_BASELINE:
        baseline = (PAPER_BASELINE[dataset] if args.original_source == "paper"
                    else original_result(args.original_results_root / dataset))
        measured = measured_result(args.root / dataset / "subject_specific", dataset)
        for fusion, values in (("Batch-averaged", baseline), ("Subject-specific", measured)):
            rows.append({"Dataset": DISPLAY_NAMES[dataset], "Training fusion": fusion,
                         "Initial EC AUC (%)": values[0], "Full PR-EC ACC (%)": values[1],
                         "Full PR-EC AUC (%)": values[2]})
        sources[dataset] = {
            "baseline": "user-supplied paper Table R8 baseline" if args.original_source == "paper"
                        else str(args.original_results_root / dataset / "experiment_summary.csv"),
            "subject_specific": str(args.root / dataset / "subject_specific/classification_folds.csv"),
            "std_ddof": 0,
        }
    write_csv(output, rows)
    write_json(output.with_suffix(".sources.json"), sources)
    print(output)


if __name__ == "__main__":
    main()
