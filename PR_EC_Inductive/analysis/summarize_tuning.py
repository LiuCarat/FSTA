"""Rank validation-only inductive tuning runs without reading test outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASET_DIRS = ("abide-i", "abide-ii", "adhd200")
OBJECTIVES = ("original", "refined", "qc_refined")
PARAMETERS = (
    "graph_mode",
    "window_length",
    "stride",
    "epochs",
    "stf_checkpoint",
    "loss_alpha",
    "batch_size",
    "d_model",
    "d_inner_hid",
    "d_k",
    "d_v",
    "n_head",
    "dropout",
    "n_warmup_steps",
    "lr_mul",
    "weight_decay",
    "no_filters",
    "reference_k",
    "fusion_beta",
    "reference_bandwidth",
    "categorical_penalty",
    "continuous_weights",
    "refiner_epochs",
    "refiner_lr",
    "gate_max",
    "gate_l1_weight",
    "anchor_weight",
    "variance_weight",
    "variance_retention",
    "qsr_epochs",
    "qsr_lr",
    "qsr_hidden_channels",
    "qsr_eta",
    "qsr_r_max",
    "qsr_perturbation_scale",
    "qsr_gate_max",
    "qsr_gate_weight",
    "qsr_variance_weight",
    "qsr_variance_retention",
    "qsr_basis_ridge",
    "classifier_epochs",
    "classifier_patience",
    "classifier_lr",
)


def collect(dataset_dir: Path, objective: str, include_smoke=False) -> list[dict]:
    rows = []
    for summary_path in sorted((dataset_dir / "tuning").glob("*/tuning_summary.json")):
        if not include_smoke and "smoke" in summary_path.parent.name.lower():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = f"{objective}_AUC_mean"
        if key not in summary:
            continue
        config = summary["config"]
        row = {
            "run_name": summary_path.parent.name,
            "objective": objective,
            "validation_auc": float(summary[key]),
            "validation_auc_percent": 100.0 * float(summary[key]),
            "folds": len(summary["folds"]),
            "path": str(summary_path.parent.resolve()),
        }
        for parameter in PARAMETERS:
            row[parameter] = config.get(parameter)
        rows.append(row)
    return sorted(rows, key=lambda row: row["validation_auc"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset", choices=DATASET_DIRS, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="qc_refined")
    parser.add_argument("--outputs-dir", type=Path, default=project_dir / "outputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.outputs_dir / args.dataset
    rows = collect(dataset_dir, args.objective, args.include_smoke)
    if not rows:
        raise FileNotFoundError(
            f"No tuning summaries for {args.dataset} and objective {args.objective}"
        )
    output = args.output or dataset_dir / "tuning" / f"ranking_{args.objective}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rank | run | validation {args.objective} AUC")
    for rank, row in enumerate(rows, 1):
        print(f"{rank:4d} | {row['run_name']} | {row['validation_auc_percent']:.2f}%")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
