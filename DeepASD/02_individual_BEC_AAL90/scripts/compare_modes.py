from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "balanced_accuracy",
    "auc",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_only_dir", required=True)
    parser.add_argument("--fmri_pheno_dir", required=True)
    parser.add_argument("--fmri_pheno_qcadv_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    experiments = {
        "fmri_only": Path(args.fmri_only_dir),
        "fmri_pheno": Path(args.fmri_pheno_dir),
        "fmri_pheno_qcadv": Path(args.fmri_pheno_qcadv_dir),
    }
    rows = []
    all_folds = []
    for mode, directory in experiments.items():
        frame = pd.read_csv(directory / "classification_metrics.csv")
        frame.insert(0, "mode", mode)
        all_folds.append(frame)
        network = pd.read_csv(directory / "fold_metrics.csv")
        row = {"mode": mode}
        for metric in METRICS:
            mean = frame[metric].mean() * 100
            std = frame[metric].std(ddof=0) * 100
            row[metric] = f"{mean:.2f}±{std:.2f}"
        for metric in [
            "edge_density",
            "directionality_index",
            "reciprocity",
        ]:
            mean = network[metric].mean() * 100
            std = network[metric].std(ddof=0) * 100
            row[metric] = f"{mean:.2f}±{std:.2f}"
        rows.append(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(
        output.with_name(output.stem + "_all_folds.csv"), index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
