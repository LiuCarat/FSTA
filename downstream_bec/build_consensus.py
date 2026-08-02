import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


def safe_correlation(first, second):
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="downstream_bec/outputs/consensus")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [np.load(path, allow_pickle=False) for path in args.inputs]
    reference_ids = datasets[0]["subject_ids"].astype(str)
    reference_labels = datasets[0]["labels"]
    reference_rois = datasets[0]["roi_names"].astype(str)
    for path, dataset in zip(args.inputs[1:], datasets[1:]):
        if not np.array_equal(reference_ids, dataset["subject_ids"].astype(str)):
            raise ValueError(f"Subject order differs in {path}")
        if not np.array_equal(reference_labels, dataset["labels"]):
            raise ValueError(f"Labels differ in {path}")
        if not np.array_equal(reference_rois, dataset["roi_names"].astype(str)):
            raise ValueError(f"ROI names differ in {path}")

    all_raw_attention = np.stack([dataset["raw_attention"] for dataset in datasets])
    all_bec = np.stack([dataset["bec"] for dataset in datasets])
    consensus_raw_attention = all_raw_attention.mean(axis=0)
    consensus_bec = all_bec.mean(axis=0)

    np.savez_compressed(
        output_dir / "consensus_bec.npz",
        raw_attention=consensus_raw_attention,
        bec=consensus_bec,
        labels=reference_labels,
        subject_ids=reference_ids,
        roi_names=reference_rois,
    )

    mask = ~np.eye(consensus_bec.shape[-1], dtype=bool)
    rows = []
    for first_index, second_index in combinations(range(len(datasets)), 2):
        subject_correlations = [
            safe_correlation(
                all_bec[first_index, subject_index][mask],
                all_bec[second_index, subject_index][mask],
            )
            for subject_index in range(len(reference_ids))
        ]
        group_correlation = safe_correlation(
            all_bec[first_index].mean(axis=0)[mask],
            all_bec[second_index].mean(axis=0)[mask],
        )
        rows.append(
            {
                "first_input": args.inputs[first_index],
                "second_input": args.inputs[second_index],
                "mean_subject_correlation": float(np.nanmean(subject_correlations)),
                "std_subject_correlation": float(np.nanstd(subject_correlations)),
                "group_mean_correlation": group_correlation,
            }
        )

    with (output_dir / "bec_stability.csv").open("w", newline="") as stability_file:
        writer = csv.DictWriter(stability_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved consensus BEC and {len(rows)} pairwise stability comparisons")


if __name__ == "__main__":
    main()
