"""Re-score legacy OOF archives without retraining or choosing a direction by score."""

from __future__ import annotations

import copy
import json
import shutil

import numpy as np

from .initial_ec import (
    METRICS, PAPER_ORIENTATION, THRESHOLD_PROTOCOL, metric_summary,
    score_subjects, write_csv, write_experiment_tables,
)


def rescore_network(source_dir, output_dir, network, paper_direction):
    name = f"Network{network}_amp"
    source = source_dir / name
    destination = output_dir / name
    with np.load(source / "oof_ec.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    expected = "extract_subject_ec returns attention.T; transpose back because decoder uses features @ attention"
    if original.get("orientation_conversion") != expected:
        raise ValueError(f"{source}: expected the known legacy double-transpose convention; refusing to guess")
    summary = copy.deepcopy(original)
    if paper_direction:
        data["oof_ec"] = data["oof_ec"].transpose(0, 2, 1).copy()
        summary.update(PAPER_ORIENTATION)
        note = "Paper-definition direction correction: removed the legacy extra transpose. All metrics, including AUPRC, are recomputed; not comparable as a threshold-only change. Direction was fixed by definition, not scores."
    else:
        summary["prediction_orientation"] = "legacy_attention_not_paper_initial_ec"
        note = "Threshold-only historical audit: legacy extra-transposed predictions retained; original AUPRC values preserved exactly. This is NOT the paper-definition direction and must not be used as the corrected submission table."
    scores = score_subjects(data["oof_ec"], data["ground_truth"])
    if not paper_direction:
        previous = data["auprc_scores"].copy()
        np.testing.assert_allclose(previous, [score["auprc"] for score in scores], rtol=0, atol=1e-14)
        for score, value in zip(scores, previous):
            score["auprc"] = float(value)
    for metric in METRICS:
        data[f"{metric}_scores"] = np.array([score[metric] for score in scores])
    data["subject_thresholds"] = np.array([score["threshold"] for score in scores])
    summary.pop("threshold", None)
    summary["threshold_protocol"] = dict(THRESHOLD_PROTOCOL)
    summary["rescore_note"] = note
    summary["source_directory"] = str(source.resolve())
    summary["original_configuration"] = summary.pop("configuration", {})
    summary["metrics"] = metric_summary(scores)
    if not paper_direction:
        summary["metrics"]["auprc"] = original["metrics"]["auprc"]
    rows = [
        {"subject_index": int(data["subject_indices"][index]),
         "subject_id": int(data["subject_ids"][index]),
         "fold": int(data["fold_ids"][index]), **score}
        for index, score in enumerate(scores)
    ]
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "oof_ec.npz", **data)
    write_csv(destination / "subject_metrics.csv", rows)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    shutil.copyfile(source / "folds.json", destination / "folds.json")
    return summary


def run_rescore(args):
    paths = [args.source_dir.resolve(), args.output_dir.resolve(), args.paper_output_dir.resolve()]
    if len(set(paths)) != 3 or any(first in second.parents for first in paths for second in paths if first != second):
        raise ValueError("Source and both output directories must be separate, non-nested paths")
    for output_dir, paper_direction in ((args.output_dir, False), (args.paper_output_dir, True)):
        summaries = [rescore_network(args.source_dir, output_dir, network, paper_direction) for network in dict.fromkeys(args.networks)]
        write_experiment_tables(output_dir, summaries)
        print(f"Wrote {output_dir / 'results.md'}")
