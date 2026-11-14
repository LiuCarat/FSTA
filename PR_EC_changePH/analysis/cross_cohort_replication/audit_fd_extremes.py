#!/usr/bin/env python3
"""Audit extreme motion covariates without changing EC or original statistics."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np

import run_covariate_adjusted_replication as adjusted


def extract_confounds(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subjects = ("sub-29880", "sub-29887", "sub-29888")
    def save_member(stream, member) -> None:
        path = Path(member.name)
        selected_subject = next((subject for subject in subjects if subject in member.name), None)
        selected = selected_subject and (
            path.name.endswith("_desc-confounds_timeseries.tsv")
            or path.name.endswith("_desc-confounds_timeseries.json")
        )
        if member.isfile() and (selected or path.name == "dataset_description.json") and member.size < 20_000_000:
            target_name = path.name if selected_subject else member.name.replace("/", "__")
            target = destination / (selected_subject or "metadata") / target_name
            target.parent.mkdir(exist_ok=True)
            with stream.extractfile(member) as source, target.open("wb") as output:
                output.write(source.read())
            print(f"Extracted {member.name}", flush=True)

    with subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
                for member in stream:
                    path = Path(member.name)
                    print(f"Outer archive entry: {member.name} ({member.size} bytes)", flush=True)
                    if path.name == "EMC_1.tar.gz":
                        with stream.extractfile(member) as compressed:
                            with tarfile.open(fileobj=compressed, mode="r|gz") as nested:
                                for nested_member in nested:
                                    save_member(nested, nested_member)
                        break
                    save_member(stream, member)
        finally:
            if process.poll() is None:
                process.terminate()


def sensitivity(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    original_dir = adjusted.DEFAULT_OUTPUT_DIR.parent / "covariate_adjusted_outputs"
    with (original_dir / "original_213_comparison.csv").open() as handle:
        original = list(csv.DictReader(handle))
    original_keys = {(int(row["SourceIndex"]), int(row["TargetIndex"])) for row in original}
    original_sign = {(int(row["SourceIndex"]), int(row["TargetIndex"])): np.sign(float(row["Original_ABIDE_I_Hedges_g"])) for row in original}
    scenarios = ("raw_fd", "exclude_two_flagged", "exclude_fd_gt_2", "log1p_fd")
    results = {scenario: {} for scenario in scenarios}
    influence_rows = []
    cohort_summaries = []
    for cohort, archive_path in adjusted.DEFAULT_INPUTS.items():
        ec, labels, names = adjusted.load_archive(archive_path, "ec")
        audit, complete = adjusted.align_covariates(archive_path, adjusted.DEFAULT_PHENOTYPES[cohort], labels)
        included = [row for row in audit if row["Included"]]
        sites = np.array([row["Site"] for row in included])
        nuisance = np.column_stack([
            np.ones(len(included)), labels[complete],
            *[np.array([row[key] for row in included]) for key in ("Age", "Sex")],
            *[(sites == site).astype(float) for site in sorted(set(sites))[1:]],
        ])
        fd = np.array([row["Mean_FD"] for row in included])
        residual_fd = fd - nuisance @ np.linalg.lstsq(nuisance, fd, rcond=None)[0]
        design = np.column_stack([nuisance, fd])
        orthogonal, _ = np.linalg.qr(design, mode="reduced")
        leverage = np.sum(orthogonal ** 2, axis=1)
        partial_leverage = residual_fd ** 2 / np.sum(residual_fd ** 2)
        for index, row in enumerate(included):
            influence_rows.append({
                "Cohort": cohort, "SubjectID": row["SubjectID"], "Site": row["Site"],
                "ASD": row["ArchiveLabel"], "Mean_FD": row["Mean_FD"],
                "DesignLeverage": float(leverage[index]),
                "FDPartialLeverage": float(partial_leverage[index]),
            })
        for scenario in scenarios:
            scenario_audit = copy.deepcopy(audit)
            for row in scenario_audit:
                if scenario == "exclude_two_flagged" and row["SubjectID"] in {"sub-29887", "sub-29888"}:
                    row["Included"] = False
                if scenario == "exclude_fd_gt_2" and row["Mean_FD"] > 2:
                    row["Included"] = False
                if scenario == "log1p_fd":
                    row["Mean_FD"] = float(np.log1p(row["Mean_FD"]))
            keep = np.array([row["Included"] for row in scenario_audit], dtype=bool)
            beta, standard_error, p_values, q_values, diagnostics = adjusted.fit_edges(ec[keep], labels[keep], scenario_audit)
            source, target = np.where(~np.eye(ec.shape[1], dtype=bool))
            rows = [{
                "SourceIndex": int(source_index), "TargetIndex": int(target_index),
                "ASD_Beta": float(beta[index]), "ASD_SE": float(standard_error[index]),
                "PValue": float(p_values[index]), "QValue": float(q_values[index]),
            } for index, (source_index, target_index) in enumerate(zip(source, target))]
            adjusted.write_csv(destination / f"{scenario}_{cohort.lower()}_edges.csv", rows, list(rows[0]))
            results[scenario][cohort] = rows
            cohort_summaries.append({
                "Scenario": scenario, "Cohort": cohort, "N": int(keep.sum()),
                "ASD_N": int(np.sum(labels[keep] == 1)), "HC_N": int(np.sum(labels[keep] == 0)),
                "ExcludedFromOriginalN": int(np.sum(~keep)),
                "ExcludedIDs": ";".join(row["SubjectID"] for row in scenario_audit if not row["Included"]),
                "SignificantEdges": int(np.sum(q_values < .05)),
            })
            if scenario == "raw_fd":
                with (original_dir / f"{cohort.lower()}_edge_statistics.csv").open() as handle:
                    previous = list(csv.DictReader(handle))
                np.testing.assert_allclose(beta, [float(row["ASD_Beta"]) for row in previous], rtol=1e-10, atol=1e-14)
                np.testing.assert_allclose(q_values, [float(row["QValue"]) for row in previous], rtol=1e-10, atol=1e-14)
    summaries = []
    for scenario in scenarios:
        merged = []
        for left, right in zip(results[scenario]["ABIDE-I"], results[scenario]["ABIDE-II"]):
            key = (left["SourceIndex"], left["TargetIndex"])
            same_direction = bool(left["ASD_Beta"] * right["ASD_Beta"] > 0)
            replicated = same_direction and left["QValue"] < .05 and right["QValue"] < .05
            merged.append({
                "SourceIndex": key[0], "TargetIndex": key[1],
                "ABIDE_I_Beta": left["ASD_Beta"], "ABIDE_II_Beta": right["ASD_Beta"],
                "ABIDE_I_QValue": left["QValue"], "ABIDE_II_QValue": right["QValue"],
                "SameDirection": same_direction, "Replicated": replicated,
                "OriginallyReplicated": key in original_keys,
                "PreservesOriginalDirection": bool(key in original_keys and same_direction and np.sign(left["ASD_Beta"]) == original_sign[key]),
            })
        comparison = [row for row in merged if row["OriginallyReplicated"]]
        summary = {
            "Scenario": scenario,
            "ReplicatedN": sum(row["Replicated"] for row in merged),
            "OriginalRetainedN": sum(row["Replicated"] for row in comparison),
            "OriginalSameDirectionN": sum(row["SameDirection"] for row in comparison),
            "OriginalPreservesDirectionN": sum(row["PreservesOriginalDirection"] for row in comparison),
        }
        summaries.append(summary)
        adjusted.write_csv(destination / f"{scenario}_cross_cohort.csv", merged, list(merged[0]))
        print(summary, flush=True)
    adjusted.write_csv(destination / "sensitivity_summary.csv", summaries, list(summaries[0]))
    adjusted.write_csv(destination / "cohort_samples.csv", cohort_summaries, list(cohort_summaries[0]))
    adjusted.write_csv(destination / "subject_leverage.csv", influence_rows, list(influence_rows[0]))
    print("Flagged-subject leverage:", [row for row in influence_rows if row["SubjectID"] in {"sub-29887", "sub-29888"}], flush=True)


def summarize_confounds(destination: Path) -> None:
    import pandas as pd

    source_path = adjusted.ROOT / "dataset/ABIDE-II/scripts/extract_aal90.py"
    spec = importlib.util.spec_from_file_location("extract_aal90", source_path)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    with adjusted.DEFAULT_PHENOTYPES["ABIDE-II"].open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        phenotype = {row["FILE_ID"].strip(): row for row in reader}
    summaries = []
    for path in sorted(destination.glob("sub-*/*_desc-confounds_timeseries.tsv")):
        subject = path.parent.name
        frame = pd.read_csv(path, sep="\t")
        fd = pd.to_numeric(frame["framewise_displacement"], errors="coerce").to_numpy()
        motion = frame[list(extractor.MOTION_COLUMNS)].to_numpy(dtype=float)
        differences = np.abs(np.diff(motion, axis=0))
        power_fd = np.sum(differences[:, :3], axis=1) + 50 * np.sum(differences[:, 3:], axis=1)
        finite = np.isfinite(power_fd) & np.isfinite(fd[1:])
        recomputed = extractor.qc_values(frame)
        np.testing.assert_allclose(recomputed["func_mean_fd"], float(phenotype[subject]["func_mean_fd"]), rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(power_fd[finite], fd[1:][finite], rtol=1e-12, atol=1e-10)
        summary = {
            "SubjectID": subject, "confounds_file": str(path), "sha256": adjusted.file_hash(path),
            "n_frames": len(frame), "n_nonfinite_fd": int(np.sum(~np.isfinite(fd))),
            "csv_mean_fd": float(phenotype[subject]["func_mean_fd"]),
            "mean_fd_first_nan_zero": float(np.nan_to_num(fd, nan=0).mean()),
            "mean_fd_finite_only": float(np.nanmean(fd)),
            "median_fd": float(np.nanmedian(fd)), "max_fd": float(np.nanmax(fd)),
            "frames_fd_gt_1": int(np.sum(fd > 1)), "frames_fd_gt_10": int(np.sum(fd > 10)),
            "frames_fd_gt_100": int(np.sum(fd > 100)),
            "power_50mm_reconstruction_max_abs_error": float(np.max(np.abs(power_fd[finite] - fd[1:][finite]))),
            "max_translation_step_each_axis": np.max(differences[:, :3], axis=0).tolist(),
            "max_rotation_step_each_axis": np.max(differences[:, 3:], axis=0).tolist(),
            "top_fd_frames_zero_based": np.argsort(np.nan_to_num(fd, nan=-1))[-10:][::-1].tolist(),
            "top_fd_values": np.sort(fd[np.isfinite(fd)])[-10:][::-1].tolist(),
            "qc_from_current_extraction_code": recomputed,
            "csv_dvars": float(phenotype[subject]["func_dvars"]),
            "csv_quality": float(phenotype[subject]["func_quality"]),
        }
        json_path = path.with_suffix(".json")
        if json_path.exists():
            metadata = json.loads(json_path.read_text())
            summary["fd_metadata"] = metadata.get("framewise_displacement")
            summary["motion_metadata"] = {name: metadata.get(name) for name in extractor.MOTION_COLUMNS}
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    with (destination.parent / "raw_confounds_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2, allow_nan=False)
        handle.write("\n")


def edge_influence(destination: Path) -> None:
    ec, labels, _ = adjusted.load_archive(adjusted.DEFAULT_INPUTS["ABIDE-II"], "ec")
    audit, complete = adjusted.align_covariates(
        adjusted.DEFAULT_INPUTS["ABIDE-II"], adjusted.DEFAULT_PHENOTYPES["ABIDE-II"], labels,
    )
    included = [row for row in audit if row["Included"]]
    sites = np.array([row["Site"] for row in included])
    design = np.column_stack([
        np.ones(len(included)), labels[complete],
        *[np.array([row[key] for row in included]) for key in ("Age", "Sex", "Mean_FD")],
        *[(sites == site).astype(float) for site in sorted(set(sites))[1:]],
    ])
    source, target = np.where(~np.eye(ec.shape[1], dtype=bool))
    response = ec[complete][:, source, target]
    full_beta = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ full_beta
    rank = design.shape[1]
    residual_variance = np.sum(residual ** 2, axis=0) / (len(design) - rank)
    orthogonal, _ = np.linalg.qr(design, mode="reduced")
    leverage = np.sum(orthogonal ** 2, axis=1)
    flagged_ids = ("sub-29887", "sub-29888")
    keep = np.array([row["SubjectID"] not in flagged_ids for row in included])
    reduced_beta = np.linalg.lstsq(design[keep], response[keep], rcond=None)[0]
    fd_standard_error = np.sqrt(residual_variance * np.linalg.inv(design.T @ design)[4, 4])
    cooks = {}
    for subject in flagged_ids:
        subject_index = next(index for index, row in enumerate(included) if row["SubjectID"] == subject)
        cooks[subject] = residual[subject_index] ** 2 / (rank * residual_variance) * leverage[subject_index] / (1 - leverage[subject_index]) ** 2
    rows = [{
        "SourceIndex": int(source_index), "TargetIndex": int(target_index),
        "RawFD_ASD_Beta": float(full_beta[1, index]),
        "ExcludeTwo_ASD_Beta": float(reduced_beta[1, index]),
        "RawFD_FD_Beta": float(full_beta[4, index]),
        "ExcludeTwo_FD_Beta": float(reduced_beta[4, index]),
        "FD_BetaChangeInFullModelSE": float((reduced_beta[4, index] - full_beta[4, index]) / fd_standard_error[index]),
        **{f"CooksD_{subject}": float(values[index]) for subject, values in cooks.items()},
    } for index, (source_index, target_index) in enumerate(zip(source, target))]
    adjusted.write_csv(destination / "abide_ii_edge_influence.csv", rows, list(rows[0]))
    summary = {
        "fd_beta_cross_edge_correlation_full_vs_exclude_two": float(np.corrcoef(full_beta[4], reduced_beta[4])[0, 1]),
        "fd_beta_change_full_se_absolute_quantiles_50_90_99_100": np.quantile(np.abs((reduced_beta[4] - full_beta[4]) / fd_standard_error), [.5, .9, .99, 1]).tolist(),
        "asd_beta_cross_edge_correlation_full_vs_exclude_two": float(np.corrcoef(full_beta[1], reduced_beta[1])[0, 1]),
        "cooks_distance": {subject: {
            "quantiles_50_90_99_100": np.quantile(values, [.5, .9, .99, 1]).tolist(),
            "edges_gt_4_over_n": int(np.sum(values > 4 / len(design))),
        } for subject, values in cooks.items()},
        "note": "Cook distances are per single observation; the FD coefficient comparison deletes both flagged observations. Full-model SE scales the two-observation coefficient change, not a formal single-observation DFBETAS.",
    }
    with (destination / "edge_influence_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-archive", type=Path)
    parser.add_argument("--summarize-confounds", action="store_true")
    parser.add_argument("--edge-influence", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.extract_archive:
        extract_confounds(args.extract_archive, args.output_dir)
    elif args.summarize_confounds:
        summarize_confounds(args.output_dir)
    elif args.edge_influence:
        edge_influence(args.output_dir)
    else:
        sensitivity(args.output_dir)


if __name__ == "__main__":
    main()
