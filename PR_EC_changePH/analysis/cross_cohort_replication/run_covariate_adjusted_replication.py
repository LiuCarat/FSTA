#!/usr/bin/env python3
"""Repeat cross-cohort replication using covariate-adjusted, edgewise OLS."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from run_cross_cohort_replication import (
    ASD_LABEL,
    DEFAULT_INPUTS,
    DEFAULT_OUTPUT_DIR,
    ROOT,
    benjamini_hochberg,
    compute_effects_and_tests,
    edge_table,
    load_archive,
    merge_tables,
    write_csv,
)


FORMULA = "EC_edge ~ ASD + Age + Sex + C(Site) + Mean_FD"
DEFAULT_PHENOTYPES = {
    cohort: ROOT / f"dataset/{cohort}/Phenotypic_Processing.csv"
    for cohort in DEFAULT_INPUTS
}
MISSING_NUMBERS = {-9999.0, -999.0, -99.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abide-i", type=Path, default=DEFAULT_INPUTS["ABIDE-I"])
    parser.add_argument("--abide-ii", type=Path, default=DEFAULT_INPUTS["ABIDE-II"])
    parser.add_argument("--phenotype-i", type=Path, default=DEFAULT_PHENOTYPES["ABIDE-I"])
    parser.add_argument("--phenotype-ii", type=Path, default=DEFAULT_PHENOTYPES["ABIDE-II"])
    parser.add_argument("--ec-key", default="ec", choices=("ec", "refined_ec", "qc_refined_ec", "original_ec"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--original-edges", type=Path, default=DEFAULT_OUTPUT_DIR / "replicated_edges.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "covariate_adjusted_outputs")
    return parser.parse_args()


def numeric(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) and number not in MISSING_NUMBERS else float("nan")


def align_covariates(
    archive_path: Path, phenotype_path: Path, labels: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    with np.load(archive_path, allow_pickle=False) as archive:
        subject_ids = np.asarray(archive["subject_ids"]).astype(str).reshape(-1)
        site_ids = np.asarray(archive["site_ids"]).astype(str).reshape(-1)
    if len(subject_ids) != len(labels) or len(site_ids) != len(labels):
        raise ValueError("Subject IDs, site IDs, labels and EC have incompatible lengths")
    if len(set(subject_ids)) != len(subject_ids) or any(not value.strip() for value in subject_ids):
        raise ValueError("EC subject IDs must be nonempty and unique")
    with phenotype_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [field.strip() for field in reader.fieldnames or []]
        required = {"FILE_ID", "SUB_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "func_mean_fd"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing phenotype columns: {sorted(required - set(reader.fieldnames))}")
        phenotype = {}
        for line_number, row in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in row.items()}
            identifier = row["FILE_ID"]
            if not identifier or identifier == "no_filename":
                continue
            if identifier in phenotype:
                raise ValueError(f"Duplicate phenotype FILE_ID: {identifier}")
            phenotype[identifier] = (line_number, row)

    audit = []
    for index, (subject_id, site_id, label) in enumerate(zip(subject_ids, site_ids, labels)):
        matched = phenotype.get(subject_id)
        row = matched[1] if matched else {}
        age = numeric(row.get("AGE_AT_SCAN", ""))
        sex = numeric(row.get("SEX", ""))
        mean_fd = numeric(row.get("func_mean_fd", ""))
        site = row.get("SITE_ID", "")
        diagnosis = numeric(row.get("DX_GROUP", ""))
        reasons = []
        if matched is None:
            reasons.append("unmatched_FILE_ID")
        else:
            sub_id = row["SUB_ID"]
            archive_number = subject_id.rsplit("_", 1)[-1].removeprefix("sub-")
            phenotype_number = sub_id.removeprefix("sub-")
            if not archive_number.isdigit() or not phenotype_number.isdigit():
                raise ValueError(f"Cannot independently check SUB_ID for {subject_id}: {sub_id}")
            if int(archive_number) != int(phenotype_number):
                raise ValueError(f"FILE_ID / SUB_ID mismatch for {subject_id}: {sub_id}")
            if not np.isfinite(diagnosis):
                reasons.append("missing_DX_GROUP")
            elif diagnosis not in (1, 2) or int(diagnosis == 2) != label:
                raise ValueError(f"Diagnosis disagrees with archive label for {subject_id}")
            if site and site != site_id:
                raise ValueError(f"Site mismatch for {subject_id}: {site!r} != {site_id!r}")
        for name, value in (("Age", age), ("Sex", sex), ("Mean_FD", mean_fd)):
            if not np.isfinite(value):
                reasons.append(f"missing_{name}")
        if not site or site.lower() in {"nan", "na", "n/a", "-9999", "-999", "-99"}:
            reasons.append("missing_Site")
        if np.isfinite(age) and age <= 0:
            raise ValueError(f"Invalid age for {subject_id}: {age}")
        if np.isfinite(sex) and sex not in (1, 2):
            raise ValueError(f"Invalid sex for {subject_id}: {sex}")
        if np.isfinite(mean_fd) and mean_fd < 0:
            raise ValueError(f"Invalid mean FD for {subject_id}: {mean_fd}")
        audit.append({
            "ECIndex": index,
            "SubjectID": subject_id,
            "PhenotypeLine": matched[0] if matched else "",
            "FILE_ID": row.get("FILE_ID", ""),
            "SUB_ID": row.get("SUB_ID", ""),
            "ArchiveSite": site_id,
            "Site": site,
            "ArchiveLabel": int(label),
            "DX_GROUP": diagnosis,
            "Age": age,
            "Sex": sex,
            "Mean_FD": mean_fd,
            "Included": not reasons,
            "ExclusionReasons": ";".join(reasons),
        })
    return audit, np.asarray([row["Included"] for row in audit], dtype=bool)


def fit_edges(
    ec: np.ndarray, labels: np.ndarray, audit: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    included = [row for row in audit if row["Included"]]
    sites = np.asarray([row["Site"] for row in included])
    site_levels = sorted(set(sites))
    column_names = ["Intercept", "ASD", "Age", "Sex", "Mean_FD"]
    columns = [np.ones(len(labels)), labels]
    columns.extend(np.asarray([row[name] for row in included], dtype=float) for name in ("Age", "Sex", "Mean_FD"))
    for site in site_levels[1:]:
        column_names.append(f"C(Site)[T.{site}]")
        columns.append((sites == site).astype(float))
    design = np.column_stack(columns)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Complete cases must contain both diagnostic groups")
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError("Covariate design is rank deficient")
    degrees_of_freedom = len(labels) - rank
    if degrees_of_freedom <= 0:
        raise ValueError("No residual degrees of freedom")
    source, target = np.where(~np.eye(ec.shape[1], dtype=bool))
    response = ec[:, source, target]
    orthogonal, triangular = np.linalg.qr(design, mode="reduced")
    coefficients = np.linalg.solve(triangular, orthogonal.T @ response)
    residuals = response - design @ coefficients
    residual_variance = np.sum(residuals ** 2, axis=0) / degrees_of_freedom
    inverse_triangular = np.linalg.solve(triangular, np.eye(rank))
    asd_variance_factor = np.sum(inverse_triangular[1] ** 2)
    standard_errors = np.sqrt(residual_variance * asd_variance_factor)
    if np.any(standard_errors <= 0):
        raise ValueError("Zero residual variance encountered")
    asd_coefficients = coefficients[1]
    t_statistics = asd_coefficients / standard_errors
    p_values = 2 * student_t.sf(np.abs(t_statistics), degrees_of_freedom)
    if not np.isfinite(p_values).all():
        raise ValueError("Nonfinite OLS p values")
    q_values = benjamini_hochberg(p_values)
    verification = []
    for edge_index in np.linspace(0, len(source) - 1, 9, dtype=int):
        edge_response = response[:, edge_index]
        reference_coefficients = np.linalg.lstsq(design, edge_response, rcond=None)[0]
        reference_residuals = edge_response - design @ reference_coefficients
        reference_variance = np.dot(reference_residuals, reference_residuals) / degrees_of_freedom
        reference_se = np.sqrt(reference_variance * np.linalg.inv(design.T @ design)[1, 1])
        reference_p = 2 * student_t.sf(abs(reference_coefficients[1] / reference_se), degrees_of_freedom)
        np.testing.assert_allclose(asd_coefficients[edge_index], reference_coefficients[1], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(p_values[edge_index], reference_p, rtol=1e-8, atol=1e-12)
        verification.append({"SourceIndex": int(source[edge_index]), "TargetIndex": int(target[edge_index]), "BetaAbsError": float(abs(asd_coefficients[edge_index] - reference_coefficients[1])), "PAbsError": float(abs(p_values[edge_index] - reference_p))})
    diagnostics = {
        "formula": FORMULA,
        "design_columns": column_names,
        "site_reference": site_levels[0],
        "site_counts": dict(Counter(sites.tolist())),
        "design_rank": rank,
        "design_condition_number": float(np.linalg.cond(design)),
        "residual_df": degrees_of_freedom,
        "tested_edges": len(source),
        "single_edge_lstsq_verification": verification,
    }
    return asd_coefficients, standard_errors, p_values, q_values, diagnostics


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")
    output_dir = args.output_dir.resolve()
    protected = DEFAULT_OUTPUT_DIR.resolve()
    if output_dir == protected or protected in output_dir.parents:
        raise ValueError("Adjusted outputs must be outside the original outputs directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_tables = {}
    adjusted = {}
    summaries = {}
    reference_rois = None
    for cohort, archive_path, phenotype_path in (
        ("ABIDE-I", args.abide_i, args.phenotype_i),
        ("ABIDE-II", args.abide_ii, args.phenotype_ii),
    ):
        ec, labels, roi_names = load_archive(archive_path, args.ec_key)
        if reference_rois is not None and not np.array_equal(reference_rois, roi_names):
            raise ValueError("Cohorts do not have identical ROI order")
        reference_rois = roi_names
        original_tables[cohort] = edge_table(cohort, *compute_effects_and_tests(ec, labels), roi_names)
        audit, keep = align_covariates(archive_path, phenotype_path, labels)
        beta, standard_error, p_values, q_values, diagnostics = fit_edges(ec[keep], labels[keep], audit)
        source, target = np.where(~np.eye(ec.shape[1], dtype=bool))
        rows = []
        for edge_index, (source_index, target_index) in enumerate(zip(source, target)):
            rows.append({
                "Source": str(roi_names[source_index]), "Target": str(roi_names[target_index]),
                "SourceIndex": int(source_index), "TargetIndex": int(target_index),
                "ASD_Beta": float(beta[edge_index]), "ASD_SE": float(standard_error[edge_index]),
                "ASD_T": float(beta[edge_index] / standard_error[edge_index]),
                "PValue": float(p_values[edge_index]), "QValue": float(q_values[edge_index]),
            })
        adjusted[cohort] = rows
        write_csv(output_dir / f"{cohort.lower()}_edge_statistics.csv", rows, list(rows[0]))
        write_csv(output_dir / f"{cohort.lower()}_subject_audit.csv", audit, list(audit[0]))
        summary = {
            "archive_path": str(archive_path.resolve()), "archive_sha256": file_hash(archive_path),
            "phenotype_path": str(phenotype_path.resolve()), "phenotype_sha256": file_hash(phenotype_path),
            "archive_n": len(labels), "matched_n": sum(bool(row["FILE_ID"]) for row in audit),
            "included_n": int(keep.sum()), "excluded_missing_n": int((~keep).sum()),
            "exclusion_reasons": dict(Counter(reason for row in audit for reason in str(row["ExclusionReasons"]).split(";") if reason)),
            "included_asd_n": int((labels[keep] == ASD_LABEL).sum()),
            "included_hc_n": int((labels[keep] != ASD_LABEL).sum()),
            "significant_edges": int((q_values < args.alpha).sum()),
            "diagnosis_label_mapping": "DX_GROUP=2 -> archive ASD=1; DX_GROUP=1 -> archive HC=0 (existing project convention)",
            "identity_checks": "Unique exact FILE_ID join; independent numeric SUB_ID check; DX_GROUP and SITE_ID agreement with archive",
            **diagnostics,
        }
        summaries[cohort] = summary
        print(f"{cohort}: matched={summary['matched_n']}/{len(labels)}, excluded={summary['excluded_missing_n']}, ASD={summary['included_asd_n']}, HC={summary['included_hc_n']}, significant={summary['significant_edges']}")

    baseline = merge_tables(original_tables["ABIDE-I"], original_tables["ABIDE-II"], args.alpha)
    baseline_by_key = {(row["SourceIndex"], row["TargetIndex"]): row for row in baseline}
    original_keys = {key for key, row in baseline_by_key.items() if row["Replicated"]}
    with args.original_edges.open(newline="", encoding="utf-8-sig") as handle:
        saved_rows = list(csv.DictReader(handle))
    saved_keys = {(int(row["SourceIndex"]), int(row["TargetIndex"])) for row in saved_rows}
    if len(saved_keys) != len(saved_rows) or saved_keys != original_keys:
        raise ValueError("Recomputed original replicated edges disagree with saved original edges")
    for row in saved_rows:
        original = baseline_by_key[(int(row["SourceIndex"]), int(row["TargetIndex"]))]
        if any(row[name] != original[name] for name in ("Source", "Target")):
            raise ValueError("Original edge ROI names do not match EC archives")

    merged = []
    for row_i, row_ii in zip(adjusted["ABIDE-I"], adjusted["ABIDE-II"]):
        key = (row_i["SourceIndex"], row_i["TargetIndex"])
        if key != (row_ii["SourceIndex"], row_ii["TargetIndex"]):
            raise ValueError("Cohort directed edge order mismatch")
        beta_i, beta_ii = row_i["ASD_Beta"], row_ii["ASD_Beta"]
        same_direction = bool(beta_i != 0 and beta_ii != 0 and np.sign(beta_i) == np.sign(beta_ii))
        replicated = bool(same_direction and row_i["QValue"] < args.alpha and row_ii["QValue"] < args.alpha)
        original = baseline_by_key[key]
        preserves_original = bool(
            same_direction and np.sign(beta_i) == np.sign(original["ABIDE_I_Hedges_g"])
            and np.sign(beta_ii) == np.sign(original["ABIDE_II_Hedges_g"])
        )
        merged.append({
            **{name: row_i[name] for name in ("Source", "Target", "SourceIndex", "TargetIndex")},
            "ABIDE_I_ASD_Beta": beta_i, "ABIDE_II_ASD_Beta": beta_ii,
            "ABIDE_I_PValue": row_i["PValue"], "ABIDE_II_PValue": row_ii["PValue"],
            "ABIDE_I_QValue": row_i["QValue"], "ABIDE_II_QValue": row_ii["QValue"],
            "SameDirection": same_direction, "Replicated": replicated,
            "OriginallyReplicated": key in original_keys,
            "Original_ABIDE_I_Hedges_g": original["ABIDE_I_Hedges_g"],
            "Original_ABIDE_II_Hedges_g": original["ABIDE_II_Hedges_g"],
            "PreservesOriginalDirectionBothCohorts": preserves_original,
        })
    replicated_rows = [row for row in merged if row["Replicated"]]
    comparison = [row for row in merged if row["OriginallyReplicated"]]
    fields = list(merged[0])
    write_csv(output_dir / "all_edges_cross_cohort.csv", merged, fields)
    write_csv(output_dir / "replicated_edges.csv", replicated_rows, fields)
    write_csv(output_dir / "original_213_comparison.csv", comparison, fields)
    result = {
        "formula": FORMULA, "ec_key": args.ec_key, "alpha": args.alpha,
        "inference": "OLS with intercept, treatment-coded site fixed effects and conventional two-sided residual-df t tests; BH-FDR independently across all off-diagonal directed edges in each cohort",
        "original_edges_path": str(args.original_edges.resolve()),
        "original_edges_sha256": file_hash(args.original_edges),
        "original_replication_reproduced": True,
        "original_replicated_n": len(original_keys),
        "adjusted_replicated_n": len(replicated_rows),
        "original_retained_n": sum(row["Replicated"] for row in comparison),
        "original_same_direction_between_cohorts_n": sum(row["SameDirection"] for row in comparison),
        "original_preserves_original_direction_both_cohorts_n": sum(row["PreservesOriginalDirectionBothCohorts"] for row in comparison),
        "original_retained_preserves_original_direction_n": sum(row["Replicated"] and row["PreservesOriginalDirectionBothCohorts"] for row in comparison),
        "adjusted_replicated_positive_n": sum(row["ABIDE_I_ASD_Beta"] > 0 for row in replicated_rows),
        "adjusted_replicated_negative_n": sum(row["ABIDE_I_ASD_Beta"] < 0 for row in replicated_rows),
        "cohorts": summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key.endswith("_n")}, indent=2))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
