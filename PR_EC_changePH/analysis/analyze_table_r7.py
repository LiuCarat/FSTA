"""Read-only W6 ASD edge analysis; never imports or executes model training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.stats import false_discovery_control, ttest_ind

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PR_EC_changePH.analysis.cross_cohort_replication import run_cross_cohort_replication as legacy

OUTPUT_ROOT = ROOT / "PR_EC_changePH/outputs/rebuttal_w6"
COHORTS = {"abide-i": "abide", "abide-ii": "abide_ii"}
REPRESENTATIONS = {
    "initial": "Initial EC",
    "original_ph": "Full PR-EC (Original PH)",
    "age_sex_site": "Full PR-EC (Age/Sex/Site)",
}
TABLE_FIELDS = [
    "EC representation", "ABIDE-I mean |g|", "ABIDE-II mean |g|",
    "Replicated edges", "ASD-enhanced", "ASD-reduced",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [field.strip() for field in reader.fieldnames or []]
        return list(reader)


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_same(left, right, description):
    if not np.array_equal(left, right):
        raise ValueError(f"Alignment failed: {description}")


def canonical_labels(path, subject_ids):
    records = {}
    for row in read_csv(path):
        identifier = row.get("FILE_ID", "").strip()
        if not identifier or identifier == "no_filename":
            continue
        if identifier in records:
            raise ValueError(f"Duplicate phenotype FILE_ID: {identifier}")
        records[identifier] = row
    labels = []
    for subject_id in subject_ids:
        row = records.get(str(subject_id))
        if row is None:
            row = records.get(str(subject_id).rsplit("/", 1)[-1])
        if row is None or float(row["DX_GROUP"]) not in (1, 2):
            raise ValueError(f"Missing or unknown diagnosis: {subject_id}")
        labels.append(int(float(row["DX_GROUP"]) == 2))
    return np.asarray(labels, dtype=np.int64)


def load_initial(path, phenotype_path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("ec", "labels", "subject_ids", "site_ids")}
        if "roi_names" in archive:
            data["roi_names"] = archive["roi_names"].astype(str)
    ec = data["ec"]
    if ec.ndim != 3 or ec.shape[1:] != (90, 90) or not np.isfinite(ec).all():
        raise ValueError(f"Expected finite [subjects,90,90] EC: {path}")
    if len(np.unique(data["subject_ids"])) != len(ec):
        raise ValueError(f"Duplicate subject IDs: {path}")
    if any(data[key].shape != (len(ec),) for key in ("labels", "subject_ids", "site_ids")):
        raise ValueError(f"Subject metadata shape mismatch: {path}")
    labels = canonical_labels(phenotype_path, data["subject_ids"])
    check_same(data["labels"], labels, f"Initial labels versus phenotype {path}")
    if min(np.count_nonzero(labels == label) for label in (0, 1)) < 2:
        raise ValueError("Each diagnosis group must contain at least two subjects")
    if "roi_names" not in data:
        data["roi_names"] = np.asarray([f"ROI_{index + 1:03d}" for index in range(90)])
        data["roi_metadata"] = "Initial archive has no ROI names; inherited first-90 index order; verified against embedded original_ec"
    else:
        data["roi_metadata"] = "Explicit ROI names present"
    return data


def load_refined(path, initial, reference_folds=None):
    with np.load(path, allow_pickle=False) as archive:
        for key in ("subject_ids", "labels", "site_ids", "roi_names"):
            check_same(initial[key], archive[key], f"{key}: {path}")
        check_same(initial["ec"], archive["original_ec"], f"embedded original_ec: {path}")
        if str(archive["representation"].item()) != "qsr_refined":
            raise ValueError(f"Not standalone QSR OOF EC: {path}")
        ec = archive["ec"]
        check_same(ec, archive["qc_refined_ec"], f"ec and qc_refined_ec: {path}")
        folds = archive["fold_ids"]
        if folds.shape != (len(ec),) or set(folds.tolist()) != set(range(1, 11)):
            raise ValueError(f"Invalid ten-fold OOF metadata: {path}")
        if reference_folds is not None:
            check_same(folds, reference_folds, f"OOF test folds: {path}")
    if ec.shape != initial["ec"].shape or not np.isfinite(ec).all():
        raise ValueError(f"Invalid refined EC: {path}")
    if not np.all(np.diagonal(ec, axis1=1, axis2=2) == 0):
        raise ValueError(f"Unexpected nonzero diagonal: {path}")
    return ec, folds


def edge_statistics(dataset, ec, labels, roi_names):
    ec = np.asarray(ec, dtype=np.float64)
    mask = ~np.eye(ec.shape[1], dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        asd_mean, hc_mean, hedges_g, q_values = legacy.compute_effects_and_tests(ec, labels)
    test = ttest_ind(ec[labels == 1][:, mask], ec[labels == 0][:, mask], axis=0,
                     equal_var=False, nan_policy="raise")
    if not np.isfinite(test.pvalue).all() or not np.isfinite(hedges_g[mask]).all():
        raise ValueError("Nonfinite off-diagonal statistics; no silent replacement is allowed")
    independent_q = false_discovery_control(test.pvalue, method="bh")
    np.testing.assert_allclose(q_values[mask], independent_q, rtol=1e-12, atol=1e-14)
    p_values = np.full_like(q_values, np.nan)
    t_values = np.full_like(q_values, np.nan)
    p_values[mask], t_values[mask] = test.pvalue, test.statistic
    rows = legacy.edge_table(dataset, asd_mean, hc_mean, hedges_g, q_values, roi_names)
    for row in rows:
        edge = (row["SourceIndex"], row["TargetIndex"])
        row["PValue"] = float(p_values[edge])
        row["WelchT"] = float(t_values[edge])
    return rows


def merge_statistics(tables):
    merged = legacy.merge_tables(tables["abide-i"], tables["abide-ii"], 0.05)
    lookups = {cohort: {(row["SourceIndex"], row["TargetIndex"]): row for row in rows}
               for cohort, rows in tables.items()}
    for row in merged:
        edge = (row["SourceIndex"], row["TargetIndex"])
        row["ABIDE_I_PValue"] = lookups["abide-i"][edge]["PValue"]
        row["ABIDE_II_PValue"] = lookups["abide-ii"][edge]["PValue"]
    return merged


def retained_edges(baseline, candidate):
    lookup = {(row["SourceIndex"], row["TargetIndex"]): row for row in candidate}
    comparisons = []
    for row in baseline:
        if not row["Replicated"]:
            continue
        new = lookup[(row["SourceIndex"], row["TargetIndex"])]
        agrees = all(np.sign(row[f"{cohort}_Hedges_g"]) == np.sign(new[f"{cohort}_Hedges_g"])
                     for cohort in ("ABIDE_I", "ABIDE_II"))
        comparisons.append({
            "SourceIndex": row["SourceIndex"], "TargetIndex": row["TargetIndex"],
            "Source": row["Source"], "Target": row["Target"],
            "OriginalClass": row["ReplicationClass"],
            "RetainedReplicated": bool(new["Replicated"]),
            "SameOriginalDirectionBothCohorts": bool(agrees),
            "RetainedAndOriginalDirection": bool(new["Replicated"] and agrees),
            **{f"Original_{cohort}_{metric}": row[f"{cohort}_{metric}"]
               for cohort in ("ABIDE_I", "ABIDE_II") for metric in ("Hedges_g", "QValue")},
            **{f"AgeSexSite_{cohort}_{metric}": new[f"{cohort}_{metric}"]
               for cohort in ("ABIDE_I", "ABIDE_II") for metric in ("Hedges_g", "QValue")},
        })
    return comparisons


def run(output):
    output = output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Analysis outputs must stay under PR_EC_changePH/outputs/rebuttal_w6")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing analysis results: {output}")
    paths = {}
    configs = {}
    for cohort, prefix in COHORTS.items():
        original_root = ROOT / "PR_EC/outputs/seed42" / cohort
        age_root = OUTPUT_ROOT / cohort / "age_sex_site"
        paths[cohort] = {
            "initial": original_root / f"{prefix}_subject_ec.npz",
            "original_ph": original_root / f"{prefix}_qsr_refined_subject_ec.npz",
            "age_sex_site": age_root / f"{prefix}_qsr_refined_subject_ec.npz",
            "phenotype": ROOT / "dataset" / ("ABIDE-I" if cohort == "abide-i" else "ABIDE-II") / "Phenotypic_Processing.csv",
            "original_summary": original_root / "summary.json",
            "age_summary": age_root / "summary.json",
        }
        for path in paths[cohort].values():
            if not path.is_file():
                raise FileNotFoundError(f"Required input missing; no training or substitution: {path}")
        configs[cohort] = {}
        for setting, key in (("original", "original_summary"), ("age_sex_site", "age_summary")):
            config = json.loads(paths[cohort][key].read_text())["config"]
            if config["seed"] != 42 or config["n_splits"] != 10:
                raise ValueError(f"Not seed42 / ten-fold source: {paths[cohort][key]}")
            if config.get("phenotype_feature_set", "original") != setting:
                raise ValueError(f"Wrong PH configuration: {paths[cohort][key]}")
            for name in ("ec_path", "qsr_refined_ec_path"):
                expected = paths[cohort]["initial" if name == "ec_path" else ("original_ph" if setting == "original" else "age_sex_site")]
                actual = Path(config[name])
                if not actual.is_absolute():
                    actual = ROOT / actual
                relocated_original = (setting == "original" and actual.resolve() ==
                                      (ROOT / "PR_EC/outputs" / cohort / expected.name).resolve())
                if actual.resolve() != expected.resolve() and not relocated_original:
                    raise ValueError(f"Config and selected EC path disagree: {name}")
            configs[cohort][setting] = config
    paper_path = ROOT / "PR_EC/analysis/cross_cohort_replication/outputs/replicated_edges.csv"
    tracked = [path for cohort in paths.values() for path in cohort.values()]
    tracked += [Path(__file__), Path(legacy.__file__)]
    if paper_path.is_file():
        tracked.append(paper_path)
    before = {str(path): digest(path) for path in tracked}
    output.mkdir(parents=True, exist_ok=True)
    statistics = {key: {} for key in REPRESENTATIONS}
    audit, subject_rows = {}, []
    common_rois = None
    for cohort in COHORTS:
        selected = paths[cohort]
        initial = load_initial(selected["initial"], selected["phenotype"])
        if common_rois is not None:
            check_same(common_rois, initial["roi_names"], "cross-cohort ROI index order")
        common_rois = initial["roi_names"]
        original, folds = load_refined(selected["original_ph"], initial)
        new, new_folds = load_refined(selected["age_sex_site"], initial, folds)
        for index, identifier in enumerate(initial["subject_ids"]):
            subject_rows.append({"dataset": cohort, "subject_id": str(identifier),
                                 "label": int(initial["labels"][index]),
                                 "diagnosis": "ASD" if initial["labels"][index] else "TC",
                                 "site": str(initial["site_ids"][index]),
                                 "original_ph_test_fold": int(folds[index]),
                                 "age_sex_site_test_fold": int(new_folds[index])})
        audit[cohort] = {
            "subjects": len(initial["ec"]), "ASD": int((initial["labels"] == 1).sum()),
            "TC": int((initial["labels"] == 0).sum()), "n_rois": 90, "tested_directed_edges": 8010,
            "subject_ids_labels_sites_exactly_equal": True, "labels_match_phenotype_DX2_ASD_DX1_TC": True,
            "embedded_initial_ec_exactly_equal": True, "oof_folds_exactly_equal": True,
            "roi_order": initial["roi_metadata"],
            "direction": "Stored EC[source_index,target_index], following original analysis; no additional transpose or symmetrization. Initial extraction already transposes attention.",
            "source_note": "ROI names are index placeholders, not independent anatomical verification; first 90 input ROI columns follow shared preprocessing.",
            "original_cache_relocation": "Original summary retains pre-seed42 paths; selected explicit seed42 archives are checked using full subject metadata and exact embedded initial EC, not replaced by current files at old paths.",
            "source_paths": {key: str(value) for key, value in selected.items()},
        }
        for representation, ec in (("initial", initial["ec"]), ("original_ph", original), ("age_sex_site", new)):
            rows = edge_statistics(cohort, ec, initial["labels"], initial["roi_names"])
            statistics[representation][cohort] = rows
            destination = output / representation
            destination.mkdir(exist_ok=True)
            write_csv(destination / f"{cohort}_edge_statistics.csv", rows)
        del initial, original, new
    merged_tables, table, cohort_summaries = {}, [], []
    for representation, cohorts in statistics.items():
        merged = merge_statistics(cohorts)
        merged_tables[representation] = merged
        replicated = [row for row in merged if row["Replicated"]]
        destination = output / representation
        write_csv(destination / "all_edges_cross_cohort.csv", merged)
        write_csv(destination / "replicated_edges.csv", replicated, list(merged[0]))
        means = {}
        for cohort, rows in cohorts.items():
            means[cohort] = float(np.mean([row["AbsHedges_g"] for row in rows]))
            cohort_summaries.append({"representation": representation, "dataset": cohort,
                                     "ASD_N": audit[cohort]["ASD"], "TC_N": audit[cohort]["TC"],
                                     "edges": len(rows), "mean_absolute_g": means[cohort],
                                     "q_lt_0_05": sum(row["QValue"] < 0.05 for row in rows)})
        table.append(dict(zip(TABLE_FIELDS, [REPRESENTATIONS[representation], means["abide-i"], means["abide-ii"],
                                            len(replicated), sum(row["ABIDE_I_Hedges_g"] > 0 for row in replicated),
                                            sum(row["ABIDE_I_Hedges_g"] < 0 for row in replicated)])))
    comparisons = retained_edges(merged_tables["original_ph"], merged_tables["age_sex_site"])
    retention = {
        "original_replicated": len(comparisons),
        "age_sex_site_replicated": table[2]["Replicated edges"],
        "retained_replicated": sum(row["RetainedReplicated"] for row in comparisons),
        "retained_replicated_same_original_direction": sum(row["RetainedAndOriginalDirection"] for row in comparisons),
        "same_original_direction_among_all_original_edges_without_significance_requirement": sum(row["SameOriginalDirectionBothCohorts"] for row in comparisons),
    }
    retention["retained_fraction"] = retention["retained_replicated"] / retention["original_replicated"] if comparisons else None
    retention["lost_original_edges"] = retention["original_replicated"] - retention["retained_replicated"]
    retention["new_replicated_edges"] = retention["age_sex_site_replicated"] - retention["retained_replicated"]
    paper_check = {"paper_reported": [213, 111, 102],
                   "computed": [table[1][field] for field in TABLE_FIELDS[3:]],
                   "counts_match": [table[1][field] for field in TABLE_FIELDS[3:]] == [213, 111, 102]}
    if paper_path.is_file():
        archived = read_csv(paper_path)
        current = [row for row in merged_tables["original_ph"] if row["Replicated"]]
        archived_lookup = {(int(row["SourceIndex"]), int(row["TargetIndex"])): row for row in archived}
        current_lookup = {(row["SourceIndex"], row["TargetIndex"]): row for row in current}
        paper_check["archived_edges_count"] = len(archived)
        paper_check["same_edge_set_as_archived_paper"] = set(archived_lookup) == set(current_lookup)
        shared = set(archived_lookup) & set(current_lookup)
        paper_check["archived_overlap"] = len(shared)
        paper_check["archived_overlap_same_direction"] = sum(all(
            np.sign(float(archived_lookup[edge][f"{cohort}_Hedges_g"])) == np.sign(current_lookup[edge][f"{cohort}_Hedges_g"])
            for cohort in ("ABIDE_I", "ABIDE_II")) for edge in shared)
        paper_check["max_abs_g_difference_on_archived_overlap"] = {
            cohort: max((abs(float(archived_lookup[edge][f"{cohort}_Hedges_g"]) - current_lookup[edge][f"{cohort}_Hedges_g"])
                         for edge in shared), default=None) for cohort in ("ABIDE_I", "ABIDE_II")}
    after = {str(path): digest(path) for path in tracked}
    if before != after:
        raise RuntimeError("An input changed during analysis; outputs must not be trusted")
    write_csv(output / "table_r7.csv", table)
    write_csv(output / "cohort_summary.csv", cohort_summaries)
    write_csv(output / "original_edges_age_sex_site_retention.csv", comparisons)
    write_csv(output / "subject_alignment.csv", subject_rows)
    summary = {
        "table_r7": table, "retention": retention, "paper_check": paper_check, "alignment_audit": audit,
        "methods": {"test": "two-sided Welch t-test", "effect": "ASD-minus-TC pooled-SD Hedges g with exact gamma correction, reused original code",
                    "fdr": "BH over all 8010 off-diagonal directed edges separately within cohort and representation",
                    "replication": "both cohort q < 0.05 and same nonzero g sign", "mean_abs_g": "all 8010 off-diagonal edges, not just significant edges",
                    "covariates": "none; identical unadjusted original Welch analysis", "ec": "Initial or test-only OOF QSR EC; no refitting or training"},
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
        "inputs_sha256_before": before, "inputs_sha256_after": after, "inputs_unchanged": True,
        "source_configs": configs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"table_r7": table, "retention": retention, "paper_check": paper_check, "output": str(output)}, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New empty directory under PR_EC_changePH/outputs/rebuttal_w6")
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
