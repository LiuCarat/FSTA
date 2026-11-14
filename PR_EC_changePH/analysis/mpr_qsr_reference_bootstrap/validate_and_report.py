#!/usr/bin/env python3
"""Independently audit saved sampling/OOF/statistics and render the pilot report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from mpr_qsr_reference_bootstrap import (
    MTG_THA_EDGES, ORIGINAL_EDGES, DEFAULT_PHENOTYPES,
    align_covariates, file_hash, fit_edges, read_edges, write_csv,
)


def audit(output):
    provenance = json.loads((output / "provenance.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    original = {
        (int(row["SourceIndex"]), int(row["TargetIndex"])): row
        for row in read_edges(ORIGINAL_EDGES)
    }
    assert len(original) == 213
    assert file_hash(ORIGINAL_EDGES) == provenance["original_edges_sha256"]
    checks, all_mtg = [], []
    for cohort, local_name, archive_name in (
        ("ABIDE-I", "abide_i", "abide-i"), ("ABIDE-II", "abide_ii", "abide-ii"),
    ):
        baseline_audit = read_edges(output / "baseline_adjusted" / f"{archive_name}_subject_audit.csv")
        numeric_subject_ids = [int(row["SUB_ID"].removeprefix("sub-")) for row in baseline_audit]
        assert len(set(numeric_subject_ids)) == len(numeric_subject_ids)
        cohort_provenance = provenance["cohorts"][local_name]
        initial_path = Path(cohort_provenance["config"]["ec_path"])
        assert file_hash(initial_path) == cohort_provenance["initial_ec_sha256"]
        baseline_path = Path(cohort_provenance["config"]["qsr_refined_ec_path"])
        assert file_hash(baseline_path) == cohort_provenance["original_qsr_ec_sha256"]
        with np.load(initial_path, allow_pickle=False) as initial, np.load(baseline_path, allow_pickle=False) as baseline:
            subjects = baseline["subject_ids"].astype(str)
            assert len(set(subjects)) == len(subjects)
            subject_to_index = {subject: index for index, subject in enumerate(subjects)}
            for repeat in range(1, summary["repeats"] + 1):
                repeat_dir = output / f"repeat_{repeat:02d}"
                sample_info = json.loads((repeat_dir / "bootstrap_training_subjects.json").read_text())
                with np.load(output / f"{local_name}_fixed_partitions.npz", allow_pickle=False) as partitions:
                    coverage = np.zeros(len(subjects), dtype=int)
                    for fold in range(1, 11):
                        train_pool = partitions[f"fold_{fold}_train_pool"]
                        validation = partitions[f"fold_{fold}_val"]
                        test = partitions[f"fold_{fold}_test"]
                        sampled = np.array([subject_to_index[subject] for subject in sample_info[local_name][str(fold)]])
                        assert len(sampled) == len(train_pool)
                        assert np.isin(sampled, train_pool).all()
                        assert not np.intersect1d(sampled, test).size
                        assert not np.intersect1d(sampled, validation).size
                        np.testing.assert_array_equal(np.sort(np.r_[train_pool, validation, test]), np.arange(len(subjects)))
                        coverage[test] += 1
                        checks.append({
                            "cohort": cohort, "repeat": repeat, "fold": fold,
                            "train_pool_n": len(train_pool), "sampled_n": len(sampled),
                            "unique_sampled_n": len(np.unique(sampled)), "validation_n": len(validation),
                            "test_n": len(test), "train_test_overlap_n": 0, "train_validation_overlap_n": 0,
                        })
                    np.testing.assert_array_equal(coverage, np.ones(len(subjects), dtype=int))
                archive_path = repeat_dir / f"{archive_name}_qsr_bootstrap_oof.npz"
                with np.load(archive_path, allow_pickle=False) as archive:
                    for field in ("subject_ids", "site_ids", "labels", "fold_ids", "roi_names"):
                        np.testing.assert_array_equal(archive[field], baseline[field])
                    np.testing.assert_array_equal(archive["original_ec"], initial["ec"])
                    ec = archive["ec"]
                    assert np.isfinite(ec).all()
                    assert np.all(np.diagonal(ec, axis1=1, axis2=2) == 0)
                    assert not np.array_equal(ec, baseline["ec"])
                    labels = archive["labels"]
                subject_audit, keep = align_covariates(archive_path, DEFAULT_PHENOTYPES[cohort], labels)
                beta, _, p_values, q_values, _ = fit_edges(ec[keep], labels[keep], subject_audit)
                saved = read_edges(repeat_dir / "covariate_adjusted_replication" / f"{archive_name}_edge_statistics.csv")
                assert len(saved) == 8010
                for field, values in (("ASD_Beta", beta), ("PValue", p_values), ("QValue", q_values)):
                    np.testing.assert_allclose([float(row[field]) for row in saved], values, rtol=1e-10, atol=1e-14)
    for repeat in range(1, summary["repeats"] + 1):
        repeat_dir = output / f"repeat_{repeat:02d}"
        rows = read_edges(repeat_dir / "covariate_adjusted_replication/all_edges_cross_cohort.csv")
        assert len(rows) == 8010
        reference = json.loads((repeat_dir / "repeat_summary.json").read_text())
        assert sum(row["Replicated"] == "True" for row in rows) == reference["adjusted_replicated_n"]
        original_rows = [row for row in rows if row["OriginallyReplicated"] == "True"]
        assert len(original_rows) == 213
        assert sum(row["Replicated"] == "True" for row in original_rows) == reference["original_213_retained_n"]
        mtg_rows = [row for row in original_rows if (int(row["SourceIndex"]), int(row["TargetIndex"])) in MTG_THA_EDGES]
        assert len(mtg_rows) == 8
        all_mtg.extend({"repeat": repeat, **row} for row in mtg_rows)
    write_csv(output / "fold_sampling_audit.csv", checks, list(checks[0]))
    frequencies = []
    baseline_comparison = {
        (int(row["SourceIndex"]), int(row["TargetIndex"])): row
        for row in read_edges(output / "baseline_adjusted/original_213_comparison.csv")
    }
    names = {76: "THA.L", 77: "THA.R", 84: "MTG.L", 85: "MTG.R"}
    for source, target in sorted(MTG_THA_EDGES):
        key = source, target
        rows = [row for row in all_mtg if (int(row["SourceIndex"]), int(row["TargetIndex"])) == key]
        frequency = {"Source": names[source], "Target": names[target], "SourceIndex": source, "TargetIndex": target, "repeats": len(rows)}
        frequency["Baseline_ABIDE_I_QValue"] = float(baseline_comparison[key]["ABIDE_I_QValue"])
        frequency["Baseline_ABIDE_II_QValue"] = float(baseline_comparison[key]["ABIDE_II_QValue"])
        frequency["BaselineReplicated"] = baseline_comparison[key]["Replicated"] == "True"
        for cohort in ("I", "II"):
            sign = np.sign(float(original[key][f"ABIDE_{cohort}_Hedges_g"]))
            frequency[f"ABIDE_{cohort}_OriginalDirectionRate"] = float(np.mean([np.sign(float(row[f"ABIDE_{cohort}_ASD_Beta"])) == sign for row in rows]))
            frequency[f"ABIDE_{cohort}_FDRPassFrequency"] = float(np.mean([float(row[f"ABIDE_{cohort}_QValue"]) < 0.05 for row in rows]))
        frequency["BothOriginalDirectionRate"] = float(np.mean([row["PreservesOriginalDirectionBothCohorts"] == "True" for row in rows]))
        frequency["ReplicationPassFrequency"] = float(np.mean([row["Replicated"] == "True" for row in rows]))
        frequencies.append(frequency)
    write_csv(output / "mtg_thalamus_detailed_frequencies.csv", frequencies, list(frequencies[0]))
    write_csv(output / "replicate_summary.csv", summary["replicates"], list(summary["replicates"][0]))
    descriptive = []
    for metric in ("adjusted_replicated_n", "original_213_retained_n", "original_213_direction_consistency_rate",
                   "original_213_between_cohort_direction_rate", "mtg_thalamus_direction_consistency_rate",
                   "mtg_thalamus_significance_pass_rate"):
        values = [row[metric] for row in summary["replicates"]]
        descriptive.append({
            "metric": metric, "repeats": len(values), "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)), "max": float(np.max(values)),
        })
    write_csv(output / "descriptive_summary.csv", descriptive, list(descriptive[0]))
    audit_summary = {
        "passed": True, "audited_training_folds": len(checks),
        "audited_oof_archives": summary["repeats"] * 2,
        "fixed_test_folds_match_original": True, "initial_ec_unchanged": True,
        "original_files_hashes_unchanged": True, "train_test_subject_overlap_n": 0,
        "unique_numeric_sub_ids_within_each_cohort": True,
        "all_edge_ols_p_q_recomputed_and_matched": True,
    }
    (output / "validation.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    report = [
        "# W3：MPR/QSR 训练参考重采样稳定性（5 次检查）", "",
        "## 方法与边界", "",
        "固定原始初始 EC 和保存的十折测试/验证划分。在每折原训练子集中，以被试为单位、等样本量、有放回独立抽样；重复被试作为多重观测保留。每次重新拟合训练数据标准化、融合 MPR 参考、QC scaler/协变量模式/QC basis，并完整重新训练 QSR；不训练初始 EC 编码器、MPR 消融 refiner 或疾病分类器。测试被试仅用于推断。QSR 初始化和扰动 RNG 沿用原折 seed，以隔离训练参考选择变化。", "",
        "这是**训练参考选择与 QSR 重新拟合的条件敏感性分析**，不是充分考虑全部被试抽样依赖性的正式 Bootstrap 置信区间。OOF 折间训练集重叠，5 次仅用于流程检查，不报告 CI 或独立抽样推断。", "",
        "复用 `cross_cohort_replication` 的 `align_covariates` / `fit_edges`：每个队列独立拟合 `EC_edge ~ ASD + Age + Sex + C(Site) + Mean_FD`，采用常规 OLS t 检验，并对每队列全部 8,010 条非对角有向边作 BH-FDR。跨队列重复要求两队列 q < 0.05 且非零同号。原 213 条来自固定原始 Welch 重复连接文件，已从原始 EC 复算核对。", "",
        "原方向保持率：在两个队列均保留原 213 条文件中的 Hedges-g 符号（不要求显著）。另保存两队列当前 beta 同号率。原连接保留数只要求当前重复发现，方向反转与否另行统计。", "",
        "注意：严格沿用项目既有标签 `DX_GROUP=2 -> labels=1`、`DX_GROUP=1 -> labels=0`；未重新解释其临床编码。沿用既有未裁剪 Mean FD 的协变量模型，ABIDE-II 两个极端 FD 记录仍存在；本实验不替代独立 FD 敏感性分析，也不能声称对异常 FD 稳健。", "",
        "## 每次结果", "",
        "| 次数 | 跨队列重复连接 | 原 213 条保留 | 原方向保持率（双队列） | 当前队列间同向率（213 条） | MTG–丘脑原方向保持率 | MTG–丘脑重复发现通过率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["replicates"]:
        report.append(f"| {row['repeat']} | {row['adjusted_replicated_n']} | {row['original_213_retained_n']} | {row['original_213_direction_consistency_rate']:.2%} | {row['original_213_between_cohort_direction_rate']:.2%} | {row['mtg_thalamus_direction_consistency_rate']:.2%} | {row['mtg_thalamus_significance_pass_rate']:.2%} |")
    baseline_rows = read_edges(output / "baseline_adjusted/original_213_comparison.csv")
    baseline_retained = sum(row["Replicated"] == "True" for row in baseline_rows)
    baseline_direction = sum(row["PreservesOriginalDirectionBothCohorts"] == "True" for row in baseline_rows)
    baseline_mtg_retained = sum(row["BaselineReplicated"] for row in frequencies)
    report.extend(["", f"相同未裁剪 FD 协变量基线：原 QSR EC 重复 {provenance['baseline_adjusted']['adjusted_replicated_n']} 条；原 213 条保留 {baseline_retained} 条；双队列保持原方向 {baseline_direction}/213；MTG–丘脑 8 条中 {baseline_mtg_retained} 条满足调整后的重复标准。不要将协变量调整相对于原 Welch 213 条的减少全部归因于参考重采样。", "", "描述性范围：" + "；".join(f"{row['metric']}: mean={row['mean']:.4g}, SD={row['sample_sd']:.4g}, min–max={row['min']:.4g}–{row['max']:.4g}" for row in descriptive[:2]) + "。这些是 5 次训练参考检查的描述性统计，不是置信区间。"])
    report.extend(["", "## MTG–丘脑 8 条有向连接", "", "AAL 90 ROI 的 0-based 索引：THA.L=76、THA.R=77、MTG.L=84、MTG.R=85，顺序与项目现有 Panel A 一致。", "", "| 连接 | ABIDE-I 原方向保持率 | ABIDE-II 原方向保持率 | 双队列原方向保持率 | ABIDE-I q<0.05 频率 | ABIDE-II q<0.05 频率 | 跨队列重复频率 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in frequencies:
        report.append(f"| {row['Source']} → {row['Target']} | {row['ABIDE_I_OriginalDirectionRate']:.0%} | {row['ABIDE_II_OriginalDirectionRate']:.0%} | {row['BothOriginalDirectionRate']:.0%} | {row['ABIDE_I_FDRPassFrequency']:.0%} | {row['ABIDE_II_FDRPassFrequency']:.0%} | {row['ReplicationPassFrequency']:.0%} |")
    report.extend(["", "## 验证与文件", "", f"审计通过：{len(checks)} 个折内训练样本均无训练–验证/测试原始被试重叠；每被试每次恰有一次 OOF 输出；10 个新 OOF 文件均匹配原始被试、标签、初始 EC、ROI 及测试折。重新独立计算全部 beta/p/q，与保存 CSV 一致。原始文件 SHA-256 均未变化。", "", "`replicate_summary.csv` / `summary.json` 为逐次及描述性汇总；`mtg_thalamus_detailed_frequencies.csv` 为 8 条边逐队列频率；`mtg_thalamus_per_repeat.csv` 为每次 beta/p/q；`fold_sampling_audit.csv` / `validation.json` 为独立审计；每个 `repeat_XX/` 保存新 OOF、训练抽样被试、QSR 训练指标及完整协变量统计；`provenance.json` / `*_fixed_partitions.npz` 保存原参数、哈希和固定分区。", ""])
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(audit_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
