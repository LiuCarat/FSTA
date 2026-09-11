#!/usr/bin/env python3
"""Summarize whole-brain reproducibility from the merged edge table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "Graph_BEC/analysis/cross_cohort_replication/outputs/all_edges_cross_cohort.csv"
DEFAULT_OUTPUT_DIR = ROOT / "Graph_BEC/analysis/cross_cohort_replication/outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90 * 89:
        raise ValueError(f"Expected 8010 directed edges, found {len(rows)}")
    return rows


def fisher_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return r, r
    z = np.arctanh(r)
    standard_error = 1 / np.sqrt(n - 3)
    critical = 1.959963984540054
    return tuple(float(value) for value in np.tanh(
        z + np.array([-1, 1]) * critical * standard_error
    ))


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")
    rows = load_table(args.input)
    g_i = np.asarray([float(row["ABIDE_I_Hedges_g"]) for row in rows])
    g_ii = np.asarray([float(row["ABIDE_II_Hedges_g"]) for row in rows])
    q_i = np.asarray([float(row["ABIDE_I_QValue"]) for row in rows])
    q_ii = np.asarray([float(row["ABIDE_II_QValue"]) for row in rows])
    rank_i = np.asarray([int(row["ABIDE_I_Rank"]) for row in rows])
    rank_ii = np.asarray([int(row["ABIDE_II_Rank"]) for row in rows])

    both_significant = (q_i < args.alpha) & (q_ii < args.alpha)
    same_direction = g_i * g_ii > 0
    replicated = both_significant & same_direction
    opposite_significant = both_significant & ~same_direction
    neither_significant = (q_i >= args.alpha) & (q_ii >= args.alpha)
    i_only = (q_i < args.alpha) & (q_ii >= args.alpha)
    ii_only = (q_i >= args.alpha) & (q_ii < args.alpha)
    regression = linregress(g_i, g_ii)
    pearson = pearsonr(g_i, g_ii)
    spearman = spearmanr(g_i, g_ii)
    rank_spearman = spearmanr(rank_i, rank_ii)

    summary = {
        "n_directed_edges": len(rows),
        "alpha": args.alpha,
        "effect_size": {
            "pearson_r": float(pearson.statistic),
            "pearson_p_value_descriptive": float(pearson.pvalue),
            "pearson_95_ci": fisher_ci(float(pearson.statistic), len(rows)),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value_descriptive": float(spearman.pvalue),
            "spearman_95_ci": fisher_ci(float(spearman.statistic), len(rows)),
            "regression_slope_II_on_I": float(regression.slope),
            "regression_intercept": float(regression.intercept),
            "mean_g_abide_i": float(g_i.mean()),
            "mean_g_abide_ii": float(g_ii.mean()),
            "mean_absolute_g_abide_i": float(np.abs(g_i).mean()),
            "mean_absolute_g_abide_ii": float(np.abs(g_ii).mean()),
            "mean_absolute_pairwise_difference": float(np.abs(g_ii - g_i).mean()),
            "rmse_pairwise_difference": float(np.sqrt(np.mean((g_ii - g_i) ** 2))),
        },
        "direction": {
            "same_direction_count": int(same_direction.sum()),
            "same_direction_rate": float(same_direction.mean()),
            "opposite_direction_count": int((~same_direction).sum()),
            "opposite_direction_rate": float((~same_direction).mean()),
        },
        "fdr_significance": {
            "abide_i_q_below_alpha": int((q_i < args.alpha).sum()),
            "abide_ii_q_below_alpha": int((q_ii < args.alpha).sum()),
            "both_q_below_alpha": int(both_significant.sum()),
            "abide_i_only": int(i_only.sum()),
            "abide_ii_only": int(ii_only.sum()),
            "neither": int(neither_significant.sum()),
            "both_significant_same_direction": int(replicated.sum()),
            "both_significant_same_direction_rate_of_all_edges": float(replicated.mean()),
            "both_significant_same_direction_rate_of_both_significant": float(replicated.sum() / both_significant.sum()),
            "both_significant_opposite_direction": int(opposite_significant.sum()),
        },
        "rank_stability_descriptive": {
            "spearman_rho": float(rank_spearman.statistic),
            "p_value_descriptive": float(rank_spearman.pvalue),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "rq1_whole_brain_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    e = summary["effect_size"]
    d = summary["direction"]
    s = summary["fdr_significance"]
    r = summary["rank_stability_descriptive"]
    report = f"""# RQ1: Whole-brain reproducibility

Input: `all_edges_cross_cohort.csv`  
Edges: **{len(rows)} directed non-diagonal BEC edges**  
Threshold: **BH-FDR q < {args.alpha:g}** within each cohort

## Results

| Metric | Value |
|---|---:|
| Pearson correlation of Hedges' g | {e['pearson_r']:.3f} (95% CI {e['pearson_95_ci'][0]:.3f}–{e['pearson_95_ci'][1]:.3f}) |
| Spearman correlation of Hedges' g | {e['spearman_rho']:.3f} (95% CI {e['spearman_95_ci'][0]:.3f}–{e['spearman_95_ci'][1]:.3f}) |
| Same direction across all edges | {d['same_direction_count']}/{len(rows)} ({d['same_direction_rate']:.1%}) |
| Both cohorts q < {args.alpha:g} | {s['both_q_below_alpha']}/{len(rows)} ({s['both_q_below_alpha']/len(rows):.1%}) |
| Both significant and same direction | {s['both_significant_same_direction']}/{len(rows)} ({s['both_significant_same_direction_rate_of_all_edges']:.1%}) |
| Same direction among both-significant edges | {s['both_significant_same_direction_rate_of_both_significant']:.1%} |
| Within-cohort rank Spearman correlation | {r['spearman_rho']:.3f} |

## Interpretation

The two cohorts show a **positive but moderate whole-brain agreement** in signed
effect sizes (`r = {e['pearson_r']:.3f}`, Spearman `rho = {e['spearman_rho']:.3f}`),
not a near-perfect replication of all edge effects. The raw direction agreement is
{d['same_direction_rate']:.1%}; after requiring independent-cohort FDR significance
in both cohorts, {s['both_significant_same_direction']} edges remain replicated with
the same direction. Thus, RQ1 is best answered as: **there is a reproducible
whole-brain effect pattern at the aggregate level, but reproducibility is selective
rather than universal across all 8010 edges.**

The low rank correlation (`rho = {r['spearman_rho']:.3f}`) further indicates that
within-cohort effect-size rank is unstable and should not define the replicated set.

The correlation p-values are reported descriptively because edges are biologically
and statistically dependent; the effect-size correlation, direction concordance,
and dual-cohort FDR overlap are the primary evidence.
"""
    report_path = args.output_dir / "rq1_whole_brain_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
