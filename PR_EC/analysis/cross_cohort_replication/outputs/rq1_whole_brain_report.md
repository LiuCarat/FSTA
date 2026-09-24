# RQ1: Whole-brain reproducibility

Input: `all_edges_cross_cohort.csv`
Edges: **8010 directed non-diagonal EC edges**
Threshold: **BH-FDR q < 0.05** within each cohort

## Results

| Metric | Value |
|---|---:|
| Pearson correlation of Hedges' g | 0.351 (95% CI 0.332–0.370) |
| Spearman correlation of Hedges' g | 0.330 (95% CI 0.311–0.350) |
| Same direction across all edges | 4840/8010 (60.4%) |
| Both cohorts q < 0.05 | 224/8010 (2.8%) |
| Both significant and same direction | 213/8010 (2.7%) |
| Same direction among both-significant edges | 95.1% |
| Within-cohort rank Spearman correlation | 0.096 |

## Interpretation

The two cohorts show a **positive but moderate whole-brain agreement** in signed
effect sizes (`r = 0.351`, Spearman `rho = 0.330`),
not a near-perfect replication of all edge effects. The raw direction agreement is
60.4%; after requiring independent-cohort FDR significance
in both cohorts, 213 edges remain replicated with
the same direction. Thus, RQ1 is best answered as: **there is a reproducible
whole-brain effect pattern at the aggregate level, but reproducibility is selective
rather than universal across all 8010 edges.**

The low rank correlation (`rho = 0.096`) further indicates that
within-cohort effect-size rank is unstable and should not define the replicated set.

The correlation p-values are reported descriptively because edges are biologically
and statistically dependent; the effect-size correlation, direction concordance,
and dual-cohort FDR overlap are the primary evidence.
