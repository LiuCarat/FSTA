# RQ1: Whole-brain reproducibility

Input: `all_edges_cross_cohort.csv`  
Edges: **8010 directed non-diagonal BEC edges**  
Threshold: **BH-FDR q < 0.05** within each cohort

## Results

| Metric | Value |
|---|---:|
| Pearson correlation of Hedges' g | 0.373 (95% CI 0.354–0.392) |
| Spearman correlation of Hedges' g | 0.351 (95% CI 0.332–0.370) |
| Same direction across all edges | 4890/8010 (61.0%) |
| Both cohorts q < 0.05 | 173/8010 (2.2%) |
| Both significant and same direction | 168/8010 (2.1%) |
| Same direction among both-significant edges | 97.1% |
| Within-cohort rank Spearman correlation | 0.099 |

## Interpretation

The two cohorts show a **positive but moderate whole-brain agreement** in signed
effect sizes (`r = 0.373`, Spearman `rho = 0.351`),
not a near-perfect replication of all edge effects. The raw direction agreement is
61.0%; after requiring independent-cohort FDR significance
in both cohorts, 168 edges remain replicated with
the same direction. Thus, RQ1 is best answered as: **there is a reproducible
whole-brain effect pattern at the aggregate level, but reproducibility is selective
rather than universal across all 8010 edges.**

The low rank correlation (`rho = 0.099`) further indicates that
within-cohort effect-size rank is unstable and should not define the replicated set.

The correlation p-values are reported descriptively because edges are biologically
and statistically dependent; the effect-size correlation, direction concordance,
and dual-cohort FDR overlap are the primary evidence.
