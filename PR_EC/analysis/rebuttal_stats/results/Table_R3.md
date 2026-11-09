# Table R3. Paired AUC differences across ten folds

Values are AUC percentage-point change, 10-fold mean delta [95% paired t CI], positive folds/10.
The CI is computed from the ten same-fold paired differences, not from two independent method summaries.
Fold training sets overlap; these fold-level CIs are stability summaries, not independent-sample significance tests.
`QC-refined − MPR-only` compares two independently configured reference-based branches; it is not a strict isolated QSR causal contribution.

| Dataset | MPR − Initial | QC-refined − MPR-only | QC-refined − Initial |
| --- | --- | --- | --- |
| ABIDE-I | -0.41 [-1.49, +0.67], 5/10 | +1.07 [-0.11, +2.24], 8/10 | +0.66 [-0.55, +1.87], 4/10 |
| ABIDE-II | -1.14 [-2.64, +0.36], 3/10 | +1.44 [+0.06, +2.83], 7/10 | +0.30 [-0.16, +0.77], 2/10 |
| ADHD200 | +0.19 [-0.68, +1.06], 5/10 | +0.34 [-0.45, +1.13], 7/10 | +0.53 [-0.56, +1.62], 4/10 |

Format: `ΔAUC percentage points [95% CI], positive folds/10`.
A positive fold means the left-hand representation has larger test AUC in that same fold.
Statistics use sample SD (`ddof=1`) and a two-sided t critical value with df=9.

## Missing outputs
