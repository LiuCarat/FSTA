# Current fold-specific inductive results (5-fold, provisional)

Values are percentages and are means over the five available folds. Confidence intervals are two-sided paired t intervals over the five same-fold AUC differences. These outputs are not the final 10-fold Table R4.

| Dataset | Method | ACC | SEN | AUC | PRE | F1 | ΔAUC vs. Initial [95% CI] | ΔAUC vs. MPR [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ABIDE-I | Initial EC | 55.46 | 58.38 | 61.28 | 51.63 | 54.02 | — | — |
|  | +MPR | 55.57 | 58.82 | 60.03 | 52.59 | 54.48 | -1.25 [-3.35, 0.85] | — |
|  | Full PR-EC | 58.21 | 60.55 | 63.41 | 54.95 | 57.41 | 2.14 [-1.77, 6.04] | 3.38 [-0.21, 6.98] |
| ABIDE-II | Initial EC | 55.69 | 69.24 | 60.56 | 52.77 | 59.49 | — | — |
|  | +MPR | 55.69 | 68.60 | 60.57 | 52.78 | 59.27 | 0.00 [-0.01, 0.02] | — |
|  | Full PR-EC | 55.79 | 71.63 | 61.24 | 52.81 | 60.32 | 0.68 [-1.20, 2.56] | 0.67 [-1.21, 2.56] |
| ADHD200 | Initial EC | 51.54 | 65.00 | 56.49 | 40.43 | 49.03 | — | — |
|  | +MPR | 54.95 | 46.43 | 56.32 | 45.24 | 40.44 | -0.17 [-1.95, 1.62] | — |
|  | Full PR-EC | 51.66 | 72.86 | 60.20 | 40.62 | 51.80 | 3.71 [-4.41, 11.83] | 3.88 [-4.23, 11.98] |
