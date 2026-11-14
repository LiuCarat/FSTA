# Rebuttal R3 paired AUC statistics

Run from the repository root:

```bash
python PR_EC/analysis/rebuttal_stats/paired_auc_r3.py
```

The script reads existing `PR_EC/outputs/*/experiment_summary.csv` files only. It
requires exactly folds 1--10 and computes, within each dataset, the same-fold
paired differences:

- `refined_AUC - original_AUC`;
- `qc_refined_AUC - refined_AUC`;
- `qc_refined_AUC - original_AUC`.

AUC differences are reported in percentage points. The 95% interval is a two-sided
paired t interval over the 10 fold-level differences (`df=9`, sample SD with
`ddof=1`). This describes consistency across the saved folds; folds have overlapping
training sets, so it is not an independent-sample significance claim.

Outputs:

- `results/Table_R3.md`: rebuttal-ready table;
- `results/Table_R3.json`: full precision summary;
- `results/{dataset}_fold_differences.csv`: raw fold AUCs are not duplicated, but
  each same-fold delta is retained in AUC units and percentage points.

The current repository contains ABIDE-I and ABIDE-II experiment summaries. It does
not contain `PR_EC/outputs/adhd200/experiment_summary.csv`, so ADHD200 is reported as
unavailable rather than fabricated.

## Subject-level paired bootstrap

The current archives save EC matrices, labels and fold IDs, but not classifier test
probabilities. Therefore a subject-level OOF bootstrap cannot be computed from the
existing outputs without rerunning the classifier stage. The compatible changes are:

1. `PR_EC/downstream/classifier.py::train_classifier`: add an optional
   `return_details=False` argument. When true, return the test probabilities, test
   labels, fold validation probabilities and selected validation threshold in a
   third dictionary; keep the current two-value return as the default.
2. `PR_EC/workflow.py::run_fold`: for each representation, collect the test
   probabilities and original subject IDs when the optional save flag is enabled.
   The classifier must still select its threshold from validation labels only.
3. `PR_EC/workflow.py::run_cross_validation` and `PR_EC/utils/output.py::save_results`:
   concatenate the held-out predictions by original subject index and save one NPZ
   per dataset containing `subject_id`, `fold_id`, `label`, `original_prob`,
   `refined_prob`, and `qc_refined_prob`.

Then a separate bootstrap script can resample the same OOF subjects with
class-stratified bootstrap (e.g. 2000 draws) and recompute ROC AUC differences for
each draw. That CI is an OOF subject-level estimand and must be reported separately
from the 10-fold mean-difference CI; it must not be substituted for it.
