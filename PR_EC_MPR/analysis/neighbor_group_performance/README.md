# Neighborhood-Grouped Performance Analysis

This supplementary analysis asks when fold-local neighbor information is associated with stronger or weaker downstream classification performance. It does not change Initial EC estimation, MPR refinement, QSR training, cross-validation splits, random seeds, neighbor count, or BrainNetCNN model selection.

## Why evaluation must be rerun

The existing `PR_EC_MPR/outputs/*` directories contain the cached Initial EC matrices, OOF MPR/QSR-refined EC matrices, and fold-level aggregate metrics. They do not contain subject-level test probabilities, classifier checkpoints, or fold-local neighbor records. The script therefore:

1. reuses the existing Initial EC archive without retraining Initial EC;
2. reconstructs the original ten folds and fold-local preprocessing;
3. reruns the unchanged MPR, QSR, and downstream BrainNetCNN stages;
4. records test probabilities and neighborhood statistics during that rerun.

The cached OOF refined matrices are not used to refit one global classifier, because that would differ from the original fold-local training protocol.

## Definitions

- **Initial EC**: the original fold-standardized EC representation.
- **MPR-only**: the existing `refined` representation produced by the PGR/MPR refiner.
- **Full PR-EC**: the existing `qc_refined` representation produced by the QSR stage in the current pipeline.
- **Neighbor match**: weighted mean FC cosine similarity between a query and its final selected neighbors. FC upper-triangle features are standardized with the current fold's training mean and standard deviation, then cosine similarity is clipped to `[0, 1]`, exactly as in the fusion retrieval path. The similarity is measured before graph row normalization and averaged with the final fused aggregation weights.
- **Neighbor motion**: weighted mean neighbor `func_mean_fd`, using the same final fused aggregation weights. Missing training-neighbor FD values are filled with the current training fold median only.
- **Groups**: each fold's training-subject median is fixed as the threshold for its test subjects. Matching and motion are grouped independently. Diagnosis and predictions are not used to set thresholds.
- **Mean edge variance**: variance is first computed across subjects for every directed edge; the diagonal is excluded; remaining edge variances are averaged. MPR and Full PR-EC retention rates use Initial EC from the same group as denominator. All three representations use the same fold-level EC scaling.
- **Group AUC difference**: within the same fold and group, `MPR-only AUC - Initial EC AUC` or `Full PR-EC AUC - Initial EC AUC`. A single-class group receives `NaN` and an explicit reason rather than zero.

Training subjects select neighbors only from their own training fold and exclude themselves. Validation and test subjects select neighbors only from that fold's training subjects. Test labels are used only after prediction for AUC and class-count reporting.

## Mean FD fields

All three phenotype files contain `func_mean_fd`, which is also the first default QSR QC column:

- ABIDE-I: `func_mean_fd`; the source CSV contains some missing entries, handled fold-locally if any occur among aligned subjects.
- ABIDE-II: `func_mean_fd`.
- ADHD200: `func_mean_fd`.

The exact aligned-subject missing count and every fold's training missing count/imputation median are saved in the audit outputs.

## Run

GPU execution is mandatory; the script intentionally has no CPU fallback. Use the existing `default` environment by its environment path:

```bash
conda run -p /data/users/liulin/miniconda3/envs/default \
  python -m PR_EC_MPR.analysis.neighbor_group_performance.run_analysis \
  --dataset all --gpu-id auto
```

Run one dataset with `--dataset abide-i`, `--dataset abide-ii`, or `--dataset adhd200`. Set `--gpu-id 0` to select a specific GPU.

## Outputs

Outputs are written under `outputs/<dataset>/`:

- `subject_level.csv`: subject ID, dataset, fold, true label, both neighborhood metrics, training-fold thresholds, both group assignments, and three test probabilities.
- `fold_group_results.csv`: group counts, patient/control counts, three AUCs, paired AUC differences, mean edge variances, variance-retention rates, missing-FD audit, and AUC failure reason.
- `dataset_summary.csv`: the required four group rows with fold AUC mean/SD, paired AUC-difference mean/SD, variance-retention mean/SD, total OOF counts, and valid-fold counts.
- `neighbor_audit.csv`: every selected training/test neighbor, final aggregation weight, pre-normalization FC similarity, raw neighbor FD, and fold-imputed FD.
- `group_auc_summary.png`: descriptive visualization of group AUCs and within-group AUC differences.
- `run_audit.json`: source configuration, GPU, aligned FD missingness, and leakage-check status.

`outputs/all_datasets_summary.csv` combines the four summary rows from ABIDE-I, ABIDE-II, and ADHD200.

This is a descriptive downstream analysis. Lower matching, higher neighbor motion, classification changes, and variance-retention changes must not be interpreted as causal negative-transfer mechanisms or as direct evidence of true EC recovery accuracy.
