# Partial Correlation FC baseline

This baseline replaces the subject-level FSTA/Graph-BEC BEC with a sparse
partial-correlation matrix estimated from each subject's ROI time series using a regularized precision
matrix. The default is Ledoit-Wolf shrinkage; Graphical Lasso is available as
an optional sparse estimator. The downstream evaluation is kept identical to Graph-BEC:

- same subject labels and subject order;
- same stratified train/validation/test folds;
- fold-local BEC/FC standardization;
- same Directed BrainNetCNN classifier;
- same classifier hyperparameters and metrics.

For each subject, a regularized precision matrix `Theta` is estimated. The
partial correlation is computed as:

```text
P[i,j] = -Theta[i,j] / sqrt(Theta[i,i] * Theta[j,j])
```

The diagonal is set to zero. Zero-variance ROIs are excluded from the fit and
represented by all-zero rows and columns. The archive uses the common
`subject_bec.npz` layout, with the matrix stored in the `bec` field.

## ABIDE-I

```bash
python Graph_BEC/baseline/Partial-Correlation-FC/run_partial_correlation_fc.py \
  --dataset abide \
  --gpu-id auto
```

## ADHD200

```bash
python Graph_BEC/baseline/Partial-Correlation-FC/run_partial_correlation_fc.py \
  --dataset adhd200 \
  --gpu-id auto
```

## ABIDE-II

```bash
python Graph_BEC/baseline/Partial-Correlation-FC/run_partial_correlation_fc.py \
  --dataset abide_ii \
  --gpu-id auto
```

The ABIDE-II profile uses `dataset/ABIDE-II/Phenotypic_Processing.csv` and
`dataset/ABIDE-II/cpac/filt_noglobal/*_rois_aal.1D` by default. The output
archive is `subject_partial_correlation_fc_abide_ii.npz`.

The archive and metrics are saved under:

```text
Graph_BEC/baseline/Partial-Correlation-FC/outputs/
```

Dataset-specific files prevent ABIDE and ADHD200 from overwriting each other:

```text
subject_partial_correlation_fc_abide.npz
subject_partial_correlation_fc_abide_ii.npz
subject_partial_correlation_fc_adhd200.npz
metrics_abide.csv / metrics_abide_ii.csv / metrics_adhd200.csv
summary_abide.json / summary_abide_ii.json / summary_adhd200.json
```

## Main options

```bash
--gl-alpha 0.1
--gl-max-iter 200
--max-subjects 20
--n-splits 10
--classifier-epochs 100
--classifier-patience 20
--classifier-lr 1e-3
--classifier-repeats 1
--generation-only
--classification-only
--regenerate-fc
```

`--estimator ledoit-wolf` is the recommended stable default. Use
`--estimator graphical-lasso` when you specifically need a sparse precision
matrix; then `--gl-alpha` controls sparsity and `--gl-max-iter` controls the
solver budget. Select these settings before the final run or through a
training-only validation scheme; do not tune them using test folds.

Generate only:

```bash
python Graph_BEC/baseline/Partial-Correlation-FC/run_partial_correlation_fc.py \
  --dataset abide \
  --generation-only
```

Classify an existing archive:

```bash
python Graph_BEC/baseline/Partial-Correlation-FC/run_partial_correlation_fc.py \
  --dataset abide_ii \
  --classification-only \
  --fc-path Graph_BEC/baseline/Partial-Correlation-FC/outputs/subject_partial_correlation_fc_abide_ii.npz
```
