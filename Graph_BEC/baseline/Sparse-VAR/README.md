# Sparse VAR baseline

This baseline replaces Graph-BEC's learned BEC with a subject-level sparse
vector autoregressive representation. For each subject, ROI `j` is regressed
on lagged values of every ROI using an elastic-net penalty:

```text
y_t = A_1 y_(t-1) + ... + A_p y_(t-p) + e_t
```

The lagged coefficient matrices are weighted and averaged into one signed,
directed `ROI x ROI` BEC. The generated BEC is then passed to the same
fold-safe scaling, stratified splits, BrainNetCNN classifier, and metrics as
the other baselines.

## Run

From the repository root:

```bash
python Graph_BEC/baseline/Sparse-VAR/run_sparse_var.py \
  --dataset abide \
  --data-root /path/to/abide \
  --lags 1 \
  --alpha 0.05 \
  --l1-ratio 1.0
```

Useful modes are `--generation-only`, `--classification-only`, and
`--regenerate-bec`. The archive contains `bec`, labels, subject/site IDs, and
`var_coefficients` with shape `[subjects, lags, roi, roi]`.

`alpha` controls sparsity; larger values produce fewer directed edges.
`l1-ratio=1` is the LASSO-like sparse VAR setting, while lower values add an
L2 stabilizer. The default `lags=1` is recommended for short fMRI runs.
