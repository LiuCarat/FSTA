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

## ABIDE-II

ABIDE-II is supported through the shared loader. The default input layout is
`dataset/ABIDE-II/cpac/filt_noglobal/*_rois_aal.1D`, with labels loaded from
the ABIDE-II phenotype file. Use a separate output path for this dataset:

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/Sparse-VAR/run_sparse_var.py \
  --dataset abide_ii \
  --data-root ./dataset/ABIDE-II \
  --lags 1 \
  --alpha 0.03 \
  --l1-ratio 1.0 \
  --max-iter 10000 \
  --tol 1e-4 \
  --output-dir Graph_BEC/baseline/Sparse-VAR/outputs/abide_ii \
  --bec-path Graph_BEC/baseline/Sparse-VAR/outputs/abide_ii/subject_sparse_var_bec_abide_ii.npz \
  --regenerate-bec
```

Sparse-VAR has no neural-network `epoch`. Its generation-side convergence
control is `--max-iter`, applied independently to each target ROI regression.
For ABIDE-II, keep `lags=1` initially because each subject has only about
120--160 time points and 90 ROIs. Start with `max-iter=10000`; reduce to
`3000` for a speed check, and increase to `20000` only if convergence warnings
appear. For regularization, compare `alpha=0.01`, `0.03`, and `0.1` while
keeping the other settings fixed.

The `--classifier-epochs` and `--classifier-patience` options only control the
downstream Graph-BEC classifier, not Sparse-VAR BEC generation. The existing
defaults `100` and `20` are appropriate starting values.

Quick smoke test:

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/Sparse-VAR/run_sparse_var.py \
  --dataset abide_ii \
  --data-root ./dataset/ABIDE-II \
  --max-subjects 2 \
  --max-iter 1000 \
  --n-splits 2 \
  --generation-only \
  --output-dir Graph_BEC/baseline/Sparse-VAR/outputs/smoke_abide_ii \
  --bec-path Graph_BEC/baseline/Sparse-VAR/outputs/smoke_abide_ii/subject_sparse_var_bec_abide_ii.npz \
  --regenerate-bec
```

## ADHD200

ADHD200 is supported through the shared Graph-BEC loader. The default input
files are `dataset/ADHD200/Phenotypic_Processing.csv` and
`dataset/ADHD200/cpac/filt_noglobal/*_rois_aal.1D`. `DX=0` is the control
group, while `DX=1/2/3` is mapped to the patient group. The loader reads 116
source ROIs and keeps the first 90, matching Sparse-VAR's BEC dimensions.
ADHD200 outputs default to `Graph_BEC/baseline/Sparse-VAR/outputs/adhd200/`.

Quick generation smoke test:

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/Sparse-VAR/run_sparse_var.py \
  --dataset adhd200 \
  --data-root ./dataset/ADHD200 \
  --max-subjects 2 \
  --lags 1 \
  --max-iter 1000 \
  --generation-only \
  --gpu-id cpu \
  --output-dir Graph_BEC/baseline/Sparse-VAR/outputs/smoke_adhd200 \
  --bec-path Graph_BEC/baseline/Sparse-VAR/outputs/smoke_adhd200/subject_sparse_var_bec_adhd200.npz \
  --regenerate-bec
```

For a full ADHD200 run, remove `--max-subjects 2` and increase `--max-iter`
as needed. `--lags 1` is recommended initially because ADHD200 subjects have
short and variable scan lengths.
