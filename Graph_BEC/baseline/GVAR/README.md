# GVAR baseline

This baseline uses the original GVAR self-explaining neural autoregressive
model (`SENNGC`) to fit one subject at a time from fMRI ROI time series. It
exports a signed directed `ROI x ROI` BEC and optionally uses the shared
Graph-BEC classifier.

## Install

```bash
conda activate default
cd /data/users/liulin/PythonCode/ST-MRI/FSTA
python -m pip install -r Graph_BEC/baseline/GVAR/requirements.txt
```

PyTorch is also required by the GVAR model and the downstream classifier. The
original GVAR implementation was designed for CUDA; use a CUDA-compatible
PyTorch installation for practical runtimes. `--cpu` is available for small
checks but is usually slow for 90 ROIs.

## Test

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/GVAR/run_gvar.py \
  --dataset abide \
  --data-root ./dataset/ABIDE-I \
  --order 1 \
  --epochs 2 \
  --max-subjects 2 \
  --generation-only
```

## Full BEC generation

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/GVAR/run_gvar.py \
  --dataset abide \
  --data-root ./dataset/ABIDE-I \
  --order 1 \
  --epochs 100 \
  --workers 4 \
  --generation-only
```

The default output is:

```text
Graph_BEC/baseline/GVAR/outputs/subject_gvar_bec_abide.npz
```

## ABIDE-II

ABIDE-II uses the shared Graph-BEC loader and the default layout under
`dataset/ABIDE-II/cpac/filt_noglobal/`. Use a dataset-specific output path so
that ABIDE-I results are not overwritten:

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/GVAR/run_gvar.py \
  --dataset abide_ii \
  --data-root ./dataset/ABIDE-II \
  --order 1 \
  --epochs 200 \
  --batch-size 128 \
  --workers 1 \
  --gpu-id auto \
  --output-dir Graph_BEC/baseline/GVAR/outputs/abide_ii \
  --bec-path Graph_BEC/baseline/GVAR/outputs/abide_ii/subject_gvar_bec_abide_ii.npz \
  --regenerate-bec
```

For ABIDE-II, `epochs=100` is a reasonable smoke-test value and `epochs=200`
is a practical starting point for the full BEC generation. Compare
`epochs=100`, `200`, and `300` on the same subjects; use `300` only if the
resulting BECs are still changing materially. The runner trains one GVAR per
subject, so increasing epochs increases total runtime approximately linearly.

Quick smoke test:

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/GVAR/run_gvar.py \
  --dataset abide_ii \
  --data-root ./dataset/ABIDE-II \
  --epochs 2 \
  --batch-size 128 \
  --max-subjects 2 \
  --generation-only \
  --gpu-id auto \
  --output-dir Graph_BEC/baseline/GVAR/outputs/smoke_abide_ii \
  --bec-path Graph_BEC/baseline/GVAR/outputs/smoke_abide_ii/subject_gvar_bec_abide_ii.npz \
  --regenerate-bec
```

## Classification

```bash
PYTHONUNBUFFERED=1 python Graph_BEC/baseline/GVAR/run_gvar.py \
  --dataset abide \
  --data-root ./dataset/ABIDE-I \
  --classification-only \
  --gpu-id auto
```

The output files are `metrics_abide.csv`, `metrics_abide.json`, and
`summary_abide.json` under `Graph_BEC/baseline/GVAR/outputs`.

The method reported in tables should be `GVAR`. `gvar_coefficients` stores
median signed generalized coefficient matrices with shape
`[subjects, order, roi, roi]`; the lag matrices are weighted by
`--lag-decay` to form `bec`. The default `--workers 1` is serial, while larger
values fit independent subjects in separate CPU/GPU processes and preserve
subject ordering in the archive. Do not use more GPU workers than available
GPU memory allows.
