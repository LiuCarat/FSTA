# NAVAR Graph-BEC Baseline

This directory contains the original NAVAR model implementation and a thin
Graph-BEC experiment runner. NAVAR is a Neural Additive Vector Autoregression
model for learning directed dependencies in multivariate time series.

The original model is kept in `NAVAR.py`; the reusable training function is in
`train_NAVAR.py`; and `dataloader.py` prepares lagged time-series examples.
`run_navar_baseline.py` is the only experiment entry point used by Graph-BEC.

## Baseline protocol

For every subject, the runner:

1. loads the subject's ROI time series;
2. trains an independent NAVAR model;
3. computes the standard deviation of the learned additive contributions;
4. sets diagonal self-contributions to zero;
5. stores the resulting directed matrix in a subject-level `.npz` archive;
6. evaluates the archive with the shared Graph-BEC downstream classifier.

The matrix convention follows NAVAR: `[source, target]` is the contribution
from a source ROI to a target ROI. The archive fields are:

- `bec`: matrices with shape `[subjects, ROIs, ROIs]`;
- `navar_scores`: the same matrices under a method-specific name;
- `labels`, `subject_ids`, `site_ids`: subject metadata;
- `navar_config`: NAVAR parameters used to generate the archive;
- `roi_names`: generated ROI labels.

## Installation

```bash
python -m pip install -r Graph_BEC/baseline/NAVAR/requirements.txt
```

## Generate and classify

```bash
python Graph_BEC/baseline/NAVAR/run_navar_baseline.py \
  --dataset abide \
  --data-root ./dataset/ABIDE-I
```

Generate only subject-level BEC matrices:

```bash
python Graph_BEC/baseline/NAVAR/run_navar_baseline.py \
  --dataset abide \
  --data-root ./dataset/ABIDE-I \
  --generation-only
```

Classify an existing archive:

```bash
python Graph_BEC/baseline/NAVAR/run_navar_baseline.py \
  --dataset abide \
  --classification-only
```

The default archive is saved to
`outputs/subject_navar_bec_<dataset>.npz`. Metrics are saved to
`outputs/metrics_navar_<dataset>.json`,
`outputs/metrics_navar_<dataset>.csv`, and
`outputs/summary_navar_<dataset>.json`.

Use `--regenerate-bec` after changing any NAVAR fitting parameter. Use
`--gpu-id cpu` to force CPU execution; otherwise NAVAR fitting and the
downstream classifier use CUDA when available.

The original NAVAR paper and supplementary material are retained in `paper/`;
the original DREAM/CauseMe example data are retained in `experiments/` as
reference assets, but they are not used by the Graph-BEC runner.
