# Pearson FC baseline

This baseline replaces the subject-level FSTA/Graph-BEC BEC with a standard
Pearson correlation matrix computed from each subject's ROI time series. The
downstream evaluation is kept identical to Graph-BEC:

- same subject labels and subject order;
- same stratified train/validation/test folds;
- fold-local BEC/FC standardization;
- same Directed BrainNetCNN classifier;
- same classifier hyperparameters and metrics.

The generated archive uses the common `subject_bec.npz` layout, with the
matrix stored in the `bec` field so it can be passed through the existing
classifier code. Pearson FC is symmetric and has its diagonal set to zero.

## ABIDE-I

From the repository root:

```bash
python Graph_BEC/baseline/Pearson-FC/run_pearson_fc.py \
  --dataset abide \
  --gpu-id auto
```

## ADHD200

```bash
python Graph_BEC/baseline/Pearson-FC/run_pearson_fc.py \
  --dataset adhd200 \
  --gpu-id auto
```

Because Pearson FC has no learned generation stage, the matrix generation is
fast. The FC archive and classifier outputs are written under. Files are dataset-specific so ABIDE and ADHD200 do not overwrite each other:

```text
Graph_BEC/baseline/Pearson-FC/outputs/
```

Useful options:

```bash
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

For example, generate only:

```bash
python Graph_BEC/baseline/Pearson-FC/run_pearson_fc.py \
  --dataset abide \
  --generation-only
```

Then classify an existing archive:

```bash
python Graph_BEC/baseline/Pearson-FC/run_pearson_fc.py \
  --dataset abide \
  --classification-only \
  --fc-path Graph_BEC/baseline/Pearson-FC/outputs/subject_fc_abide.npz
```
