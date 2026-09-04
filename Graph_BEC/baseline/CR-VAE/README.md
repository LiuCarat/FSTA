# CR-VAE

`run_abide_classifier.py` evaluates whether subject-level CR-VAE causal
matrices can distinguish ASD from TC. It does not modify the Graph-BEC main
program.

## Evaluation protocol

CR-VAE's `GC()` matrix belongs to a fitted model rather than an individual
input window. A single CR-VAE trained on all ABIDE subjects would therefore
produce one shared matrix, which cannot serve as subject-level classification
features. This runner instead uses the following protocol:

1. Fit one CR-VAE model to each subject's time series.
2. Extract the continuous `GC(threshold=False)` matrix as that subject's BEC.
3. Save all matrices once to `outputs/subject_bec.npz`.
4. Keep those BECs frozen during stratified 10-fold classification.
5. Fit the BEC scaler and BrainNetCNN classifier only on each fold's training
   data, with validation-based early stopping.

The archive contains `bec`, `labels`, `subject_ids`, and `site_ids`, matching
the core layout of the Graph-BEC `subject_bec.npz` file.

## Run generation and classification

From the repository root:

```bash
python Graph_BEC/baseline/CR-VAE/run_abide_classifier.py \
  --data-root dataset/ABIDE-I \
  --gpu-id auto \
  --n-splits 10 \
  --classifier-epochs 100 \
  --classifier-patience 20 \
  --classifier-lr 1e-3 \
  --classifier-repeats 1
```

The original CR-VAE Phase-I function defines `batch_size=2048`, and the
Henon demo uses that value implicitly because it does not pass `batch_size`.
For ABIDE's 90-ROI model, this runner defaults to the safer
`--crvae-batch-size 256`; the other main Phase-I defaults remain
`context=20`, `hidden=64`, `max_iter=1000`, `lr=5e-2`, `lambda=0.1`, and
`ridge=0`. You can reproduce the original batch setting explicitly with
`--crvae-batch-size 2048` if your GPU has enough memory.

Generation is checkpointed after every subject by default. Each checkpoint is
written to a temporary file and atomically replaces the archive, so an
interrupted run keeps all completed subjects and does not leave a partially
written `.npz`. Re-running the same command resumes an incomplete archive or
reuses a complete one. Use `--checkpoint-every N` to trade restart granularity
for fewer archive writes.

For a much faster first result, use `--fast`. It keeps the CR-VAE model and
the requested `context=20`, `hidden=64`, `lr=0.05`, and `lambda=0.1`, but uses
100 iterations and batch size 64. The archive is checkpointed every 10
subject by default; use `--checkpoint-every 1` explicitly if you want to make
that behavior clear in a command.

## Run the two stages separately

Generate BECs once:

```bash
python Graph_BEC/baseline/CR-VAE/run_abide_classifier.py \
  --data-root dataset/ABIDE-I \
  --gpu-id auto \
  --fast \
  --bec-path Graph_BEC/baseline/CR-VAE/outputs/subject_bec_fast.npz \
  --generation-only
```

Run only 10-fold classification using the frozen archive:

```bash
python Graph_BEC/baseline/CR-VAE/run_abide_classifier.py \
  --bec-path Graph_BEC/baseline/CR-VAE/outputs/subject_bec.npz \
  --gpu-id auto \
  --classification-only
```

Use `--regenerate-bec` to discard the logical checkpoint and regenerate BECs
from subject 1. Classification outputs are saved as `metrics.csv`,
`metrics.json`, and `summary.json` in the CR-VAE output directory.

## ABIDE-II

ABIDE-II is supported through the shared loader. Its default input layout is
`dataset/ABIDE-II/cpac/filt_noglobal/*_rois_aal.1D`, with labels matched from
the ABIDE-II phenotype records. Use a dataset-specific archive and output
directory:

```bash
python Graph_BEC/baseline/CR-VAE/run_abide_classifier.py \
  --dataset abide_ii \
  --data-root dataset/ABIDE-II \
  --gpu-id auto \
  --crvae-context 20 \
  --crvae-hidden 64 \
  --crvae-max-iter 500 \
  --crvae-batch-size 256 \
  --crvae-lr 5e-2 \
  --crvae-lambda 0.1 \
  --n-splits 10 \
  --classifier-epochs 100 \
  --classifier-patience 20 \
  --output-dir Graph_BEC/baseline/CR-VAE/outputs/abide_ii \
  --bec-path Graph_BEC/baseline/CR-VAE/outputs/abide_ii/subject_bec_abide_ii.npz \
  --regenerate-bec
```

CR-VAE generation does not use neural-network `epoch`; `--crvae-max-iter`
controls the number of Phase-I optimization iterations for each subject. For
ABIDE-II, use `100` for a smoke test, `300` as a fast pilot, and `500` as the
recommended starting point for the full BEC archive. Compare `300`, `500`, and
`1000` on the same subjects before selecting the final value. Because the
runner fits one model per subject, runtime grows approximately linearly with
`--crvae-max-iter`.

The classifier's `--classifier-epochs` is a separate parameter and only
controls the downstream 10-fold evaluation; `100` with patience `20` remains a
reasonable default.

Quick smoke test:

```bash
python Graph_BEC/baseline/CR-VAE/run_abide_classifier.py \
  --dataset abide_ii \
  --data-root dataset/ABIDE-II \
  --gpu-id auto \
  --max-subjects 2 \
  --crvae-max-iter 10 \
  --crvae-batch-size 64 \
  --generation-only \
  --output-dir Graph_BEC/baseline/CR-VAE/outputs/smoke_abide_ii \
  --bec-path Graph_BEC/baseline/CR-VAE/outputs/smoke_abide_ii/subject_bec_abide_ii.npz \
  --regenerate-bec
```
