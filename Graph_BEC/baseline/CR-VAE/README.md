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

Generation is checkpointed every 10 subjects by default. Re-running the same
command resumes an incomplete archive or reuses a complete one.

For a much faster first result, use `--fast`. It keeps the CR-VAE model and
the requested `context=20`, `hidden=64`, `lr=0.05`, and `lambda=0.1`, but uses
100 iterations and batch size 64. The archive is checkpointed every 10
subjects by default, which avoids 871 compressed-file writes. Use
`--checkpoint-every 1` if you prefer maximum restart safety.

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
