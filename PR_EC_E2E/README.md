# End-to-End Neighbor Aggregation control

This is an **additional supervised control**, not a supervised replacement for
PR-EC. Use the independent entry point below. The copied legacy `main_*.py`
entry points still dispatch to `PR_EC` and are **not** E2E experiment commands.

## Protocol

- Read the original `PR_EC/outputs/<dataset>/summary.json` configuration and
  cached Initial EC. Never train STF-EC or write to the original outputs.
- Reuse the original stratified ten-fold train/validation/test splitter,
  fold-local EC/phenotype standardization and PH+FC fusion graph. Verify test
  membership, subject ordering and labels against the original OOF EC archive.
- Retain exactly the nonzero Top-K candidate support of the original graph;
  MPR weights select candidates but **are not used as attention weights**.
  All neighbor ECs come exclusively from the current training fold. Mask padded
  candidates and exclude self-neighbors for training subjects.
- Summarize each directed EC with incoming/outgoing means and RMS (4N values).
  Separate bias-free query/key projections (32 dimensions, tanh) learn scaled
  dot-product softmax attention over the K candidates. The value is the full
  Initial EC matrix. Fuse as `self + sigmoid(gate) * (attention_value - self)`;
  the scalar gate starts at 0 (mixing coefficient 0.5).
- Jointly fit this small aggregator and the unchanged DirectedBrainNetCNN with
  BCE. Preserve the source classifier's epochs, patience, batch size, Adam LR,
  weight decay (1e-4), dropout (0.3), seed, validation BCE early stopping, and
  validation Youden threshold. Initialize BrainNetCNN before the aggregator so
  its initial parameters match the source classifier for the same seed.
- No QC data, QC sensitivity maps, pseudo-targets, perturbations or test-driven
  model selection are used. The frozen Initial EC is shared with PR-EC; this
  does not retroactively change its original encoder pretraining protocol.

## Run

Use a Python environment containing PyTorch, NumPy, SciPy and scikit-learn.
From the repository root, complete the real single-fold check first:

```bash
python PR_EC_E2E/run_e2e.py --datasets abide-i --folds 1
```

Inspect `outputs/neighbor_attention_v1/abide-i/fold_01/audit_repeat_1.json`:
query/key/gate and classifier must receive finite nonzero classification
gradients, all must update, and attention weights must change after Adam.
The audit also checks split disjointness, train-only retrieval, self-exclusion,
candidate support and that changing held-out features cannot affect training
standardization or training neighbors. Fitting has no test-data arguments.

Only after passing this check, run all three datasets (reusing completed folds):

```bash
python PR_EC_E2E/run_e2e.py --resume
```

After all thirty folds complete, independently recompute the metrics and check
the saved neighbor sets, split coverage, source hashes and gradient audits:

```bash
python PR_EC_E2E/verify_results.py
```

This also writes a combined three-dataset comparison, `verification.json`,
and subject-aligned `oof_predictions.npz` files without changing fold results.

`--gpu-id cpu` selects CPU; `--threads` only controls CPU execution, not the
training protocol. Run IDs use distinct `--output-root` directories; existing
results cannot be overwritten. Resume requires identical source hashes,
implementation, hyperparameters and runtime settings. Interrupted incomplete
fold directories must be preserved or moved before retrying; they are never
silently overwritten.

## Outputs

Each dataset directory contains a provenance manifest, per-fold checkpoints,
training/gradient/leakage audits, neighbor indices/masks and original graph
weights, and test subject IDs/labels/probabilities per classifier repeat.
`fold_results.csv` includes paired `E2E - Full PR-EC` AUC differences.
`summary.json` and `comparison.md` report ACC, SEN, AUC, Precision and F1 as
fold mean ± population SD (`ddof=0`, matching the original reporting rule).
Partial runs are explicitly marked as fewer than ten folds.

Initial EC, MPR-only and Full PR-EC controls use **archived per-fold metrics**
(`original`, `refined`, `qc_refined`), not OOF ECs retrained as if they were
fold-local train data. The source experiment did not save classifier test
probabilities; its original AUCs suffice for fold-paired differences, but not
for a subject-level paired DeLong test. No such test is claimed here.
For classifier repeats, metrics are averaged across repeats, matching PR-EC;
saved test probabilities retain the repeat axis rather than silently ensembling.
