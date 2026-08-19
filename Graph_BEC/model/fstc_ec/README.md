# Self-contained FSTC-EC

This directory contains the factorized FSTA-EC model, its FSTA attention
components, ABIDE-I ROI loader, window sampler, runtime helpers, losses, and
training entry point. It does not import `Graph_BEC.baseline.FSTA_EC`.

## Train inside this repository

```bash
python -m Graph_BEC.model.fstc_ec.train --checkpoint-selection best
```

## Export and train elsewhere

Copy this directory as a Python package, then run it from its parent directory:

```bash
python -m fstc_ec.train \
  --data-root /path/to/ABIDE-I \
  --checkpoint /path/to/factorized_fsta_ec.pt \
  --output /path/to/factorized_fsta_ec_subject_ec.npz \
  --checkpoint-selection best
```

Required Python packages are `torch` and `numpy`. The data loader expects the
ABIDE-I CPAC `filt_noglobal` AAL ROI layout used by the original trainer.
