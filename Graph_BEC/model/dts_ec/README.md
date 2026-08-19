# DTS-EC

**DTS-EC** means **Decoupled Temporal-Spatial Effective Connectivity**. It
uses a Fourier-temporal encoder, a directed spatial EC adapter, and
EC-controlled signal-flow reconstruction. The Fourier encoder is adapted from
FSTA; DTS-EC does not import the FSTA-EC baseline implementation.

## Train inside this repository

```bash
python -m Graph_BEC.model.dts_ec.train --checkpoint-selection best
```

## Export and train elsewhere

Copy this directory as a Python package, then run it from its parent directory:

```bash
python -m dts_ec.train \
  --data-root /path/to/ABIDE-I \
  --checkpoint /path/to/dts_ec.pt \
  --output /path/to/dts_ec_subject_ec.npz \
  --checkpoint-selection best
```

Required Python packages are `torch` and `numpy`. The data loader expects the
ABIDE-I CPAC `filt_noglobal` AAL ROI layout used by the original trainer.
