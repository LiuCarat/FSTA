# DTS-EC

**DTS-EC** combines spectral filtering, temporal dynamics modeling, and directed effective connectivity estimation.
It uses a learnable spectral filter, a shared ROI-wise temporal dynamics mixer,
a directed EC adapter, and EC-controlled signal-flow reconstruction.

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

## Capacity controls

- `--hidden-dim`: Fourier and Temporal Mixer channel width (default `32`).
- `--temporal-dim`: temporal representation width before the EC adapter;
  defaults to `--hidden-dim`.
- `--ec-dim`: source/target and signal-flow width (default `32`).
- `--decoder-hidden-dim`: flow-decoder MLP width; use `0` for the original
  linear `ec_dim → 1` decoder, or for example `32` for `ec_dim → 32 → 1`.
