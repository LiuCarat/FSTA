# VarCoNet baseline

This directory contains the minimal VarCoNet encoder used as a
functional-connectivity baseline. The runnable entry point is
`run_varconet.py`; unrelated paper experiments for HCP, BolT, competing
methods, atlas parcellation and fingerprinting are intentionally omitted.

The entry point reads the repository's standard ROI files under
`cpac/filt_noglobal/`, uses the shared dataset phenotype loader, and evaluates
with the shared fold-local Directed BrainNetCNN classifier. The encoder is
trained on each training fold using two randomly cropped/noisy views and the
original VarCoNet InfoNCE objective.

## ABIDE-I

```bash
python Graph_BEC/baseline/VarCoNet/run_varconet.py \
  --dataset abide --gpu-id auto
```

## ABIDE-II

```bash
python Graph_BEC/baseline/VarCoNet/run_varconet.py \
  --dataset abide_ii --gpu-id auto
```

## ADHD200

ADHD200 uses `dataset/ADHD200/Phenotypic_Processing.csv` and
`dataset/ADHD200/cpac/filt_noglobal/*_rois_aal.1D` by default. `DX=0` is the
control group and `DX=1/2/3` is mapped to the patient group. The shared loader
keeps the first 90 of the 116 source ROIs, matching VarCoNet's input size.
Representations and metrics are written to `outputs/adhd200/` by default.

Run a short generation smoke test:

```bash
python Graph_BEC/baseline/VarCoNet/run_varconet.py \
  --dataset adhd200 \
  --data-root ./dataset/ADHD200 \
  --max-subjects 2 \
  --encoder-epochs 2 \
  --generation-only \
  --gpu-id cpu \
  --output-dir Graph_BEC/baseline/VarCoNet/outputs/smoke_adhd200
```

Useful smoke-test options are `--max-subjects 20 --n-splits 2
--encoder-epochs 2 --classifier-epochs 2 --gpu-id cpu`.
Representations and metrics are written to `outputs/<dataset>/`.
