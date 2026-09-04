# VarCoNet baseline

This directory contains the minimal VarCoNet encoder used as a
functional-connectivity baseline. The runnable entry point is
`run_varconet.py`; unrelated paper experiments for HCP, BolT, competing
methods, atlas parcellation and fingerprinting are intentionally omitted.

The entry point reads the repository's standard ABIDE ROI files under
`cpac/filt_noglobal/`, uses the shared ABIDE phenotype loader, and evaluates
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

Useful smoke-test options are `--max-subjects 20 --n-splits 2
--encoder-epochs 2 --classifier-epochs 2 --gpu-id cpu`.
Representations and metrics are written to `outputs/<dataset>/`.
