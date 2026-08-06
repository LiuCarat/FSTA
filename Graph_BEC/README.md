# FSTA-Graph-BEC

```text
ROI time series -> FSTA -> initial directed BEC
                              + AGE / SEX / FIQ patient graph
                              -> gated BEC refinement
                              -> group statistics + frozen BrainNetCNN probe
```

## Simplified layout

- `FSTA_Graph_BEC.py`: experiment entry.
- `data.py`: dataset loading, windows, preprocessing, splits, deterministic seeds.
- `phenotype.py`: phenotype loading and train-only patient graph.
- `model/pgr_bec_static.py`: Static subject-specific edge gate.
- `downstream/`: BrainNetCNN and evaluation.
- `analysis/`: BEC and edge statistics.

## Reproducible raw / BEC comparison

The same fold uses a stage-local seed. Original and refined BrainNetCNN probes
also start from the same initialization and batch order. Threshold selection
uses validation labels, never test labels.

Historical `subject_bec.npz` archives were extracted from the final FSTA epoch,
so raw mode defaults to `--fsta-checkpoint final`. Use `best` only when creating
a new archive and compare it only with another `best` run.

Raw mode prints `raw-vs-archive BEC` when `--bec-path` contains the same subjects.
Only when `max_abs` is approximately zero should raw and BEC downstream results
be expected to match exactly.

```bash
python Graph_BEC/FSTA_Graph_BEC.py --input-mode raw --seed 42
python Graph_BEC/FSTA_Graph_BEC.py --input-mode bec --seed 42
```

## Refinement variants

- `--refiner-mode static`: new subject-specific gate from `[A, N, |N-A|]`.

## Recommended robust comparison

Do not select the classifier seed that gives the largest refined AUC. Run
paired repeats: original and refined BEC use exactly the same classifier seed
in every repeat, and report the mean paired AUC change.
