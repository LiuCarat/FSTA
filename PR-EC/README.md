# PR-EC

This directory contains the public PR-EC pipeline:

```text
raw fMRI
→ Individual-EC
→ MPR patient-similarity graph
→ QSR Refinement
→ PR-EC
→ downstream classification
```

The pipeline does not save the intermediate Individual-EC archive. It keeps
Individual-EC in memory and writes only the final PR-EC archive.

## Data layout

Run the commands from the repository root. The expected dataset locations are:

```text
dataset/
├── ABIDE-I/
│   ├── cpac/filt_noglobal/rois_aal/
│   └── Phenotypic_Processing_filled.csv
├── ABIDE-II/
│   ├── cpac/filt_noglobal/rois_aal/
│   └── Phenotypic_Processing.csv
└── ADHD200/
    ├── cleaned/AAL_TCs_filtfix/
    └── adhd200_preprocessed_phenotypics.tsv
```

Raw datasets and generated outputs are not included in this repository.

## Run

```bash
python PR-EC/main_abide_i.py
python PR-EC/main_abide_ii.py
python PR-EC/main_adhd200.py
```

Use `--gpu-id cpu` for CPU execution or provide a CUDA device identifier such
as `--gpu-id 0`. The main files contain the dataset-specific defaults.

The final archive path is configured by `--pr-ec-path` and is also printed as:

```text
PR_EC_PATH=...
```

The archive contains the final `pr_ec` matrices, labels, subject IDs, site IDs,
fold IDs, ROI names, and the `PR-EC` representation label.

## Dependencies

The implementation requires Python 3.10 or newer and the following packages:

```text
numpy
scikit-learn
torch
```

Install a PyTorch build appropriate for the target CPU or CUDA environment.
