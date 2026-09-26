# Population-Regularized Learning of Individualized Brain Effective Connectivity (PR-EC)

## Introduction

PR-EC estimates individualized brain effective connectivity (EC) from resting-state functional magnetic resonance imaging (rs-fMRI) and further regularizes individual connectivity using population connectivity information.

The final output is saved as an .npz archive containing the estimated PR-EC matrices, labels, subject IDs, site IDs, fold IDs, ROI names, and representation information.

The resulting individualized PR-EC matrices can be directly used for downstream analyses, such as brain-network-based classification.

## Dataset

The current implementation supports the following rs-fMRI datasets:

ABIDE-I
ABIDE-II
ADHD-200

For ABIDE-I, the rs-fMRI data were preprocessed using the Configurable Pipeline for the Analysis of Connectomes (CPAC). The ROI time series parcellated with the AAL atlas are used as input to PR-EC.

For ABIDE-II and ADHD-200, the imaging data were organized according to the Brain Imaging Data Structure (BIDS) and preprocessed using fMRIPrep. The preprocessed rs-fMRI data were subsequently parcellated using the AAL atlas to obtain ROI-level time series as input to PR-EC.

Raw imaging data and generated outputs are not included in this anonymous repository.

## Data Layout

Run all commands from the repository root. The expected input data structure is:

```
dataset/
├── ABIDE-I/
│   ├── cpac/filt_noglobal/rois_aal/
│   └── Phenotypic_Processing.csv
├── ABIDE-II/
│   ├── cpac/filt_noglobal/rois_aal/
│   └── Phenotypic_Processing.csv
└── ADHD200/
    ├── cpac/filt_noglobal/rois_aal/
    └── Phenotypic_Processing.csv
```

The directories above contain the ROI-level time series used by the current implementation. Dataset paths can be adjusted according to the local environment.

## Quick Start

Run the experiments from the repository root.

For **ABIDE-I**:

```
python PR-EC/main_abide_i.py
```

For **ABIDE-II**:

```
python PR-EC/main_abide_ii.py
```

For **ADHD-200**:

```
python PR-EC/main_adhd200.py
```

GPU execution can be specified using:

```
python PR-EC/main_abide_i.py --gpu-id 0
```

For CPU execution:

```
python PR-EC/main_abide_i.py --gpu-id cpu
```

Dataset-specific parameters and default settings are provided in the corresponding main files.

## Output

The output path of the final PR-EC archive can be configured using `--pr-ec-path`. After execution, the path is also printed as:

```
pr_ec_path=...
```

The resulting `.npz` archive contains the final `pr_ec` matrices together with the corresponding labels, subject IDs, site IDs, fold IDs, ROI names, and the `PR-EC` representation label.

The `pr_ec` matrices represent individualized brain effective connectivity estimated by PR-EC and can be directly used as connectivity features for downstream analyses.

## Dependencies

The experiments were conducted using Python 3.11.15 with the following main dependencies:
```
numpy==2.4.4
scikit-learn==1.8.0
torch==2.11.0
```
The GPU experiments were conducted with CUDA 12.8.

To install the required Python packages, run:
```
pip install -r requirements.txt
```
For GPU execution, please install a PyTorch build compatible with the local CUDA environment. CPU execution is also supported.
