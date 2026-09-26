# Population-Regularized Learning of Individualized Brain Effective Connectivity (PR-EC)

## Introduction

PR-EC estimates individualized brain effective connectivity (EC) from resting-state functional magnetic resonance imaging (rs-fMRI) and further regularizes individual connectivity using population connectivity information.

The resulting individualized PR-EC matrices can be directly used for downstream analyses, such as brain-network-based classification.

## Dataset

The current implementation supports the following rs-fMRI datasets:

- **ABIDE-I:** [official ABIDE-I page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
- **ABIDE-II:** [official ABIDE-II page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html)
- **ADHD-200:** [official ADHD-200 page](https://fcon_1000.projects.nitrc.org/indi/adhd200/)

**The original datasets are not included in this repository. Please download them separately from their official sources using the links above.**

For datasets that require additional preprocessing, such as ABIDE-II and ADHD-200, please follow the preprocessing procedure described in the Experimental Details section of the paper. The imaging data should be organized according to the Brain Imaging Data Structure (BIDS) and preprocessed using fMRIPrep. The preprocessed rs-fMRI data should then be parcellated using the AAL atlas to obtain ROI-level time series as input to PR-EC.

For ABIDE-I, the implementation uses the ROI-level time series derived from the CPAC preprocessing pipeline with the AAL atlas.

Please ensure that the resulting ROI-level time series and phenotypic files follow the directory structure described below before running the experiments.

## Data Layout

Run all commands from the repository root. The expected input data structure is:

```
dataset/
├── ABIDE-I/
│   ├── cpac/filt_noglobal/*_rois_aal.1D
│   └── Phenotypic_Processing.csv
├── ABIDE-II/
│   ├── cpac/filt_noglobal/*_rois_aal.1D
│   └── Phenotypic_Processing.csv
└── ADHD200/
    ├── cpac/filt_noglobal/*_rois_aal.1D
    └── Phenotypic_Processing.csv
```

For ABIDE-I and ABIDE-II, the phenotype file must contain the subject identifier, diagnosis (`DX_GROUP`), and site identifier (`SITE_ID`). For ADHD-200, the default loader expects `ScanDir ID`, `DX`, `Site`, `Gender`, `Age`, `Full4 IQ`, and `Handedness` columns. Dataset paths can be adjusted with `--data-root` and `--phenotype-csv`.

## Quick Start

Run the experiments from the repository root.

For **ABIDE-I**:

```
python pr_ec/main_abide_i.py
```

For **ABIDE-II**:

```
python pr_ec/main_abide_ii.py
```

For **ADHD-200**:

```
python pr_ec/main_adhd200.py
```

GPU execution can be specified using:

```
python pr_ec/main_abide_i.py --gpu-id 0
```

For CPU execution:

```
python pr_ec/main_abide_i.py --gpu-id cpu
```

Dataset-specific parameters and default settings are provided in the corresponding main files.

## Output

The output path of the final PR-EC archive can be configured using `--pr-ec-path`. By default, outputs are written under `pr_ec/outputs/` and the selected archive path is printed as:

```
pr_ec_path=...
```

`.npz` archive containing the QSR-refined EC matrix, labels, subject IDs, site IDs, fold IDs, ROI names, and representation information.

`experiment_summary.csv` stores fold-level downstream classification metrics.

`summary.json` stores the experiment configuration, Individual-EC training metrics, fold results, and summary statistics.
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
