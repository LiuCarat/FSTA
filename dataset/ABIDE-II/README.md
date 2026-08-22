# ABIDE-II preprocessing

## 1. Build the phenotype table

The current phenotype candidate list contains 677 subjects with complete `DX`, `Age`, `Sex`, `FIQ`, and `PIQ`:

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  dataset/ABIDE-II/prepare_phenotype.py \
  --input ABIDEII/phenotype/ABIDEII_phenotype_merged.csv \
  --eligible ABIDEII/phenotype/phenotype_eligible_subjects.csv \
  --output dataset/ABIDE-II/Phenotypic_Processing.csv \
  --require-fiq-piq
```

The output is an ABIDE-I-like canonical table with `SITE_ID`, `SUB_ID`, `FILE_ID`, `DX_GROUP`, `AGE_AT_SCAN`, `SEX`, `FIQ`, `VIQ`, and `PIQ`.

## 2. Create one BIDS view

The downloader stores data under site directories. fMRIPrep expects one BIDS root, so create a symlink view:

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  dataset/ABIDE-II/prepare_bids_layout.py \
  --raw-root ABIDEII/raw \
  --bids-root ABIDEII/bids
```

This does not duplicate the NIfTI files. It creates subject symlinks and writes `abideii_bids_manifest.tsv`.

## 3. Run fMRIPrep with Docker

Set the FreeSurfer license path first:

```bash
export FS_LICENSE="$HOME/.freesurfer.txt"
```

Then run:

```bash
dataset/ABIDE-II/run_fmriprep.sh ABIDEII/bids ABIDEII/derivatives/fmriprep ABIDEII/work
```

The script requests `MNI152NLin6Asym`, which is needed for a common AAL atlas space. fMRIPrep is resumable; rerunning the command reuses completed subjects.

## 4. Extract CPAC-like AAL90 `.1D`

The extraction is not performed directly on the fMRIPrep BOLD. It implements the
ABIDE-I `cpac / filt_noglobal` intent using fMRIPrep outputs:

- 24-parameter motion nuisance regression when the corresponding fMRIPrep columns exist;
- white-matter and CSF nuisance regression;
- no global-signal regression;
- FD scrubbing at `0.2 mm` by default;
- band-pass filtering from `0.01` to `0.1 Hz` by default;
- linear detrending and no post-extraction z-score;
- AAL116 extraction followed by retaining the first 90 ROI columns.

The exact CPAC implementation and fMRIPrep confound definitions are not byte-for-byte
identical, but this keeps the same `filt_noglobal` preprocessing intent. The strategy
is implemented in `extract_aal90.py` and should be kept fixed for all ABIDE-II subjects.

```bash
/data/users/liulin/miniconda3/envs/default/bin/python \
  dataset/ABIDE-II/extract_aal90.py \
  --fmriprep-root ABIDEII/derivatives/fmriprep \
  --atlas /path/to/AAL116_MNI.nii.gz \
  --manifest ABIDEII/bids/abideii_bids_manifest.tsv \
  --output-root dataset/ABIDE-II/AAL_TCs_filtfix
```

Optional parameters:

```bash
--fd-threshold 0.2 \
--low-cutoff 0.01 \
--high-cutoff 0.1 \
--min-timepoints 78
```

The output is:

```text
dataset/ABIDE-II/AAL_TCs_filtfix/{SITE}/{SUB_ID}/{SUB_ID}_aal_TCs.1D
```

Each file should have shape `[time_points_after_scrubbing, 90]`. Subjects with fewer than
78 remaining time points are skipped by default. The script prints the number of censored
frames for every subject.
