# ABIDE-II preprocessing

This directory keeps one preprocessing path:

```text
ABIDE-II raw BIDS -> fMRIPrep -> 24P+aCompCor denoising -> AAL90 .1D
```

The temporal policy is fixed: linear detrending, 0.01–0.1 Hz band-pass filtering,
no global signal regression, and no frame deletion. FD and DVARS are recorded as
subject-level QC only.

## 1. Prepare the BIDS view and phenotype

The downloader stores each site separately. Merge those subjects into one symlinked
BIDS view and create the canonical Graph-BEC phenotype table:

```bash
python dataset/ABIDE-II/prepare_abide2.py
```

Outputs:

```text
ABIDEII/bids/
ABIDEII/bids/abideii_bids_manifest.tsv
dataset/ABIDE-II/ABIDEII_phenotype_graphbec.csv
```

The phenotype converter maps ABIDE-II `dx_group=1` (ASD) to Graph-BEC `DX_GROUP=2`
and `dx_group=2` (control) to `DX_GROUP=1`.

## 2. Run fMRIPrep

Set a valid FreeSurfer license, then run:

```bash
export FS_LICENSE="$HOME/.freesurfer.txt"
dataset/ABIDE-II/run_fmriprep.sh \
  ABIDEII/bids \
  ABIDEII/derivatives/fmriprep \
  ABIDEII/work
```

The wrapper requests `MNI152NLin6Asym`, uses `--fs-no-reconall`, and is resumable.

## 3. Extract denoised AAL90 time series

Provide an AAL116 label image in the same MNI template family as the fMRIPrep output:

```bash
python dataset/ABIDE-II/extract_aal90.py \
  --fmriprep-root ABIDEII/derivatives/fmriprep \
  --atlas /path/to/AAL116_MNI.nii.gz \
  --manifest ABIDEII/bids/abideii_bids_manifest.tsv \
  --phenotype dataset/ABIDE-II/ABIDEII_phenotype_graphbec.csv
```

The extractor builds Friston-24 motion regressors, adds the first five
`a_comp_cor_*` components when available, and performs nuisance regression,
linear detrending, and 0.01–0.1 Hz filtering. It does not include global signal
and never deletes frames. AAL labels are resampled with nearest-neighbor
interpolation; labels 1–90 are retained. FD/DVARS metrics are written into the
phenotype table as subject-level QC.

Output:

```text
dataset/ABIDE-II/cpac/filt_noglobal/sub-29008_rois_aal.1D
dataset/ABIDE-II/cpac/filt_noglobal/sub-29007_rois_aal.1D
...
```

Each `.1D` file is `[T, 90]`. Graph-BEC accepts both these AAL90 files and legacy
ABIDE-I AAL116 files. The old `prepare_bids_layout.py` and `prepare_phenotype.py`
scripts were merged into `prepare_abide2.py`.

## 4. Download existing ABIDE-fMRIPrep derivatives without raw MRI

`download_abide_fmriprep.py` uses DataLad/git-annex and intentionally retrieves only
files needed for the later denoising and ROI extraction:

```text
resting preprocessed BOLD in MNI152NLin2009cAsym
confounds_timeseries.tsv
confounds_timeseries.json
BOLD JSON
brain mask
```

It does not retrieve raw BOLD, T1w, fieldmaps, FreeSurfer outputs, CIFTI/GIfTI, or
HTML reports. Install `datalad` and `git-annex` first:

```bash
conda install -c conda-forge datalad git-annex
```

Test one BNI subject first. `29008` is automatically converted to the repository ID
`sub-v2s0x29008`:

```bash
python dataset/ABIDE-II/download_abide_fmriprep.py \
  --site BNI_1 \
  --output-root ABIDEII/derivatives/abide-fmriprep \
  --subject 29008
```

Download all selected subjects from an existing `SITE_ID,subject` CSV:

```bash
python dataset/ABIDE-II/download_abide_fmriprep.py \
  --site BNI_1 \
  --output-root ABIDEII/derivatives/abide-fmriprep \
  --subject-file ABIDEII/phenotype/download_subjects.csv
```

Download all subjects in a site repository:

```bash
python dataset/ABIDE-II/download_abide_fmriprep.py \
  --site BNI_1 \
  --output-root ABIDEII/derivatives/abide-fmriprep \
  --all-subjects
```

Check the selected file list before retrieving content:

```bash
python dataset/ABIDE-II/download_abide_fmriprep.py \
  --site BNI_1 \
  --output-root ABIDEII/derivatives/abide-fmriprep \
  --subject 29008 \
  --dry-run
```

The downloaded repository keeps the DataLad layout. Later, point
`extract_aal90.py` at the repository and use the matching template space:

```bash
python dataset/ABIDE-II/extract_aal90.py \
  --fmriprep-root ABIDEII/derivatives/abide-fmriprep/v2s0 \
  --atlas /path/to/AAL116_MNI152NLin2009cAsym.nii.gz \
  --manifest ABIDEII/bids/abideii_bids_manifest.tsv \
  --phenotype dataset/ABIDE-II/ABIDEII_phenotype_graphbec.csv \
  --space MNI152NLin2009cAsym
```

## 5. Low-memory one-subject-at-a-time fMRIPrep

For the downloaded site-separated raw data, use
`run_fmriprep_onebyone.sh`. It does not require a permanent `bids` directory:

```bash
bash dataset/ABIDE-II/run_fmriprep_onebyone.sh sub-29563
```

Without a subject argument it scans all raw sites and processes only subjects with
exactly one BOLD and one T1w:

```bash
bash dataset/ABIDE-II/run_fmriprep_onebyone.sh
```

For each subject the script creates a temporary one-subject BIDS view, runs Docker
with one process and one OpenMP thread, retains only the following derivatives under
`dataset/ABIDE-II/fmriprep/`, and deletes the temporary BIDS/work/output directory:

```text
*space-MNI152NLin6Asym_desc-preproc_bold.nii.gz
*desc-confounds_timeseries.tsv
*desc-confounds_timeseries.json
*space-MNI152NLin6Asym_desc-brain_mask.nii.gz
*_bold.json
```

It skips a subject if its retained preprocessed BOLD already exists, so rerunning
resumes at subject granularity. The default memory cap is 12 GB and can be changed:

```bash
FMRIPREP_MEM_MB=16000 FMRIPREP_NPROCS=2 \
  bash dataset/ABIDE-II/run_fmriprep_onebyone.sh sub-29563
```

The temporary root can be placed on a local scratch disk:

```bash
FMRIPREP_TMP_ROOT=/path/to/local/scratch/abide2-fmriprep \
  bash dataset/ABIDE-II/run_fmriprep_onebyone.sh
```

The retained derivatives still need `extract_aal90.py` for the final temporal
pipeline: Friston-24, aCompCor, linear detrending, 0.01–0.1 Hz filtering, no GSR,
and AAL90 extraction. fMRIPrep alone does not produce the final `.1D` files.
