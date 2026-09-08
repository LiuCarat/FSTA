# FSTA-EC baseline

This baseline contains its own FSTA-EC implementation and does not import the
implementation under `Graph_BEC/model/fsta_ec`. It trains one FSTA model on
the selected subjects, extracts one Original-BEC matrix for every subject,
saves a reusable `.npz` archive, and evaluates the archive with the shared
downstream classifier.

## Files

- `run_fsta_ec_baseline.py`: archive generation and classification entry point.
- `arguments.py`: baseline-specific model arguments.
- `fsta_training.py`, `utils/utils.py`: training and BEC extraction.
- `model/`: independent FSTA-EC model implementation in the original layout.
- `requirements.txt`: baseline dependencies.
- `__init__.py`: package marker.

The old dataset-specific training scripts are not kept here; the baseline
runner uses the shared Graph-BEC data loader and classifier only. The training
objective is the original reconstruction loss plus the `alpha_sp` sparsity
term; there is no entropy loss parameter in this baseline.

## Install

```bash
python -m pip install -r Graph_BEC/baseline/FSTA-EC/requirements.txt
```

## Run

Generate subject-level BECs and classify them in one command:

```bash
python Graph_BEC/baseline/FSTA-EC/run_fsta_ec_baseline.py \
  --dataset abide \
  --data-root ./dataset/ABIDE-I \
  --gpu-id auto
```

For a short smoke run:

```bash
python Graph_BEC/baseline/FSTA-EC/run_fsta_ec_baseline.py \
  --dataset abide \
  --data-root ./dataset/ABIDE-I \
  --epochs 2 \
  --max-subjects 2 \
  --generation-only
```

Use `--generation-only` to only create the archive. Use
`--classification-only` with `--bec-path` to classify an existing archive.
The default archive is
`Graph_BEC/baseline/FSTA-EC/outputs/subject_fsta_ec_bec_<dataset>.npz`.

The archive contains `bec`, labels, subject IDs, site IDs, ROI names, and
subject reconstruction errors. The method reported in comparison tables
should be `FSTA-EC` or `FSTA-EC Original-BEC`.

## ABIDE-II

The same runner supports ABIDE-II through the existing ABIDE-II profile:

```bash
python Graph_BEC/baseline/FSTA-EC/run_fsta_ec_baseline.py \
  --dataset abide_ii \
  --data-root ./dataset/ABIDE-II \
  --gpu-id auto
```

It reads `dataset/ABIDE-II/Phenotypic_Processing.csv` and
`dataset/ABIDE-II/cpac/filt_noglobal/*_rois_aal.1D` by default, and writes the
archive and metrics to `Graph_BEC/baseline/FSTA-EC/outputs/abide_ii/` by
default.

## ADHD200

ADHD200 is supported through the shared Graph-BEC loader. The default input
files are `dataset/ADHD200/Phenotypic_Processing.csv` and
`dataset/ADHD200/cpac/filt_noglobal/*_rois_aal.1D`. `DX=0` is used as the
control group, while `DX=1/2/3` is mapped to the patient group. The loader
reads 116 source ROIs and keeps the first 90, matching FSTA-EC's model
configuration. ADHD200 archives and metrics default to the separate
`Graph_BEC/baseline/FSTA-EC/outputs/adhd200/` directory.

ADHD200 has subjects with different scan lengths. The runner therefore uses a
default `window-length=75` for ADHD200 (all currently prepared subjects have
at least 76 frames), while ABIDE keeps its original default of `80`. You can
override this with `--window-length` if needed.

Run the full baseline with:

```bash
python Graph_BEC/baseline/FSTA-EC/run_fsta_ec_baseline.py \
  --dataset adhd200 \
  --data-root ./dataset/ADHD200 \
  --gpu-id auto \
  --epochs 51
```

For a quick generation smoke test:

```bash
python Graph_BEC/baseline/FSTA-EC/run_fsta_ec_baseline.py \
  --dataset adhd200 \
  --data-root ./dataset/ADHD200 \
  --max-subjects 2 \
  --epochs 2 \
  --generation-only \
  --output-dir Graph_BEC/baseline/FSTA-EC/outputs/smoke_adhd200 \
  --bec-path Graph_BEC/baseline/FSTA-EC/outputs/smoke_adhd200/subject_fsta_ec_bec_adhd200.npz \
  --regenerate-bec
```
