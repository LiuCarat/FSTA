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
